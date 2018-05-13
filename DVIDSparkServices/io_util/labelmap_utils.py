import os
import csv
import warnings
import subprocess

import numpy as np
import pandas as pd

from neuclease.merge_table import load_edge_csv

import logging
logger = logging.getLogger(__name__)

try:
    with warnings.catch_warnings():
        # Importing graph_tool results in warnings about duplicate C++/Python conversion functions.
        # Ignore those warnings
        warnings.filterwarnings("ignore", "to-Python converter")
        import graph_tool as gt

    _graph_tool_available = True
except ImportError:
    _graph_tool_available = False
    

from ..util import Timer

LabelMapSchema = \
{
    "description": "A label mapping file to apply to segmentation after reading or before writing.",
    "type": "object",
    "default": {},
    "properties": {
        "file": {
            "description": "Path to a file of labelmap data",
            "type": "string" ,
            "default": ""
        },
        "file-type": {
            "description": "Path to a CSV file containing supervoxel graph or mapping\n",
            "type": "string",
            "enum": [
                        # CSV file containing the direct mapping.  Columns: orig,new
                        "label-to-body",
                        
                        # CSV file containing a list of merged edges between adjacent
                        # supervoxels within every body (but no inter-body edges).
                        # Rows are sv1,sv2
                        "body-rag",

                        # CSV file containing a list of label merges.
                        # Same format as a body-rag file, but with only a subset of the edges.
                        # (A label-to-body mapping is derived from this, via connected components analysis.)
                        "equivalence-edges",
                        
                     "__invalid__"],
            "default": "__invalid__"
        },
        "apply-when": {
            "type": "string",
            "enum": ["reading", "writing", "reading-and-writing"],
            "default": "reading-and-writing"
        }
    }
}


def load_labelmap(labelmap_config, working_dir):
    """
    Load a labelmap file as specified in the given labelmap_config,
    which must conform to LabelMapSchema.
    
    If the labelmapfile exists on gbuckets, it will be downloaded first.
    If it is gzip-compressed, it will be unpacked.
    
    The final downloaded/uncompressed file will be saved into working_dir,
    and the final path will be overwritten in the labelmap_config.
    """
    path = labelmap_config["file"]

    # path is [gs://][/]path/to/file.csv[.gz]

    # If the file is in a gbucket, download it first (if necessary)
    if path.startswith('gs://'):
        filename = path.split('/')[-1]
        downloaded_path = working_dir + '/' + filename
        if not os.path.exists(downloaded_path):
            cmd = f'gsutil -q cp {path} {downloaded_path}'
            logger.info(cmd)
            subprocess.check_call(cmd, shell=True)
        path = downloaded_path

    if not labelmap_config["file"].startswith("/"):
        path = os.path.normpath( os.path.join(working_dir, labelmap_config["file"]) )

    # Now path is /path/to/file.csv[.gz]
    
    if not os.path.exists(path) and os.path.exists(path + '.gz'):
        path = path + '.gz'

    # If the file is compressed, decompress it
    if os.path.splitext(path)[1] == '.gz':
        uncompressed_path = path[:-3] # drop '.gz'
        if not os.path.exists(uncompressed_path):
            subprocess.check_call(f"gunzip {path}", shell=True)
            assert os.path.exists(uncompressed_path), \
                "Tried to uncompress the labelmap CSV file... where did it go?"
        path = uncompressed_path # drop '.gz'

    # Now path is /path/to/file.csv
    # Overwrite the final downloaded/upacked location
    labelmap_config['file'] = path

    # Mapping is only loaded into numpy once, on the driver
    if labelmap_config["file-type"] == "label-to-body":
        logger.info(f"Loading label-to-body mapping from {path}")
        with Timer("Loading mapping", logger):
            mapping_pairs = load_edge_csv(path)

    elif labelmap_config["file-type"] in ("equivalence-edges", "body-rag"):
        logger.info(f"Loading equivalence mapping from {path}")
        with Timer("Loading mapping", logger):
            mapping_pairs = segment_to_body_mapping_from_edge_csv(path)

        # Export mapping to disk in case anyone wants to view it later
        output_dir, basename = os.path.split(path)
        mapping_csv_path = f'{output_dir}/LABEL-TO-BODY-{basename}'
        if not os.path.exists(mapping_csv_path):
            with open(mapping_csv_path, 'w') as f:
                csv.writer(f).writerows(mapping_pairs)
    else:
        raise RuntimeError(f"Unknown labelmap file-type: {labelmap_config['file-type']}")

    return mapping_pairs


def groups_from_edges(edges):
    """
    The given list of edges [(node_a, node_b),(node_a, node_b),...] encode a graph.
    Find the connected components in the graph and return them as a dict:
    
    { group_id : [node_id, node_id, node_id] }
    
    ...where each group_id is the minimum node_id of the group.
    """
    import networkx as nx
    g = nx.Graph()
    g.add_edges_from(edges)
    
    groups = {}
    for segment_set in nx.connected_components(g):
        # According to Jeremy, group_id == the min segment_id of the group.
        groups[min(segment_set)] = list(sorted(segment_set))

    return groups


def segment_to_body_mapping_from_edge_csv(csv_path, output_csv_path=None):
    """
    Load and return the a segment-to-body mapping (a.k.a. an "equivalence_mapping" in Brainmaps terminology,
    from the given csv_path of equivalence edges (or complete merge graph).

    That is, compute the groups "connected components" of the graph,
    and map each node to its owning group.
    
    Each row represents an edge. For example:
    
        123,456
        123,789
        789,234
        
    The CSV file may optionally contain a header row.
    Also, it may contain more than two columns, but only the first two columns are used.
    
    Args:
        csv_path:
            Path to a csv file whose first two columns are edge pairs
        
        output_csv_path:
            (Optional.) If provided, also write the results to a CSV file.
        
    Returns:
        ndarray with two columns representing node and group

    Note: The returned array is NOT merely the parsed CSV.
          It has been transformed from equivalence edges to node mappings,
          via a connected components step.
    """
    edges = load_edge_csv(csv_path)
    mapping = mapping_from_edges(edges)
    
    if output_csv_path:
        equivalence_mapping_to_csv(mapping, output_csv_path)
        
    return mapping

def mapping_from_edges(edges, sort_by='body', as_series=False, remove_identities=False):
    """
    Compute the connected components of the graph defined by 'edges'
    (a 2-column ndarray representing edges between segments),
    and return the mapping of segment ID -> body ID,
    where body ID is defined as the minimum segment ID in each connected component.

    Args:
        
        sort_by: How to sort the rows of the resulting mapping.
                 Choices: 'segment', 'body'.

        as_series:
            If True, return a pandas.Series (where segment is the index, body is the data).
            If False, return a 2-column np.ndarray, with segment in the first column and body in the second column.

        remove_identities:
            If True, remove rows with identity mappings (leave them implicit)
    Returns:
        The computed mapping, as either an ndarray or Series, as requested via as_series.
    """
    assert sort_by in ('segment', 'body')

    if _graph_tool_available:
        mapping = _mapping_from_edges_gt(edges)
    else:
        mapping = _mapping_from_edges_nx(edges)

    if sort_by == 'body':
        mapping.sort_values(inplace=True)
    else:
        mapping.sort_index(inplace=True)

    if remove_identities:
        mapping = pd.DataFrame(mapping).query('sv != body')['body']

    if as_series:
        return mapping
    
    return np.array((mapping.index, mapping.values)).transpose()

def _mapping_from_edges_gt(edges):
    """
    Helper function for mapping_from_edges()
    
    Compute the connected components of the graph defined by 'edges',
    and return the mapping of segment ID -> body ID
    as an UNSORTED pandas.Series (where segment is the index).
    
    Note: This version uses graph-tool to compute the connected components, which is very fast.
          However, graph-tool is not installed by default due to licensing restrictions.
    """
    from graph_tool.topology import label_components

    g = gt.Graph(directed=False)
    sv_pmap = g.add_edge_list( edges, hashed=True )
    cc_pmap, _hist = label_components(g)
    
    df = pd.DataFrame({'sv': sv_pmap.get_array(),
                       'cc_index': cc_pmap.get_array()})
    
    # Convert component labels to body IDs (min segment ID in the component)
    cc_body_ids = df.groupby('cc_index').min()
    cc_body_ids.columns = ['body']

    mapped_cc_df = df.merge( cc_body_ids, how='inner', left_on='cc_index', right_index=True, copy=False )

    mapping = pd.Series(index=mapped_cc_df['sv'].values.astype(edges.dtype),
                        data=mapped_cc_df['body'].values.astype(edges.dtype))
    mapping.name = 'body'
    mapping.index.name = 'sv'

    return mapping

def _mapping_from_edges_nx(edges):
    """
    Compute the connected components of the graph defined by 'edges',
    and return the mapping of segment ID -> body ID
    as an UNSORTED pandas.Series (where segment is the index).
    
    Note: This version uses networkx to compute the connected components, which is SLOW.
          Computing the hemibrain mapping takes ~45 minutes.
    """
    segments = pd.unique(edges.reshape(-1))
    mapping = pd.Series(index=segments, data=edges.dtype.type(0))
    mapping.name = 'body'
    mapping.index.name = 'sv'

    import networkx as nx
    g = nx.Graph()
    g.add_edges_from(edges)
    
    for segment_set in nx.connected_components(g):
        mapping[segment_set] = min(segment_set)

    return mapping

def mapping_from_groups(groups):
    """
    Given a dict of { group_id: [node_a, node_b,...] },
    Return a reverse-mapping in the form of an ndarray:
        
        [[node_a, group_id],
         [node_b, group_id],
         [node_c, group_id],
         ...
        ]
    """
    element_count = sum(map(len, groups.values()))
    
    def generate():
        for group_id, members in groups.items():
            for member in members:
                yield member
                yield group_id

    mapping = np.fromiter( generate(), np.uint64, 2*element_count ).reshape(-1,2)
    return mapping


def equivalence_mapping_to_csv(mapping_pairs, output_path):
    if not os.path.exists(output_path):
        with open(output_path, 'w') as f:
            csv.writer(f).writerows(mapping_pairs)

def compare_mappings(old_mapping, new_mapping):
    """
    Compare two mappings and return the set of bodies that were added, dropped, or changed.
    
    Args:
        old_mapping: Ndarray of shape (N,2), where first column is SV and second column is body.
        new_mapping: Ditto.
        
    Returns:
        (dropped_bodies, added_bodies, changed_size_bodies, changed_content_bodies)
        
        where:
            dropped_bodies: set of body IDs that don't exist in new_mapping
            
            added_bodies: set of body IDs that don't exist in old_mappign
            
            chagned_size_bodies: set of body IDs that exist in both mappings,
                                 but whose supervoxel lists differ in length.

            changed_content_bodies: set of body IDs that exist in both mappings,
                                    and whose supervoxel lists have matching length,
                                    but whose supervoxels are not identical.
    """
    for mapping in (old_mapping, new_mapping):
        assert isinstance(mapping, np.ndarray)
        assert mapping.ndim == 2
        assert mapping.shape[1] == 2
    
    old_df = pd.DataFrame(old_mapping, columns=['sv', 'body'])
    new_df = pd.DataFrame(new_mapping, columns=['sv', 'body'])

    old_bodies = set(pd.unique(old_df['body']))
    new_bodies = set(pd.unique(new_df['body']))

    dropped_bodies = old_bodies - new_bodies
    added_bodies = new_bodies - old_bodies

    old_mapping_common = old_df.query('body not in @dropped_bodies').copy()
    new_mapping_common = new_df.query('body not in @added_bodies').copy()
    
    old_mapping_common.sort_values(['body', 'sv'], inplace=True)
    new_mapping_common.sort_values(['body', 'sv'], inplace=True)
    
    old_common_counts = old_mapping_common.groupby('body').agg({'sv': 'size'})
    new_common_counts = new_mapping_common.groupby('body').agg({'sv': 'size'})
    old_common_counts.columns = ['sv_count']
    new_common_counts.columns = ['sv_count']
    assert (old_common_counts.shape == new_common_counts.shape), \
        f"{old_common_counts.shape} != {new_common_counts.shape}"

    changed_size_rows = (old_common_counts['sv_count'].values != new_common_counts['sv_count'].values)
    changed_size_bodies = set(old_common_counts.iloc[(changed_size_rows,)].index)
    
    old_mapping_common = old_mapping_common.query('body not in @changed_size_bodies')
    new_mapping_common = new_mapping_common.query('body not in @changed_size_bodies')
    
    # Now that we've removed all bodies that were not common to the two
    # mappings or that had different sizes in the two mappings, the mappings are directly comparable.
    assert old_mapping_common.shape == new_mapping_common.shape
    assert (old_mapping_common['body'].values == new_mapping_common['body'].values).all()
    
    changed_content_rows = (old_mapping_common['sv'].values != new_mapping_common['sv'].values)
    changed_content_bodies = set(old_mapping_common.iloc[(changed_content_rows,)]['body'])
    
    return dropped_bodies, added_bodies, changed_size_bodies, changed_content_bodies


def erode_leaf_nodes(edges, rounds=1):
    """
    Given a graph encoded in the given array of edge pairs.
    """
    import pandas as pd
    
    edge_df = pd.DataFrame(edges, columns=['u', 'v'], copy=True)
    
    for _ in range(rounds):
        leaf_nodes = find_all_leaf_nodes( edges )
        leaf_edges = edge_df.eval('u in @leaf_nodes or v in @leaf_nodes')
        edge_df[leaf_edges] = 0

    return edge_df.query('u != 0 and v != 0')

def edges_for_group(edges, group_id):
    import pandas as pd
    import networkx as nx
    g = nx.Graph()
    g.add_edges_from(edges)
    
    group_nodes = nx.node_connected_component(g, group_id) # (Used in query below)

    edge_df = pd.DataFrame(edges, columns=['u', 'v'])
    group_edges = edge_df.query('u in @group_nodes or v in @group_nodes')
    return group_edges.values

def find_leaf_nodes_for_group(edges, group_id):
    """
    Exctract the graph of nodes for the given group,
    then partition into the set of leaf and non-leaf nodes.
    
    Note: Obviously, 'edges' must include the complete merge graph for each body,
          not merely an equivalence mapping or a label-to-body mapping.
    """
    group_edges = edges_for_group(edges, group_id)
    return find_all_leaf_nodes(group_edges.values)

def find_all_leaf_nodes(edges):
    """
    For a graph with the given edges [ndarray, shape == (N,2)],
    return the list of nodes with only a single connection.

    Note: Obviously, 'edges' must include the complete merge graph for each body,
          not merely an equivalence mapping or a label-to-body mapping.
    """
    import pandas as pd
    # Leaf nodes are simply those nodes only referred to in only a single edge
    endpoint_counts = pd.Series(edges.ravel('K')).value_counts()
    leaf_counts = endpoint_counts[ endpoint_counts == 1 ]
    return np.array(leaf_counts.index)





