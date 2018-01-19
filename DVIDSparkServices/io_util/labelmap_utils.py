import os
import csv
import subprocess
from itertools import chain

import numpy as np

import logging
logger = logging.getLogger(__name__)

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

    # path is [gs://]/path/to/file.csv[.gz]

    # If the file is in a gbucket, download it first (if necessary)
    if path.startswith('gs://'):
        filename = path.split('/')[-1]
        downloaded_path = working_dir + '/' + filename
        if not os.path.exists(downloaded_path):
            cmd = f'gsutil -q cp {path} {downloaded_path}'
            logger.info(cmd)
            subprocess.check_call(cmd, shell=True)
        path = downloaded_path

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
        with open(path, 'r') as csv_file:
            rows = csv.reader(csv_file)
            all_items = chain.from_iterable(rows)
            mapping_pairs = np.fromiter(all_items, np.uint64).reshape(-1,2)
    elif labelmap_config["file-type"] == "equivalence-edges":
        mapping_pairs = equivalence_mapping_from_edge_csv(path)

        # Export mapping to disk in case anyone wants to view it later
        output_dir, basename = os.path.split(path)
        mapping_csv_path = f'{output_dir}/LABEL-TO-BODY-{basename}'
        if not os.path.exists(mapping_csv_path):
            with open(mapping_csv_path, 'w') as f:
                csv.writer(f).writerows(mapping_pairs)

    return mapping_pairs


def load_edge_csv(csv_path):
    """
    Load and return the given edge list CSV file as a numpy array.
    
    Each row represents an edge. For example:
    
        123,456
        123,789
        789,234
    
    The CSV file may optionally contain a header row.
    Also, it may contain more than two columns, but only the first two columns are used.
    
    Returns:
        ndarray with shape (N,2)
    """
    with open(csv_path, 'r') as csv_file:
        # Is there a header?
        has_header = csv.Sniffer().has_header(csv_file.read(1024))
        csv_file.seek(0)
        rows = iter(csv.reader(csv_file))
        if has_header:
            # Skip header
            _header = next(rows)
        
        # We only care about the first two columns
        all_items = chain.from_iterable( (row[0], row[1]) for row in rows )
        edges = np.fromiter(all_items, np.uint64).reshape(-1,2) # implicit conversion from str -> uint64

    return edges


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


def equivalence_mapping_from_edge_csv(csv_path, output_csv_path=None):
    """
    Load and return the equivalence_mapping from the given csv_path of equivalence edges.
    
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
    groups = groups_from_edges(edges)
    mapping = mapping_from_groups(groups)
    
    if output_csv_path:
        equivalence_mapping_to_csv(mapping, output_csv_path)
        
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

