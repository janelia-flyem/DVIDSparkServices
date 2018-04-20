import logging

import h5py
import numpy as np
import pandas as pd

from DVIDSparkServices.util import Timer
from DVIDSparkServices.io_util.labelmap_utils import mapping_from_edges

logger = logging.getLogger(__name__)

MERGE_TABLE_DTYPE = [('id_a', '<u8'),
                     ('id_b', '<u8'),
                     ('xa', '<u4'),
                     ('ya', '<u4'),
                     ('za', '<u4'),
                     ('xb', '<u4'),
                     ('yb', '<u4'),
                     ('zb', '<u4'),
                     ('score', '<f4')]


def swap_cols(table, rows, name_a, name_b):
    """
    Swap two columns of a structured array, in-place.
    """
    col_a = table[name_a][rows]
    col_b = table[name_b][rows]
    
    # Swap dtypes to avoid assignment error
    col_a, col_b = col_a.view(col_b.dtype), col_b.view(col_a.dtype)

    table[name_a][rows] = col_b
    table[name_b][rows] = col_a


def load_and_normalize_merge_table(path, drop_duplicate_edges=True, sort=None):
    """
    Load the merge table given by the given path (extension '.npy'),
    and then 'normalize' it by ensuring that id_a <= id_b for all rows,
    swapping fields as needed.
    
    If drop_duplicate_edges=True, duplicate edges will be dropped,
    without regard to any of the other columns (e.g. two rows with
    identical edges but different scores are still considered duplicates).
    """
    merge_table = np.load(path)
    assert merge_table.dtype == MERGE_TABLE_DTYPE
    return normalize_merge_table(merge_table, drop_duplicate_edges, sort)

def normalize_merge_table(merge_table, drop_duplicate_edges=True, sort=None):
    """
    'Normalize' the given merge table by ensuring that id_a <= id_b for all rows,
    swapping fields as needed.
    
    If drop_duplicate_edges=True, duplicate edges will be dropped,
    without regard to any of the other columns (e.g. two rows with
    identical edges but different scores are still considered duplicates).
    """
    assert merge_table.dtype == MERGE_TABLE_DTYPE

    # Group the A coords and the B coords so they can be swapped together
    grouped_dtype = [('id_a', '<u8'),
                     ('id_b', '<u8'),
                     ('loc_a', [('xa', '<u4'), ('ya', '<u4'), ('za', '<u4')]),
                     ('loc_b', [('xb', '<u4'), ('yb', '<u4'), ('zb', '<u4')]),
                     ('score', '<f4')]

    swap_rows = merge_table['id_a'] > merge_table['id_b']
    merge_table_grouped = merge_table.view(grouped_dtype)
    
    swap_cols(merge_table_grouped, swap_rows, 'id_a', 'id_b')
    swap_cols(merge_table_grouped, swap_rows, 'loc_a', 'loc_b')

    assert (merge_table['id_a'] <= merge_table['id_b']).all()

    if drop_duplicate_edges:
        edge_df = pd.DataFrame( {'id_a': merge_table['id_a'], 'id_b': merge_table['id_b']} )
        dupe_rows = edge_df.duplicated(keep='last')
        if dupe_rows.sum() > 0:
            merge_table = merge_table[~dupe_rows]
    
    if sort is not None:
        merge_table.sort(order=sort)
    
    return merge_table

def extract_edges(merge_table, as_records=False):
    """
    Extract a copy of only the edge columns of the merge table.
    Returns C-order.
    """
    assert merge_table.dtype == MERGE_TABLE_DTYPE
    table_view = merge_table.view([('edges', MERGE_TABLE_DTYPE[:2]), ('other', MERGE_TABLE_DTYPE[2:])])
    edges = table_view['edges'].copy('C')
    if as_records:
        return edges
    return edges.view((np.uint64, 2))

def load_supervoxel_sizes(h5_path):
    """
    Load the stored supervoxel size table from hdf5 and return the result as a pd.Series, with sv as the index.
    
    h5_path: A file with two datasets: sv_ids and sv_sizes
    """
    with h5py.File(h5_path, 'r') as f:
        sv_sizes = pd.Series(index=f['sv_ids'][:], data=f['sv_sizes'][:])
    sv_sizes.name = 'voxel_count'
    sv_sizes.index.name = 'sv'

    logger.info(f"Volume contains {len(sv_sizes)} supervoxels and {sv_sizes.values.sum()/1e12:.1f} Teravoxels in total")    

    # Sorting by supervoxel ID may give better performance during merges later
    sv_sizes.sort_index(inplace=True)
    return sv_sizes


def compute_comparison_mapping_table(old_edges, new_edges, sv_sizes=None):
    """
    Given two agglomeration encoded via old_edges and new_edges
    (in which vertex IDs correspond to supervoxel IDs),
    compute the connected components for both graphs,
    and also the CC of their graph intersection.
    
    Returns the mapping from SV to body (CC) for all three graphs as a pd.DataFrame.
    Each body ID is defined as the minimum SV ID in the body, so of course there
    will be no correspondence between body IDs in the different mappings.
    
    If sv_sizes is provided, the size of each supervoxel is appended as a column in the DataFrame.
    Any supervoxel IDs missing from sv_sizes ("phantom" supervoxels) are presumed to be of size 0.
    
    Args:
        old_edges: ndarray, shape (N,2)
        
        new_edges: ndarray, shape (M,2)
        
        sv_sizes: (Optional)
                  Must be a pd.Series as returned by load_supervoxel_sizes(),
                  i.e. sv is the index and size is the value.
    Returns:
        pd.DataFrame, indexed by sv with columns:
        "old_body", "new_body", "intersection_component", and "voxel_count" (if sv_sizes was provided)
    """
    # We require C-order arrays, since we'll be fiddling with dtype views that change the shape of the arrays.
    # https://mail.scipy.org/pipermail/numpy-svn/2015-December/007404.html
    old_edges = old_edges.astype(np.uint64, order='C', copy=False)
    new_edges = new_edges.astype(np.uint64, order='C', copy=False)

    # Edges must be pre-normalized
    assert (old_edges[:, 0] <= old_edges[:, 1]).all()
    assert (new_edges[:, 0] <= new_edges[:, 1]).all()
    
    with Timer("Removing duplicate edges"):
        # Pre-sorting should speed up drop_duplicates()
        old_edges.view([('u', np.uint64), ('v', np.uint64)]).sort()
        new_edges.view([('u', np.uint64), ('v', np.uint64)]).sort()
    
        old_edges = pd.DataFrame(old_edges, copy=False).drop_duplicates().values
        new_edges = pd.DataFrame(new_edges, copy=False).drop_duplicates().values
    
    with Timer("Computing intersection"):
        all_edges = np.concatenate((old_edges, new_edges))
        all_edges.view([('u', np.uint64), ('v', np.uint64)]).sort()
        duplicate_markers = pd.DataFrame(all_edges, copy=False).duplicated().values
        common_edges = all_edges[duplicate_markers]
        del all_edges
    
    with Timer("Ensuring identical SV sets"):
        old_svs = set(pd.unique(old_edges.flat))
        new_svs = set(pd.unique(new_edges.flat))
        common_svs = set(pd.unique(common_edges.flat))
        
        # Append identity rows for SVs missing from either graph
        missing_from_old = np.fromiter(new_svs.union(common_svs) - old_svs, dtype=np.uint64)
        missing_from_new = np.fromiter(old_svs.union(common_svs) - new_svs, dtype=np.uint64)
        missing_from_common = np.fromiter(new_svs.union(old_svs) - common_svs, dtype=np.uint64)
    
        if len(missing_from_old) > 0:
            old_missing_edges = np.concatenate((missing_from_old[:, None], missing_from_old[:, None]), axis=1)
            old_edges = np.concatenate((old_edges, old_missing_edges))
    
        if len(missing_from_new) > 0:
            new_missing_edges = np.concatenate((missing_from_new[:, None], missing_from_new[:, None]), axis=1)
            new_edges = np.concatenate((new_edges, new_missing_edges))
    
        if len(missing_from_common) > 0:
            common_missing_edges = np.concatenate((missing_from_common[:, None], missing_from_common[:, None]), axis=1)
            common_edges = np.concatenate((common_edges, common_missing_edges))
    
    with Timer("Computing old mapping"):
        old_mapping = mapping_from_edges(old_edges, sort_by='segment', as_series=True)
    
    with Timer("Computing new mapping"):
        new_mapping = mapping_from_edges(new_edges, sort_by='segment', as_series=True)
    
    with Timer("Computing intersection mapping"):
        intersection_mapping = mapping_from_edges(common_edges, sort_by='segment', as_series=True)
    
    assert len(old_mapping.index) == len(new_mapping.index) == len(intersection_mapping.index)
    
    sv_table = pd.DataFrame( { "old_body": old_mapping,
                               "new_body": new_mapping,
                               "intersection_component": intersection_mapping },
                               copy=False )
    sv_table.index.name = "sv"

    if sv_sizes is not None:
        sv_table = sv_table.merge(pd.DataFrame(sv_sizes), 'left', left_index=True, right_index=True, copy=False)
        
        # Fix 'phantom' supervoxels (mentioned in the merge graph(s), but not present in the volume)
        sv_table['voxel_count'].fillna(0, inplace=True)
        sv_table['voxel_count'] = sv_table['voxel_count'].astype(np.uint64)
    return sv_table


def compute_component_table(sv_table):
    """
    Reduce the given sv_table as returned by compute_comparison_mapping_table()
    into a table indexed by 'component', i.e. groups of supervoxels that have the
    have body label in both the 'old' and 'new' graphs.
    """
    aggs = { "old_body": "first", "new_body": "first" }
    if 'voxel_count' in sv_table.columns:
        aggs["voxel_count"] = "sum"

    # For a given intersection component, all rows will be identical
    # (except the index), so we can reduce the table now to simplify the analysis.
    component_table = sv_table.groupby("intersection_component").agg(aggs)
    return component_table


def compute_split_merge_stats(component_table):
    """
    Determine which bodies were involved in splits and/or merges,
    and return the subset of component_table that includes only those bodies.
    
    Also return basic statistics for each old body that was split and each new body that was merged.
    
    Arg:
        component_table (DataFrame)
            Index: component
            Columns: 'old_body', 'new_body', 'voxel_count'
    
    Returns three DataFrames:
    
        affected_components:
            The subset rows in of component_table whose components are involved in splits or merges.

        split_body_stats:
            Index: old_body
            Columns: ['num_components', 'body_voxels', 'largest_component_voxels', 'remaining_voxels']
                (where 'remaining_voxels' is body_voxels - largest_component_voxels)

        merge_body_stats:
            Index: new_body
            Columns: ['num_components', 'body_voxels', 'largest_component_voxels', 'remaining_voxels']
                (where 'remaining_voxels' is body_voxels - largest_component_voxels)
    """
    # Compute number of components in each old body and the body size.
    # More than 1 component indicates the old body was split.
    # (Includes bodies that were split and then merged with other bodies.)
    old_body_stats = component_table.groupby('old_body').agg({"voxel_count": ['size', 'sum', 'max']})
    old_body_stats.columns = ['num_components', 'body_voxels', 'largest_component_voxels']
    old_body_stats['remaining_voxels'] = old_body_stats['body_voxels'] - old_body_stats['largest_component_voxels']
    
    # Compute number of components in each new body and the body size.
    # More than 1 component indicates the new body is the result of a merger.
    # (Includes bodies whose components were split from other bodies.)
    new_body_stats = component_table.groupby('new_body').agg({"voxel_count": ['size', 'sum', 'max']})
    new_body_stats.columns = ['num_components', 'body_voxels', 'largest_component_voxels']
    new_body_stats['remaining_voxels'] = new_body_stats['body_voxels'] - new_body_stats['largest_component_voxels']
    
    split_body_stats = old_body_stats.query('num_components > 1')
    merge_body_stats = new_body_stats.query('num_components > 1')
    affected_components = component_table.query('(old_body in @split_body_stats.index) or (new_body in @merge_body_stats.index)')
    
    return affected_components, split_body_stats, merge_body_stats


def frequencies_by_size_thresholds(col):
    d = { '>= 1 Gv':                            (col >= 1e9).sum(),
          '100 Mv - 1 Gv':  ((100e6 <= col) & (col <   1e9)).sum(),
          '10 Mv - 100 Mv':  ((10e6 <= col) & (col < 100e6)).sum(),
          '1 Mv - 10 Mv':     ((1e6 <= col) & (col <  10e6)).sum(),
          '100 kv - 1 Mv':  ((100e3 <= col) & (col <   1e6)).sum(),
          '10 kv - 100 kv':  ((10e3 <= col) & (col < 100e3)).sum(),
          '< 10 kv':                           (col <= 10e3).sum(),
          'TOTAL':                                        len(col) }

    return pd.DataFrame(list(zip(d.keys(), d.values())), columns=['size range', 'body count'])


