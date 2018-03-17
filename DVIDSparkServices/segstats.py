import logging
from itertools import chain
from functools import partial
from collections import namedtuple

import numpy as np
import vigra
import dvidutils

from DVIDSparkServices.io_util.brick import box_intersection
from DVIDSparkServices.util import box_to_slicing, Timer
from DVIDSparkServices.reconutils.downsample import downsample_binary_3d_suppress_zero

BLOCK_WIDTH = 64

# See COLUMNS_INFO, below


def _concatenate_coordinate_lists(series):
    """
    Concatenate the given pd.Series() of coordinate lists
    and then sort them.
    
    Note: Duplicates are NOT dropped.
    """
    concatenated = np.array(list(chain(*series)))
    sort_order = np.lexsort(np.transpose(concatenated)[::-1, :]) # The lexsort() API is so dumb
    sorted_list = concatenated[sort_order, :]
    return list(sorted_list)


ColumnInfo = namedtuple('ColumnInfo', 'dtype agg_func')
COLUMNS_INFO = {
    'segment':              ColumnInfo(np.uint64, None),   # 'segment' column is not usually aggregated.
    'voxel_count':          ColumnInfo(np.uint64, np.sum),
    'compressed_bytes':     ColumnInfo(np.uint64, np.sum),
    
    # For this statistic, each DataFrame entry will contain a whole list of block coordinates.
    # This is impractical for large volumes, but maybe useful for small ones?
    'block_list':           ColumnInfo(list, _concatenate_coordinate_lists),

    # These are a little tricky because pd.groupby().agg() complains
    # if we attempt to return np.ndarray from an aggregation function.
    # Depending on what you're doing, it may be faster to use z0,y0,x0,z1,y1,x1 columns (below).
    'bounding_box_start':   ColumnInfo(list, lambda s: list(np.amin(list(s.values), axis=0))),
    'bounding_box_stop':    ColumnInfo(list, lambda s: list(np.amax(list(s.values), axis=0))),
    
    # Altnernative columns for bounding_box_start/stop,
    # using individual columns for each dimension
    'z0': ColumnInfo(np.int64, np.amin),
    'y0': ColumnInfo(np.int64, np.amin),
    'x0': ColumnInfo(np.int64, np.amin),
    
    'z1': ColumnInfo(np.int64, np.amax),
    'y1': ColumnInfo(np.int64, np.amax),
    'x1': ColumnInfo(np.int64, np.amax),
}

def aggregate_segment_stats_from_bricks(bricks_rdd, column_names):
    """
    Compute segment statistics for an RDD of segmentation Bricks.
    
    bricks_rdd: An RDD of Brick objects.
    column_names: The list of statistics to compute. (See COLUMNS_INFO for choices.)
    """
    return bricks_rdd.map( partial(stats_df_from_brick, column_names) ) \
                     .treeReduce( merge_stats_dfs, depth=4 )


def aggregate_segment_stats_from_rows(rows_rdd, column_names):
    """
    rows_rdd: Where each RDD item is a tuple:
                (segment, col1, col2, col3, col4), where col1, col2, etc.
                are any of the columns from COLUMNS_INFO above.
                The order of your tuples should be specified in the list 'column_names'.
    """
    return rows_rdd.map( partial(stats_df_from_rows, column_names) ) \
                   .treeReduce( merge_stats_dfs, depth=4 )


def merge_stats_dfs(*stats_dfs):
    """
    Merge two DataFrames whose columns are some subset of those listed in COLUMNS_INFO.
    The 'segment' column must be present.
    Rows with identical segments will be aggregated into a single row.
    Their data will be reduced according to their columns' agg_func entries in COLUMNS_INFO.
    """
    import pandas as pd
    combined_df = pd.concat(stats_dfs, ignore_index=True)

    agg_funcs = { k: v.agg_func for k,v in COLUMNS_INFO.items()
                  if k in combined_df.columns }
    del agg_funcs['segment']
    
    # TODO: This may be slow for large DFs, since most segments will have only one row,
    #       but the agg function will be called on them anyway.
    #       Using merge() to perform the concat/groupby/agg simultaneously might be faster (?)
    #       but's tedious to code and requires dtype conversions, sentinel values, etc.
    #       One possibility for optimization here is to split combined_df into unique/non-unique
    #       segments first, and only run groupby() on the non-unique portion.

    # Use as_index=False to preserve 'segment' column in output.
    grouped_df = combined_df.groupby('segment', as_index=False).agg( agg_funcs )
    return grouped_df


def stats_df_from_brick(column_names, brick, exclude_zero=True, exclude_halo=True):
    """
    For a given brick, return a DataFrame of statistics for the segments it contains.
    
    Args:
    
        column_names (list):
            Which statistics to compute. Anything from COLUMNS_INFO
            is permitted, except compressed_bytes.
            The 'segment' column must be first in the list.
        
        brick (Brick):
            The brick to process
        
        exclude_zero (bool):
            Discard statistics for segment=0.
        
        exclude_halo (bool):
            Exclude voxels that lie outside the Brick's logical_box.
    
    Returns:
        pd.DataFrame, with df.columns == column_names
    """
    import pandas as pd
    assert column_names[0] == 'segment'

    volume = brick.volume
    if exclude_halo and (brick.physical_box != brick.logical_box).any():
        internal_box = box_intersection( brick.logical_box, brick.physical_box ) - brick.physical_box[0]
        volume = volume[box_to_slicing(*internal_box)]
        volume = np.asarray(volume, order='C')

    # We always compute segment and voxel_count
    TRIVIAL_COLUMNS = set(['segment', 'voxel_count'])
    counts = pd.Series(volume.ravel('K')).value_counts(sort=False)
    segment_ids = counts.index.values
    assert segment_ids.dtype == volume.dtype
    
    # Other columns are computed only if needed
    if set(column_names) - TRIVIAL_COLUMNS:
        # Must remap to consecutive segments before calling extractRegionFeatures()
        remapped_ids = np.arange(len(segment_ids), dtype=np.uint32)
        mapper = dvidutils.LabelMapper( segment_ids, remapped_ids )
        remapped_vol = mapper.apply(volume)
        assert remapped_vol.dtype == np.uint32
        remapped_vol = vigra.taggedView( remapped_vol, 'zyx' )

        # Compute (local) bounding boxes.
        acc = vigra.analysis.extractRegionFeatures( np.zeros(remapped_vol.shape, np.float32), remapped_vol,
                                                    ["Count", "Coord<Minimum >", "Coord<Maximum >"]  )
        assert (acc["Count"] == counts.values).all()
        
        # Use int64: int32 is dangerous because multiplying them together quickly overflows
        local_bb_starts = acc["Coord<Minimum >"].astype(np.int64)
        local_bb_stops = (1 + acc["Coord<Maximum >"]).astype(np.int64)

        global_bb_starts = local_bb_starts + brick.physical_box[0]
        global_bb_stops = local_bb_stops + brick.physical_box[0]

        if 'block_list' in column_names:
            block_lists = []
            for remapped_id, start, stop in zip(remapped_ids, local_bb_starts, local_bb_stops):
                local_box = np.array((start, stop))
                binary = (remapped_vol[box_to_slicing(*local_box)] == remapped_id)
                
                # This downsample function respects block-alignment, since we're providing the local_box
                reduced, block_bb = downsample_binary_3d_suppress_zero(binary, BLOCK_WIDTH, local_box)
                
                local_block_indexes = np.transpose(reduced.nonzero())
                local_block_starts = BLOCK_WIDTH * (block_bb[0] + local_block_indexes)
                global_block_starts = brick.physical_box[0] + local_block_starts
                block_lists.append(global_block_starts)
    
    # Segment is always first.
    df = pd.DataFrame(columns=column_names)
    df['segment'] = segment_ids

    # Append columns in-order
    for column in column_names:
        if column == 'voxel_count':
            df['voxel_count'] = counts.values
        
        if column == 'block_list':
            df['block_list'] = block_lists
        
        if column == 'bounding_box_start':
            df['bounding_box_start'] = list(global_bb_starts) # Must convert to list or pandas complains about non-1D-data.
        
        if column == 'bounding_box_stop':
            df['bounding_box_stop'] = list(global_bb_stops) # ditto

        if column in ('z0', 'y0', 'x0'):
            df[column] = global_bb_starts[:, ('z0', 'y0', 'x0').index(column)]

        if column in ('z1', 'y1', 'x1'):
            df[column] = global_bb_stops[:, ('z1', 'y1', 'x1').index(column)]
        
        if column == 'compressed_bytes':
            raise RuntimeError("Can't compute compressed_bytes in this function.")

    if exclude_zero:
        df.drop(df.index[df.segment == 0], inplace=True)

    return df

def stats_df_from_rows(column_names, rows):
    """
    Convert the given rows (list-of-tuples) into a pd.DataFrame,
    whose columns are specified in column_names.
    The 'segment' column must come first.
    """
    import pandas as pd

    columns = list(zip(*rows))
    assert len(columns) == len(column_names)
    assert column_names[0] == 'segment'

    df = pd.DataFrame(columns=column_names)

    for name, col in zip(column_names, columns):        
        df[name] = col

    return df

def write_stats( stats_df, output_path, logger=None ):
    if not output_path.endswith('.pkl.xz'):
        output_path += '.pkl.xz'

    if logger is None:
        logger = logging.getLogger(__name__)
    
    stats_bytes = stats_df.memory_usage().sum()
    stats_gb = stats_bytes / 1e9
    with Timer(f"Saving segment statistics", logger):
        logger.info(f"Writing stats ({stats_gb:.3f} GB) to {output_path}")
        stats_df.to_pickle(output_path)
