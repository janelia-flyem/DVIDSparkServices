import os
import sys
import argparse
import datetime
import logging
import multiprocessing
    
import requests
from tqdm import tqdm # progress bar

import h5py
import numpy as np
import pandas as pd
from numba import jit

from dvidutils import LabelMapper # Fast integer array mapping in C++

import DVIDSparkServices # We implicitly rely on initialize_excepthook()

from DVIDSparkServices.util import Timer, default_dvid_session
from DVIDSparkServices.io_util.labelmap_utils import load_edge_csv
from DVIDSparkServices.dvid.metadata import DataInstance

# The labelops_pb2 file was generated with the following commands:
# $ cd DVIDSparkServices/dvid
# $ protoc --python_out=. labelops.proto
# $ sed -i '' s/labelops_pb2/DVIDSparkServices.dvid.labelops_pb2/g labelops_pb2.py
from DVIDSparkServices.dvid.labelops_pb2 import LabelIndex, LabelIndices, MappingOps, MappingOp

logger = logging.getLogger(__name__)


STATS_DTYPE = [('body_id',    np.uint64),
               ('segment_id', np.uint64),
               ('z',          np.int32),
               ('y',          np.int32),
               ('x',          np.int32),
               ('count',      np.uint32)]

AGGLO_MAP_COLUMNS = ['segment_id', 'body_id']


def main():
    """
    Command-line wrapper interface for ingest_label_indexes(), and/or ingest_mapping(), below.
    """
    logger.setLevel(logging.INFO)
    
    # No need to add a handler -- root logger already has a handler via DVIDSparkServices.__init__
    #handler = logging.StreamHandler(sys.stdout)
    #logger.addHandler(handler)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--last-mutid', '-i', required=False, type=int)
    parser.add_argument('--agglomeration-mapping', '-m', required=False,
                        help='A CSV file with two columns, mapping supervoxels to agglomerated bodies. Any missing entries implicitly identity-mapped.')
    parser.add_argument('--operation', default='indexes', choices=['indexes', 'mappings', 'both'],
                        help='Whether to load the LabelIndexes, MappingOps, or both.')
    parser.add_argument('--num-threads', '-n', default=1, type=int,
                        help='How many threads to use when ingesting label indexes (does not currently apply to mappings)')
    parser.add_argument('--batch-size', '-b', default=20_000, type=int,
                        help='Data is grouped in batches to the server. This is the batch size, as measured in ROWS of data to be processed for each batch.')
    parser.add_argument('--no-progress-bar', action='store_true')
    parser.add_argument('server')
    parser.add_argument('uuid')
    parser.add_argument('labelmap_instance')
    parser.add_argument('supervoxel_block_stats_h5', nargs='?', # not required if only ingesting mapping
                        help=f'An HDF5 file with a single dataset "stats", with dtype: {STATS_DTYPE[1:]} (Note: No column for body_id)')

    args = parser.parse_args()

    # Read agglomeration file
    segment_to_body_df = None
    if args.agglomeration_mapping:
        with Timer("Loading agglomeration mapping", logger):
            mapping_pairs = load_edge_csv(args.agglomeration_mapping)
            segment_to_body_df = pd.DataFrame(mapping_pairs, columns=AGGLO_MAP_COLUMNS)

    if args.last_mutid is None:
        # By default, use 0 if we're ingesting
        # supervoxels (no agglomeration), otherwise use 1
        if args.agglomeration_mapping:
            args.last_mutid = 1
        else:
            args.last_mutid = 0

    # Upload label indexes
    if args.operation in ('indexes', 'both'):
        if not args.supervoxel_block_stats_h5:
            raise RuntimeError("You must provide a supervoxel_block_stats_h5 file if you want to ingest LabelIndexes")

        # Read block stats file
        block_sv_stats = load_stats_h5_to_records(args.supervoxel_block_stats_h5)
    
        with Timer(f"Grouping {len(block_sv_stats)} blockwise supervoxel counts and loading LabelIndices", logger):
            ingest_label_indexes( args.server,
                                  args.uuid,
                                  args.labelmap_instance,
                                  args.last_mutid,
                                  block_sv_stats,
                                  segment_to_body_df,
                                  batch_rows=args.batch_size,
                                  num_threads=args.num_threads,
                                  show_progress_bar=not args.no_progress_bar )

    # Upload mappings
    if args.operation in ('mappings', 'both'):
        if not args.agglomeration_mapping:
            raise RuntimeError("Can't load mappings without an agglomeration-mapping file.")
        
        with Timer(f"Loading mapping ops", logger):
            ingest_mapping( args.server,
                            args.uuid,
                            args.labelmap_instance,
                            args.last_mutid,
                            segment_to_body_df,
                            args.batch_size,
                            show_progress_bar=not args.no_progress_bar )

    logger.info(f"DONE.")


def ingest_mapping( server,
                    uuid,
                    instance_name,
                    mutid,
                    segment_to_body_df,
                    batch_size=100_000,
                    show_progress_bar=True,
                    session=None ):
    """
    Ingest the forward-map (supervoxel-to-body) into DVID via the .../mappings endpoint
    
    Args:
        server, uuid, instance_name:
            DVID instance info
    
        mutid:
            The mutation ID to use for all mappings
        
        segment_to_body_df:
            DataFrame.  Must have columns ['segment_id', 'body_id']
        
        batch_size:
            Approximately how many mapping pairs to pack into a single REST call.
        
        show_progress_bar:
            Show progress information as an animated bar on the console.
            Otherwise, progress log messages are printed to the console.
    
    """
    assert list(segment_to_body_df.columns) == AGGLO_MAP_COLUMNS
    instance_info = DataInstance(server, uuid, instance_name)
    if instance_info.datatype != 'labelmap':
        raise RuntimeError(f"DVID instance is not a labelmap: {instance_name}")

    segment_to_body_df.sort_values(['body_id', 'segment_id'], inplace=True)

    
    if not server.startswith('http://'):
        server = 'http://' + server

    if session is None:
        session = requests.Session()

    def send_mapping_ops(mappings):
        ops = MappingOps()
        ops.mappings.extend(mappings)
        payload = ops.SerializeToString()
        r = session.post(f'{server}/api/node/{uuid}/{instance_name}/mappings', data=payload)
        r.raise_for_status()

    if show_progress_bar:
        # Console-based progress
        progress_bar = tqdm(total=len(segment_to_body_df), disable=not show_progress_bar)
    else:
        # Log-based progress
        progress_bar = LoggerProgressBar(len(segment_to_body_df), logger)

    with progress_bar:
        batch_ops_so_far = 0
        mappings = []
        for body_id, body_df in segment_to_body_df.groupby('body_id'):
            op = MappingOp()
            op.mutid = mutid
            op.mapped = body_id
            op.original.extend(body_df['segment_id'])

            # Add to this chunk of ops
            mappings.append(op)

            # Send if chunk is full
            if batch_ops_so_far >= batch_size:
                send_mapping_ops(mappings)
                progress_bar.update(batch_ops_so_far)
                mappings = [] # reset
                batch_ops_so_far = 0

            batch_ops_so_far += len(op.original)
        
        # send last chunk
        if mappings:
            send_mapping_ops(mappings)
            progress_bar.update(batch_ops_so_far)


def ingest_label_indexes( server,
                          uuid,
                          instance_name,
                          last_mutid,
                          block_sv_stats,
                          segment_to_body_df=None,
                          batch_rows=1_000_000,
                          num_threads=1,
                          show_progress_bar=True ):
    """
    Ingest the label indexes for a particular agglomeration.
    
    Args:
        server, uuid, instance_name:
            DVID instance info
    
        last_mutid:
            The mutation ID to use for all indexes
        
        block_sv_stats:
            numpy structured array of blockwise supervoxel counts, with dtype:
            ['body_id', 'segment_id', 'z', 'y', 'x', 'count']
        
        segment_to_body_df:
            If loading an agglomeration, must be a 2-column DataFrame, mapping supervoxel-to-body.
            If loading unagglomerated supervoxels, set to None (identity mapping is used).
        
        batch_size:
            How many LabelIndex structures to include in each /indices REST call.
        
        num_threads:
            How many threads to use, for parallel loading.
        
        show_progress_bar:
            Show progress information as an animated bar on the console.
            Otherwise, progress log messages are printed to the console.
    """
    instance_info = DataInstance(server, uuid, instance_name)
    if instance_info.datatype != 'labelmap':
        raise RuntimeError(f"DVID instance is not a labelmap: {instance_name}")
    bz, by, bx = instance_info.blockshape_zyx
    assert bz == by == bx == 64, "I hard-coded a block-width of 64 in this code, below."

    endpoint = f'{server}/api/node/{uuid}/{instance_name}/indices'

    gen = generate_stats_batches(block_sv_stats, segment_to_body_df, batch_rows)

    if show_progress_bar:
        # Console-based progress
        progress_bar = tqdm(total=len(block_sv_stats), disable=not show_progress_bar)
    else:
        # Log-based progress
        progress_bar = LoggerProgressBar(len(block_sv_stats), logger)

    pool = multiprocessing.Pool(num_threads)
    processor = StatsBatchProcessor(last_mutid, endpoint)
    with progress_bar, pool:
        for next_stats_batch_total_rows in pool.imap(processor.process_batch, gen):
            progress_bar.update(next_stats_batch_total_rows)


def generate_stats_batches( block_sv_stats, segment_to_body_df=None, batch_rows=100_000 ):
    """
    Generator.
    For the given array of with dtype=STATS_DTYPE, sort the array by [body_id,z,y,x] (IN-PLACE),
    and then break it up into subarrays of rows with contiguous body_id.
    
    The subarrays are then yielded in batches, where the total rowcount across all subarrays in
    each batch has approximately batch_rows.
    
    Yields:
        (batch, batch_total_rowcount)
    """
    with Timer("Assigning body IDs", logger):
        _overwrite_body_id_column(block_sv_stats, segment_to_body_df)
    
    with Timer(f"Sorting {len(block_sv_stats)} block stats", logger):
        block_sv_stats.sort(order=['body_id', 'z', 'y', 'x', 'segment_id', 'count'])

    def gen():
        next_stats_batch = []
        next_stats_batch_total_rows = 0
    
        for body_group in groupby_presorted(block_sv_stats, block_sv_stats['body_id'][:, None]):
            next_stats_batch.append(body_group)
            next_stats_batch_total_rows += len(body_group)
            if next_stats_batch_total_rows >= batch_rows:
                yield (next_stats_batch, next_stats_batch_total_rows)
                del next_stats_batch
                next_stats_batch = []
                next_stats_batch_total_rows = 0
    
        # last batch
        if next_stats_batch:
            yield (next_stats_batch, next_stats_batch_total_rows)

    return gen()


def _overwrite_body_id_column(block_sv_stats, segment_to_body_df=None):
    """
    Given a stats array with 'columns' as defined in STATS_DTYPE,
    overwrite the body_id column according to the given agglomeration mapping DataFrame.
    
    If no mapping is given, simply copy the segment_id column into the body_id column.
    
    Args:
        block_sv_stats: numpy.ndarray, with dtype=STATS_DTYPE
        segment_to_body_df: pandas.DataFrame, with columns ['segment_id', 'body_id']
    """
    assert block_sv_stats.dtype == STATS_DTYPE

    assert STATS_DTYPE[0][0] == 'body_id'
    assert STATS_DTYPE[1][0] == 'segment_id'
    
    block_sv_stats = block_sv_stats.view( [STATS_DTYPE[0], STATS_DTYPE[1], ('other_cols', STATS_DTYPE[2:])] )

    if segment_to_body_df is None:
        # No agglomeration
        block_sv_stats['body_id'] = block_sv_stats['segment_id']
    else:
        assert list(segment_to_body_df.columns) == AGGLO_MAP_COLUMNS
        
        # This could be done via pandas merge(), followed by fillna(), etc.,
        # but I suspect LabelMapper is faster and more frugal with RAM.
        mapper = LabelMapper(segment_to_body_df['segment_id'].values, segment_to_body_df['body_id'].values)
        del segment_to_body_df
    
        # Remap in batches to save RAM
        batch_size = 1_000_000
        for chunk_start in range(0, len(block_sv_stats), batch_size):
            chunk_stop = min(chunk_start+batch_size, len(block_sv_stats))
            chunk_segments = block_sv_stats['segment_id'][chunk_start:chunk_stop]
            block_sv_stats['body_id'][chunk_start:chunk_stop] = mapper.apply(chunk_segments, allow_unmapped=True)


class StatsBatchProcessor:
    """
    Function object to take batches of grouped stats,
    convert them into the proper protobuf structure,
    and send them to an endpoint.
    
    Defined here as a class instead of a simple function to enable
    pickling (for multiprocessing), even when this file is run as __main__.
    """
    def __init__(self, last_mutid, endpoint):
        self.last_mutid = last_mutid
        self.endpoint = endpoint

        self.user = os.environ["USER"]
        self.mod_time = datetime.datetime.now().isoformat()
        if not self.endpoint.startswith('http://'):
            self.endpoint = 'http://' + self.endpoint

        self._session = None # Created lazily, after pickling

    @property
    def session(self):
        if self._session is None:
            self._session = default_dvid_session('ingest_label_indexes')
        return self._session
    
    def process_batch(self, batch_and_rowcount):
        """
        Takes a batch of grouped stats rows and sends it to dvid in the appropriate protobuf format.
        """
        next_stats_batch, next_stats_batch_total_rows = batch_and_rowcount
        labelindex_batch = list(map(self.label_index_for_body, next_stats_batch))
        self.post_labelindex_batch(labelindex_batch)
        return next_stats_batch_total_rows

    def label_index_for_body(self, body_group):
        """
        Load body_group (a subarray with dtype=STATS_DTYPE
        and a only single unique body_id) into a LabelIndex protobuf structure.
        """
        body_id = body_group[0]['body_id']

        label_index = LabelIndex()
        label_index.label = body_id
        label_index.last_mutid = self.last_mutid
        label_index.last_mod_user = self.user
        label_index.last_mod_time = self.mod_time
        
        body_dtype = STATS_DTYPE[0]
        segment_dtype = STATS_DTYPE[1]
        coords_dtype = ('coord_cols', STATS_DTYPE[2:5])
        count_dtype = STATS_DTYPE[5]
        assert body_dtype[0] == 'body_id'
        assert segment_dtype[0] == 'segment_id'
        assert np.dtype(coords_dtype[1]).names == ('z', 'y', 'x')
        assert count_dtype[0] == 'count'
        
        body_group = body_group.view([body_dtype, segment_dtype, coords_dtype, count_dtype])
        coord_cols = body_group['coord_cols'].view((np.int32, 3)).reshape(-1, 3)
        for block_group in groupby_presorted(body_group, coord_cols):
            coord = block_group['coord_cols'][0]
            encoded_block_id = _encode_block_id(coord)
            label_index.blocks[encoded_block_id].counts.update( zip(block_group['segment_id'], block_group['count']) )
    
        return label_index


    def post_labelindex_batch(self, batch_indexes):
        """
        Send a batch (list) of LabelIndex objects to dvid.
        """
        label_indices = LabelIndices()
        label_indices.indices.extend(batch_indexes)
        payload = label_indices.SerializeToString()
        
        r = self.session.post(self.endpoint, data=payload)
        r.raise_for_status()


@jit(nopython=True)
def _encode_block_id(coord):
    """
    Helper function for StatsBatchProcessor.label_index_for_body()
    Encodes a coord (structured array of z,y,x)
    into a uint64 in the format DVID expects.
    """
    encoded_block_id = np.uint64(0)
    encoded_block_id |= (np.uint64(coord.z // 64) << 42)
    encoded_block_id |= (np.uint64(coord.y // 64) << 21)
    encoded_block_id |= np.uint64(coord.x // 64)
    return encoded_block_id


def load_stats_h5_to_records(h5_path):
    """
    Read a block segment statistics HDF5 file.
    The file should contain a dataset named 'stats', whose dtype
    is the same as STATS_DTYPE, EXCEPT that it has no 'body_id' column. 
    
    A complete stats array is returned, with full STATS_DTYPE,
    including an UNINITIALIZED column for 'body_id'.
    """
    with h5py.File(h5_path, 'r') as f:
        dset = f['stats']
        with Timer(f"Allocating RAM for {len(dset)} block stats rows", logger):
            block_sv_stats = np.empty(dset.shape, dtype=[('body_col', [STATS_DTYPE[0]]), ('other_cols', STATS_DTYPE[1:])])

        with Timer(f"Loading block stats into RAM", logger):
            h5_batch_size = 1_000_000
            for batch_start in range(0, len(dset), h5_batch_size):
                batch_stop = min(batch_start + h5_batch_size, len(dset))
                block_sv_stats['other_cols'][batch_start:batch_stop] = dset[batch_start:batch_stop]
        
        block_sv_stats = block_sv_stats.view(STATS_DTYPE)
    return block_sv_stats


@jit(nopython=True)
def groupby_presorted(a, sorted_cols):
    """
    Given an array of data and some sorted reference columns to use for grouping,
    yield subarrays of the data, according to runs of identical rows in the reference columns.
    
    JIT-compiled with numba.
    For pre-sorted structured array input, this is much faster than pandas.groupby().
    
    Args:
        a: ND array, any dtype, shape (N,) or (N,...)
        sorted_cols: ND array, at least 2D, any dtype, shape (N,...),
                     not necessarily the same shape as 'a', except for the first dimension.
                     Must be pre-ordered so that identical rows are contiguous,
                     and therefore define the group boundaries.

    Note: The contents of 'a' need not be related in any way to sorted_cols.
          The sorted_cols array is just used to determine the split points,
          and the corresponding rows of 'a' are returned.

    Examples:
    
        a = np.array( [[0,0,0],
                       [1,0,0],
                       [2,1,0],
                       [3,1,1],
                       [4,2,1]] )

        # Group by second column
        groups = list(groupby_presorted(a, a[:,1:2]))
        assert (groups[0] == [[0,0,0], [1,0,0]]).all()
        assert (groups[1] == [[2,1,0], [3,1,1]]).all()
        assert (groups[2] == [[4,2,1]]).all()
    
        # Group by third column
        groups = list(groupby_presorted(a, a[:,2:3]))
        assert (groups[0] == [[0,0,0], [1,0,0], [2,1,0]]).all()
        assert (groups[1] == [[3,1,1], [4,2,1]]).all()

        # Group by external column
        col = np.array([10,10,40,40,40]).reshape(5,1) # must be at least 2D
        groups = list(groupby_presorted(a, col))
        assert (groups[0] == [[0,0,0], [1,0,0]]).all()
        assert (groups[1] == [[2,1,0], [3,1,1],[4,2,1]]).all()
        
    """
    assert sorted_cols.ndim >= 2
    assert sorted_cols.shape[0] == a.shape[0]

    if len(a) == 0:
        return

    start = 0
    vals = sorted_cols[0]
    for stop in range(len(sorted_cols)):
        next_vals = sorted_cols[stop]
        if (next_vals != vals).any():
            yield a[start:stop]
            start = stop
            vals = next_vals

    yield a[start:len(sorted_cols)] # last group


class LoggerProgressBar:
    """
    Bare-bones drop-in replacement for tqdm that logs periodic progress messages,
    for when you don't want tqdm's animated progress bar. 
    """
    def __init__(self, total, logger, step_percent=5):
        self.total = total
        self.milestones = list(np.arange(0.0, 1.001, step_percent/100))
        self.progress = 0
        self.logger = logger
    
    def update(self, increment):
        self.progress += increment
        while self.milestones and self.progress >= self.milestones[0] * self.total:
            self.logger.info(f"Progress: {int(self.milestones[0] * 100):02d}%")
            self.milestones = self.milestones[1:]

    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        import yaml
        test_dir = os.path.dirname(DVIDSparkServices.__file__) + '/../integration_tests/test_copyseg/temp_data'
        with open(f'{test_dir}/config.yaml', 'r') as f:
            config = yaml.load(f)

        dvid_config = config['outputs'][0]['dvid']
        
        ##
        mapping_file = f'{test_dir}/../LABEL-TO-BODY-mod-100-labelmap.csv'
        block_stats_file = f'{test_dir}/block-statistics.h5'

        # SPECIAL DEBUGGING TEST
        #mapping_file = f'{test_dir}/../LABEL-TO-BODY-mod-100-labelmap.csv'
        #block_stats_file = f'/tmp/block-statistics-testvol.h5'
        
        sys.argv += (#f"--operation=indexes"
                     f"--operation=both"
                     f" --agglomeration-mapping={mapping_file}"
                     f" --num-threads=4"
                     f" --batch-size=1000"
                     #f" --no-progress-bar"
                     f" {dvid_config['server']}"
                     f" {dvid_config['uuid']}"
                     f" {dvid_config['segmentation-name']}"
                     f" {block_stats_file}".split())

    main()



