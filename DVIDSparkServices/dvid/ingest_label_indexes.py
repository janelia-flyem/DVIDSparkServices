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


STATS_DTYPE = [('body_id', np.uint64), ('segment_id', np.uint64), ('z', np.int32), ('y', np.int32), ('x', np.int32), ('count', np.uint32)]
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
                                  show_progress_bar=True )

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
                            show_progress_bar=True )

    logger.info(f"DONE.")


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
            Show progress information on the console.
    """
    instance_info = DataInstance(server, uuid, instance_name)
    if instance_info.datatype != 'labelmap':
        raise RuntimeError(f"DVID instance is not a labelmap: {instance_name}")
    bz, by, bx = instance_info.blockshape_zyx
    assert bz == by == bx == 64, "I hard-coded a block-width of 64 in this code, below."

    endpoint = f'{server}/api/node/{uuid}/{instance_name}/indices'

    gen = generate_stats_batches(block_sv_stats, segment_to_body_df, batch_rows)
    processor = StatsBatchProcessor(last_mutid, endpoint)

    pool = multiprocessing.Pool(num_threads)
    progress_bar = tqdm(total=len(block_sv_stats), disable=not show_progress_bar)
    with progress_bar, pool:
        for next_stats_batch_total_rows in pool.imap(processor.process_batch, gen):
            progress_bar.update(next_stats_batch_total_rows)

class StatsBatchProcessor:
    def __init__(self, last_mutid, endpoint):
        self.last_mutid = last_mutid
        self.endpoint = endpoint

        self.session = default_dvid_session('ingest_label_indexes')
        self.user = os.environ["USER"]
        self.mod_time = datetime.datetime.now().isoformat()
        if not self.endpoint.startswith('http://'):
            self.endpoint = 'http://' + self.endpoint
    
    def process_batch(self, args):
        next_stats_batch, next_stats_batch_total_rows = args
        labelindex_batch = populate_labelindex_batch(next_stats_batch, self.last_mutid, self.user, self.mod_time)
        send_labelindex_batch(self.session, self.endpoint, labelindex_batch)
        return next_stats_batch_total_rows

def generate_stats_batches( block_sv_stats, segment_to_body_df=None, batch_rows=100_000 ):
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


def populate_labelindex_batch(stats_batch, last_mutid, user, mod_time):
    batch_indexes = []
    for body_group in stats_batch:
        body_id = body_group[0]['body_id']
        label_index = label_index_for_body(body_id, body_group, last_mutid, user, mod_time)
        batch_indexes.append(label_index)
    return batch_indexes

def send_labelindex_batch(session, endpoint, batch_indexes):
    """
    Send a batch (list) of LabelIndex objects to dvid.
    """
    label_indices = LabelIndices()
    label_indices.indices.extend(batch_indexes)
    payload = label_indices.SerializeToString()
    
    r = session.post(endpoint, data=payload)
    r.raise_for_status()


def _overwrite_body_id_column(block_sv_stats, segment_to_body_df=None):
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


def load_stats_h5_to_records(h5_path):
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

def label_index_for_body(body_id, body_group, last_mutid, user, mod_time):
    label_index = LabelIndex()
    label_index.label = body_id
    label_index.last_mutid = last_mutid
    label_index.last_mod_user = user
    label_index.last_mod_time = mod_time
    
    body_dtype = STATS_DTYPE[0]
    segment_dtype = STATS_DTYPE[1]
    coords_dtype = ('coord_cols', STATS_DTYPE[2:5])
    count_dtype = STATS_DTYPE[5]
    assert body_dtype[0] == 'body_id'
    assert segment_dtype[0] == 'segment_id'
    assert np.dtype(coords_dtype[1]).names == ('z', 'y', 'x')
    assert count_dtype[0] == 'count'
    
    body_group = body_group.view([body_dtype, segment_dtype, coords_dtype, count_dtype])

    # These are initialized outside the loop and re-used to save object initialization time    
    encoded_block_id = np.uint64(0)
    block_index = np.zeros((3,), np.uint64)
    _42 = np.uint64(42)
    _21 = np.uint64(21)
    
    coord_cols = body_group['coord_cols'].view((np.int32, 3)).reshape(-1, 3)
    for block_group in groupby_presorted(body_group, coord_cols):
        coord = block_group['coord_cols'][0]

        block_index[0] = (coord['z'] // 64)
        block_index[1] = (coord['y'] // 64)
        block_index[2] = (coord['x'] // 64)

        encoded_block_id ^= encoded_block_id # reset to np.uint64(0) without instantiating a new np.uint64
        encoded_block_id |= (block_index[0] << _42)
        encoded_block_id |= (block_index[1] << _21)
        encoded_block_id |= block_index[2]

        label_index.blocks[encoded_block_id].counts.update( zip(block_group['segment_id'], block_group['count']) )

    return label_index


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
            Show progress information on the console.
    
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

    progress_bar = tqdm(total=len(segment_to_body_df), disable=not show_progress_bar)
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



if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        import yaml
        import DVIDSparkServices
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
        
        sys.argv += (f"--operation=indexes"
                     #f"--operation=both"
                     #f" --agglomeration-mapping={mapping_file}"
                     f" --num-threads=4"
                     f" --batch-size=1000"
                     f" {dvid_config['server']}"
                     f" {dvid_config['uuid']}"
                     f" {dvid_config['segmentation-name']}"
                     f" {block_stats_file}".split())

    main()



