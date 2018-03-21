import os
import sys
import time
import argparse
import datetime
import logging
import threading

import requests
from tqdm import tqdm # progress bar

import numpy as np
import pandas as pd

from dvidutils import LabelMapper # Fast integer array mapping in C++

import DVIDSparkServices # We implicitly rely on initialize_excepthook()

from DVIDSparkServices.util import Timer
from DVIDSparkServices.io_util.labelmap_utils import load_edge_csv
from DVIDSparkServices.dvid.metadata import DataInstance

# The labelops_pb2 file was generated with the following command: 
# $ protoc --python_out=. labelops.proto
from DVIDSparkServices.dvid.labelops_pb2 import LabelIndex, LabelIndices, MappingOps, MappingOp

logger = logging.getLogger(__name__)

SUPERVOXEL_STATS_COLUMNS = ['segment_id', 'z', 'y', 'x', 'count']
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
    parser.add_argument('server')
    parser.add_argument('uuid')
    parser.add_argument('labelmap_instance')
    parser.add_argument('supervoxel_block_stats_csv', nargs='?', # not required if only ingesting mapping
                        help=f'A CSV file with columns: {SUPERVOXEL_STATS_COLUMNS}')

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
        if not args.supervoxel_block_stats_csv:
            raise RuntimeError("You must provide a supervoxel_block_stats_csv file if you want to ingest LabelIndexes")

        # Read block stats file
        with Timer("Loading supervoxel block statistics file", logger):
            dtypes = { 'segment_id': np.uint64,
                       'z': np.int32,
                       'y': np.int32,
                       'x':np.int32,
                       'count': np.uint32 }
            block_sv_stats_df = pd.read_csv(args.supervoxel_block_stats_csv, engine='c', dtype=dtypes)
    
        with Timer(f"Grouping {len(block_sv_stats_df)} blockwise supervoxel counts and loading LabelSetIndexes", logger):
            ingest_label_indexes( args.server,
                                  args.uuid,
                                  args.labelmap_instance,
                                  args.last_mutid,
                                  block_sv_stats_df,
                                  segment_to_body_df,
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
                            show_progress_bar=True )

    logger.info(f"DONE.")


def ingest_label_indexes( server,
                          uuid,
                          instance_name,
                          last_mutid,
                          block_sv_stats_df,
                          segment_to_body_df=None,
                          batch_size=1000,
                          num_threads=1,
                          show_progress_bar=True ):
    """
    Ingest the label indexes for a particular agglomeration.
    
    Args:
        server, uuid, instance_name:
            DVID instance info
    
        last_mutid:
            The mutation ID to use for all indexes
        
        block_sv_stats_df:
            DataFrame of blockwise supervoxel counts, with columns:
            ['segment_id', 'z', 'y', 'x', 'count']
        
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
    assert num_threads > 0
    if not server.startswith('http://'):
        server = 'http://' + server
    instance_info = DataInstance(server, uuid, instance_name)
    if instance_info.datatype != 'labelmap':
        raise RuntimeError(f"DVID instance is not a labelmap: {instance_name}")
    blockshape_zyx = instance_info.blockshape_zyx

    block_sv_stats_df = _append_body_id_column(block_sv_stats_df, segment_to_body_df)

    progress_lock = threading.Lock()
    progress_bar = tqdm(total=len(block_sv_stats_df), disable=not show_progress_bar)
    with progress_bar:
        generator_lock = threading.Lock()
        batch_generator = _gen_label_index_batches( block_sv_stats_df, blockshape_zyx, last_mutid, batch_size )
        
        session = requests.Session()
        def send_batch(batch):
            """
            Send a batch (list) of LabelIndex objects to dvid.
            """
            with Timer() as timer:
                label_indices = LabelIndices()
                label_indices.indices.extend(batch)
                payload = label_indices.SerializeToString()
            serializing_time = timer.seconds
            
            with Timer() as timer:
                r = session.post(f'{server}/api/node/{uuid}/{instance_name}/indices', data=payload)
                r.raise_for_status()
            sending_time = timer.seconds
            
            return (serializing_time, sending_time)
            
        failed = False
        total_serializing_time = 0.0
        total_sending_time = 0.0
        total_generation_time = 0.0
        
        start_time = time.time()
        def worker_impl():
            """
            Worker thread run loop.
            """
            nonlocal failed
            nonlocal total_sending_time
            nonlocal total_serializing_time
            nonlocal total_generation_time
            
            try:
                while not failed:
                    with generator_lock:
                        # Fetch next batch (will raise StopIteration when we run out of batches)
                        with Timer() as timer:
                            next_batch, batch_entries = next(batch_generator)
                        batch_generation_time = timer.seconds
                    
                    serializing_time, sending_time = send_batch(next_batch)

                    with progress_lock:
                        total_serializing_time += serializing_time
                        total_sending_time += sending_time
                        total_generation_time += batch_generation_time
                        progress_bar.update(batch_entries)
            except StopIteration:
                pass
            except Exception:
                failed = True
                raise   # This will still print a traceback, despite the fact that we're in a Thread,
                        # thanks to DVIDSparkServices.__init__.initialize_excepthook()
        
        try:
            # Launch the workers and wait for them to finish
            workers = [threading.Thread(target=worker_impl) for _ in range(num_threads)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
        except:
            logger.info("Loading LabelIndices failed.")
            failed = True
            raise
        else:
            logger.info("Loading LabelIndices completed.")
        finally:
            logger.info(f"Total batch generation time: {total_generation_time:.2f} seconds (serialized)")
            logger.info(f"Total serialization time: {total_serializing_time:.2f} seconds (parallelized??)")
            logger.info(f"Total sending time: {total_sending_time:.2f} seconds (parallelized)")
            unaccounted_time = (time.time() - start_time) - total_generation_time - total_serializing_time - total_sending_time
            logger.info(f"Total unaccounted time: {unaccounted_time:.2f} seconds")
        
        if failed:
            raise RuntimeError("One or more worker threads encountered an error while ingesting label indexes.  See traceback above.")
        

def _append_body_id_column(block_sv_stats_df, segment_to_body_df=None):
    assert list(block_sv_stats_df.columns) == SUPERVOXEL_STATS_COLUMNS

    if segment_to_body_df is None:
        # No agglomeration
        block_sv_stats_df['body_id'] = block_sv_stats_df['segment_id']
    else:
        assert list(segment_to_body_df.columns) == AGGLO_MAP_COLUMNS
        
        # This could be done via pandas merge(), followed by fillna(), etc.,
        # but I suspect LabelMapper is faster and more frugal with RAM.
        mapper = LabelMapper(segment_to_body_df['segment_id'].values, segment_to_body_df['body_id'].values)
        del segment_to_body_df
    
        block_sv_stats_df['body_id'] = 0
    
        batch_size = 1_000_000
        for chunk_start in range(0, len(block_sv_stats_df), batch_size):
            chunk_stop = min(chunk_start+batch_size, len(block_sv_stats_df))
            chunk_segments = block_sv_stats_df.loc[chunk_start:chunk_stop, 'segment_id'].values
            block_sv_stats_df.loc[chunk_start:chunk_stop, 'body_id'] = mapper.apply(chunk_segments, allow_unmapped=True)
    
    return block_sv_stats_df


def _gen_label_index_batches( block_sv_stats_df, blockshape_zyx, last_mutid=0, batch_size=100 ):
    """
    Generator produces LabelIndex batches. NOT THREADSAFE.
    
    The resulting batches are lists of (list-of-LabelIndex, num_batch_entries)
    """
    index_generator = _gen_label_indexes(block_sv_stats_df, blockshape_zyx, last_mutid)
    try:
        while True:
            next_batch = []
            batch_entries = 0

            for _ in range(batch_size):
                label_index, num_entries = next(index_generator) # raises StopIteration when none left
                next_batch.append( label_index )
                batch_entries += num_entries

            yield (next_batch, batch_entries)

    except StopIteration:
        if next_batch:
            # Last batch (if smaller than batch_size)
            yield (next_batch, batch_entries)


def _gen_label_indexes(block_sv_stats_df, blockshape_zyx, last_mutid=0):
    """
    Generator.  Helper function for ingest_label_indexes().  NOT THREADSAFE.
    
    Append a 'body_id' column onto block_sv_stats_df, initialized via the mapping specified via segment_to_body_df.
    Then group the rows of block_sv_stats_df by body_id, and yields a LabelIndex for each group.
    
    For each group, this generator yields:

        (label_index, num_entries)
        
        ...where body_id is the body_id is uint64, label_index is a LabelIndex (protobuf structure),
        and num_entries indicates how many total 'counts' entries there are in the label_index
        (for progress tracking purposes).
    """
    user = os.environ["USER"]
    mod_time = datetime.datetime.now().isoformat()

    for body_id, body_group_df in block_sv_stats_df.groupby('body_id'):
        label_index = LabelIndex()
        label_index.label = body_id
        label_index.last_mutid = last_mutid
        label_index.last_mod_user = user
        label_index.last_mod_time = mod_time
        
        num_entries = 0
        for coord_zyx, block_df in body_group_df.groupby(['z', 'y', 'x']):
            num_entries += len(block_df)

            assert not (coord_zyx % blockshape_zyx).any()
            block_index_zyx = (np.array(coord_zyx) // blockshape_zyx).astype(np.int32)

            # Must leave room for the sign bit!
            assert (np.abs(block_index_zyx) < 2**20).all()
            
            # block key is encoded as 3 21-bit (signed) integers packed into a uint64
            # (leaving the MSB set to 0)
            encoded_block_index = np.uint64(0)
            encoded_block_index |= np.uint64(block_index_zyx[0] << 42)
            encoded_block_index |= np.uint64(block_index_zyx[1] << 21)
            encoded_block_index |= np.uint64(block_index_zyx[2] << 0)
            
            for sv, count in zip(block_df['segment_id'], block_df['count']):
                label_index.blocks[encoded_block_index].counts[sv] = count

        yield (label_index, num_entries)


def ingest_mapping( server,
                    uuid,
                    instance_name,
                    mutid,
                    segment_to_body_df,
                    batch_size=100_000,
                    show_progress_bar=True ):
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

    session = requests.Session()
    
    if not server.startswith('http://'):
        server = 'http://' + server

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
        mapping_file = f'{test_dir}/../LABEL-TO-BODY-mod-100-labelmap.csv'
        sys.argv += (#f"--operation=indexes"
                     f"--operation=both"
                     f" --agglomeration-mapping={mapping_file}"
                     f" --num-threads=4"
                     f" {dvid_config['server']}"
                     f" {dvid_config['uuid']}"
                     f" {dvid_config['segmentation-name']}"
                     f" {test_dir}/block-statistics.csv".split())

    main()




