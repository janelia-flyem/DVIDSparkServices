import os
import sys
import argparse
import datetime
import logging

import requests
from tqdm import tqdm # progress bar

import numpy as np
import pandas as pd

from dvidutils import LabelMapper # Fast integer array mapping in C++

from DVIDSparkServices.util import Timer
from DVIDSparkServices.io_util.labelmap_utils import load_edge_csv

from DVIDSparkServices.dvid.metadata import DataInstance

# The labelops_pb2 file was generated with the following command: 
# $ protoc --python_out=. labelops.proto
from DVIDSparkServices.dvid.labelops_pb2 import LabelIndex, MappingOps, MappingOp

logger = logging.getLogger(__name__)

SUPERVOXEL_STATS_COLUMNS = ['segment_id', 'z', 'y', 'x', 'count']
AGGLO_MAP_COLUMNS = ['segment_id', 'body_id']

def gen_labelset_indexes(block_sv_stats_df, blockshape_zyx, segment_to_body_df=None, last_mutid=0):
    """
    Generator.  Helper function for ingest_label_indexes().
    
    Append a 'body_id' column onto block_sv_stats_df, initialized via the mapping specified via segment_to_body_df.
    Then group the rows of block_sv_stats_df by body_id, and yields a LabelIndex for each group.
    
    For each group, this generator yields:

        (body_id, label_index, group_entries_total)
        
        ...where body_id is the body_id is uint64, label_index is a LabelIndex (protobuf structure),
        and group_entries_total is an integer for tracking progress.
    """
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
    
        chunk_size = 1_000_000
        for chunk_start in range(0, len(block_sv_stats_df), chunk_size):
            chunk_stop = min(chunk_start+chunk_size, len(block_sv_stats_df))
            chunk_segments = block_sv_stats_df.loc[chunk_start:chunk_stop, 'segment_id'].values
            block_sv_stats_df.loc[chunk_start:chunk_stop, 'body_id'] = mapper.apply(chunk_segments, allow_unmapped=True)

    user = os.environ["USER"]
    mod_time = datetime.datetime.now().isoformat()

    for body_id, body_group_df in block_sv_stats_df.groupby('body_id'):
        group_entries_total = 0
        label_set_index = LabelIndex()
        label_set_index.last_mutid = last_mutid
        label_set_index.last_mod_user = user
        label_set_index.last_mod_time = mod_time
        
        for coord_zyx, block_df in body_group_df.groupby(['z', 'y', 'x']):
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
                label_set_index.blocks[encoded_block_index].counts[sv] = count

            group_entries_total += len(block_df)
        yield (body_id, label_set_index, group_entries_total)

def ingest_label_indexes(server, uuid, instance_name, last_mutid, block_sv_stats_df, segment_to_body_df=None, show_progress_bar=True):
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
        
         show_progress_bar:
             Show progress information on the console.
    """
    instance_info = DataInstance(server, uuid, instance_name)
    if instance_info.datatype != 'labelmap':
        raise RuntimeError(f"DVID instance is not a labelmap: {instance_name}")
    
    blockshape_zyx = instance_info.blockshape_zyx
    
    session = requests.Session()
    
    if not server.startswith('http://'):
        server = 'http://' + server

    with tqdm(total=len(block_sv_stats_df), disable=not show_progress_bar) as progress_bar:
        for body_id, label_set_index, group_entries_total in gen_labelset_indexes(block_sv_stats_df, blockshape_zyx, segment_to_body_df, last_mutid):
            payload = label_set_index.SerializeToString()
            r = session.post(f'{server}/api/node/{uuid}/{instance_name}/index/{body_id}', data=payload)
            r.raise_for_status()
            progress_bar.update(group_entries_total)

def ingest_mapping(server, uuid, instance_name, mutid, segment_to_body_df, chunk_size=100_000, show_progress_bar=True):
    """
    Ingest the forward-map (supervoxel-to-body) into DVID via the .../mappings endpoint
    
    Args:
        server, uuid, instance_name:
            DVID instance info
    
        mutid:
            The mutation ID to use for all mappings
        
        segment_to_body_df:
            DataFrame.  Must have columns ['segment_id', 'body_id']
        
        chunk_size:
            How many ops to pack into a single REST call.
        
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
        session.post(f'{server}/api/node/{uuid}/{instance_name}/mappings', data=payload)

    with tqdm(total=len(segment_to_body_df), disable=not show_progress_bar) as progress_bar:
        segments_progress = 0
        mappings = []
        for body_id, body_df in segment_to_body_df.groupby('body_id'):
            op = MappingOp()
            op.mutid = mutid
            op.mapped = body_id
            op.original.extend(body_df['segment_id'])

            # Add to this chunk's mappings list
            mappings.append(op)

            # Send if chunk is full
            if len(mappings) == chunk_size:
                send_mapping_ops(mappings)
                progress_bar.update(segments_progress)
                mappings = [] # reset

            segments_progress += len(body_df['segment_id'])
        
        # send last chunk
        if mappings:
            send_mapping_ops(mappings)
            progress_bar.update(segments_progress)


def main():
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    #logging.getLogger('requests').setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--last-mutid', '-i', required=False)
    parser.add_argument('--agglomeration-mapping', '-m', required=False,
                        help='A CSV file with two columns, mapping supervoxels to agglomerated bodies. Any missing entries implicitly identity-mapped.')
    parser.add_argument('--operation', default='indexes', choices=['indexes', 'mappings', 'both'],
                        help='Whether to load the LabelIndexes, MappingOps, or both.')
    parser.add_argument('server')
    parser.add_argument('uuid')
    parser.add_argument('labelmap_instance')
    parser.add_argument('supervoxel_block_stats_csv',
                        help=f'A CSV file with columns: {SUPERVOXEL_STATS_COLUMNS}')

    args = parser.parse_args()

    segment_to_body_df = None
    if args.agglomeration_mapping:
        with Timer("Loading agglomeration mapping", logger):
            mapping_pairs = load_edge_csv(args.agglomeration_mapping)
            segment_to_body_df = pd.DataFrame(mapping_pairs, columns=AGGLO_MAP_COLUMNS)

    if args.last_mutid is None:
        # By default, use 0 if we're ingesting supervoxels (no agglomeration),
        # otherwise use 1
        if args.agglomeration_mapping:
            args.last_mutid = 1
        else:
            args.last_mutid = 0

    if args.operation in ('indexes', 'both'):
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
                                  True )

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
        sys.argv += f"--operation=both --agglomeration-mapping={mapping_file} {dvid_config['server']} {dvid_config['uuid']} {dvid_config['segmentation-name']} {test_dir}/block-statistics.csv".split()
    main()




