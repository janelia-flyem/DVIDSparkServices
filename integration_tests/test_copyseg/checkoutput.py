import sys
import glob

import yaml
import requests

import numpy as np
import pandas as pd

from libdvid import DVIDNodeService
from dvidutils import LabelMapper

import DVIDSparkServices
from DVIDSparkServices.sparkdvid.sparkdvid import sparkdvid
from DVIDSparkServices.dvid.ingest_label_indexes import ingest_label_indexes, ingest_mapping

dirpath = sys.argv[1]
#import os
#dirpath = os.path.dirname(__file__)

configs = glob.glob(dirpath + "/temp_data/config.*")
assert len(configs) == 1, "Why does the temp_dir have more than one config.* file?"

with open(configs[0], 'r') as f:
    config = yaml.load(f)

input_service = DVIDNodeService( str(config['input']['dvid']['server']),
                                 str(config['input']['dvid']['uuid']) )
input_name = config['input']['dvid']['segmentation-name']
input_bb_xyz = config['input']['geometry']['bounding-box']
input_bb_zyx = np.array(input_bb_xyz)[:,::-1]
input_shape = input_bb_zyx[1] - input_bb_zyx[0]

input_volume = input_service.get_labels3D(input_name, input_shape, input_bb_zyx[0])

##
## Check actual segmentation
##
index = 0 # Only one output in this test
output_service = DVIDNodeService( str(config['outputs'][index]['dvid']['server']),
                                  str(config['outputs'][index]['dvid']['uuid']))
output_name = config['outputs'][index]['dvid']['segmentation-name']
output_bb_xyz = config['outputs'][index]['geometry']['bounding-box']
output_bb_zyx = np.array(output_bb_xyz)[:,::-1]
output_shape = output_bb_zyx[1] - output_bb_zyx[0]

output_volume = output_service.get_labels3D(output_name, output_shape, output_bb_zyx[0])

if not (input_volume == output_volume).all():
    print(f"DEBUG: FAIL: output volume {index} does not correspond to input volume!")
    sys.exit(1)

##
## Check exported statistics
##
stats_path = dirpath + f'/temp_data/block-statistics.csv'
df = pd.read_csv(stats_path)

# Must deduplicate ourselves (workflow doesn't do it)
duplicate_rows = df.duplicated(['segment_id', 'z', 'y', 'x'], keep='last')
df = df[~duplicate_rows]

# Overwrite the duplicate-less version just to keep this test
# file size limited after several sequential tests.
df.to_csv(stats_path, header=True, index=False)

segment_counts_from_stats = df[['segment_id', 'count']].groupby('segment_id').sum()['count']
assert 0 not in segment_counts_from_stats, "Segment 0 should not be included in block statistics"
segment_counts_from_stats.sort_index(inplace=True)

segment_counts_from_output_seg = pd.Series(output_volume.reshape(-1)).value_counts(sort=False)
segment_counts_from_output_seg = segment_counts_from_output_seg[segment_counts_from_output_seg.index != 0]
segment_counts_from_output_seg.sort_index(inplace=True)

#from IPython.core.debugger import set_trace
#set_trace()

assert (segment_counts_from_output_seg == segment_counts_from_stats).all()

print("DEBUG: CopySegmentation test passed.")
sys.exit(0)


###
### LabelIndexes are no longer ingested in this script;
### See IngestLabelIndices (and its checkoutput.py file).
###

# ##
# ## Now ingest SUPERVOXEL label indexes (for the first output), and try a sparsevol query to prove that it works.
# ##
# block_stats_path = dirpath + '/temp_data/block-statistics.csv'
# dvid_config = config['outputs'][0]['dvid']
# 
# dtypes = { 'segment_id': np.uint64,
#            'z': np.int32,
#            'y': np.int32,
#            'x':np.int32,
#            'count': np.uint32 }
# 
# block_sv_stats_df = pd.read_csv(block_stats_path, engine='c', dtype=dtypes)
# block_sv_stats_df['segment_id'] = block_sv_stats_df['segment_id'].astype(np.uint64)
# 
# last_mutid = 0
# ingest_label_indexes( dvid_config['server'],
#                       dvid_config['uuid'],
#                       dvid_config['segmentation-name'],
#                       last_mutid,
#                       block_sv_stats_df,
#                       num_threads=2 )
# 
# # Find segment with the most blocks touched (not necessarily the most voxels)
# largest_sv = block_sv_stats_df['segment_id'].value_counts(ascending=False).argmax()
# print(f"Fetching sparsevol for label {largest_sv}")
# 
# # Fetch the /sparsevol-coarse representation for it.
# fetched_block_coords = sparkdvid.get_coarse_sparsevol( dvid_config['server'],
#                                                        dvid_config['uuid'],
#                                                        dvid_config['segmentation-name'],
#                                                        largest_sv )
# 
# fetched_block_coords = np.asarray(sorted(map(tuple, fetched_block_coords)))
# expected_block_coords = block_sv_stats_df.query('segment_id == @largest_sv')[['z', 'y', 'x']].sort_values(['z', 'y', 'x']).values
# expected_block_coords //= 64 # dvid block width
# 
# assert (fetched_block_coords == expected_block_coords).all()
# 
# 
# ##
# ## Now RE-ingest, for BODY label indexes (for the first output), and try a sparsevol query to prove that it works.
# ##
# ## (Normally, we would want to close the original node and branch
# ##  from it before ingesting body indexes, but we'll skip that for this test.)
# ##
# ## We use fake label-to-body mapping for this test: It's just a table of mod-10 values for each object.
# ##
# block_sv_stats_df = pd.read_csv(block_stats_path, engine='c', dtype=dtypes)
# block_sv_stats_df['segment_id'] = block_sv_stats_df['segment_id'].astype(np.uint64)
# segment_to_body_df = pd.read_csv(f'{dirpath}/LABEL-TO-BODY-mod-100-labelmap.csv', names=['segment_id', 'body_id'], header=None)
# 
# last_mutid = 1
# ingest_label_indexes( dvid_config['server'],
#                       dvid_config['uuid'],
#                       dvid_config['segmentation-name'],
#                       last_mutid,
#                       block_sv_stats_df,
#                       segment_to_body_df,
#                       num_threads=2 )
# 
# # Use label 42 for this test.
# BODY_ID = 42
# print(f"Fetching sparsevol for label {BODY_ID}")
# 
# # Fetch the /sparsevol-coarse representation for it.
# fetched_block_coords = sparkdvid.get_coarse_sparsevol( dvid_config['server'],
#                                                        dvid_config['uuid'],
#                                                        dvid_config['segmentation-name'],
#                                                        BODY_ID )
# 
# fetched_block_coords = np.asarray(sorted(map(tuple, fetched_block_coords)))
# expected_block_coords = ( block_sv_stats_df.query('segment_id % 100 == @BODY_ID')[['z', 'y', 'x']]
#                           .drop_duplicates()
#                           .sort_values(['z', 'y', 'x']).values )
# expected_block_coords //= 64 # dvid block width
# 
# 
# # print("EXPECTED BLOCKS:")
# # print(expected_block_coords)
# # 
# # print("FETCHED BLOCKS:")
# # print(fetched_block_coords)
# 
# assert (fetched_block_coords == expected_block_coords).all()
# 
# # Ingest the mapping and verify with /label/<coord>
# 
# server = dvid_config['server']
# uuid = dvid_config['uuid']
# instance = dvid_config['segmentation-name']
# 
# print("Ingesting mapping")
# ingest_mapping( server,
#                 uuid,
#                 instance,
#                 1, # mutid
#                 segment_to_body_df,
#                 batch_size=100_000,
#                 show_progress_bar=True )
# 
# mapper = LabelMapper(segment_to_body_df['segment_id'], segment_to_body_df['body_id'])
# expected_mapped_vol = mapper.apply(output_volume)
# 
# # Find a coordinate that should map to our body, but didn't start that way.
# coords = ((expected_mapped_vol == BODY_ID) & (output_volume != BODY_ID)).nonzero()
# coords = np.array(coords).transpose()
# coords += output_bb_zyx[0]
# 
# first_coord = coords[0]
# first_coord_str = '_'.join(map(str, first_coord[::-1])) # zyx -> xyz
# 
# r = requests.get(f"http://{server}/api/node/{uuid}/{instance}/label/{first_coord_str}")
# r.raise_for_status()
# fetched_label = r.json()["Label"]
# assert fetched_label == BODY_ID, f"Expected {BODY_ID}, got {fetched_label}"

