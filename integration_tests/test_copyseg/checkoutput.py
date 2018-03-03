import sys
import glob
import yaml
import numpy as np
import pandas as pd
from libdvid import DVIDNodeService
from IPython.core.history import sqlite3

dirpath = sys.argv[1]

configs = glob.glob(dirpath + "/temp_data/config.*")
assert len(configs) == 1, "Why does the temp_dir have more than one config.* file?"

with open(configs[0], 'r') as f:
    config = yaml.load(f)

input_service = DVIDNodeService(str(config['input']['dvid']['server']), str(config['input']['dvid']['uuid']))
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
stats_path = dirpath + f'/temp_data/block-statistics.sqlite'
with sqlite3.connect(stats_path) as conn:
    df = pd.read_sql('SELECT * from block_stats', conn)
segment_counts_from_stats = df[['segment_id', 'count']].groupby('segment_id').sum()['count']
assert 0 not in segment_counts_from_stats, "Segment 0 should not be included in block statistics"
segment_counts_from_stats.sort_index(inplace=True)

segment_counts_from_output_seg = pd.Series(output_volume.reshape(-1)).value_counts(sort=False)
segment_counts_from_output_seg = segment_counts_from_output_seg[segment_counts_from_output_seg.index != 0]
segment_counts_from_output_seg.sort_index(inplace=True)

from IPython.core.debugger import set_trace
set_trace()

assert (segment_counts_from_output_seg == segment_counts_from_stats).all()

#     
#     ##
#     ## Check exported statistics
#     ##
#     def check_segment_stats(segment):
#         binary = (output_volume == segment)
#         row = df[df.segment == segment].iloc[0]
#         assert row['voxel_count'] == binary.sum()
# 
#         coords = np.transpose(binary.nonzero())
#         assert (row['bounding_box_start'] == coords.min(axis=0)).all()
#         assert (row['bounding_box_stop'] == 1+coords.max(axis=0)).all()
#         
#         if 'block_list' in df.columns:
#             block_indexes = coords // 64
#             sorted_blocks = 64*np.array(sorted(set(map(tuple, block_indexes))))
#             assert (row['block_list'] == sorted_blocks).all()
# 
#     # Check the 10 largest and 10 smallest segments
#     sorted_df = df.sort_values('voxel_count', ascending=False)
#     largest_segments = list(sorted_df['segment'][:10])
#     smallest_segments = list(sorted_df['segment'][-10:])
#     for segment in largest_segments + smallest_segments:
#         if segment != 0:
#             check_segment_stats(segment)

print("DEBUG: CopySegmentation test passed.")
sys.exit(0)
