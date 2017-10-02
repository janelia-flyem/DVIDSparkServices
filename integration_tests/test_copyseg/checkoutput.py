import sys
import json
import numpy as np
from libdvid import DVIDNodeService

dirpath = sys.argv[1]

with open(dirpath + "/temp_data/config.json") as f:
    config = json.load(f)

input_service = DVIDNodeService(str(config['input']['server']), str(config['input']['uuid']))
input_name = config['input']['segmentation-name']
input_bb_xyz = config['input']['bounding-box']
input_bb_zyx = np.array(input_bb_xyz)[:,::-1]
input_shape = input_bb_zyx[1] - input_bb_zyx[0]

input_volume = input_service.get_labels3D(input_name, input_shape, input_bb_zyx[0])

output_service = DVIDNodeService(str(config['output']['server']), str(config['output']['uuid']))
output_name = config['output']['segmentation-name']
output_bb_xyz = config['output']['bounding-box']
output_bb_zyx = np.array(output_bb_xyz)[:,::-1]
output_shape = output_bb_zyx[1] - output_bb_zyx[0]

output_volume = output_service.get_labels3D(output_name, output_shape, output_bb_zyx[0])

if not (input_volume == output_volume).all():
    print("DEBUG: FAIL: output volume does not correspond to input volume!")
    sys.exit(1)
else:
    print("DEBUG: CopySegmentation test passed.")
    sys.exit(0)
