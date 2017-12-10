import sys
import glob
import yaml
import numpy as np
from libdvid import DVIDNodeService

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

for index in range(len(config['outputs'])):
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

print("DEBUG: CopySegmentation test passed.")
sys.exit(0)
