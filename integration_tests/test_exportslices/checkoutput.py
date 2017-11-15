import os
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

input_service = DVIDNodeService(str(config['input']['source']['server']), str(config['input']['source']['uuid']))
input_name = config['input']['source']['grayscale-name']
input_bb_xyz = config['input']['geometry']['bounding-box']
input_bb_zyx = np.array(input_bb_xyz)[:,::-1]
input_shape = input_bb_zyx[1] - input_bb_zyx[0]

input_volume = input_service.get_gray3D(input_name, input_shape, input_bb_zyx[0])

os.chdir(f'{dirpath}/temp_data')

import vigra
for z in range(input_bb_zyx[0,0], input_bb_zyx[1,0]):
    z_slice = vigra.impex.readImage(config['output']['source']['slice-path-format'].format(z))
    assert (z_slice.withAxes('yx') == input_volume[z-input_bb_zyx[0]]).all()

print("DEBUG: ExportSlices test passed.")
sys.exit(0)
