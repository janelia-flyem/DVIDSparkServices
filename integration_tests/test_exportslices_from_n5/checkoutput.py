import os
import sys
import glob
import yaml
import numpy as np
import z5py

from DVIDSparkServices.util import bb_to_slicing

dirpath = sys.argv[1]

configs = glob.glob(dirpath + "/temp_data/config.*")
assert len(configs) == 1, "Why does the temp_dir have more than one config.* file?"

with open(configs[0], 'r') as f:
    config = yaml.load(f)


n5_file = z5py.File(dirpath + '/../resources/volume-256.n5')

dset_name = config['input']['n5']['dataset-name']
input_bb_xyz = config['input']['geometry']['bounding-box']
input_bb_zyx = np.array(input_bb_xyz)[:,::-1]
input_shape = input_bb_zyx[1] - input_bb_zyx[0]

input_volume = n5_file[dset_name][bb_to_slicing(*input_bb_zyx.tolist())]
assert (input_volume.shape == input_shape).all(), "Error reading reference N5 volume -- bad shape??"

os.chdir(f'{dirpath}/temp_data')

import vigra
for z in range(input_bb_zyx[0,0], input_bb_zyx[1,0]):
    z_slice = vigra.impex.readImage(config['output']['slice-files']['slice-path-format'].format(z))
    assert (z_slice.withAxes('yx') == input_volume[z-input_bb_zyx[0]]).all()

print("DEBUG: ExportSlices test passed.")
sys.exit(0)
