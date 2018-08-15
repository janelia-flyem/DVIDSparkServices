import sys
import glob

import yaml

import numpy as np
import pandas as pd

from libdvid import DVIDNodeService

dirpath = sys.argv[1]
#import os
#dirpath = os.path.dirname(__file__)

configs = glob.glob(f"{dirpath}/temp_data/config.*")
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

points = pd.read_csv(f'{dirpath}/points.csv', header=0)[['z', 'y', 'x']]
points.sort_values(['z', 'y', 'x'], inplace=True)

offset_points = points - input_bb_zyx[0]
points['label'] = input_volume[tuple(offset_points.values.transpose())]

results = pd.read_csv(dirpath + '/temp_data/point-samples.csv')
if (results[['z', 'y', 'x', 'label']].values != points[['z', 'y', 'x', 'label']].values).all():
    print("DEBUG: FAIL: point sample file does not match expected!")
    sys.exit(1)

if 'coord_sum' not in results.columns:
    print("DEBUG: FAIL: extraneous input columns were not preserved in the output!")
    sys.exit(1)
    
if not (results['coord_sum'] == results[['x', 'y', 'z']].sum(axis=1)).all():
    print("DEBUG: FAIL: extraneous input columns were not preserved correctly!")
    sys.exit(1)
