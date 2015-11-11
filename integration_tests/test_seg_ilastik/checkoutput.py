from __future__ import print_function
import sys
import json
import numpy as np
from gala.evaluate import rand_index
from libdvid import DVIDNodeService

dirpath = sys.argv[1]

reference_segmentation_path = dirpath + "/outputs/reference-segmentation.npz"
reference_segmentation_xyz = np.load(reference_segmentation_path).items()[0][1].transpose()

with open(dirpath + "/temp_data/config.json") as f:
    config = json.load(f)

node_service = DVIDNodeService(str(config['dvid-info']['dvid-server']), str(config['dvid-info']['uuid']))
test_segmentation_xyz = node_service.get_labels3D(str(config['dvid-info']['segmentation-name']), reference_segmentation_xyz.shape, (0,0,0))

score = rand_index(reference_segmentation_xyz, test_segmentation_xyz)
if abs(score - 1.0) < 0.01:
    print("DEBUG: Segmentation output matches reference with rand_index: {}".format(score))
    sys.exit(0)
else:
    print("DEBUG: FAIL: Segmentation output does not match reference. rand_index: {}".format(score))
    sys.exit(1)
