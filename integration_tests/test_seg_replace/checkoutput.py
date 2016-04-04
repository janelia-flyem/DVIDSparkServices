from __future__ import print_function
import sys
import json
import numpy as np
from libdvid import DVIDNodeService

dirpath = sys.argv[1]

with open(dirpath + "/temp_data/config.json") as f:
    config = json.load(f)

node_service = DVIDNodeService(str(config['dvid-info']['dvid-server']), str(config['dvid-info']['uuid']))
test_segmentation_zyx = node_service.get_labels3D(str(config['dvid-info']['segmentation-name']), (256, 256, 256), (0,0,0))

unique_labels = np.unique(test_segmentation_zyx)
if unique_labels.max() > 1e9:
    print("DEBUG: FAIL: Replace test produced very high label values. Max was {}".format(unique_labels.max()))
    sys.exit(1)

print("DEBUG: Body replace test passed.")
sys.exit(0)
