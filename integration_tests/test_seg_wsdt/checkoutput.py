from __future__ import print_function
import sys
import json
import numpy as np
from DVIDSparkServices.reconutils.misc import compute_vi
from libdvid import DVIDNodeService

dirpath = sys.argv[1]

with open(dirpath + "/temp_data/config.json") as f:
    config = json.load(f)

node_service = DVIDNodeService(str(config['dvid-info']['dvid-server']), str(config['dvid-info']['uuid']))
test_segmentation_xyz = node_service.get_labels3D(str(config['dvid-info']['segmentation-name']), (256, 256, 256), (0,0,0))

unique_labels = np.unique(test_segmentation_xyz)
if unique_labels.max() > 1e9:
    print("DEBUG: FAIL: Watershed produced very high label values. Max was {}".format(unique_labels.max()))
    sys.exit(1)
if ((unique_labels != np.arange(1, len(unique_labels)+1)).all()): 
    print("DEBUG: FAIL: Watershed produced non-sequential label values.")
    sys.exit(1)

print("DEBUG: Watershed test passed.")
sys.exit(0)
