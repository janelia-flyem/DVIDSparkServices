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
    print("DEBUG: FAIL: Watershed produced very high label values. Max was {}".format(unique_labels.max()))
    sys.exit(1)

# if ((unique_labels != np.arange(1, len(unique_labels)+1)).all()): 
#     print("DEBUG: FAIL: Watershed produced non-sequential label values.")
#     np.save( dirpath + '/temp_data/test_segmentation_zyx.npy', test_segmentation_zyx )
#     sys.exit(1)

# For test_seg_with_roi, the ROI does not include the corners
corner_data = np.concatenate([ test_segmentation_zyx[0:64, 0:64, 0:64],
                               test_segmentation_zyx[0:64, 0:64, 192:256],
                               test_segmentation_zyx[0:64, 192:256, 0:64],
                               test_segmentation_zyx[0:64, 192:256, 192:256],
                               test_segmentation_zyx[192:256, 0:64, 0:64],
                               test_segmentation_zyx[192:256, 0:64, 192:256],
                               test_segmentation_zyx[192:256, 192:256, 0:64],
                               test_segmentation_zyx[192:256, 192:256, 192:256] ])

if corner_data.any():
    print("DEBUG: FAIL: The ROI in this test excludes the corner blocks, so the watershed output should be zero there.")
    sys.exit(1)

print("DEBUG: Watershed test passed.")
sys.exit(0)
