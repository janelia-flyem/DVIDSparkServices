from __future__ import print_function
import sys
import json
import numpy as np
from DVIDSparkServices.reconutils.misc import compute_vi
from libdvid import DVIDNodeService

dirpath = sys.argv[1]

reference_segmentation_path = dirpath + "/outputs/reference-segmentation.npz"
reference_segmentation_zyx = np.load(reference_segmentation_path).items()[0][1]

with open(dirpath + "/temp_data/config.json") as f:
    config = json.load(f)

node_service = DVIDNodeService(str(config['dvid-info']['dvid-server']), str(config['dvid-info']['uuid']))
test_segmentation_zyx = node_service.get_labels3D(str(config['dvid-info']['segmentation-name']), reference_segmentation_zyx.shape, (0,0,0))

merge, split = compute_vi(reference_segmentation_zyx, test_segmentation_zyx)
score = merge+split

if score < 0.01:
    print("DEBUG: Segmentation output matches reference with vi: {} + {} = {}".format(merge, split, score))
    sys.exit(0)
else:
    print("DEBUG: FAIL: Segmentation output does not match reference. vi {} + {} = {}".format(merge, split, score))
    np.save( dirpath + '/temp_data/test_segmentation_zyx.npy', test_segmentation_zyx )
    sys.exit(1)
