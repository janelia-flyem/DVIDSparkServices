import csv
import sys
import glob
import yaml
from itertools import chain
import numpy as np
from libdvid import DVIDNodeService

dirpath = sys.argv[1]

#import os
#import DVIDSparkServices
#dirpath = os.path.dirname(DVIDSparkServices.__file__) + '/../integration_tests/test_copyseg_remapped'

configs = glob.glob(dirpath + "/temp_data/config.*")
assert len(configs) == 1, "Why does the temp_dir have more than one config.* file?"

with open(configs[0], 'r') as f:
    config = yaml.load(f)

server = config['dvid-info']['dvid']['server']
uuid = config['dvid-info']['dvid']['uuid']
meshes_instance = config['dvid-info']['dvid']['meshes-destination']

import requests
r = requests.get(f'http://{server}/api/node/{uuid}/{meshes_instance}/keys')
r.raise_for_status()

keys = r.json()
encoded_body_ids = np.fromiter(map(lambda k: int(k[:-len(".tar")]), keys), int)

keyEncodeLevel0 = 10000000000000
stored_bodies = encoded_body_ids - keyEncodeLevel0

expected_bodies = config["mesh-config"]["storage"]["subset-bodies"]
assert set(stored_bodies) == set(expected_bodies), \
    f"Stored bodies didn't match expected subset: {stored_bodies.tolist()} vs {expected_bodies}"
