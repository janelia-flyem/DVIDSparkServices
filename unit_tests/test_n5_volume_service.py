import os
import unittest

import numpy as np

import DVIDSparkServices
from DVIDSparkServices.io_util.volume_service import N5VolumeServiceReader, GrayscaleVolumeSchema

from DVIDSparkServices.json_util import validate_and_inject_defaults
from DVIDSparkServices.util import box_to_slicing

TEST_VOLUME_N5 = os.path.dirname(DVIDSparkServices.__file__) + '/../integration_tests/resources/volume-256.n5'
TEST_VOLUME_RAW = os.path.dirname(DVIDSparkServices.__file__) + '/../integration_tests/resources/grayscale-256-256-256-uint8.bin'

class TestN5VolumeService(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        with open(TEST_VOLUME_RAW, 'rb') as f_raw:
            cls.RAW_VOLUME_DATA = np.frombuffer(f_raw.read(), dtype=np.uint8).reshape((256,256,256))

        cls.VOLUME_CONFIG = {
          "n5": {
            "path": TEST_VOLUME_N5,
            "dataset-name": "s0"
          }
        }
        
        validate_and_inject_defaults(cls.VOLUME_CONFIG, GrayscaleVolumeSchema)
    
    def test_full_volume(self):
        reader = N5VolumeServiceReader(self.VOLUME_CONFIG, os.getcwd())
        assert (reader.bounding_box_zyx == [(0,0,0), (256,256,256)]).all()
        full_from_n5 = reader.get_subvolume(reader.bounding_box_zyx)
        assert full_from_n5.shape == self.RAW_VOLUME_DATA.shape
        assert (full_from_n5 == self.RAW_VOLUME_DATA).all()

    def test_slab(self):
        box = np.array([(64,0,0), (128,256,256)])
        
        slab_from_raw = self.RAW_VOLUME_DATA[box_to_slicing(*box)]

        reader = N5VolumeServiceReader(self.VOLUME_CONFIG, os.getcwd())
        slab_from_n5 = reader.get_subvolume(box)

        assert slab_from_n5.shape == slab_from_raw.shape, \
            f"Wrong shape: Expected {slab_from_raw.shape}, Got {slab_from_n5.shape}"
        assert (slab_from_n5 == slab_from_raw).all()

    def test_multiscale(self):
        reader = N5VolumeServiceReader(self.VOLUME_CONFIG, os.getcwd())
        assert (reader.bounding_box_zyx == [(0,0,0), (256,256,256)]).all()
        
        full_from_n5 = reader.get_subvolume(reader.bounding_box_zyx // 4, 2)
        
        assert (full_from_n5.shape == np.array(self.RAW_VOLUME_DATA.shape) // 4).all()
        assert (full_from_n5 == self.RAW_VOLUME_DATA[::4, ::4, ::4]).all()

if __name__ == "__main__":
    unittest.main()
