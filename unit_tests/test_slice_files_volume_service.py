import os
import unittest

import numpy as np

import DVIDSparkServices
from DVIDSparkServices.io_util.volume_service import SliceFilesVolumeServiceReader, SliceFilesVolumeServiceWriter
from DVIDSparkServices.workflows.common_schemas import GrayscaleVolumeSchema

from DVIDSparkServices.json_util import validate_and_inject_defaults
from DVIDSparkServices.util import box_to_slicing

TEST_VOLUME_SLICE_PATH_FORMAT = os.path.dirname(DVIDSparkServices.__file__) + '/../integration_tests/resources/volume-256-pngs/{:05d}.png'
TEST_VOLUME_RAW = os.path.dirname(DVIDSparkServices.__file__) + '/../integration_tests/resources/grayscale-256-256-256-uint8.bin'

class TestSliceFilesVolumeService(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        with open(TEST_VOLUME_RAW, 'rb') as f_raw:
            cls.RAW_VOLUME_DATA = np.frombuffer(f_raw.read(), dtype=np.uint8).reshape((256,256,256))

        cls.VOLUME_CONFIG = {
          "slice-files": {
            "slice-path-format": TEST_VOLUME_SLICE_PATH_FORMAT
          }
        }
        
        validate_and_inject_defaults(cls.VOLUME_CONFIG, GrayscaleVolumeSchema)
    
    def test_full_volume(self):
        try:
            reader = SliceFilesVolumeServiceReader(self.VOLUME_CONFIG, os.getcwd())
        except SliceFilesVolumeServiceReader.NoSlicesFoundError:
            raise RuntimeError("Test data could not be found. "
                               "It is supposed to be generated when you run the INTEGRATION TESTS. "
                               "Please run (or at least start) the integration tests first.")
        
        assert (reader.bounding_box_zyx == [(0,0,0), (256,256,256)]).all()
        full_from_slices = reader.get_subvolume(reader.bounding_box_zyx)
        assert full_from_slices.shape == self.RAW_VOLUME_DATA.shape
        assert (full_from_slices == self.RAW_VOLUME_DATA).all()

    def test_slab(self):
        box = np.array([(64,0,0), (128,256,256)])
        
        slab_from_raw = self.RAW_VOLUME_DATA[box_to_slicing(*box)]

        reader = SliceFilesVolumeServiceReader(self.VOLUME_CONFIG, os.getcwd())
        slab_from_slices = reader.get_subvolume(box)

        assert slab_from_slices.shape == slab_from_raw.shape, \
            f"Wrong shape: Expected {slab_from_raw.shape}, Got {slab_from_slices.shape}"
        assert (slab_from_slices == slab_from_raw).all()


if __name__ == "__main__":
    unittest.main()
