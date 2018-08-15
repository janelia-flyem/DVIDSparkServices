import os
import unittest

import numpy as np

from neuclease.util import box_to_slicing

import DVIDSparkServices
from DVIDSparkServices.io_util.volume_service import SliceFilesVolumeServiceReader, SliceFilesVolumeServiceWriter, GrayscaleVolumeSchema
from DVIDSparkServices.json_util import validate_and_inject_defaults

TEST_VOLUME_SLICE_PATH_FORMAT = os.path.dirname(DVIDSparkServices.__file__) + '/../integration_tests/resources/volume-256-pngs/{:05d}.png'
TEST_VOLUME_RAW = os.path.dirname(DVIDSparkServices.__file__) + '/../integration_tests/resources/grayscale-256-256-256-uint8.bin'

class TestSliceFilesVolumeServiceReader(unittest.TestCase):
    
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
    
    def test_read_full_volume(self):
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

    def test_read_slab(self):
        box = np.array([(64,0,0), (128,256,256)])
        
        slab_from_raw = self.RAW_VOLUME_DATA[box_to_slicing(*box)]

        reader = SliceFilesVolumeServiceReader(self.VOLUME_CONFIG, os.getcwd())
        slab_from_slices = reader.get_subvolume(box)

        assert slab_from_slices.shape == slab_from_raw.shape, \
            f"Wrong shape: Expected {slab_from_raw.shape}, Got {slab_from_slices.shape}"
        assert (slab_from_slices == slab_from_raw).all()

class TestSliceFilesVolumeServiceWriter(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        with open(TEST_VOLUME_RAW, 'rb') as f_raw:
            cls.RAW_VOLUME_DATA = np.frombuffer(f_raw.read(), dtype=np.uint8).reshape((256,256,256))

        cls.VOLUME_CONFIG = {
          "slice-files": {
            "slice-path-format": '/tmp/test-slices/{:05d}.png'
          },
          "geometry": {
              "bounding-box": [[0,0,0],
                               list(cls.RAW_VOLUME_DATA.shape[::-1])]
            }
        }
        
        validate_and_inject_defaults(cls.VOLUME_CONFIG, GrayscaleVolumeSchema)

    def test_write_full_volume(self):
        writer = SliceFilesVolumeServiceWriter(self.VOLUME_CONFIG, os.getcwd())
        writer.write_subvolume(self.RAW_VOLUME_DATA, (0,0,0))
        
        reader = SliceFilesVolumeServiceReader(self.VOLUME_CONFIG, os.getcwd())
        written_vol = reader.get_subvolume(writer.bounding_box_zyx)
        assert (written_vol == self.RAW_VOLUME_DATA).all()

    def test_write_slab(self):
        writer = SliceFilesVolumeServiceWriter(self.VOLUME_CONFIG, os.getcwd())

        slab_box = writer.bounding_box_zyx.copy()
        slab_box[:,0] = [64,128]

        writer.write_subvolume(self.RAW_VOLUME_DATA[64:128], (64,0,0))
        
        reader = SliceFilesVolumeServiceReader(self.VOLUME_CONFIG, os.getcwd())
        written_vol = reader.get_subvolume(slab_box)
        assert (written_vol == self.RAW_VOLUME_DATA[64:128]).all()


if __name__ == "__main__":
    unittest.main()
