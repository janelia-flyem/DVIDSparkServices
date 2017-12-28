import os
import unittest

import numpy as np

import DVIDSparkServices
from DVIDSparkServices.io_util.volume_service import N5VolumeServiceReader, GrayscaleVolumeSchema, TransposedVolumeService
from DVIDSparkServices.json_util import validate_and_inject_defaults
from DVIDSparkServices.util import box_to_slicing

TEST_VOLUME_N5 = os.path.dirname(DVIDSparkServices.__file__) + '/../integration_tests/resources/volume-256.n5'
TEST_VOLUME_RAW = os.path.dirname(DVIDSparkServices.__file__) + '/../integration_tests/resources/grayscale-256-256-256-uint8.bin'

class TestTransposedVolumeService(unittest.TestCase):
    
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
        # First, N5 alone
        n5_reader = N5VolumeServiceReader(self.VOLUME_CONFIG, os.getcwd())
        assert (n5_reader.bounding_box_zyx == [(0,0,0), (256,256,256)]).all()
        full_from_n5 = n5_reader.get_subvolume(n5_reader.bounding_box_zyx)
        assert full_from_n5.shape == self.RAW_VOLUME_DATA.shape
        assert (full_from_n5 == self.RAW_VOLUME_DATA).all()

        # Now use transposed reader, but with identity transposition
        transposed_reader = TransposedVolumeService(n5_reader, ['z', 'y', 'x'])
        assert (transposed_reader.bounding_box_zyx == n5_reader.bounding_box_zyx).all()
        assert (transposed_reader.preferred_message_shape == n5_reader.preferred_message_shape).all()
        assert transposed_reader.block_width == n5_reader.block_width
        assert transposed_reader.dtype == n5_reader.dtype
        
        full_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx)
        assert (full_transposed == full_from_n5).all()

        # Now transpose x and y (reflect across diagonal line at y=x)
        transposed_reader = TransposedVolumeService(n5_reader, ['z', 'x', 'y'])
        assert (transposed_reader.bounding_box_zyx == n5_reader.bounding_box_zyx[:, (0,2,1)]).all()
        assert (transposed_reader.preferred_message_shape == n5_reader.preferred_message_shape[((0,2,1),)]).all()
        assert transposed_reader.block_width == n5_reader.block_width
        assert transposed_reader.dtype == n5_reader.dtype
        
        full_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx)
        assert (full_transposed == full_from_n5.transpose(0, 2, 1)).all()

        # Invert x and y (but don't transpose)
        # Equivalent to 180 degree rotation
        transposed_reader = TransposedVolumeService(n5_reader, ['z', '1-y', '1-x'])
        assert (transposed_reader.bounding_box_zyx == n5_reader.bounding_box_zyx).all()
        assert (transposed_reader.preferred_message_shape == n5_reader.preferred_message_shape).all()
        assert transposed_reader.block_width == n5_reader.block_width
        assert transposed_reader.dtype == n5_reader.dtype
        
        full_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx)
        assert (full_transposed == full_from_n5[:,::-1, ::-1]).all()

        # XY 90 degree rotation, clockwise about the Z axis
        transposed_reader = TransposedVolumeService(n5_reader, TransposedVolumeService.XY_CLOCKWISE_90)
        assert (transposed_reader.bounding_box_zyx == n5_reader.bounding_box_zyx).all()
        assert (transposed_reader.preferred_message_shape == n5_reader.preferred_message_shape).all()
        assert transposed_reader.block_width == n5_reader.block_width
        assert transposed_reader.dtype == n5_reader.dtype
        
        full_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx)
        assert (full_transposed == full_from_n5[:, ::-1, :].transpose(0,2,1)).all()
        
        # Check the corners of the first plane: should be rotated clockwise
        z_slice_n5 = full_from_n5[0]
        z_slice_transposed = full_transposed[0]
        assert z_slice_n5[0,0] == z_slice_transposed[0,-1]
        assert z_slice_n5[0,-1] == z_slice_transposed[-1,-1]
        assert z_slice_n5[-1,-1] == z_slice_transposed[-1,0]
        assert z_slice_n5[-1,0] == z_slice_transposed[0,0]

        # Verify that subvolume requests work correctly
        box = [[10,20,30], [20, 40, 60]]
        subvol_transposed = transposed_reader.get_subvolume(box)
        assert (subvol_transposed == full_transposed[box_to_slicing(*box)]).all()

        # XZ degree rotation, clockwise about the Y axis
        transposed_reader = TransposedVolumeService(n5_reader, TransposedVolumeService.XZ_CLOCKWISE_90)
        assert (transposed_reader.bounding_box_zyx == n5_reader.bounding_box_zyx).all()
        assert (transposed_reader.preferred_message_shape == n5_reader.preferred_message_shape).all()
        assert transposed_reader.block_width == n5_reader.block_width
        assert transposed_reader.dtype == n5_reader.dtype
        
        full_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx)
        assert (full_transposed == full_from_n5[::-1, :, :].transpose(2,1,0)).all()
        
        # Check the corners of the first plane: should be rotated clockwise
        y_slice_n5 = full_from_n5[:, 0, :]
        y_slice_transposed = full_transposed[:, 0, :]
        assert y_slice_n5[0,0] == y_slice_transposed[0,-1]
        assert y_slice_n5[0,-1] == y_slice_transposed[-1,-1]
        assert y_slice_n5[-1,-1] == y_slice_transposed[-1,0]
        assert y_slice_n5[-1,0] == y_slice_transposed[0,0]

        # YZ degree rotation, clockwise about the X axis
        transposed_reader = TransposedVolumeService(n5_reader, TransposedVolumeService.YZ_CLOCKWISE_90)
        assert (transposed_reader.bounding_box_zyx == n5_reader.bounding_box_zyx).all()
        assert (transposed_reader.preferred_message_shape == n5_reader.preferred_message_shape).all()
        assert transposed_reader.block_width == n5_reader.block_width
        assert transposed_reader.dtype == n5_reader.dtype
        
        full_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx)
        assert (full_transposed == full_from_n5[::-1, :, :].transpose(1,0,2)).all()
        
        # Check the corners of the first plane: should be rotated clockwise
        x_slice_n5 = full_from_n5[:, :, 0]
        x_slice_transposed = full_transposed[:, :, 0]
        assert x_slice_n5[0,0] == x_slice_transposed[0,-1]
        assert x_slice_n5[0,-1] == x_slice_transposed[-1,-1]
        assert x_slice_n5[-1,-1] == x_slice_transposed[-1,0]
        assert x_slice_n5[-1,0] == x_slice_transposed[0,0]

if __name__ == "__main__":
    unittest.main()
