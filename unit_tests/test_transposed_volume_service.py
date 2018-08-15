import os
import unittest

import numpy as np

import DVIDSparkServices
from DVIDSparkServices.io_util.volume_service import N5VolumeServiceReader, GrayscaleVolumeSchema, TransposedVolumeService
from DVIDSparkServices.json_util import validate_and_inject_defaults
from neuclease.util import box_to_slicing

TEST_VOLUME_N5 = os.path.dirname(DVIDSparkServices.__file__) + '/../integration_tests/resources/volume-256.n5'
TEST_VOLUME_RAW = os.path.dirname(DVIDSparkServices.__file__) + '/../integration_tests/resources/grayscale-256-256-256-uint8.bin'

class TestTransposedVolumeService(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        with open(TEST_VOLUME_RAW, 'rb') as f_raw:
            cls.RAW_VOLUME_DATA = np.frombuffer(f_raw.read(), dtype=np.uint8).reshape((256,256,256))
            cls.RAW_VOLUME_DATA = cls.RAW_VOLUME_DATA[:256, :200, :100]

        cls.VOLUME_CONFIG = {
          "n5": {
            "path": TEST_VOLUME_N5,
            "dataset-name": "s0"
          },
          "geometry": {
              "bounding-box": [[0,0,0], [100,200,256]]
          }
        }
        
        validate_and_inject_defaults(cls.VOLUME_CONFIG, GrayscaleVolumeSchema)

    def test_full_volume(self):
        # First, N5 alone
        n5_reader = N5VolumeServiceReader(self.VOLUME_CONFIG, os.getcwd())
        assert (n5_reader.bounding_box_zyx == [(0,0,0), (256,200,100)]).all()
        full_from_n5 = n5_reader.get_subvolume(n5_reader.bounding_box_zyx)
        assert full_from_n5.shape == self.RAW_VOLUME_DATA.shape
        assert (full_from_n5 == self.RAW_VOLUME_DATA).all()

        # Check API
        transposed_reader = TransposedVolumeService(n5_reader, ['x', 'y', 'z'])
        assert transposed_reader.base_service == n5_reader
        assert len(transposed_reader.service_chain) == 2
        assert transposed_reader.service_chain[0] == transposed_reader
        assert transposed_reader.service_chain[1] == n5_reader
        
        # Now use transposed reader, but with identity transposition
        transposed_reader = TransposedVolumeService(n5_reader, ['x', 'y', 'z'])
        assert (transposed_reader.bounding_box_zyx == n5_reader.bounding_box_zyx).all()
        assert (transposed_reader.preferred_message_shape == n5_reader.preferred_message_shape).all()
        assert transposed_reader.block_width == n5_reader.block_width
        assert transposed_reader.dtype == n5_reader.dtype
        
        full_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx)
        assert (full_transposed == full_from_n5).all()
        assert full_transposed.flags.c_contiguous

        # Now transpose x and y (reflect across diagonal line at y=x)
        transposed_reader = TransposedVolumeService(n5_reader, ['y', 'x', 'z'])
        assert (transposed_reader.bounding_box_zyx == n5_reader.bounding_box_zyx[:, (0,2,1)]).all()
        assert (transposed_reader.preferred_message_shape == n5_reader.preferred_message_shape[((0,2,1),)]).all()
        assert transposed_reader.block_width == n5_reader.block_width
        assert transposed_reader.dtype == n5_reader.dtype
        
        full_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx)
        assert (full_transposed == full_from_n5.transpose(0, 2, 1)).all()
        assert full_transposed.flags.c_contiguous

        # Invert x and y (but don't transpose)
        # Equivalent to 180 degree rotation
        transposed_reader = TransposedVolumeService(n5_reader, ['1-x', '1-y', 'z'])
        assert (transposed_reader.bounding_box_zyx == n5_reader.bounding_box_zyx).all()
        assert (transposed_reader.preferred_message_shape == n5_reader.preferred_message_shape).all()
        assert transposed_reader.block_width == n5_reader.block_width
        assert transposed_reader.dtype == n5_reader.dtype
        
        full_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx)
        assert (full_transposed == full_from_n5[:,::-1, ::-1]).all()
        assert full_transposed.flags.c_contiguous

        # XY 90 degree rotation, clockwise about the Z axis
        transposed_reader = TransposedVolumeService(n5_reader, TransposedVolumeService.XY_CLOCKWISE_90)
        assert (transposed_reader.bounding_box_zyx == n5_reader.bounding_box_zyx[:, (0,2,1)]).all()
        assert (transposed_reader.preferred_message_shape == n5_reader.preferred_message_shape[((0,2,1),)]).all()
        assert transposed_reader.block_width == n5_reader.block_width
        assert transposed_reader.dtype == n5_reader.dtype
        
        full_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx)
        assert (full_transposed == full_from_n5[:, ::-1, :].transpose(0,2,1)).all()
        assert full_transposed.flags.c_contiguous
        
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
        assert subvol_transposed.flags.c_contiguous

        # XZ degree rotation, clockwise about the Y axis
        transposed_reader = TransposedVolumeService(n5_reader, TransposedVolumeService.XZ_CLOCKWISE_90)
        assert (transposed_reader.bounding_box_zyx == n5_reader.bounding_box_zyx[:, (2,1,0)]).all()
        assert (transposed_reader.preferred_message_shape == n5_reader.preferred_message_shape[((2,1,0),)]).all()
        assert transposed_reader.block_width == n5_reader.block_width
        assert transposed_reader.dtype == n5_reader.dtype
        
        full_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx)
        assert (full_transposed == full_from_n5[::-1, :, :].transpose(2,1,0)).all()
        assert full_transposed.flags.c_contiguous
        
        # Check the corners of the first plane: should be rotated clockwise
        y_slice_n5 = full_from_n5[:, 0, :]
        y_slice_transposed = full_transposed[:, 0, :]
        assert y_slice_n5[0,0] == y_slice_transposed[0,-1]
        assert y_slice_n5[0,-1] == y_slice_transposed[-1,-1]
        assert y_slice_n5[-1,-1] == y_slice_transposed[-1,0]
        assert y_slice_n5[-1,0] == y_slice_transposed[0,0]

        # YZ degree rotation, clockwise about the X axis
        transposed_reader = TransposedVolumeService(n5_reader, TransposedVolumeService.YZ_CLOCKWISE_90)
        assert (transposed_reader.bounding_box_zyx == n5_reader.bounding_box_zyx[:, (1,0,2)]).all()
        assert (transposed_reader.preferred_message_shape == n5_reader.preferred_message_shape[((1,0,2),)]).all()
        assert transposed_reader.block_width == n5_reader.block_width
        assert transposed_reader.dtype == n5_reader.dtype
        
        full_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx)
        assert (full_transposed == full_from_n5[::-1, :, :].transpose(1,0,2)).all()
        assert full_transposed.flags.c_contiguous
        
        # Check the corners of the first plane: should be rotated clockwise
        x_slice_n5 = full_from_n5[:, :, 0]
        x_slice_transposed = full_transposed[:, :, 0]
        assert x_slice_n5[0,0] == x_slice_transposed[0,-1]
        assert x_slice_n5[0,-1] == x_slice_transposed[-1,-1]
        assert x_slice_n5[-1,-1] == x_slice_transposed[-1,0]
        assert x_slice_n5[-1,0] == x_slice_transposed[0,0]

        # Multiple rotations (the hemibrain N5 -> DVID transform)
        transposed_reader = TransposedVolumeService(n5_reader, ['1-z', 'x', 'y'])
        assert (transposed_reader.bounding_box_zyx == n5_reader.bounding_box_zyx[:, (1,2,0)]).all()
        assert (transposed_reader.preferred_message_shape == n5_reader.preferred_message_shape[((1,2,0),)]).all()
        assert transposed_reader.block_width == n5_reader.block_width
        assert transposed_reader.dtype == n5_reader.dtype
        
        full_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx)
        assert (full_transposed == full_from_n5[::-1, :, :].transpose(1,2,0)).all()
        assert full_transposed.flags.c_contiguous

    def test_multiscale(self):
        SCALE = 2
        n5_reader = N5VolumeServiceReader(self.VOLUME_CONFIG, os.getcwd())

        # No transpose
        transposed_reader = TransposedVolumeService(n5_reader, TransposedVolumeService.NO_TRANSPOSE)
        from_n5 = n5_reader.get_subvolume(n5_reader.bounding_box_zyx // 2**SCALE, SCALE)
        from_transposed = transposed_reader.get_subvolume(n5_reader.bounding_box_zyx // 2**SCALE, SCALE)
        assert (from_transposed == from_n5).all() 

        # XZ degree rotation, clockwise about the Y axis
        transposed_reader = TransposedVolumeService(n5_reader, TransposedVolumeService.XY_CLOCKWISE_90)
        from_n5 = n5_reader.get_subvolume(n5_reader.bounding_box_zyx // 2**SCALE, SCALE)
        from_transposed = transposed_reader.get_subvolume(transposed_reader.bounding_box_zyx // 2**SCALE, SCALE)
        assert (from_transposed == from_n5[:,::-1,:].transpose(0,2,1)).all() 

if __name__ == "__main__":
    unittest.main()
