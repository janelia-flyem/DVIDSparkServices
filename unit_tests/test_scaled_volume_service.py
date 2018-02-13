import os
import copy
import unittest

import numpy as np

import DVIDSparkServices
from DVIDSparkServices.io_util.volume_service import VolumeService, N5VolumeServiceReader, GrayscaleVolumeSchema, ScaledVolumeService
from DVIDSparkServices.json_util import validate_and_inject_defaults
from DVIDSparkServices.util import box_to_slicing
from DVIDSparkServices.reconutils.downsample import downsample_raw
from skimage.util.shape import view_as_blocks

TEST_VOLUME_N5 = os.path.dirname(DVIDSparkServices.__file__) + '/../integration_tests/resources/volume-256.n5'
TEST_VOLUME_RAW = os.path.dirname(DVIDSparkServices.__file__) + '/../integration_tests/resources/grayscale-256-256-256-uint8.bin'

class TestScaledVolumeService(unittest.TestCase):
    
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
              "bounding-box": [[0,0,0], [100,200,256]],
              "available-scales": [0] # Ensure only the first scale is used.
          }
        }
        
        validate_and_inject_defaults(cls.VOLUME_CONFIG, GrayscaleVolumeSchema)

        # First, N5 alone
        n5_reader = N5VolumeServiceReader(cls.VOLUME_CONFIG, os.getcwd())
        assert (n5_reader.bounding_box_zyx == [(0,0,0), (256,200,100)]).all()
        full_from_n5 = n5_reader.get_subvolume(n5_reader.bounding_box_zyx)
        assert full_from_n5.shape == cls.RAW_VOLUME_DATA.shape
        assert (full_from_n5 == cls.RAW_VOLUME_DATA).all()
    
        cls.n5_reader = n5_reader
        cls.full_from_n5 = full_from_n5
    
    def test_full_volume_no_scaling(self):
        n5_reader = self.n5_reader
        full_from_n5 = self.full_from_n5
        
        scaled_reader = ScaledVolumeService(n5_reader, 0)
        assert (scaled_reader.bounding_box_zyx == n5_reader.bounding_box_zyx).all()
        assert (scaled_reader.preferred_message_shape == n5_reader.preferred_message_shape).all()
        assert scaled_reader.block_width == n5_reader.block_width
        assert scaled_reader.dtype == n5_reader.dtype

        full_scaled = scaled_reader.get_subvolume(scaled_reader.bounding_box_zyx)
        assert (full_scaled == full_from_n5).all()
        assert full_scaled.flags.c_contiguous

    def test_full_volume_downsample_1(self):
        n5_reader = self.n5_reader
        full_from_n5 = self.full_from_n5

        # Scale 1
        scaled_config = copy.deepcopy(self.VOLUME_CONFIG)
        scaled_config["rescale-level"] = 1
        scaled_reader = VolumeService.create_from_config(scaled_config, '.')
        
        assert (scaled_reader.bounding_box_zyx == n5_reader.bounding_box_zyx // 2).all()
        assert (scaled_reader.preferred_message_shape == n5_reader.preferred_message_shape // 2).all()
        assert scaled_reader.block_width == n5_reader.block_width // 2
        assert scaled_reader.dtype == n5_reader.dtype

        full_scaled = scaled_reader.get_subvolume(scaled_reader.bounding_box_zyx)
        assert (full_scaled == downsample_raw(full_from_n5, 1)[-1]).all()
        assert full_scaled.flags.c_contiguous
        
    def test_full_volume_upsample_1(self):
        n5_reader = self.n5_reader
        full_from_n5 = self.full_from_n5

        # Scale -1
        scaled_reader = ScaledVolumeService(n5_reader, -1)
        assert (scaled_reader.bounding_box_zyx == n5_reader.bounding_box_zyx * 2).all()
        assert (scaled_reader.preferred_message_shape == n5_reader.preferred_message_shape * 2).all()
        assert scaled_reader.block_width == n5_reader.block_width * 2
        assert scaled_reader.dtype == n5_reader.dtype

        full_scaled = scaled_reader.get_subvolume(scaled_reader.bounding_box_zyx)
        assert (full_from_n5 == full_scaled[::2,::2,::2]).all()
        assert full_scaled.flags.c_contiguous

    def test_subvolume_no_scaling(self):
        n5_reader = self.n5_reader
        full_from_n5 = self.full_from_n5
        
        box = np.array([[13, 15, 20], [100, 101, 91]])
        subvol_from_n5 = full_from_n5[box_to_slicing(*box)].copy('C')
        
        scaled_reader = ScaledVolumeService(n5_reader, 0)
        subvol_scaled = scaled_reader.get_subvolume(box)

        assert (subvol_scaled.shape == box[1] - box[0]).all()
        assert subvol_from_n5.shape == subvol_scaled.shape, \
            f"{subvol_scaled.shape} != {subvol_from_n5.shape}"
        assert (subvol_scaled == subvol_from_n5).all()
        assert subvol_scaled.flags.c_contiguous

    def test_subvolume_downsample_1(self):
        n5_reader = self.n5_reader
        full_from_n5 = self.full_from_n5
        
        down_box = np.array([[13, 15, 20], [20, 40, 41]])
        up_box = 2*down_box
        up_subvol_from_n5 = full_from_n5[box_to_slicing(*up_box)]
        down_subvol_from_n5 = downsample_raw(up_subvol_from_n5, 1)[-1]
        
        # Scale 1
        scaled_reader = ScaledVolumeService(n5_reader, 1)
        subvol_scaled = scaled_reader.get_subvolume(down_box)

        assert (subvol_scaled.shape == down_box[1] - down_box[0]).all()
        assert down_subvol_from_n5.shape == subvol_scaled.shape, \
            f"{subvol_scaled.shape} != {down_subvol_from_n5.shape}"
        assert (subvol_scaled == down_subvol_from_n5).all()
        assert subvol_scaled.flags.c_contiguous

    def test_subvolume_upsample_1(self):
        n5_reader = self.n5_reader
        full_from_n5 = self.full_from_n5

        up_box = np.array([[13, 15, 20], [100, 101, 91]])
        full_upsampled_vol = np.empty( 2*np.array(full_from_n5.shape), dtype=n5_reader.dtype )
        up_view = view_as_blocks(full_upsampled_vol, (2,2,2))
        up_view[:] = full_from_n5[:, :, :, None, None, None]
        up_subvol_from_n5 = full_upsampled_vol[box_to_slicing(*up_box)]
                
        # Scale -1
        scaled_reader = ScaledVolumeService(n5_reader, -1)
        subvol_scaled = scaled_reader.get_subvolume(up_box)

        assert (subvol_scaled.shape == up_box[1] - up_box[0]).all()
        assert up_subvol_from_n5.shape == subvol_scaled.shape, \
            f"{subvol_scaled.shape} != {up_subvol_from_n5.shape}"
        assert (subvol_scaled == up_subvol_from_n5).all()
        assert subvol_scaled.flags.c_contiguous


if __name__ == "__main__":
    unittest.main()
