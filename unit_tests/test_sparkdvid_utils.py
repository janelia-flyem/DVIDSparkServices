import unittest

import numpy as np

from libdvid import DVIDNodeService, DVIDServerService

from neuclease.util import box_to_slicing
from DVIDSparkServices.reconutils.downsample import downsample_binary_3d_suppress_zero
from DVIDSparkServices.sparkdvid.sparkdvid import sparkdvid

TEST_DVID_SERVER = "http://127.0.0.1:8000"

class Test_get_union_block_mask_for_bodies(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        server_service = DVIDServerService(TEST_DVID_SERVER)
        cls.uuid = server_service.create_new_repo("foo", "bar")
        cls.instance = 'labels_test_mask_utils'

        cls.labels = np.zeros( (256,256,256), np.uint64 )

        cls.labels[50, 50, 40:140] = 1
        cls.labels[50, 60:200, 30] = 2
        
        ns = DVIDNodeService( TEST_DVID_SERVER, cls.uuid )
        ns.create_label_instance(cls.instance, 64)
        ns.put_labels3D(cls.instance, cls.labels, (0,0,0))
    
    def test_get_union_mask_for_bodies(self):
        union_mask, box, blocksize = sparkdvid.get_union_block_mask_for_bodies(TEST_DVID_SERVER, self.uuid, self.instance, [1,2])
        expected, _  = downsample_binary_3d_suppress_zero( self.labels.astype(bool), 64 )
        assert blocksize == (64, 64, 64)
        assert (expected[box_to_slicing(*box)] == union_mask).all()

if __name__ == "__main__":
    unittest.main()
