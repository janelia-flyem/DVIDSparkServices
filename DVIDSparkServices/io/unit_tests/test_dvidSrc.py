import unittest
import numpy as np

from DVIDSparkServices.io.partitionSchema import volumePartition, VolumeOffset, VolumeSize, PartitionDims, partitionSchema

from DVIDSparkServices.io.dvidSrc import dvidSrc
from libdvid import DVIDNodeService, DVIDServerService

dvidserver = "http://127.0.0.1:8000"

class TestdvidSrc(unittest.TestCase):
    """Tests read from DVID and patching.

    Note: DVID server must be available at 127.0.0.1:8000.
    """
   
    def test_dvidfetchgray(self):
        """Check reading grayscale from DVID from partitions.

        This also checks basic iteration and overwrite of
        previous data.
        """
        
        service = DVIDServerService(dvidserver)
        uuid = service.create_new_repo("foo", "bar")
        
        ns = DVIDNodeService(dvidserver, uuid)
        ns.create_grayscale8("gray")
  
        arr = np.random.randint(255, size=(64,64,64)).astype(np.uint8)

        # load gray data
        ns.put_gray3D("gray", arr, (0,0,0))

        # read data
        schema = partitionSchema(PartitionDims(32,64,64))
        volpart = volumePartition(0, VolumeOffset(0,0,0))
        overwrite = np.random.randint(255, size=(64,64,64)).astype(np.uint8)
        partitions = schema.partition_data([(volpart, overwrite)])

        dvidreader = dvidSrc(dvidserver, uuid, "gray", partitions, maskonly=False)

        newparts = dvidreader.extract_volume()
        self.assertEqual(len(newparts), 2) 

        for (part, vol) in newparts:
            if part.get_offset().z == 0:
                match = np.array_equal(arr[0:32,:,:], vol)
                self.assertTrue(match)
            else:
                match = np.array_equal(arr[32:64,:,:], vol)
                self.assertTrue(match)

        # test iteration
        dvidreader2 = dvidSrc(dvidserver, uuid, "gray", partitions, maskonly=False)
        
        newparts2 = []
        for newpart in dvidreader2:
            self.assertEqual(len(newpart), 1) 
            newparts2.extend(newpart)
        self.assertEqual(len(newparts2), 2) 

        for (part, vol) in newparts2:
            if part.get_offset().z == 0:
                match = np.array_equal(arr[0:32,:,:], vol)
                self.assertTrue(match)
            else:
                match = np.array_equal(arr[32:64,:,:], vol)
                self.assertTrue(match)


    def test_dvidpadgray(self):
        """Check padding data with DVID grayscale.
        """
        
        service = DVIDServerService(dvidserver)
        uuid = service.create_new_repo("foo", "bar")
        
        ns = DVIDNodeService(dvidserver, uuid)
        ns.create_grayscale8("gray")
  
        arr = np.random.randint(255, size=(58,58,58)).astype(np.uint8)

        arr2 = np.zeros((64,64,64), np.uint8)
        arr2[0:58,0:58,0:58] = arr
        # load gray data
        ns.put_gray3D("gray", arr2, (0,0,0))
        
        # load shifted data for comparison
        arr2[6:64,6:64,6:64] = arr

        # read and pad data
        schema = partitionSchema(PartitionDims(32,64,64), enablemask=True, padding=8 )
        volpart = volumePartition(0, VolumeOffset(6,6,6))
        partitions = schema.partition_data([(volpart, arr)])

        # fetch with mask
        dvidreader = dvidSrc(dvidserver, uuid, "gray", partitions)

        newparts = dvidreader.extract_volume()
        self.assertEqual(len(newparts), 2) 

        for (part, vol) in newparts:
            if part.get_offset().z == 0:
                match = np.array_equal(arr2[0:32,:,:], vol)
                self.assertTrue(match)
            else:
                match = np.array_equal(arr2[32:64,:,:], vol)
                self.assertTrue(match)

    def test_dvidpadlabels(self):
        """Check padding data with DVID labels.
        """
        
        service = DVIDServerService(dvidserver)
        uuid = service.create_new_repo("foo", "bar")
        
        ns = DVIDNodeService(dvidserver, uuid)
        ns.create_labelblk("labels")
  
        arr = np.random.randint(12442, size=(58,58,58)).astype(np.uint64)

        arr2 = np.zeros((64,64,64), np.uint64)
        arr2[0:58,0:58,0:58] = arr
        # load gray data
        ns.put_labels3D("labels", arr2, (0,0,0))
        
        # load shifted data for comparison
        arr2[6:64,6:64,6:64] = arr

        # read and pad data
        schema = partitionSchema(PartitionDims(32,64,64), enablemask=True, padding=8, blank_delimiter=99999)
        volpart = volumePartition(0, VolumeOffset(6,6,6))
        partitions = schema.partition_data([(volpart, arr)])

        # fetch with mask
        dvidreader = dvidSrc(dvidserver, uuid, "labels", partitions)

        newparts = dvidreader.extract_volume()
        self.assertEqual(len(newparts), 2) 

        for (part, vol) in newparts:
            if part.get_offset().z == 0:
                match = np.array_equal(arr2[0:32,:,:], vol)
                self.assertTrue(match)
            else:
                match = np.array_equal(arr2[32:64,:,:], vol)
                self.assertTrue(match)



if __name__ == "main":
    unittest.main()
