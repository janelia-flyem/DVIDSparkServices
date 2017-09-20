import unittest
import numpy as np

from DVIDSparkServices.io_util.partitionSchema import volumePartition, VolumeOffset, PartitionDims, partitionSchema

class TestVolumePartition(unittest.TestCase):
    def test_partitionhash(self):
        """Check hashing function for volumePartition.
        """
        part1 = volumePartition(7,  VolumeOffset(3,1,5))
        part2 = volumePartition(7,  VolumeOffset(3,2,5))
        part3 = volumePartition(8,  VolumeOffset(3,1,5))

        # hash should be equal even with different offsets
        self.assertEqual(hash(part1), hash(part2))
        
        # hash should be different if index is different
        self.assertNotEqual(hash(part1), hash(part3))

    def test_partitioneq(self):
        """Check equivalence function for volumePartition.
        """
        part1 = volumePartition(7,  VolumeOffset(3,1,5))
        part2 = volumePartition(7,  VolumeOffset(3,2,5))
        part3 = volumePartition(8,  VolumeOffset(3,1,5))

        # should be equal even with different offsets
        self.assertEqual(part1, part2)
        
        # should be different if index is different
        self.assertNotEqual(part1, part3)

class TestPartitionSchema(unittest.TestCase):
    def test_createtiles(self):
        """Take a 3D volume and transform into a series of slices.
        """

        arr = np.random.randint(255, size=(45,25,13)).astype(np.uint8)
        #arr = np.random.randint(255, size=(2,3,4)).astype(np.uint8)
        schema = partitionSchema(PartitionDims(1,0,0))
        volpart = volumePartition(0, VolumeOffset(0,0,0))

        res = schema.partition_data([[volpart, arr]])
        self.assertEqual(len(res), 45)
        
        for tilepair in res:
            part, tile = tilepair
            zplane = part.get_offset().z
            match = np.array_equal(tile[0, :, :], arr[zplane, :, :])
            self.assertEqual(match, True)

    def test_createtilesshifted(self):
        """Take a 3d volume with offset and transform into a series of slices.

        The size of each tile should be the same even with the shift.
        """

        arr = np.random.randint(255, size=(45,25,13)).astype(np.uint8)
        #arr = np.random.randint(255, size=(2,3,4)).astype(np.uint8)
        schema = partitionSchema(PartitionDims(1,0,0))
        volpart = volumePartition(0, VolumeOffset(4,2,3))

        res = schema.partition_data([[volpart, arr]])
        self.assertEqual(len(res), 45)

        for tilepair in res:
            part, tile = tilepair
            zplane = part.get_offset().z - 4
            self.assertEqual(part.get_offset().x, 0)
            self.assertEqual(part.get_offset().y, 0)
            match = np.array_equal(tile[0, :, :], arr[zplane, :, :])
            self.assertEqual(match, True)

    def test_createvolshiftandpad(self):
        """Take a 3d volume with offset and transform into a series of subvolumes with padding.

        The data is moved to the proper partition and is padded so that the x, y, z
        of the stored data is lined up to a grid determined by the padding specified.
        Also, tests >8bit data and the data mask.
        """

        arr = np.random.randint(1025, size=(4,6,4)).astype(np.uint16)
        #arr = np.random.randint(255, size=(2,3,4)).astype(np.uint8)
        schema = partitionSchema(PartitionDims(2,0,0), blank_delimiter=1111, padding=2, enablemask=True)
        volpart = volumePartition(0, VolumeOffset(1,1,1))

        res = schema.partition_data([[volpart, arr]])
        self.assertEqual(len(res), 3) # 3 partitions with zsize=2 each

        arrcomp = np.zeros((6,8,6), dtype=np.uint16)
        arrcomp[:] = 1111
        arrcomp[1,1:7,1:5] = arr[0,:,:] 
        arrcomp[2,1:7,1:5] = arr[1,:,:] 
        arrcomp[3,1:7,1:5] = arr[2,:,:] 
        arrcomp[4,1:7,1:5] = arr[3,:,:] 

        for partpair in res:
            part, partvol = partpair
            zidx = part.get_offset().z
            
            # make a mask
            mask = arrcomp[zidx:zidx+2,:,:].copy()
            mask[mask != 1111] = 1
            mask[mask == 1111] = 0
           
            match = np.array_equal(arrcomp[zidx:zidx+2,:,:], partvol)
            matchmask = np.array_equal(mask, part.mask)
            self.assertEqual(match, True)
            self.assertEqual(matchmask, True)


    def test_fixedpartitionandreloffset(self):
        """Test rigid partition sizes and relative offsets.

        No padding should be needed and mask should be None.
        """

        arr = np.random.randint(1025, size=(4,6,4)).astype(np.uint16)
        schema = partitionSchema(PartitionDims(2,8,8), blank_delimiter=1111, padding=2, enablemask=True)
        volpart = volumePartition(0, VolumeOffset(1,1,1), reloffset=VolumeOffset(1,1,1))

        res = schema.partition_data([[volpart, arr]])
        self.assertEqual(len(res), 2) # 2 partitions with zsize=2 each

        for partpair in res:
            part, partvol = partpair
            zidx = part.get_offset().z
            
            # this mask should be all 1
            self.assertEqual(part.get_reloffset().x, 2)
            self.assertEqual(part.get_reloffset().y, 2)
            self.assertEqual(part.get_reloffset().z, 0)

            match = np.array_equal(arr[zidx-2:zidx,:,:], partvol)
            self.assertEqual(match, True)
            self.assertEqual(part.mask, None)

    def test_creatinglargevol(self):
        """Take small partitions and group into one large partition.
        """

        arr = np.random.randint(1025, size=(4,6,4)).astype(np.uint16)
        schema = partitionSchema(PartitionDims(2,0,0), blank_delimiter=1111, padding=2)
        volpart = volumePartition(0, VolumeOffset(1,1,1))

        res = schema.partition_data([(volpart, arr)])
        self.assertEqual(len(res), 3) # 3 partitions with zsize=2 each

        # make a new volume and pad
        arrcomp = np.zeros((6,8,6), dtype=np.uint16)
        arrcomp[:] = 1111
        arrcomp[1,1:7,1:5] = arr[0,:,:] 
        arrcomp[2,1:7,1:5] = arr[1,:,:] 
        arrcomp[3,1:7,1:5] = arr[2,:,:] 
        arrcomp[4,1:7,1:5] = arr[3,:,:] 

        # reverse procedure should be same as the original
        schemaglb = partitionSchema(PartitionDims(0,0,0))
        res2 = schemaglb.partition_data(res)
        self.assertEqual(len(res2), 1) # 3 partitions with zsize=2 each
       
        match = np.array_equal(arrcomp, res2[0][1])
        self.assertEqual(match, True)

    def test_badpadding(self):
        """Check that incorrect padding specification results in error.
        """
        def doit():
            partitionSchema(PartitionDims(2,0,0), blank_delimiter=1111, padding=3, enablemask=True)
        
        self.assertRaises(AssertionError, doit)
    

if __name__ == "__main__":
    unittest.main()
