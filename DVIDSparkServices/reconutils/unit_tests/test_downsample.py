import unittest

import numpy as np
from DVIDSparkServices.util import bb_to_slicing
from DVIDSparkServices.reconutils.downsample import downsample_raw, downsample_3Dlabels, downsample_labels_3d

class Testdownsample(unittest.TestCase):
    """Tests array downsampling routines. 
    """
   
    def test_downsample_raw_blank(self):
        """Tests downsample_raw dimensions and pyramid creation.
        """
 
        data = np.zeros((100, 60, 9))
        data_1 = np.zeros((50, 30, 5))
        data_2 = np.zeros((25, 15, 3))
 
        res = downsample_raw(data, numlevels=2) 
 
        # check that level 1 matches
        match = np.array_equal(data_1, res[0])
        self.assertTrue(match)
 
        # check that level 2 matches
        match = np.array_equal(data_2, res[1])
        self.assertTrue(match)
 
    def test_downsample_3Dlabels_bad(self):
        """Tests that illegal input gracefully fails in label downsample.
        """
 
        # does not support 2D data
        founderror = False
        try:
            data2d = np.zeros((32,32))
            res = downsample_3Dlabels(data2d)
        except ValueError, err:
            founderror = True 
        self.assertTrue(founderror)
 
 
        # support multiple of two
        founderror = False
        try:
            data2d = np.zeros((32,32,50))
            res = downsample_3Dlabels(data2d,1)
        except ValueError, err:
            founderror = True 
        self.assertFalse(founderror)
 
        # does not support non multiple of two
        founderror = False
        try:
            data2d = np.zeros((32,32,50))
            res = downsample_3Dlabels(data2d,2)
        except ValueError, err:
            founderror = True 
        self.assertTrue(founderror)
 
    def test_downsample_3Dlabels(self):
        """Tests label downsampling algorithm.
        """
 
        # creates label array with some downsampling corner cases
        data = np.array([[[1, 0, 2, 2],
                          [0, 1, 2, 2],
                          [2, 3, 5, 7],
                          [5, 3, 6, 0]],
 
                         [[1, 1, 2, 8],
                          [0, 0, 0, 0],
                          [3, 1, 1, 2],
                          [2, 7, 3, 4]],
 
                         [[8, 8, 0, 0],
                          [8, 8, 0, 0],
                          [0, 0, 0, 0],
                          [8, 0, 0, 0]],
 
                         [[8, 8, 0, 0],
                          [8, 8, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]], dtype=np.uint8)
        data_1 = np.array([[[0, 2],
                            [3, 0]],
 
                           [[8, 0],
                            [0, 0]]], dtype=np.uint8)
        data_2 = np.array([[[0]]], dtype=np.uint8)
     
        res = downsample_3Dlabels(data,2)
 
        # check that level 1 matches
        match = np.array_equal(data_1, res[0])
        self.assertTrue(match)
 
        # check that level 2 matches
        match = np.array_equal(data_2, res[1])
        self.assertTrue(match)

class TestRecuderFunctions(unittest.TestCase):
    
    def setUp(self):
        _ = 0        
        #         0      3      6      9
        data = [[[1,1,_, 3,3,_, _,_,_, _], # 0
                 [1,_,_, _,3,4, _,_,_, _],
                 [_,_,2, 4,4,4, _,5,_, _],

                 [_,_,_, 2,2,8, 1,1,_, _], # 3
                 [_,7,6, 2,2,8, _,9,9, 1],
                 [7,7,6, 8,8,8, 9,9,1, _],

                 [_,_,_, _,_,_, _,_,_, _], # 6
                 [_,_,_, _,_,_, _,_,_, _],
                 [_,_,_, _,_,_, _,_,_, _],

                 [_,2,_, _,4,_, _,6,_, 8], # 9
                 [1,_,1, 3,_,3, 5,_,5, 7]]]# 10

        self.data = np.asarray(data)
    
    def test_downsample_labels_3d(self):
        downsampled, box = downsample_labels_3d(self.data, (1,3,3))
        assert (box == [(0,0,0), (1,4,4)]).all()
        
        _ = 0
        expected = [[[1,4,5,_],
                     [7,8,9,1],
                     [_,_,_,_],
                     [1,3,5,7]]]

        assert downsampled.shape == (1,4,4)
        assert (downsampled == expected).all()

    def downsample_binary_3d(self):
        downsampled, box = downsample_labels_3d(self.data, (1,3,3))
        assert (box == [(0,0,0), (1,4,4)]).all()
        
        _ = 0
        expected = [[[1,1,1,_],
                     [1,1,1,1],
                     [_,_,_,_],
                     [1,1,1,1]]]

        assert downsampled.shape == (1,4,4)
        assert (downsampled == expected).all()

    def test_downsample_labels_3d_WITH_OFFSET(self):
        # Take a subset of the data, and tell the downsampling function where it came from.
        data_box = [(0, 1, 2),
                    (1, 10, 9)]
        offset_data = self.data[bb_to_slicing(*data_box)]
        downsampled, box = downsample_labels_3d(offset_data, (1,3,3), data_box)
        assert (box == [(0,0,0), (1,4,3)]).all()
        
        _ = 0
        expected = [[[2,4,5],
                     [6,8,9],
                     [_,_,_],
                     [_,4,6]]]

        assert downsampled.shape == (1,4,3)
        assert (downsampled == expected).all()


if __name__ == "__main__":
    unittest.main()
