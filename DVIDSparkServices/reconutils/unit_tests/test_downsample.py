import unittest

import numpy as np
from DVIDSparkServices.reconutils.downsample import downsample_raw, downsample_3Dlabels

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
        
if __name__ == "main":
    unittest.main()
