from __future__ import division
import unittest

import numpy as np
import scipy.sparse
import vigra

from numpy_allocation_tracking.decorators import assert_mem_usage_factor

from DVIDSparkServices.reconutils.morpho import split_disconnected_bodies, matrix_argmax, object_masks_for_labels, assemble_masks
from DVIDSparkServices.util import bb_to_slicing
from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray
from DVIDSparkServices.reconutils.downsample import downsample_box

class TestSplitDisconnectedBodies(unittest.TestCase):
    
    def test_basic(self):

        _ = 2 # for readability in the array below

        # Note that we multiply these by 10 for this test!
        orig = [[ 1,1,1,1,_,_,3,3,3,3 ],
                [ 1,1,1,1,_,_,3,3,3,3 ],
                [ 1,1,1,1,_,_,3,3,3,3 ],
                [ 1,1,1,1,_,_,3,3,3,3 ],
                [ _,_,_,_,_,_,_,_,_,_ ],
                [ 0,0,_,_,4,4,_,_,0,0 ],  # Note that the zeros here will not be touched.
                [ _,_,_,_,4,4,_,_,_,_ ],
                [ 1,1,1,_,_,_,_,3,3,3 ],
                [ 1,1,1,_,1,1,_,3,3,3 ],
                [ 1,1,1,_,1,1,_,3,3,3 ]]
        
        orig = np.array(orig).astype(np.uint64)
        orig *= 10
        
        split, mapping = split_disconnected_bodies(orig)
        
        assert ((orig == 20) == (split == 20)).all(), \
            "Label 2 is a single component and therefore should remain untouched in the output"

        assert ((orig == 40) == (split == 40)).all(), \
            "Label 4 is a single component and therefore should remain untouched in the output"
        
        assert (split[:4,:4] == 10).all(), \
            "The largest segment in each split object is supposed to keep it's label"
        assert (split[:4,-4:] == 30).all(), \
            "The largest segment in each split object is supposed to keep it's label"

        lower_left_label = split[-1,1]
        lower_right_label = split[-1,-1]
        bottom_center_label = split[-1,5]

        assert lower_left_label != 10, "Split object was not relabeled"
        assert bottom_center_label != 10, "Split object was not relabeled"
        assert lower_right_label != 30, "Split object was not relabeled"

        assert lower_left_label in (41,42,43), "New labels are supposed to be consecutive with the old"
        assert lower_right_label in (41,42,43), "New labels are supposed to be consecutive with the old"
        assert bottom_center_label in (41,42,43), "New labels are supposed to be consecutive with the old"

        assert (split[-3:,:3] == lower_left_label).all()
        assert (split[-3:,-3:] == lower_right_label).all()
        assert (split[-2:,4:6] == bottom_center_label).all()

        assert set(mapping.keys()) == set([10,30,41,42,43]), "mapping: {}".format( mapping )

        assert (vigra.analysis.applyMapping(split, mapping, allow_incomplete_mapping=True) == orig).all(), \
            "Applying mapping to the relabeled image did not recreate the original image."

    def test_mem_usage(self):
        a = 1 + np.arange(100**2, dtype=np.uint32).reshape((100,100)) // 10
        split, mapping = assert_mem_usage_factor(10)(split_disconnected_bodies)(a)
        assert (a == split).all()
        assert mapping == {}


class TestSparseMatrixUtilityFunctions(unittest.TestCase):
 
    def test_matrix_argmax(self):
        """
        Test the special argmax function for sparse matrices.
        """
        data = [[0,3,0,2,0],
                [0,0,1,0,5],
                [0,0,0,2,0]]
        data = np.array(data).astype(np.uint32)
         
        m = scipy.sparse.coo_matrix(data, data.nonzero())
         
        assert (matrix_argmax(m, axis=1) == [1,4,3]).all()  
        assert (matrix_argmax(m, axis=1) == matrix_argmax(data, axis=1)).all()


class Test_object_masks_for_labels(unittest.TestCase):

    def setUp(self):
        _ = 0
        data_slice = [[_,_,_,_,_,_,_,_,_,_],
                      [_,2,_,_,_,_,3,3,_,_],
                      [_,2,2,_,_,_,_,3,_,_],
                      [_,_,2,_,_,_,_,3,_,_],
                      [_,_,_,2,_,_,_,3,_,_],
                      [_,_,_,_,2,_,_,3,_,_],
                      [_,_,_,_,_,2,_,3,_,_],
                      [_,_,_,_,_,2,_,_,4,4], # Notice: object 4 touches the volume edge
                      [_,_,_,_,_,2,_,_,_,_]]

        data_slice = np.asarray(data_slice)
        
        # Put the data slice in the middle of the test volume,
        # so nothing touches the volume edge in the Z-dimension.
        segmentation = np.zeros( (3,) + data_slice.shape, dtype=np.uint64 )
        segmentation[1] = data_slice
        
        self.segmentation = segmentation
        
    def test_basic(self):
        label_ids_and_masks = object_masks_for_labels( self.segmentation,
                                                       box=None,
                                                       minimum_object_size=1,
                                                       always_keep_border_objects=False,
                                                       compress_masks=False )

        # Result isn't necessarily sorted
        masks_dict = dict( label_ids_and_masks )
        assert set(masks_dict.keys()) == set([2,3,4])

        for label in (2,3,4):
            full_mask = (self.segmentation == label)
            bb_start = np.transpose( full_mask.nonzero() ).min(axis=0)
            bb_stop  = np.transpose( full_mask.nonzero() ).max(axis=0) + 1
            
            box, mask, count = masks_dict[label]
            
            assert (np.asarray(box) == (bb_start, bb_stop)).all()
            assert (mask == full_mask[bb_to_slicing(bb_start, bb_stop)]).all(), \
                "Incorrect mask for label {}: \n {}".format( label, full_mask )
            assert count == full_mask[bb_to_slicing(bb_start, bb_stop)].sum()
    
    def test_minimum_object_size(self):
        # Exclude object 4, which is too small
        label_ids_and_masks = object_masks_for_labels( self.segmentation,
                                                       box=None,
                                                       minimum_object_size=3,
                                                       always_keep_border_objects=False,
                                                       compress_masks=False )

        # Result isn't necessarily sorted
        masks_dict = dict( label_ids_and_masks )
        assert set(masks_dict.keys()) == set([2,3])

        for label in (2,3):
            full_mask = (self.segmentation == label)
            bb_start = np.transpose( full_mask.nonzero() ).min(axis=0)
            bb_stop  = np.transpose( full_mask.nonzero() ).max(axis=0) + 1
            
            box, mask, count = masks_dict[label]
            
            assert (np.asarray(box) == (bb_start, bb_stop)).all()
            assert (mask == full_mask[bb_to_slicing(bb_start, bb_stop)]).all(), \
                "Incorrect mask for label {}: \n {}".format( label, full_mask )
            assert count == full_mask[bb_to_slicing(bb_start, bb_stop)].sum()

    def test_always_keep_border_objects(self):
        # Object 4 is too small, but it's kept anyway because it touches the border.
        label_ids_and_masks = object_masks_for_labels( self.segmentation,
                                                       box=None,
                                                       minimum_object_size=3,
                                                       always_keep_border_objects=True, # Keep border objects
                                                       compress_masks=False )

        # Result isn't necessarily sorted
        masks_dict = dict( label_ids_and_masks )
        assert set(masks_dict.keys()) == set([2,3,4])

        for label in (2,3,4):
            full_mask = (self.segmentation == label)
            bb_start = np.transpose( full_mask.nonzero() ).min(axis=0)
            bb_stop  = np.transpose( full_mask.nonzero() ).max(axis=0) + 1
            
            box, mask, count = masks_dict[label]
            
            assert (np.asarray(box) == (bb_start, bb_stop)).all()
            assert (mask == full_mask[bb_to_slicing(bb_start, bb_stop)]).all(), \
                "Incorrect mask for label {}: \n {}".format( label, full_mask )
            assert count == full_mask[bb_to_slicing(bb_start, bb_stop)].sum()
    
    def test_compressed_output(self):
        label_ids_and_masks = object_masks_for_labels( self.segmentation,
                                                       box=None,
                                                       minimum_object_size=1,
                                                       always_keep_border_objects=False,
                                                       compress_masks=True )

        # Result isn't necessarily sorted
        masks_dict = dict( label_ids_and_masks )
        assert set(masks_dict.keys()) == set([2,3,4])

        for label in (2,3,4):
            full_mask = (self.segmentation == label)
            bb_start = np.transpose( full_mask.nonzero() ).min(axis=0)
            bb_stop  = np.transpose( full_mask.nonzero() ).max(axis=0) + 1
            
            box, compressed_mask, count = masks_dict[label]
            assert isinstance(compressed_mask, CompressedNumpyArray)
            mask = compressed_mask.deserialize()
            
            assert (np.asarray(box) == (bb_start, bb_stop)).all()
            assert (mask == full_mask[bb_to_slicing(bb_start, bb_stop)]).all(), \
                "Incorrect mask for label {}: \n {}".format( label, full_mask )
            assert count == full_mask[bb_to_slicing(bb_start, bb_stop)].sum()

class Test_assemble_masks(unittest.TestCase):

    def test_basic(self):
        _ = 0
        #                  0 1 2 3 4  5 6 7 8 9
        complete_mask = [[[_,_,_,_,_, _,_,_,_,_], # 0
                          [_,1,_,_,_, _,1,1,_,_], # 1
                          [_,1,1,_,_, 1,1,1,_,_], # 2
                          [_,_,1,_,_, _,_,1,_,_], # 3
                          [_,_,_,1,_, _,_,1,_,_], # 4

                          [_,_,_,1,1, _,1,1,_,_], # 5
                          [_,_,_,_,_, 1,1,1,_,_], # 6
                          [_,_,_,_,_, 1,_,_,1,_], # 7
                          [_,_,_,_,_, 1,_,_,_,_]]]# 8

        complete_mask = np.asarray(complete_mask, dtype=bool)

        boxes = []
        boxes.append( ((0,1,1), (1,5,4)) )
        boxes.append( ((0,1,5), (1,5,8)) )
        boxes.append( ((0,5,3), (1,6,5)) )
        boxes.append( ((0,5,5), (1,9,9)) )
        
        masks = [ complete_mask[ bb_to_slicing(*box)] for box in boxes ]

        combined_bounding_box, combined_mask, downsample_factor = assemble_masks( boxes, masks, downsample_factor=1, minimum_object_size=1 )
        assert (combined_bounding_box == ( (0,1,1), (1,9,9) )).all()
        assert (combined_mask == complete_mask[bb_to_slicing(*combined_bounding_box)]).all()
        assert downsample_factor == 1

    def test_with_downsampling(self):
        _ = 0
        #                  0 1 2 3 4  5 6 7 8 9
        complete_mask = [[[_,_,_,_,_, _,_,_,_,_], # 0
                          [_,1,_,_,_, _,1,1,_,_], # 1
                          [_,1,1,_,_, 1,1,1,_,_], # 2
                          [_,_,1,_,_, _,_,1,_,_], # 3
                          [_,_,_,1,_, _,_,1,_,_], # 4

                          [_,_,_,1,1, _,1,1,_,_], # 5
                          [_,_,_,_,_, 1,1,1,_,_], # 6
                          [_,_,_,_,_, 1,_,_,1,_], # 7
                          [_,_,_,_,_, 1,_,_,_,_]]]# 8

        complete_mask = np.asarray(complete_mask, dtype=bool)

        boxes = []
        boxes.append( ((0,1,1), (1,5,4)) )
        boxes.append( ((0,1,5), (1,5,8)) )
        boxes.append( ((0,5,3), (1,6,5)) )
        boxes.append( ((0,5,5), (1,9,9)) )
        
        masks = [ complete_mask[ bb_to_slicing(*box)] for box in boxes ]

        combined_bounding_box, combined_mask, downsample_factor = assemble_masks( boxes, masks, downsample_factor=2, minimum_object_size=1, suppress_zero=False )

        expected_downsampled_mask = [[[1,_,_,1,_],
                                      [0,_,_,1,_],
                                      [0,1,1,1,_],
                                      [0,_,1,_,_],
                                      [0,_,1,_,_]]]
        expected_downsampled_mask = np.asarray(expected_downsampled_mask)

        assert (combined_bounding_box == ((0,1,1), (1,9,9)) ).all()
        assert (combined_mask == expected_downsampled_mask).all()
        assert downsample_factor == 2

    def test_auto_downsampling_choice(self):
        complete_mask = np.ones( (100,100,100), dtype=np.bool )
        box = ((0,0,0), (100,100,100))

        # Restrict RAM usage to less than 1/8 of the full mask, so even downsampling by 2 isn't enough.
        # The function will be forced to use a downsampling factor of 3.
        RAM_LIMIT = (complete_mask.size / 8.) - 1
        
        combined_bounding_box, combined_mask, downsample_factor = \
            assemble_masks( [box],
                            [complete_mask],
                            downsample_factor=-1, # 'auto'
                            minimum_object_size=1,
                            max_combined_mask_size=RAM_LIMIT)

        assert (combined_bounding_box == box ).all()
        assert combined_mask.all()
        assert downsample_factor == 3
        
    def test_with_downsampling_and_pad(self):
        _ = 0
        #                  0 1 2 3 4  5 6 7 8 9
        complete_mask = [[[_,_,_,_,_, _,_,_,_,_], # 0
                          [_,1,_,_,_, _,1,1,_,_], # 1
                          [_,1,1,_,_, 1,1,1,_,_], # 2
                          [_,_,1,_,_, _,_,1,_,_], # 3
                          [_,_,_,1,_, _,_,1,_,_], # 4

                          [_,_,_,1,1, _,1,1,_,_], # 5
                          [_,_,_,_,_, 1,1,1,_,_], # 6
                          [_,_,_,_,_, 1,_,_,1,_], # 7
                          [_,_,_,_,_, 1,_,_,_,_]]]# 8

        complete_mask = np.asarray(complete_mask, dtype=bool)

        boxes = []
        boxes.append( ((0,1,1), (1,5,4)) )
        boxes.append( ((0,1,5), (1,5,8)) )
        boxes.append( ((0,5,3), (1,6,5)) )
        boxes.append( ((0,5,5), (1,9,9)) )
        
        masks = [ complete_mask[ bb_to_slicing(*box)] for box in boxes ]

        pad = 2
        combined_bounding_box, combined_mask, downsample_factor = assemble_masks( boxes, masks, downsample_factor=2, minimum_object_size=1, suppress_zero=False, pad=pad )

        expected_downsampled_mask = [[[1,_,_,1,_],
                                      [0,_,_,1,_],
                                      [0,1,1,1,_],
                                      [0,_,1,_,_],
                                      [0,_,1,_,_]]]

        expected_downsampled_mask = np.pad(expected_downsampled_mask, pad, 'constant', constant_values=0)

        expected_downsampled_mask = np.asarray(expected_downsampled_mask)

        combined_box_without_pad = np.array([(0,1,1), (1,9,9)])
        padding_in_full_res_space = [(-4, -4, -4), (4, 4, 4)]
        assert (combined_bounding_box == (combined_box_without_pad + padding_in_full_res_space) ).all()
        assert (combined_mask == expected_downsampled_mask).all()
        assert downsample_factor == 2

        downsampled_box = downsample_box(combined_bounding_box, np.array((2,2,2)))
        assert (combined_mask.shape == (downsampled_box[1] - downsampled_box[0])).all(), \
            "Output mask shape is not consistent with the combined box"

if __name__ == "__main__":
    unittest.main()
