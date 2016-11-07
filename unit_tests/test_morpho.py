import unittest

import numpy as np
import vigra

from DVIDSparkServices.reconutils.morpho import split_disconnected_bodies

class TestSplitDisconnectedBodies(unittest.TestCase):
    
    def test_basic(self):

        _ = 2 # for readability in the array below

        # Note that we multiply these by 10 for this test!
        orig = [[ 1,1,1,1,_,_,3,3,3,3 ],
                [ 1,1,1,1,_,_,3,3,3,3 ],
                [ 1,1,1,1,_,_,3,3,3,3 ],
                [ 1,1,1,1,_,_,3,3,3,3 ],
                [ _,_,_,_,_,_,_,_,_,_ ],
                [ _,_,_,_,4,4,_,_,_,_ ],
                [ _,_,_,_,4,4,_,_,_,_ ],
                [ 1,1,1,_,_,_,_,3,3,3 ],
                [ 1,1,1,_,1,1,_,3,3,3 ],
                [ 1,1,1,_,1,1,_,3,3,3 ]]
        
        orig = np.array(orig).astype(np.uint64)
        orig *= 10
        
        split, mapping = split_disconnected_bodies(orig)
        
        assert ((orig == 20) == (split == 20)).all(), \
            "Label 2 was not split and should remain the same everywhere"

        assert ((orig == 40) == (split == 40)).all(), \
            "Label 2 was not split and should remain the same everywhere"
        
        assert (split[:4,:4] == 10).all(), \
            "The largest segment in each split object supposed to keep it's label"
        assert (split[:4,-4:] == 30).all(), \
            "The largest segment in each split object supposed to keep it's label"

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


        assert set(mapping.keys()) == set([41,42,43]), "mapping: {}".format( mapping )

        assert (vigra.analysis.applyMapping(split, mapping, allow_incomplete_mapping=True) == orig).all(), \
            "Applying mapping to the relabeled image did not recreate the original image."



if __name__ == "__main__":
    unittest.main()
