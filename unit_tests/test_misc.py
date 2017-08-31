from __future__ import division
import os
import numpy as np

from numpy_allocation_tracking.decorators import assert_mem_usage_factor

import DVIDSparkServices
from DVIDSparkServices.reconutils.misc import select_channels, normalize_channels_in_place, \
                                              find_large_empty_regions, naive_membrane_predictions, \
                                              seeded_watershed

import logging
logger = logging.getLogger("unit_tests.test_misc")

def _load_grayscale():
    grayscale_path = os.path.split(DVIDSparkServices.__file__)[0] + '/../integration_tests/resources/grayscale-256-256-256-uint8.bin'
    with open(grayscale_path, 'rb') as f:
        grayscale_bytes =  f.read()

    grayscale_flat = np.frombuffer(grayscale_bytes, dtype=np.uint8)
    grayscale = grayscale_flat.reshape((256, 256, 256), order='C')
    return grayscale

def test_find_large_empty_regions():
    grayscale = _load_grayscale().copy()
    
    assert grayscale.min() > 0, \
        "I thought the test grayscale doesn't have zero-pixels..."
    
    # Create two 'large' empty regions
    grayscale[:10, :10, :10] = 0
    grayscale[-10:, -10:, -10:] = 0

    # Create a small empty region (that should be ignored)
    grayscale[100,100:105,100:105] = 0

    mask = find_large_empty_regions(grayscale, min_background_voxel_count=100)

    # Were the large regions detected?
    assert (mask[:10, :10, :10] == 0).all()
    assert (mask[-10:, -10:, -10:] == 0).all()

    # Was the rest of it left alone?
    expected = np.ones_like( grayscale, dtype=bool )
    expected[:10, :10, :10] = 0
    expected[-10:, -10:, -10:] = 0

    assert (mask[100,100:105,100:105] == 1).all()
    assert (mask == expected).all()


def test_select_channels():
    a = np.zeros((100,200,10), dtype=np.float32)
    a[:] = np.arange(10)[None, None, :]
    
    assert select_channels(a, None) is a
    assert (select_channels(a, [2,3,5]) == np.array([2,3,5])[None, None, :]).all()
    
    combined = select_channels(a, [1,2,3,[4,5]])
    assert combined.shape == (100,200,4)
    assert (combined[..., 0] == 1).all()
    assert (combined[..., 1] == 2).all()
    assert (combined[..., 2] == 3).all()
    assert (combined[..., 3] == (4+5)).all()

def test_normalize_channels_in_place():
    a = np.zeros((100,200,3), dtype=np.float32)
    a[..., 0] = (0.2 / 2)
    a[..., 1] = (0.3 / 2)
    a[..., 2] = (0.5 / 2)

    # erase a pixel entirely
    a[50,50,:] = 0.0
    
    normalize_channels_in_place(a)
    assert (a.sum(axis=-1) == 1.0).all()

    assert a[50,50,0] == a[50,50,1] == a[50,50,2]

class TestMemoryUsage(object):

    @classmethod
    def setupClass(cls):
        cls.grayscale = _load_grayscale()

    def test_memory_usage(self):
        # Create a volume with an empty region
        grayscale = self.grayscale.copy()
        grayscale[0:100] = 0
        
        mask = assert_mem_usage_factor(6.1)(find_large_empty_regions)(grayscale)
        pred = assert_mem_usage_factor(4.1)(naive_membrane_predictions)(grayscale, mask)
        assert_mem_usage_factor(3.1)(normalize_channels_in_place)(pred)
        supervoxels = assert_mem_usage_factor(2.1)(seeded_watershed)(pred, mask)

if __name__ == "__main__":
    import sys
    logger.addHandler( logging.StreamHandler(sys.stdout) )
    logger.setLevel(logging.DEBUG)
    
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    nose.run(defaultTest=__file__)
