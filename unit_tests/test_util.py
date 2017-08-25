import numpy as np
from DVIDSparkServices.util import runlength_encode, unicode_to_str, blockwise_boxes, nonconsecutive_bincount

def test_runlength_encode():
    mask = np.array( [[[0,1,1,0,1],
                       [0,0,0,0,0],
                       [0,1,1,1,0],
                       [1,1,0,1,1],
                       [1,1,1,1,1]]] )
    
    coords = np.transpose( np.nonzero(mask) )
    
    expected_rle = np.array([[0,0,1,2],
                             [0,0,4,4],
                             [0,2,1,3],
                             [0,3,0,1],
                             [0,3,3,4],
                             [0,4,0,4]])

    rle = runlength_encode(coords)
    assert (rle == expected_rle).all()

def test_empty_runlength_encode():
    coords = []
    rle = runlength_encode(coords)
    assert rle.shape == (0,4)

def test_unicode_to_str():
    data = {'hello': u'world',
            'number': 5,
            'list': [1,2,3],
            'dict': {'hello': u'world'}}

    new_data = unicode_to_str(data)
    assert isinstance(new_data['number'], int)
    assert isinstance(new_data['list'], list)
    assert isinstance(new_data['list'][0], int)
    assert isinstance(new_data['hello'], str)
    assert isinstance(new_data['dict']['hello'], str)


def test_blockwise_boxes():
    bb = [(10,10), (100,100)]
    block_shape = (30,40)
    boxes = np.array(list(blockwise_boxes(bb, block_shape)))
    assert boxes.tolist() == [[[10, 10], [30, 40]],
                              [[10, 40], [30, 80]],
                              [[10, 80], [30, 100]],
                              [[30, 10], [60, 40]],
                              [[30, 40], [60, 80]],
                              [[30, 80], [60, 100]],
                              [[60, 10], [90, 40]],
                              [[60, 40], [90, 80]],
                              [[60, 80], [90, 100]],
                              [[90, 10], [100, 40]],
                              [[90, 40], [100, 80]],
                              [[90, 80], [100, 100]]]

def test_nonconsecutive_bincount():
    labels = [[1,1,1,1,1],
              [2,2,2,2,0],
              [5,5,5,0,0],
              [7,7,0,0,0],
              [2**33,0,0,0,0]]

    labels = np.asarray(labels, dtype=np.uint64)
    counts = nonconsecutive_bincount(labels)
    assert counts[0] == 10
    assert counts[1] == 5
    assert counts[2] == 4
    assert counts[5] == 3
    assert counts[7] == 2
    assert counts[2**33] == 1
    assert set(counts.keys()) == {0,1,2,5,7,2**33}


    
import logging
logger = logging.getLogger("unit_tests.test_util")

if __name__ == "__main__":
    import sys
    logger.addHandler( logging.StreamHandler(sys.stdout) )
    logger.setLevel(logging.DEBUG)
    
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    nose.run(defaultTest=__file__)
