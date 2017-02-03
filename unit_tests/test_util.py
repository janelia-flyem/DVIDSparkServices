import numpy as np
from DVIDSparkServices.util import runlength_encode

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
