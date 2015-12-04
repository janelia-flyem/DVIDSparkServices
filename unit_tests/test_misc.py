import numpy as np
from DVIDSparkServices.reconutils.misc import select_channels, normalize_channels_in_place

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

if __name__ == "__main__":
    import sys
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    nose.run(defaultTest=__file__)
