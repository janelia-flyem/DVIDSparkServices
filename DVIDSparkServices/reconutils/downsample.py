"""Downsampling operations used for raw and label data.
"""
import numpy as np
from scipy import ndimage
from numba import jit
from skimage.util import view_as_blocks

def downsample_raw(data, numlevels=1):
    """Downsamples the given numpy array.

    This function returns an array of downsample
    array in powers of two.  Each dimension of the
    downsampled data is down by rounding, i.e., ceiling(dim/2).

    Args:
        data (numpy array): input data
        numlevels (int): number of downsample levels
    Returns:
        [numpy array] numpy array for each downsample level.
    """
   
    res = []
    for level in range(numlevels):
        res.append(ndimage.interpolation.zoom(data, 0.5, mode='reflect'))
        data = res[level]
    return res

@jit
def downsample_3Dlabels(data, numlevels=1):
    """Downsamples 3D label data using maximum element.
    
    This function returns an array of downsample array in powers
    of two.  For each 8 voxel subcube, the most common label is
    chosen.  If there are multiple labels with the maximum count,
    the smallest label is chosen.
    
    Currently assumes each dimension is a multiple of 2.

    Note: Using numba speeds this function up by almost 2x.  The inner
    loop is still very inefficient.  Preliminary analysis suggests
    that optimization of the internal loop could improve the speed
    by no more than an additional 2X.

    Args:
        data (numpy array): input data
        numlevels (int): number of downsample levels
    Returns:
        [numpy array] numpy array for each downsample level.
    """

    if len(data.shape) != 3:
        raise ValueError("Only supports 3D arrays")

    res = [0]
    for level in range(numlevels):
        # init new data from last shape in pyramid
        zmax, ymax, xmax = data.shape

        if ((zmax % 2) != 0) or ((ymax % 2) != 0) or ((xmax % 2) != 0):
            raise ValueError("Level: " + str(level+1) + " is not a multiple of two")

        data2 = np.zeros((zmax/2,ymax/2,xmax/2)).astype(data.dtype)

        for ziter in range(0,zmax,2):
            for yiter in range(0,ymax,2):
                for xiter in range(0,xmax,2):
                    v1 = data[ziter, yiter, xiter] 
                    v2 = data[ziter, yiter, xiter+1] 
                    v3 = data[ziter, yiter+1, xiter] 
                    v4 = data[ziter, yiter+1, xiter+1] 
                    v5 = data[ziter+1, yiter, xiter] 
                    v6 = data[ziter+1, yiter, xiter+1] 
                    v7 = data[ziter+1, yiter+1, xiter] 
                    v8 = data[ziter+1, yiter+1, xiter+1]
      
                    # mind most frequent element
                    freqs = {}
                    freqs[v2] = 0
                    freqs[v3] = 0
                    freqs[v4] = 0
                    freqs[v5] = 0
                    freqs[v6] = 0
                    freqs[v7] = 0
                    freqs[v8] = 0
                    
                    freqs[v1] = 1
                    freqs[v2] += 1
                    freqs[v3] += 1
                    freqs[v4] += 1
                    freqs[v5] += 1
                    freqs[v6] += 1
                    freqs[v7] += 1
                    freqs[v8] += 1

                    maxval = 0
                    freqkey = 0
                    for key, val in freqs.items():
                            if val > maxval or (val == maxval and key < freqkey):
                                    maxval = val
                                    freqkey = key

                    data2[ziter/2, yiter/2, xiter/2] = freqkey
        
        res.append(data2)
        data = data2

    return res[1:]


def downsample_labels_nd(a, factor):
    """
    Downsample the given ND array by binning the pixels into bins
    sized according to the downsampling 'factor'.
    
    Each bin is reduced to one pixel in the output array, whose value
    is the most common element from the bin.

    The blockshape can be given by 'factor', which must be either an int or tuple.
    If factor is tuple, it defines the blockshape.
    If factor is an int, then the blockshape will be square (or cubic, etc.)
    """
    @jit(nopython=True)
    def _flat_mode_destructive(data):
        """
        Given a flat array, return the mode.
        Beware: Overwrites flat_data.
        
        Note: We could have used scipy.stats.mode() here,
              but that implementation is insanely slow for large arrays,
              especially if there are many label values in the array.
        """
        data.sort()
        diff = np.diff(data)
        diff_bool = np.ones((len(diff)+2,), dtype=np.uint8)
        diff_bool[1:-1] = (diff != 0)
    
        diff_nonzero = diff_bool.nonzero()[0]
        run_lengths = diff_nonzero[1:] - diff_nonzero[:-1]
        max_run = np.argmax(run_lengths)
        return data[diff_nonzero[max_run]]
    
    @jit(nopython=True)
    def _blockwise_modes(data, output_shape):
        """
        Given an array of shape = output_shape + block_shape,
        compute the mode of each block.
        """
        assert len(output_shape) == len(data.shape) / 2
        
        # This assertion not allowed in nopython mode, but it's correct.
        #assert output_shape == data.shape[:len(data.shape)] 
        
        modes = np.zeros(output_shape, data.dtype)
        for block_index in np.ndindex(output_shape):
            flat_block_data = data[block_index].copy().reshape(-1)
            modes[block_index] = _flat_mode_destructive(flat_block_data)
        return modes

    # If the factor is an int, convert it to a blockshape.
    if np.issubdtype(type(factor), np.integer):
        blockshape = (factor,)*a.ndim
    else:
        blockshape = factor

    assert not (np.array(a.shape) % blockshape).any(), \
        "Downsampling factor must divide cleanly into array shape"

    v = view_as_blocks(a, blockshape)
    return _blockwise_modes(v, v.shape[:a.ndim])

if __name__ == "__main__":
    a = np.random.randint(0,4, (4,8)).view(np.uint64)
    print a
    print ""
    print downsample_labels_nd(a, 2)
    print ""

a = np.random.randint(0,4, (6,8)).view(np.uint64)
print a
print ""
print downsample_labels_nd(a, (3,4))
print ""

a = np.random.randint(0,4, (4,6,8)).view(np.uint64)
print a
print ""
print downsample_labels_nd(a, (2,3,4))
print ""
