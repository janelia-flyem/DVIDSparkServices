"""Downsampling operations used for raw and label data.
"""
import numpy as np
from scipy import ndimage
from numba import jit

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

    res = []
    sortarray = range(8)
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

    return res


