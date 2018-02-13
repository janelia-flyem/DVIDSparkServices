"""Downsampling operations used for raw and label data.
"""
from __future__ import print_function, absolute_import
from __future__ import division
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

@jit(cache=True)
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

        data2 = np.zeros((zmax//2, ymax//2, xmax//2)).astype(data.dtype)

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

                    data2[ziter//2, yiter//2, xiter//2] = freqkey
        
        res.append(data2)
        data = data2

    return res[1:]

@jit(nopython=True, cache=True)
def downsample_box( box, block_shape ):
    """
    Given a box (i.e. start and stop coordinates) and a
    block_shape (downsampling factor), return the corresponding box
    in downsampled coordinates.
    """
    assert block_shape.shape[0] == box.shape[1]
    downsampled_box = np.zeros_like(box)
    downsampled_box[0] = box[0] // block_shape
    downsampled_box[1] = (box[1] + block_shape - 1) // block_shape
    return downsampled_box

def make_blockwise_reducer_3d(reducer_func, nopython=True):
    """
    Returns a function that can reduce an array of shape (Z*Bz, Y*By, X*Bx)
    into an array of shape (Z,Y,X), by dividing the array into shapes of blocks (Bz,By,Bx)
    and calling the given 'reducer' function on each block.

    The reducer function must return a scalar.
    Ideally, the reducer should be jit-compileable with numba. If not, set nopython=False.
    
    See reduce_blockwise(), below, for details regarding the returned function.
    
    Implemented according to guidelines in numba FAQ:
    http://numba.pydata.org/numba-doc/dev/user/faq.html#can-i-pass-a-function-as-an-argument-to-a-jitted-function
    """
    @jit(nopython=nopython, cache=True)
    def _reduce_blockwise_compiled(data, block_shape, data_box, reduced_box):
        _output_shape = reduced_box[1] - reduced_box[0]
        output_shape = (_output_shape[0], _output_shape[1], _output_shape[2])
        output = np.zeros(output_shape, data.dtype)

        for block_index in np.ndindex(*output_shape):
            # block_bounds = block_shape * ( (block_index, 1+block_index) + reduced_box[0] )
            block_bounds = np.zeros( (2,3), dtype=np.int32 )
            block_bounds[0] = block_index
            block_bounds[1] = 1 + block_bounds[0]
            block_bounds[:] += reduced_box[0]
            block_bounds[:] *= block_shape
            
            block_bounds[0] = np.maximum(block_bounds[0], data_box[0])
            block_bounds[1] = np.minimum(block_bounds[1], data_box[1])
    
            z0, y0, x0 = block_bounds[0] - data_box[0]
            z1, y1, x1 = block_bounds[1] - data_box[0]
            
            block_data = data[z0:z1, y0:y1, x0:x1]

            bi_z, bi_y, bi_x = block_index
            output[bi_z, bi_y, bi_x] = reducer_func(block_data)
        return output

    def reduce_blockwise(data, block_shape, data_box=None):
        """
        Reduce the given 3D array block-by-block, returning a smaller array of scalars (one per block).
        
        Args:
         data:
             3D array, whose shape need not be exactly divisible by the block_shape
         
         block_shape:
             tuple (Bz,By,Bx)
         
         data_box:
             bounding box pair: [(z0, y0, x0), (z1, y1, x1)]
             
             If block_shape does not cleanly divide into block_shape, blocks on the edge
             of the full data array will be appropriately truncated before they are sent
             to the reducer function.  This is true for blocks on *any* side of the volume.
             
             It is assumed that blocks are aligned to some global coordinate grid,
             starting at (0,0,0), but the 'data' array might not be aligned with that grid.
             For example, the first element of the 'data' array may correspond to voxel (0, 0, 1),
             and therefore the first block will be smaller than most other blocks in the volume.
        """
        assert data.ndim == 3
    
        if data_box is None:
            data_box = np.array([(0,0,0), data.shape])
        else:
            data_box = np.asarray(data_box)
        
        assert data_box.shape == (2,3)
             
        # If the block_shape is an int, convert it to a shape.
        if np.issubdtype(type(block_shape), np.integer):
            block_shape = (block_shape, block_shape, block_shape)
        
        block_shape = np.array(block_shape)
        assert block_shape.shape == (3,)
        
        if (block_shape == 1).all():
            # Shortcut: Nothing to do.
            return data, data_box.copy()
        
        reduced_box = downsample_box(data_box, block_shape)
        reduced_output = _reduce_blockwise_compiled(data, block_shape, data_box, reduced_box)
        return reduced_output, reduced_box

    return reduce_blockwise


@jit(nopython=True, cache=True)
def flat_mode_except_zero(data):
    """
    Given an array, flatten it and return the mode, without including
    zeros, if possible.
    
    If (data == 0).all(), then 0 is returned.
    """
    data = data.copy().reshape(-1)
    data = data[data != 0]
    if data.size == 0:
        return 0
    return _flat_mode(data)


@jit(nopython=True, cache=True)
def flat_mode(data):
    """
    Given an ND array, flatten it and return the mode.
    """
    data = data.copy().reshape(-1)
    return _flat_mode(data)


@jit(nopython=True, cache=True)
def _flat_mode(data):
    """
    Given an contiguous flat array, return the mode.
    
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


@jit(nopython=True, cache=True)
def flat_binary_mode(data):
    nonzero = 0
    for index in np.ndindex(data.shape):
        z,y,x = index
        if data[z,y,x] != 0:
            nonzero += 1

    if nonzero > data.size // 2:
        return 1
    return 0

# Signature:
# reduced_output, reduced_box = f(data, block_shape, data_box=None)
downsample_labels_3d = make_blockwise_reducer_3d(flat_mode)
downsample_binary_3d = make_blockwise_reducer_3d(flat_binary_mode)

# These variants will not return zero as the block mode UNLESS it's the only value in the block.
downsample_labels_3d_suppress_zero = make_blockwise_reducer_3d(flat_mode_except_zero)
downsample_binary_3d_suppress_zero = make_blockwise_reducer_3d(np.any)

if __name__ == "__main__":
    # These work, too:
    #
    # @jit(nopython=True)
    # def flat_mode_exclude_1(data):
    #     return flat_mode(data, 1)
    # downsample_labels_3d = make_blockwise_reducer_3d(flat_mode_exclude_1)
    # 
    # from functools import partial
    # downsample_labels_3d = make_blockwise_reducer_3d(partial(flat_mode, exclude_label=1), nopython=False)

    import time
    start = time.time()

    a = np.random.randint(0,4, (1,4,8)).view(np.uint64)
    print(a)
    print("")
    downsampled, box = downsample_labels_3d(a, 2)
    assert tuple(box[1] - box[0]) == downsampled.shape
    print(downsampled)
    print("")
    print(box.tolist())
    print("")

    a = np.random.randint(0,4, (1,4,8)).view(np.uint64)
    print(a)
    print("")
    downsampled, box = downsample_binary_3d(a, 2)
    assert tuple(box[1] - box[0]) == downsampled.shape
    print(downsampled)
    print("")
    print(box.tolist())
    print("")

    a = np.array([[[0,1,0],
                   [0,0,0]]]).astype(np.uint64)
    print(a)
    print("")
    downsampled, box = downsample_labels_3d(a, (1,1,3))
    assert tuple(box[1] - box[0]) == downsampled.shape
    print(downsampled)
    print(box.tolist())
    print("")

    a = np.array([[[0,1,0],
                   [0,0,0]]]).astype(np.uint64)
    print(a)
    print("")
    downsampled, box = downsample_binary_3d(a, (1,1,3))
    assert tuple(box[1] - box[0]) == downsampled.shape
    print(downsampled)
    print(box.tolist())
    print("")

    a = np.random.randint(0,4, (1,6,8)).view(np.uint64)
    print(a)
    print("")
    downsampled, box = downsample_labels_3d(a, (1,3,4))
    assert tuple(box[1] - box[0]) == downsampled.shape
    print(downsampled)
    print(box.tolist())
    print("")
    
    a = np.random.randint(0,4, (4,6,8)).view(np.uint64)
    print(a)
    print("")
    downsampled, box = downsample_labels_3d(a, (2,3,4))
    assert tuple(box[1] - box[0]) == downsampled.shape
    print(downsampled)
    print(box.tolist())
    print("")

    print("total seconds: {:.2f}".format(time.time() - start))