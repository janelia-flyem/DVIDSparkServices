from itertools import starmap
import numpy as np
from skimage.util import view_as_blocks

def bb_to_slicing(start, stop):
    """
    For the given bounding box (start, stop),
    return the corresponding slicing tuple.

    Example:
    
        >>> assert bb_to_slicing([1,2,3], [4,5,6]) == np.s_[1:4, 2:5, 3:6]
    """
    return tuple( starmap( slice, zip(start, stop) ) )

def boxlist_to_json( bounds_list, indent=0 ):
    # The 'json' module doesn't have nice pretty-printing options for our purposes,
    # so we'll do this ourselves.
    from cStringIO import StringIO
    from os import SEEK_CUR

    buf = StringIO()
    buf.write('    [\n')
    for bounds_zyx in bounds_list:
        start_str = '[{}, {}, {}]'.format(*bounds_zyx[0])
        stop_str  = '[{}, {}, {}]'.format(*bounds_zyx[1])
        buf.write(' '*indent + '[ ' + start_str + ', ' + stop_str + ' ],\n')

    # Remove last comma, close list
    buf.seek(-2, SEEK_CUR)
    buf.write('\n')
    buf.write(' '*indent + ']')

    return buf.getvalue()

def mkdir_p(path):
    """
    Like the bash command: mkdir -p
    
    ...why the heck isn't this built-in to the Python std library?
    """
    import os, errno
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class RoiMap(object):
    """
    Little utility class to help with ROI manipulations
    """
    def __init__(self, roi_blocks):
        # Make a map of the entire ROI
        # Since roi blocks are 32^2, this won't be too huge.
        # For example, a ROI that's 10k*10k*100k pixels, this will be ~300 MB
        # For a 100k^3 ROI, this will be 30 GB (still small enough to fit in RAM on the driver)
        block_mask, (blocks_start, blocks_stop) = coordlist_to_boolmap(roi_blocks)
        blocks_shape = blocks_stop - blocks_start

        self.block_mask = block_mask
        self.blocks_start = blocks_start
        self.blocks_stop = blocks_stop
        self.blocks_shape = blocks_shape
        

def coordlist_to_boolmap(coordlist, bounding_box=None):
    """
    Convert the given list of coordinates (z,y,x) into a 3D bool array.
    
    coordlist: For example, [[0,1,2], [0,1,3], [0,2,0]]
    
    bounding_box: (Optional) Specify the bounding box that corresponds
                  to the region of interest.
                  If not provided, default is to use the smallest bounds
                  that includes the entire coordlist.
    
    Returns: boolmap (3D array, bool), and the bounding_box (start, stop) of the array.
    """
    coordlist = np.asarray(list(coordlist)) # Convert, in case coordlist was a set
    coordlist_min = np.min(coordlist, axis=0)
    coordlist_max = np.max(coordlist, axis=0)
    
    if bounding_box is None:
        start, stop = (coordlist_min, 1+coordlist_max)
    else:
        start, stop = bounding_box
        if (coordlist_min < start).any() or (coordlist_max >= stop).any():
            # Remove the coords that are outside the user's bounding-box of interest
            coordlist = filter(lambda coord: (coord - start >= 0).all() and (coord < stop).all(),
                               coordlist)
            coordlist = np.array(coordlist)

    shape = stop - start
    coords = coordlist - start
    
    boolmap = np.zeros( shape=shape, dtype=bool )
    boolmap[tuple(coords.transpose())] = 1
    return boolmap, (start, stop)

def block_mask_to_px_mask(block_mask, block_width):
    """
    Given a mask array with block-resolution (each item represents 1 block),
    upscale it to pixel-resolution.
    """
    px_mask_shape = block_width*np.array(block_mask.shape)
    px_mask = np.zeros( px_mask_shape, dtype=np.bool )
    
    # In this 6D array, the first 3 axes address the block index,
    # and the last 3 axes address px within the block
    px_mask_blockwise = view_as_blocks(px_mask, (block_width, block_width, block_width))
    assert px_mask_blockwise.shape[0:3] == block_mask.shape
    assert px_mask_blockwise.shape[3:6] == (block_width, block_width, block_width)
    
    # Now we can broadcast into it from the block mask
    px_mask_blockwise[:] = block_mask[:, :, :, None, None, None]
    return px_mask

def dense_roi_mask_for_subvolume(subvolume, border='default'):
    """
    Return a dense (pixel-level) mask for the given subvolume,
    according to the ROI blocks it lists in its 'intersecting_blocks' member.
    
    border: How much border to incorporate into the mask beyond the subvolume's own bounding box.
            By default, just use the subvolume's own 'border' attribute.
    """
    sv = subvolume
    if border == 'default':
        border = sv.border
    else:
        assert border <= sv.border, \
            "Subvolumes don't store ROI blocks outside of their known border "\
            "region, so I can't produce a mask outside that area."
    
    # subvol bounding box/shape (not block-aligned)
    sv_start_px = np.array((sv.box.z1, sv.box.y1, sv.box.x1)) - border
    sv_stop_px  = np.array((sv.box.z2, sv.box.y2, sv.box.x2)) + border
    sv_shape_px = sv_stop_px - sv_start_px
    
    # subvol bounding box/shape in block coordinates
    sv_start_blocks = sv_start_px // sv.roi_blocksize
    sv_stop_blocks = (sv_stop_px + sv.roi_blocksize-1) // sv.roi_blocksize

    intersecting_block_mask, _ = coordlist_to_boolmap(sv.intersecting_blocks, (sv_start_blocks, sv_stop_blocks))
    intersecting_dense = block_mask_to_px_mask(intersecting_block_mask, sv.roi_blocksize)

    # bounding box of the sv dense coordinates within the block-aligned intersecting_dense
    dense_start = sv_start_px % sv.roi_blocksize
    dense_stop = dense_start + sv_shape_px
    
    # Extract the pixels we want from the (block-aligned) intersecting_dense
    sv_intersecting_dense = intersecting_dense[bb_to_slicing(dense_start, dense_stop)]
    assert sv_intersecting_dense.shape == tuple(sv_shape_px)
    return sv_intersecting_dense

def runlength_encode(coord_list_zyx, assume_sorted=False):
    """
    Given an array of coordinates in the form:
        
        [[Z,Y,X],
         [Z,Y,X],
         [Z,Y,X],
         ...
        ]
        
    Return an array of run-length encodings of the form:
    
        [[Z,Y,X1,X2],
         [Z,Y,X1,X2],
         [Z,Y,X1,X2],
         ...
        ]
    
    Note: The interval [X1,X2] is INCLUSIVE, following DVID conventions, not Python conventions.
    
    Args:
        coord_list_zyx:
            Array of shape (N,3)
        
        assume_sorted:
            If True, the provided coordinates are assumed to be pre-sorted in Z-Y-X order.
            Otherwise, they are sorted before the RLEs are computed.
    
    Timing notes:
        The FIB-25 'seven_column_roi' consists of 927971 block indices.
        On that ROI, this function takes 1.65 seconds, but with numba installed,
        it takes 35 ms (after ~400 ms warmup).
        So, JIT speedup is ~45x.
    """
    coord_list_zyx = np.asarray(coord_list_zyx)
    assert coord_list_zyx.ndim == 2
    assert coord_list_zyx.shape[1] == 3
    if len(coord_list_zyx) == 0:
        return np.ndarray( (0,4), np.int64 )
    
    if not assume_sorted:
        sorting_ind = np.lexsort(coord_list_zyx.transpose()[::-1])
        coord_list_zyx = coord_list_zyx[sorting_ind]

    return _runlength_encode(coord_list_zyx)

# See conditional jit activation, below
#@numba.jit(nopython=True)
def _runlength_encode(coord_list_zyx):
    """
    Helper function for runlength_encode(), above.
    
    coord_list_zyx:
        Array of shape (N,3), of form [[Z,Y,X], [Z,Y,X], ...],
        pre-sorted in Z-Y-X order.  Duplicates permitted.
    """
    # Numba doesn't allow us to use empty lists at all,
    # so we have to initialize this list with a dummy row,
    # which we'll omit in the return value
    runs = [0,0,0,0]
    
    # Start the first run
    (prev_z, prev_y, prev_x) = current_run_start = coord_list_zyx[0]
    
    for i in range(1, len(coord_list_zyx)):
        (z,y,x) = coord = coord_list_zyx[i]

        # If necessary, end the current run and start a new one
        # (Also, support duplicate coords without breaking the current run.)
        if (z != prev_z) or (y != prev_y) or (x not in (prev_x, 1+prev_x)):
            runs += list(current_run_start) + [prev_x]
            current_run_start = coord

        (prev_z, prev_y, prev_x) = (z,y,x)

    # End the last run
    runs += list(current_run_start) + [prev_x]

    # Return as 2D array
    runs = np.array(runs).reshape((-1,4))
    return runs[1:, :] # omit dummy row (see above)

# Enable JIT if numba is available
try:
    import numba
    _runlength_encode = numba.jit(nopython=True)(_runlength_encode)
except ImportError:
    pass


def mask_roi(data, subvolume, border='default'):
    """
    masks data to 0 if outside of ROI stored in subvolume
    
    Note: This function operates on data IN-PLACE
    """
    mask = dense_roi_mask_for_subvolume(subvolume, border)
    assert data.shape == mask.shape
    data[np.logical_not(mask)] = 0
    return None # Emphasize in-place behavior


def select_item(rdd, *indexes):
    """
    Given an RDD of tuples, return an RDD listing the Nth item from each tuple.
    If the tuples are nested, you can provide multiple indexes to drill down to the element you want.
    
    For now, each index must be either 0 or 1.
    
    NOTE: Multiple calls to this function will result in redundant calculations.
          You should probably persist() the rdd before calling this function.
    
    >>> rdd = sc.parallelize([('foo', ('a', 'b')), ('bar', ('c', 'd'))])
    >>> select_item(rdd, 1, 0).collect()
    ['b', 'd']
    """
    for i in indexes:
        if i == 0:
            rdd = rdd.keys()
        else:
            rdd = rdd.values()
    return rdd

def zip_many(*rdds):
    """
    Like RDD.zip(), but supports more than two RDDs.
    It's baffling that PySpark doesn't include this by default...
    """
    assert len(rdds) >= 2

    result = rdds[0].zip(rdds[1])
    rdds = rdds[2:]

    def append_value_to_key(k_v):
        return (k_v[0] + (k_v[1],))

    while rdds:
        next_rdd, rdds = rdds[0], rdds[1:]
        result = result.zip(next_rdd).map(append_value_to_key)
    return result

def join_many(*rdds):
    """
    Like RDD.join(), but supports more than two RDDs.
    It's baffling that PySpark doesn't include this by default...
    """
    assert len(rdds) >= 2
    
    result = rdds[0].join(rdds[1])
    rdds = rdds[2:]
    
    def condense_value(k_v):
        k, (v1, v2) = k_v
        return (k, v1 + (v2,))
    
    while rdds:
        next_rdd, rdds = rdds[0], rdds[1:]
        result = result.join(next_rdd).map(condense_value, True)
    return result
