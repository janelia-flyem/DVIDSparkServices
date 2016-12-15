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
