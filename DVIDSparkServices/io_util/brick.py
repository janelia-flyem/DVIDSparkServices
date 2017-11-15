from collections import namedtuple
from itertools import starmap
from functools import partial

import numpy as np
from DVIDSparkServices.util import ndrange, extract_subvol, overwrite_subvol, box_as_tuple, box_intersection
from DVIDSparkServices import rddtools as rt
from DVIDSparkServices.util import cpus_per_worker, num_worker_nodes

# Grid:
# Describes a blocking scheme, which is simply a grid block shape,
# and an offset coordinate for the first block in the grid.
Grid = namedtuple("Grid", "block_shape offset")

class Brick:
    """
    A Brick occupies one full block of a grid, but the volume
    data it contains might not span the entire grid block.
    
    Therefore, a Brick tracks it's logical_box (the entire grid block),
    and it's physical_box (the box its volume data actually occupies
    within the grid block.)
     
    Note: Both boxes are always stored in GLOBAL coordinates.
    """
    def __init__(self, logical_box, physical_box, volume):
        self.logical_box = np.asarray(logical_box)
        self.physical_box = np.asarray(physical_box)
        self.volume = volume
        assert (self.physical_box[1] - self.physical_box[0] == self.volume.shape).all()
        assert (self.physical_box[0] >= self.logical_box[0]).all()
        assert (self.physical_box[1] <= self.logical_box[1]).all()

    def __hash__(self):
        return hash(tuple(self.logical_box[0]))

    def __str__(self):
        if (self.logical_box == self.physical_box).all():
            return f"logical & physical: {self.logical_box.tolist()}"
        return f"logical: {self.logical_box.tolist()}, physical: {self.physical_box.tolist()}"


def generate_bricks_from_volume_source( bounding_box, grid, volume_accessor_func, sc=None, rdd_partition_length=None ):
    """
    Generate an RDD or iterable of Bricks for the given bounding box and grid.
     
    Args:
        bounding_box:
            (start, stop)
 
        grid:
            Grid (see above)
 
        volume_accessor_func:
            Callable with signature: f(box) -> ndarray
            Note: The callable will be unpickled only once per partition, so initialization
                  costs after unpickling are only incurred once per partition.
 
        sc:
            SparkContext. If provided, an RDD is returned.  Otherwise, returns an ordinary Python iterable.
 
        rdd_partition_length:
            Optional. If provided, the RDD will have (approximately) this many bricks per partition.
    """
    logical_and_physical_boxes = ( (box, box_intersection(box, bounding_box))
                                  for box in boxes_from_grid(bounding_box, grid) )

    if sc:
        num_rdd_partitions = None
        if rdd_partition_length:
            logical_and_physical_boxes = list(logical_and_physical_boxes) # need len()
            num_rdd_partitions = int( np.ceil( len(logical_and_physical_boxes) / rdd_partition_length ) )

        logical_and_physical_boxes = sc.parallelize( logical_and_physical_boxes, num_rdd_partitions )

    # Use map_partitions instead of map(), to be explicit about
    # the fact that the function is re-used within each partition.
    def make_bricks( logical_and_physical_boxes ):
        logical_and_physical_boxes = list(logical_and_physical_boxes)
        if not logical_and_physical_boxes:
            return []
        logical_boxes, physical_boxes = zip( *logical_and_physical_boxes )
        volumes = map( volume_accessor_func, physical_boxes )
        return starmap( Brick, zip(logical_boxes, physical_boxes, volumes) )
    
    bricks = rt.map_partitions( make_bricks, logical_and_physical_boxes )

    if sc:
        # If we're working with a tiny volume (e.g. testing),
        # make sure we at least parallelize across all cores.
        if bricks.getNumPartitions() < cpus_per_worker() * num_worker_nodes():
            bricks = bricks.repartition( cpus_per_worker() * num_worker_nodes() )

    return bricks

def pad_brick_data_from_volume_source( padding_grid, volume_accessor_func, brick ):
    """
    Expand the given Brick's data until its physical_box is aligned with the given padding_grid.
    The data in the expanded region will be sourced from the given volume_accessor_func.
    
    Note: padding_grid need not be identical to the grid the Brick was created with,
          but it must divide evenly into that grid. 

    For instance, if padding_grid happens to be the same as the brick's own native grid,
    then the phyiscal_box is expanded to align perfectly with the logical_box on all sides: 
    
        +-------------+      +-------------+
        | physical |  |      |     same    |
        |__________|  |      |   physical  |
        |             |  --> |     and     |
        |   logical   |      |   logical   |
        |_____________|      |_____________|
    
    Args:
        brick: Brick
        padding_grid: Grid
        volume_accessor_func: Callable with signature: f(box) -> ndarray

    Returns: Brick
    
    Note: If no padding is necessary, then the original Brick is returned (no copy is made).
    """
    block_shape = padding_grid.block_shape
    assert ((brick.logical_box - padding_grid.offset) % block_shape == 0).all(), \
        f"Padding grid {padding_grid.offset} must be aligned with brick logical_box: {brick.logical_box}"
    
    # Subtract offset to calculate the needed padding
    offset_physical_box = brick.physical_box - padding_grid.offset

    if (offset_physical_box % block_shape == 0).all():
        # Internal data is already aligned to the padding_grid.
        return brick
    
    offset_padded_box = np.array([offset_physical_box[0] // block_shape * block_shape,
                                  (offset_physical_box[1] + block_shape - 1) // block_shape * block_shape])
    
    # Re-add offset
    padded_box = offset_padded_box + padding_grid.offset
    assert (padded_box[0] >= brick.logical_box[0]).all()
    assert (padded_box[1] <= brick.logical_box[1]).all()

    # Initialize a new volume of the fully-padded shape
    padded_volume_shape = padded_box[1] - padded_box[0]
    padded_volume = np.zeros(padded_volume_shape, dtype=brick.volume.dtype)

    # Overwrite the previously existing data in the new padded volume
    orig_box = brick.physical_box
    orig_box_within_padded = orig_box - padded_box[0]
    overwrite_subvol(padded_volume, orig_box_within_padded, brick.volume)
    
    # Check for a non-zero-volume halo on all six sides.
    halo_boxes = []
    for axis in range(padded_volume.ndim):
        if orig_box[0,axis] != padded_box[0,axis]:
            leading_halo_box = padded_box.copy()
            leading_halo_box[1, axis] = orig_box[0,axis]
            halo_boxes.append(leading_halo_box)

        if orig_box[1,axis] != padded_box[1,axis]:
            trailing_halo_box = padded_box.copy()
            trailing_halo_box[0, axis] = orig_box[1,axis]
            halo_boxes.append(trailing_halo_box)

    assert halo_boxes, \
        "How could halo_boxes be empty if there was padding needed?"

    for halo_box in halo_boxes:
        # Retrieve padding data for one halo side
        halo_volume = volume_accessor_func(halo_box)
        
        # Overwrite in the final padded volume
        halo_box_within_padded = halo_box - padded_box[0]
        overwrite_subvol(padded_volume, halo_box_within_padded, halo_volume)

    return Brick( brick.logical_box, padded_box, padded_volume )


def realign_bricks_to_new_grid(new_grid, original_bricks):
    """
    Given a list/RDD of Bricks which are tiled over some original grid,
    chop them up and reassemble them into a new list/RDD of Bricks,
    tiled according to the given new_grid.
    
    Requires data shuffling.
    
    Returns: RDD (or iterable):
        [ (logical_box, Brick),
          (logical_box, Brick), ...]
    """
    # For each original brick, split it up according
    # to the new logical box destinations it will map to.
    new_logical_boxes_and_brick_fragments = rt.flat_map( partial(split_brick, new_grid), original_bricks )

    # Group fragments according to their new homes
    grouped_brick_fragments = rt.group_by_key( new_logical_boxes_and_brick_fragments )
    
    # Re-assemble fragments into the new grid structure.
    new_logical_boxes_and_bricks = rt.map_values(assemble_brick_fragments, grouped_brick_fragments)
    
    return new_logical_boxes_and_bricks


def split_brick(new_grid, original_brick):
    """
    Given a single brick and a new grid to which its data should be redistributed,
    split the brick into pieces, indexed by their NEW grid locations.
    
    The brick fragments are returned as Bricks themselves, but with relatively
    small volume and physical_box members.
    
    Returns: [(box,Brick), (box, Brick), ....],
            where each Brick is a fragment (to be assembled later into the new grid's bricks),
            and 'box' is the logical_box of the Brick into which this fragment should be assembled.
    """
    new_logical_boxes_and_fragments = []
    
    # Iterate over the new boxes that intersect with the original brick
    for new_logical_box in boxes_from_grid(original_brick.physical_box, new_grid):
        # Physical intersection of original with new
        split_box = box_intersection(new_logical_box, original_brick.physical_box)
        
        # Extract portion of original volume data that belongs to this new box
        split_box_internal = split_box - original_brick.physical_box[0]
        fragment_vol = extract_subvol(original_brick.volume, split_box_internal)

        # Append key (new_logical_box) and new brick fragment,
        # to be assembled into the final brick in a later stage.
        key = box_as_tuple(new_logical_box)
        fragment_brick = Brick(new_logical_box, split_box, fragment_vol)

        new_logical_boxes_and_fragments.append( (key, fragment_brick) )

    return new_logical_boxes_and_fragments


def assemble_brick_fragments( fragments ):
    """
    Given a list of Bricks with identical logical_boxes, splice their volumes
    together into a final Brick that contains a full volume containing all of
    the fragments.
    
    Note: Brick 'fragments' are also just Bricks, whose physical_box does
          not cover the entire logical_box for the brick.
    
    Each fragment's physical_box indicates where that fragment's data
    should be located within the final returned Brick.
    
    Returns: A Brick containing the data from all fragments.
    
    Note: If the fragment physical_boxes are not disjoint, the results
          are undefined.
    """
    fragments = list(fragments)

    # All logical boxes must be the same
    logical_boxes = np.asarray([frag.logical_box for frag in fragments])
    assert (logical_boxes == logical_boxes[0]).all(), \
        "Cannot assemble brick fragments from different logical boxes. "\
        "They belong to different bricks!"
    final_logical_box = logical_boxes[0]

    # The final physical box is the min/max of all fragment physical extents.
    physical_boxes = np.array([frag.physical_box for frag in fragments])
    assert physical_boxes.ndim == 3 # (N, 2, Dim)
    assert physical_boxes.shape == ( len(fragments), 2, final_logical_box.shape[1] )
    
    final_physical_box = np.asarray( ( np.min( physical_boxes[:,0,:], axis=0 ),
                                       np.max( physical_boxes[:,1,:], axis=0 ) ) )

    assert (final_physical_box[0] >= final_logical_box[0]).all()
    assert (final_physical_box[1] <= final_logical_box[1]).all()

    final_volume_shape = final_physical_box[1] - final_physical_box[0]
    dtype = fragments[0].volume.dtype

    final_volume = np.zeros(final_volume_shape, dtype)

    for frag in fragments:
        internal_box = frag.physical_box - final_physical_box[0]
        overwrite_subvol(final_volume, internal_box, frag.volume)

    return Brick( final_logical_box, final_physical_box, final_volume )


def boxes_from_grid(bounding_box, grid):
    """
    Generator.
    
    Assuming an ND grid with boxes of size grid.block_shape, and aligned at the given grid.offset,
    iterate over all boxes of the grid that fall within or intersect the given bounding_box.
    
    Note: The returned boxes are not clipped to fall within the bounding_box.
          If either bounding_box[0] or bounding_box[1] is not aligned with the grid,
          some returned boxes will extend beyond the bounding_box.
    """
    
    if grid.offset is None or not any(grid.offset):
        # Shortcut
        yield from _boxes_from_grid_no_offset(bounding_box, grid.block_shape)
    else:
        grid_offset = np.asarray(grid.offset)
        bounding_box = bounding_box - grid.offset
        for box in _boxes_from_grid_no_offset(bounding_box, grid.block_shape):
            box += grid_offset
            yield box


def _boxes_from_grid_no_offset(bounding_box, block_shape):
    """
    Generator.
    
    Assuming an ND grid with boxes of size block_shape, and aligned at the origin (0,0,...),
    iterate over all boxes of the grid that fall within or intersect the given bounding_box.
    
    Note: The returned boxes are not clipped to fall within the bounding_box.
          If either bounding_box[0] or bounding_box[1] is not aligned with the grid
          (i.e. they are not a multiple of block_shape),
          some returned boxes will extend beyond the bounding_box.
    """
    bounding_box = np.asarray(bounding_box, dtype=int)
    block_shape = np.asarray(block_shape)

    # round down, round up
    aligned_start = ((bounding_box[0]) // block_shape) * block_shape
    aligned_stop = ((bounding_box[1] + block_shape-1) // block_shape) * block_shape

    for block_start in ndrange( aligned_start, aligned_stop, block_shape ):
        yield np.array((block_start, block_start + block_shape))


def clipped_boxes_from_grid(bounding_box, grid):
    """
    Generator.
    
    Assuming an ND grid with boxes of size grid.block_shape, and aligned at the given grid.offset,
    iterate over all boxes of the grid that fall within or intersect the given bounding_box.
    
    Returned boxes that would intersect the edge of the bounding_box are clipped so as not
    to extend beyond the bounding_box.
    """
    for box in boxes_from_grid(bounding_box, grid):
        yield box_intersection(box, bounding_box)

