import logging
from itertools import starmap
from functools import partial
import collections

import numpy as np

from DVIDSparkServices.util import ndrange, extract_subvol, overwrite_subvol, box_as_tuple, box_intersection,\
    box_to_slicing
from DVIDSparkServices import rddtools as rt
from DVIDSparkServices.util import cpus_per_worker, num_worker_nodes, persist_and_execute, unpersist
from DVIDSparkServices.io_util.labelmap_utils import load_labelmap
from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray

logger = logging.getLogger(__name__)

class Grid:
    """
    Describes a blocking scheme, which is simply a grid block shape,
    and an offset coordinate for the first block in the grid.
    """
    def __init__(self, block_shape, offset=None):
        if offset is None:
            offset = (0,)*len(block_shape)
        assert len(block_shape) == len(offset)
        self.block_shape = np.asarray(block_shape)
        self.offset = np.asarray(offset)
        self.modulus_offset = self.offset % block_shape

    def equivalent_to(self, other_grid):
        """
        Returns True if the other grid is equivalent to this one, meaning 
        it has the same block shame and it's offset is the same after modulus.
        """
        return (self.block_shape == other_grid.block_shape).all() and \
               (self.modulus_offset == other_grid.modulus_offset).all()

    def compute_logical_box(self, point):
        """
        Return the logical box that encompasses the given point.
        """
        block_index = (point - self.offset) // self.block_shape
        block_start = self.offset + (block_index * self.block_shape)
        return np.asarray( (block_start, block_start + self.block_shape) )

class SparseBlockMask:
    """
    Tiny class to hold a low-resolution binary mask and the box it corresponds to.
    """
    def __init__(self, lowres_mask, box, resolution):
        """
        Args:
            lowres_mask:
                boolean ndarray, where each voxel represents a block of full-res data.
            box:
                The volume of space covered by the mask, in FULL-RES coordinates
            resolution:
                The width (or shape) of each lowres voxel in FULL-RES coordinates.
        """
        self.lowres_mask = lowres_mask.astype(bool, copy=False)
        self.box = np.asarray(box)
        self.resolution = resolution
        if isinstance(self.resolution, collections.Iterable):
            self.resolution = np.asarray(resolution)
            
        assert (((self.box[1] - self.box[0]) // self.resolution) == self.lowres_mask.shape).all(), \
            f"Inconsistent mask shape ({lowres_mask}) and box {box.tolist()} for the given resolution ({resolution}).\n"\
            "Note: box should be specified in FULL resolution coordinates."

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
        self._volume = volume
        assert (self.physical_box[1] - self.physical_box[0] == self.volume.shape).all()
        assert (self.physical_box[0] >= self.logical_box[0]).all()
        assert (self.physical_box[1] <= self.logical_box[1]).all()
        
        # Used for pickling.
        self._compressed_volume = None

    def __hash__(self):
        return hash(tuple(self.logical_box[0]))

    def __str__(self):
        if (self.logical_box == self.physical_box).all():
            return f"logical & physical: {self.logical_box.tolist()}"
        return f"logical: {self.logical_box.tolist()}, physical: {self.physical_box.tolist()}"

    @property
    def volume(self):
        """
        The volume is decompressed lazily.
        See __getstate__() for explanation.
        """
        if self._volume is None:
            assert self._compressed_volume is not None
            self._volume = self._compressed_volume.deserialize()
        return self._volume
    
    def __getstate__(self):
        """
        Pickle representation.
        
        By default, the volume would be compressed/decompressed transparently via
        the code in CompressedNumpyArray.py, but we want decompression to be
        performed lazily.
        
        Therefore, we explicitly compress the volume here, and decompress it only
        first upon access, via the self.volume property.
        
        This avoids decompression during certain Spark operations that don't
        require actual manipulation of the voxels, notably groupByKey().
        """
        if self._volume is not None:
            self._compressed_volume = CompressedNumpyArray(self._volume)

        d = self.__dict__.copy()
        d['_volume'] = None
        return d

def generate_bricks_from_volume_source( bounding_box, grid, volume_accessor_func, sc=None, rdd_partition_length=None, sparse_boxes=None ):
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
        
        sparse_boxes:
            Optional.
            A pre-calculated list of boxes to use instead of instead of calculating
            the complete (dense) list of grid boxes within the bounding box.
            If provided, should be a list of physical boxes, and no two should occupy the same logical box.
            Note: They will still be clipped to the overall bounding_box.
    """
    if sparse_boxes is None:
        # Generate boxes from densely populated grid
        logical_and_physical_boxes = ( (box, box_intersection(box, bounding_box))
                                      for box in boxes_from_grid(bounding_box, grid) )
    else:
        # User provided list of physical boxes.
        # Clip them to the bounding box and calculate the logical boxes.
        if not hasattr(sparse_boxes, '__len__'):
            sparse_boxes = list( sparse_boxes )
        physical_boxes = np.asarray( sparse_boxes )
        assert physical_boxes.ndim == 3 and physical_boxes.shape[1:3] == (2,3)
        
        def logical_and_clipped( box ):
            # Note: Non-intersecting boxes will have non-positive shape after clipping
            clipped_box = box_intersection(box, bounding_box)
            midpoint = (clipped_box[0] + clipped_box[1]) // 2
            logical_box = grid.compute_logical_box( midpoint )
            return ( logical_box, clipped_box )

        logical_and_physical_boxes = map(logical_and_clipped, physical_boxes)

        # Drop any boxes that fall completely outside the bounding box
        # (check that physical box has positive shape)
        logical_and_physical_boxes = filter(lambda l_p: (l_p[1][1] > l_p[1][0]).all(), logical_and_physical_boxes )

    if sc:
        num_rdd_partitions = None
        if rdd_partition_length is not None:
            rdd_partition_length = max(1, rdd_partition_length)
            if not hasattr(logical_and_physical_boxes, '__len__'):
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


def apply_labelmap_to_bricks(bricks, labelmap_config, working_dir, unpersist_original=False):
    """
    Relabel the bricks with a labelmap.
    If the given config specifies not labelmap, then the original is returned.

    bricks: RDD of Bricks
    
    labelmap_config: config dict, adheres to LabelMapSchema
    
    working_dir: If labelmap is from a gbucket, it will be downlaoded to working_dir.
    
    unpersist_original: If True, unpersist (or replace) the input.
                     Otherwise, the caller is free to unpersist the returned
                     bricks without affecting the input.
    
    Returns:
        remapped_bricks - Already computed and persisted.
    """
    path = labelmap_config["file"]
    if not path:
        if not unpersist_original:
            # The caller wants to be sure that the result can be
            # unpersisted safely without affecting the bricks he passed in,
            # so we return a *new* RDD, even though it's just a copy of the original.
            return rt.map( lambda brick: brick, bricks )
        return bricks

    mapping_pairs = load_labelmap(labelmap_config, working_dir)
    remapped_bricks = apply_label_mapping(bricks, mapping_pairs)

    if unpersist_original:
        unpersist(bricks)

    return remapped_bricks


def apply_label_mapping(bricks, mapping_pairs):
    """
    Given an RDD of bricks (of label data) and a pre-loaded labelmap in
    mapping_pairs [[orig,new],[orig,new],...],
    apply the mapping to the bricks.
    
    bricks:
        RDD of Bricks containing label volumes
    
    mapping_pairs:
        Mapping as returned by load_labelmap.
        An ndarray of the form:
            [[orig,new],
             [orig,new],
             ... ],
    """
    from dvidutils import LabelMapper
    def remap_bricks(partition_bricks):
        domain, codomain = mapping_pairs.transpose()
        mapper = LabelMapper(domain, codomain)
        
        partition_bricks = list(partition_bricks)
        for brick in partition_bricks:
            # TODO: Apparently LabelMapper can't handle non-contiguous arrays right now.
            #       (It yields incorrect results)
            #       Check to see if this is still a problem in the latest version of xtensor-python.
            brick.volume = np.asarray( brick.volume, order='C' )
            
            mapper.apply_inplace(brick.volume, allow_unmapped=True)
        return partition_bricks
    
    # Use mapPartitions (instead of map) so LabelMapper can be constructed just once per partition
    remapped_bricks = rt.map_partitions( remap_bricks, bricks )
    persist_and_execute(remapped_bricks, f"Remapping bricks", logger)
    return remapped_bricks


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
    #grouped_brick_fragments = rt.group_by_key( new_logical_boxes_and_brick_fragments )
    grouped_brick_fragments = rt.frugal_group_by_key( new_logical_boxes_and_brick_fragments )
    
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

def slabs_from_box( full_res_box, slab_depth, scale=0, scaling_policy='round-out', slab_cutting_axis=0 ):
    """
    Generator.
    
    Divide a bounding box into several 'slabs' stacked along a particular axis,
    after optionally reducing the bounding box to a reduced scale.
    
    Note: The output slabs are aligned to multiples of the slab depth.
          For example, if full_res_box starts at 3 and slab_depth=10,
          then the first slab will span [3,10], and the second slab will span from [10,20].
    
    full_res_box: (start, stop)
        The original bounding-box, in full-res coordinates
    
    slab_depth: (int)
        The desired width of the output slabs.
        This will be the size of the output slabs, regardless of any scaling applied.
    
    scale:
        Reduce the bounding-box to a smaller scale before computing the output slabs.
        
    scaling_policy:
        For scale > 0, the input bounding box is reduced.
        For bounding boxes that aren't divisible by 2*scale, the start/stop coordinates must be rounded up or down.
        Choices are:
            'round-out': Expand full_res_box to the next outer multiple of 2**scale before scaling down.
            'round-in': Shrink full_res_box to the next inner multiple of 2**scale before scaling down.
            'round-down': Round down on full_res_box (both start and stop) before scaling down.
    
    slab_cutting_axes:
        Which axis to cut across to form the stacked slabs. Default is Z (assuming ZYX order).
    """
    assert scaling_policy in ('round-out', 'round-in', 'round-down')
    full_res_box = np.asarray(full_res_box)

    round_method = scaling_policy[len('round-'):]
    scaled_input_bb_zyx = round_box(full_res_box, 2**scale, round_method) // 2**scale

    slab_shape_zyx = scaled_input_bb_zyx[1] - scaled_input_bb_zyx[0]
    slab_shape_zyx[slab_cutting_axis] = slab_depth

    # This grid outlines the slabs -- each box in slab_grid is a full slab
    grid_offset = scaled_input_bb_zyx[0].copy()
    grid_offset[slab_cutting_axis] = 0 # See note about slab alignment, above.
    
    slab_grid = Grid(slab_shape_zyx, grid_offset)
    slab_boxes = clipped_boxes_from_grid(scaled_input_bb_zyx, slab_grid)
    
    return slab_boxes

def round_coord(coord, grid_spacing, how):
    """
    Round the given coordinate up or down to the nearest grid position.
    """
    assert how in ('down', 'up')
    if how == 'down':
        return (coord // grid_spacing) * grid_spacing
    if how == 'up':
        return ((coord + grid_spacing - 1) // grid_spacing) * grid_spacing

def round_box(box, grid_spacing, how='out'):
    """
    Expand/shrink the given box out to align it to a grid.

    box: (start, stop)
    grid_spacing: int or shape
    how: One of ['out', 'in', 'down', 'up'].
         Determines which direction the box corners are moved.
    """
    directions = { 'out':  ('down', 'up'),
                   'in':   ('up', 'down'),
                   'down': ('down', 'down'),
                   'up':   ('up', 'up') }

    assert how in directions.keys()
    return np.array( [ round_coord(box[0], grid_spacing, directions[how][0]),
                       round_coord(box[1], grid_spacing, directions[how][1]) ] )

def sparse_boxes_from_block_mask( sparse_block_mask, brick_grid, return_logical_boxes=False ):
    """
    Given a sparsely populated binary image (block_mask), overlay a brick_grid
    and extract the list of non-empty bricks (specified as boxes).

    TODO: Implement an alternative version of this function that accepts a
          list of coordinates, instead of a mask, and groups coordinates via pandas operations.
          (It would be more expensive for relatively dense masks, but cheaper for very sparse masks.)

    Args:
        sparse_block_mask:
            SparseBlockMask

        brick_grid:
            The desired grid to use for the output.
        
        return_logical_boxes:
            If True, the result is returned as a list of full-size "logical" boxes.
            Otherwise, each box is shrunken to the minimal size while still
            encompassing all data with its grid box (i.e. a physical box).

    Returns:
        (logical_boxes, physical_boxes) of non-empty bricks, as indicated by block_mask.    
    """
    assert isinstance(sparse_block_mask, SparseBlockMask)
    assert (brick_grid.modulus_offset == (0,0,0)).all(), \
        "TODO: This function doesn't yet support brick grids with non-zero offsets"
    assert ((brick_grid.block_shape % sparse_block_mask.resolution) == 0).all(), \
        "Brick grid must be a multiple of the block grid"

    block_mask_box = np.asarray(sparse_block_mask.box)
    
    lowres_brick_grid = Grid( brick_grid.block_shape // sparse_block_mask.resolution )
    lowres_block_mask_box = block_mask_box // sparse_block_mask.resolution
    
    logical_and_clipped_boxes = ( (box, box_intersection(box, lowres_block_mask_box))
                                   for box in boxes_from_grid(lowres_block_mask_box, lowres_brick_grid) )

    lowres_boxes = []
    
    for logical_lowres_box, clipped_lowres_box in logical_and_clipped_boxes:
        box_within_mask = clipped_lowres_box - lowres_block_mask_box[0]
        brick_mask = sparse_block_mask.lowres_mask[box_to_slicing(*box_within_mask)]
        brick_coords = np.transpose(brick_mask.nonzero()).astype(np.int32)
        if len(brick_coords) == 0:
            continue
        if return_logical_boxes:
            lowres_boxes.append( logical_lowres_box )
        else:
            physical_lowres_box = ( brick_coords.min(axis=0),
                                    brick_coords.max(axis=0) + 1 )
            
            physical_lowres_box += box_within_mask[0] + lowres_block_mask_box[0]
            
            lowres_boxes.append( physical_lowres_box )
    
    nonempty_boxes = np.array(lowres_boxes, dtype=np.int32) * sparse_block_mask.resolution
    
    if len(nonempty_boxes) == 0:
        nonempty_boxes = np.zeros((0,3), dtype=np.int32)
    
    return nonempty_boxes
