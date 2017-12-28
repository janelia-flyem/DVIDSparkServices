import numpy as np

from DVIDSparkServices import rddtools as rt
from DVIDSparkServices.util import num_worker_nodes, cpus_per_worker
from dvidutils import downsample_labels

from .brick import Brick, Grid, generate_bricks_from_volume_source, realign_bricks_to_new_grid, pad_brick_data_from_volume_source, apply_labelmap_to_bricks

class BrickWall:
    """
    Manages a (lazy) set of bricks within a Grid.
    Mostly just a convenience wrapper to simplify pipelines of transformations over RDDs of bricks.
    """
    
    ##
    ## Operations
    ##

    def realign_to_new_grid(self, new_grid):
        """
        Chop upand the Bricks in this BrickWall reassemble them into a new BrickWall,
        tiled according to the given new_grid.
        
        Note: Requires data shuffling.
        
        Returns: A a new BrickWall, with a new internal RDD for bricks.
        """
        new_logical_boxes_and_bricks = realign_bricks_to_new_grid( new_grid, self.bricks )
        new_wall = BrickWall( self.bounding_box, self.grid, _bricks=rt.values(new_logical_boxes_and_bricks) )
        return new_wall

    def fill_missing(self, volume_accessor_func, padding_grid=None):
        """
        For each brick whose physical_box does not extend to all edges of its logical_box,
        fill the missing space with data from the given volume accessor.
        
        Args:
            volume_accessor_func:
                See __init__, above.
            
            padding_grid:
                (Optional.) Need not be identical to the BrickWall's native grid,
                but must divide evenly into it. If not provided, the native grid is used.
        """
        if padding_grid is None:
            padding_grid = self.grid
            
        def pad_brick(brick):
            return pad_brick_data_from_volume_source(padding_grid, volume_accessor_func, brick)
        
        padded_bricks = rt.map( pad_brick, self.bricks )
        new_wall = BrickWall( self.bounding_box, self.grid, _bricks=padded_bricks )
        return new_wall

    def apply_labelmap(self, labelmap_config, working_dir, unpersist_original=False):
        """
        Relabel the bricks in this BrickWall with a labelmap.
        If the given config specifies not labelmap, then the original is returned.
    
        bricks: RDD of Bricks
        
        labelmap_config: config dict, adheres to LabelMapSchema
        
        working_dir: If labelmap is from a gbucket, it will be downlaoded to working_dir.
        
        unpersist_original: If True, unpersist (or replace) the input.
                            Otherwise, the caller is free to unpersist the returned
                            BrickWall without affecting the input.
        
        Returns:
            A new BrickWall, with remapped bricks.
        """
        remapped_bricks = apply_labelmap_to_bricks(self.bricks, labelmap_config, working_dir, unpersist_original)
        return BrickWall( self.bounding_box, self.grid, _bricks=remapped_bricks )

    def translate(self, offset_zyx):
        """
        Translate all bricks by the given offset.
        Does not change the brick data, just the logical/physical boxes.
        
        Also, translates the bounding box and grid.
        """
        def translate_brick(brick):
            return Brick( brick.logical_box + offset_zyx,
                          brick.physical_box + offset_zyx,
                          brick.volume )

        translated_bricks = rt.map( translate_brick, self.bricks )
        
        new_bounding_box = self.bounding_box + offset_zyx
        new_grid = Grid( self.grid.block_shape, self.grid.offset + offset_zyx )
        return BrickWall( new_bounding_box, new_grid, _bricks=translated_bricks )

    def persist_and_execute(self, description, logger=None):
        self.bricks = rt.persist_and_execute( self.bricks, description, logger )
    
    def unpersist(self):
        rt.unpersist(self.bricks)

    def label_downsample(self, block_shape):
        assert block_shape[0] == block_shape[1] == block_shape[2], \
            "Currently, downsampling must be isotropic"

        factor = block_shape[0]
        def downsample_brick(brick):
            # For consistency with DVID's on-demand downsampling, we suppress 0 pixels.
            assert (brick.physical_box % factor == 0).all()
            assert (brick.logical_box % factor == 0).all()
        
            # Old: Python downsampling
            # downsample_3Dlabels(brick.volume)
        
            # Newer: Numba downsampling
            #downsampled_volume, _ = downsample_labels_3d_suppress_zero(brick.volume, (2,2,2), brick.physical_box)
        
            # Even Newer: C++ downsampling (note: only works on aligned data.)
            downsampled_volume = downsample_labels(brick.volume, factor, suppress_zero=True)
        
            downsampled_logical_box = brick.logical_box // factor
            downsampled_physical_box = brick.physical_box // factor
            
            return Brick(downsampled_logical_box, downsampled_physical_box, downsampled_volume)

        new_bounding_box = self.bounding_box // factor
        new_grid = Grid( self.grid.block_shape // factor, self.grid.offset // factor )
        new_bricks = rt.map( downsample_brick, self.bricks )
        
        return BrickWall( new_bounding_box, new_grid, _bricks=new_bricks )

    def copy(self):
        """
        Return a duplicate of this BrickWall, with a new bricks RDD (which not persisted).
        """
        return BrickWall( self.bounding_box, self.grid, rt.map( lambda x:x, self.bricks ) )

    ##
    ## Convenience Constructor
    ##

    @classmethod
    def from_volume_service(cls, volume_service, scale=0, bounding_box_zyx=None, sc=None, target_partition_size_voxels=None):
        grid = Grid(volume_service.preferred_message_shape, (0,0,0))
        
        downsampled_box = bounding_box_zyx
        if downsampled_box is None:
            full_box = volume_service.bounding_box_zyx
            downsampled_box = np.zeros((2,3), dtype=int)
            downsampled_box[0] = full_box[0] // 2**scale # round down
            
            # Proper downsampled bounding-box would round up here...
            #downsampled_box[1] = (full_box[1] + 2**scale - 1) // 2**scale
            
            # ...but some some services probably don't do that, so we'll
            # round down to avoid out-of-bounds errors for higher scales. 
            downsampled_box[1] = full_box[1] // 2**scale

        return BrickWall( downsampled_box,
                          grid,
                          lambda box: volume_service.get_subvolume(box, scale),
                          sc,
                          target_partition_size_voxels )

    ##
    ## Generic Constructor
    ##

    def __init__(self, bounding_box, grid, volume_accessor_func=None, sc=None, target_partition_size_voxels=None, _bricks=None):
        """
        Generic constructor, taking an arbitrary volume_accessor_func.
        Specific convenience constructors for various DVID/Brainmaps/slices sources are below.
        
        Args:
            bounding_box:
                (start, stop)
     
            grid:
                Grid (see brick.py)
     
            volume_accessor_func:
                Callable with signature: f(box) -> ndarray
                Note: The callable will be unpickled only once per partition, so initialization
                      costs after unpickling are only incurred once per partition.
     
            sc:
                SparkContext. If provided, an RDD is returned.  Otherwise, returns an ordinary Python iterable.
     
            target_partition_size_voxels:
                Optional. If provided, the RDD partition lengths (i.e. the number of bricks per RDD partition)
                will be chosen to have (approximately) this many total voxels in each partition.
        """
        self.grid = grid
        self.bounding_box = bounding_box

        if _bricks:
            assert sc is None
            assert target_partition_size_voxels is None
            assert volume_accessor_func is None
            self.bricks = _bricks
        else:
            assert volume_accessor_func is not None
            rdd_partition_length = None
            if target_partition_size_voxels is None:
                num_threads = num_worker_nodes() * cpus_per_worker()
                total_voxels = np.prod(bounding_box[1] - bounding_box[0])
                voxels_per_thread = total_voxels / num_threads
                target_partition_size_voxels = (voxels_per_thread // 2) # Arbitrarily aim for 2 partitions per thread

            block_size_voxels = np.prod(grid.block_shape)
            rdd_partition_length = target_partition_size_voxels // block_size_voxels

            self.bricks = generate_bricks_from_volume_source(bounding_box, grid, volume_accessor_func, sc, rdd_partition_length)
