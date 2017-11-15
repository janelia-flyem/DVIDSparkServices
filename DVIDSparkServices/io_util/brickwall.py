import numpy as np

from dvid_resource_manager.client import ResourceManagerClient

from DVIDSparkServices.sparkdvid.sparkdvid import sparkdvid
from DVIDSparkServices.dvid.metadata import DataInstance
from DVIDSparkServices import rddtools as rt
from DVIDSparkServices.auto_retry import auto_retry

from .brainmaps import BrainMapsVolume
from .brick import Grid, generate_bricks_from_volume_source, realign_bricks_to_new_grid, pad_brick_data_from_volume_source

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
        new_wall = BrickWall( self.bounding_box, self.grid, _bricks=new_logical_boxes_and_bricks.values() )
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
        return padded_bricks

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
            if target_partition_size_voxels:
                block_size_voxels = np.prod(grid.block_shape)
                rdd_partition_length = target_partition_size_voxels // block_size_voxels
            self.bricks = generate_bricks_from_volume_source(bounding_box, grid, volume_accessor_func, sc, rdd_partition_length)
    
    ##
    ## Convenience Constructors
    ##

    @classmethod
    def from_volume_source_config(cls, volume_config, sc=None, target_partition_size_voxels=None, resource_manager_client=None):
        if volume_config["source"]["semantic-type"] == "segmentation":
            assert volume_config["apply-labelmap"]["file"] == "", \
                "BrickWall does not automatically apply labelmaps. Remove that from the config and apply it afterwards."

        if resource_manager_client is None:
            resource_manager_client = ResourceManagerClient("", "")
        
        service = volume_config["source"]["service-type"]

        if service == "dvid":
            return cls._from_dvid_config(volume_config, sc, target_partition_size_voxels, resource_manager_client)
        elif service == "brainmaps":
            return  cls._from_brainmaps_config(volume_config, sc, target_partition_size_voxels, resource_manager_client)
        elif service == "slice-files":
            return  cls._from_slice_files_config(volume_config, sc, target_partition_size_voxels, resource_manager_client)

        raise RuntimeError(f"Unsupported volume configuration: {volume_config}")
            

    @classmethod
    def _from_brainmaps_config(cls, volume_config, sc, target_partition_size_voxels, resource_manager_client):
        input_bb_zyx = np.array(volume_config["geometry"]["bounding-box"])[:,::-1]
        brick_shape_zyx = volume_config["geometry"]["message-block-shape"][::-1]
        scale = volume_config["geometry"]["scale"]
        grid = Grid(brick_shape_zyx, (0,0,0))

        # Instantiate this outside of get_brainmaps_subvolume,
        # so it can be shared across an entire partition.
        vol = BrainMapsVolume( volume_config["source"]["project"],
                               volume_config["source"]["dataset"],
                               volume_config["source"]["volume-id"],
                               volume_config["source"]["change-stack-id"],
                               dtype=np.uint64 )

        assert (input_bb_zyx[0] >= vol.bounding_box[0]).all() and (input_bb_zyx[1] <= vol.bounding_box[1]).all(), \
            f"Specified bounding box ({input_bb_zyx.tolist()}) extends outside the "\
            f"BrainMaps volume geometry ({vol.bounding_box.tolist()})"
        
        # Two-levels of auto-retry:
        # 1. Auto-retry up to three time for any reason.
        # 2. If that fails due to 504 or 503 (probably cloud VMs warming up), wait 5 minutes and try again.
        @auto_retry(1, pause_between_tries=5*60.0, logging_name=__name__,
                    predicate=lambda ex: '503' in ex.args[0] or '504' in ex.args[0])
        @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
        def get_brainmaps_subvolume(box):
            req_bytes = 8 * np.prod(box[1] - box[0])
            with resource_manager_client.access_context('brainmaps', True, 1, req_bytes):
                return vol.get_subvolume(box, scale)

        return BrickWall(input_bb_zyx, grid, get_brainmaps_subvolume, sc, target_partition_size_voxels)
    
    @classmethod
    def _from_dvid_config(cls, volume_config, sc, target_partition_size_voxels, resource_manager_client):
        input_bb_zyx = np.array(volume_config["geometry"]["bounding-box"])[:,::-1]
        brick_shape_zyx = volume_config["geometry"]["message-block-shape"][::-1]
        grid = Grid(brick_shape_zyx, (0,0,0))

        mgr_ip = resource_manager_client.server_ip
        mgr_port = resource_manager_client.server_port

        server = volume_config["source"]["server"]
        uuid = volume_config["source"]["uuid"]
        scale = volume_config["geometry"]["scale"]

        if volume_config["source"]["semantic-type"] == "grayscale":
            instance_name = volume_config["source"]["grayscale-name"]
        elif volume_config["source"]["semantic-type"] == "segmentation":
            instance_name = volume_config["source"]["segmentation-name"]

        data_instance = DataInstance(volume_config["source"]["server"], volume_config["source"]["uuid"], instance_name)
        instance_type = data_instance.datatype
        is_labels = data_instance.is_labels()        

        # Two-levels of auto-retry:
        # 1. Auto-retry up to three time for any reason.
        # 2. If that fails due to 504 or 503 (probably cloud VMs warming up), wait 5 minutes and try again.
        @auto_retry(1, pause_between_tries=5*60.0, logging_name=__name__,
                    predicate=lambda ex: '503' in ex.args[0] or '504' in ex.args[0])
        @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
        def get_voxels(box):
            shape = np.asarray(box[1]) - box[0]
            return sparkdvid.get_voxels( server,
                                         uuid,
                                         instance_name,
                                         scale,
                                         instance_type,
                                         is_labels,
                                         shape,
                                         box[0],
                                         mgr_ip,
                                         mgr_port )
    
        return BrickWall(input_bb_zyx, grid, get_voxels, sc, target_partition_size_voxels)
    
    @classmethod
    def _from_slice_files_config(cls, volume_config, sc, target_partition_size_voxels, resource_manager_client):
        raise NotImplemented

