import copy
import logging
from functools import partial

import numpy as np
from skimage.util import view_as_blocks

from dvid_resource_manager.client import ResourceManagerClient

from DVIDSparkServices import rddtools as rt
from DVIDSparkServices.util import num_worker_nodes, cpus_per_worker, replace_default_entries, Timer, box_to_slicing
from DVIDSparkServices.io_util.brick import Grid, clipped_boxes_from_grid
from DVIDSparkServices.io_util.brickwall import BrickWall
from DVIDSparkServices.io_util.volume_service import VolumeService, VolumeServiceWriter, GrayscaleVolumeSchema
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.dvid.metadata import ( is_datainstance, create_rawarray8, Compression,
                                              extend_list_value, update_extents, reload_server_metadata )

logger = logging.getLogger(__name__)

class ConvertGrayscaleVolume(Workflow):
    OptionsSchema = copy.deepcopy(Workflow.OptionsSchema)
    OptionsSchema["additionalProperties"] = False
    OptionsSchema["properties"].update(
    {
        "max-pyramid-scale": {
            "description": "The maximum scale to copy from input to output.",
            "type": "integer",
            "minimum": 0,
            "maximum": 10,
            ##
            ## NO DEFAULT: Must choose!
            #"default": -1
        },
         
        "slab-depth": {
            "description": "The volume is processed iteratively, in 'slabs'.\n"
                           "This setting determines the thickness of each slab.\n"
                           "Must be a multiple of the output brick Z-dimension.\n",
            "type": "integer",
            "minimum": 1
            ##
            ## NO DEFAULT: Must choose!
            #"default": -1
        }
    })

    Schema = \
    {
        "$schema": "http://json-schema.org/schema#",
        "title": "Convert a grayscale volume from one format to another.",
        "type": "object",
        "required": ["input", "output"],
        "properties": {
            "input": GrayscaleVolumeSchema,
            "output": GrayscaleVolumeSchema,
            "options": OptionsSchema
        }
    }

    @classmethod
    def schema(cls):
        return ConvertGrayscaleVolume.Schema


    # name of application for DVID queries
    APPNAME = "ConvertGrayscaleVolume".lower()


    def __init__(self, config_filename):
        super().__init__( config_filename, ConvertGrayscaleVolume.schema(), "Convert Grayscale Volume" )


    def _init_services(self):
        """
        Initialize the input and output services,
        and fill in 'auto' config values as needed.
        """
        input_config = self.config_data["input"]
        output_config = self.config_data["output"]
        options = self.config_data["options"]

        self.mgr_client = ResourceManagerClient( options["resource-server"], options["resource-port"] )
        self.input_service = VolumeService.create_from_config( input_config, self.config_dir, self.mgr_client )

        replace_default_entries(output_config["geometry"]["bounding-box"], self.input_service.bounding_box_zyx[:, ::-1])
        self.output_service = VolumeService.create_from_config( output_config, self.config_dir, self.mgr_client )
        assert isinstance( self.output_service, VolumeServiceWriter )

        logger.info(f"Output bounding box: {output_config['geometry']['bounding-box']}")


    def _validate_config(self):
        """
        Validate config values.
        """
        options = self.config_data["options"]

        brick_depth = self.output_service.preferred_message_shape[0]
        assert options["slab-depth"] % brick_depth == 0, \
            f'slab-depth ({options["slab-depth"]}) is not a multiple of the output brick shape ({brick_depth})'

        # Output bounding-box must match exactly (or left as auto)
        input_bb_zyx = self.input_service.bounding_box_zyx
        output_bb_zyx = self.output_service.bounding_box_zyx
        assert ((output_bb_zyx == input_bb_zyx) | (output_bb_zyx == -1)).all(), \
            "Output bounding box must match the input bounding box exactly. (No translation permitted)."


    def _prepare_output(self):
        """
        Create DVID data instances (including multi-scale) and update metadata.
        """
        output_config = self.config_data["output"]
        max_scale = self.config_data["options"]["max-pyramid-scale"]

        # Only dvid supported so far...
        assert "dvid" in output_config, "Unsupported output service type"

        server = output_config["dvid"]["server"]
        uuid = output_config["dvid"]["uuid"]
        instance_name = output_config["dvid"]["grayscale-name"]
        block_width = output_config["geometry"]["block-width"]

        if output_config["dvid"]["compression"] == "raw":
            compression = Compression.DEFAULT
        elif output_config["dvid"]["compression"] == "jpeg":
            compression = Compression.JPEG

        full_output_box_zyx = np.array(output_config["geometry"]["bounding-box"])[:, ::-1]

        for scale in range(max_scale+1):
            scaled_output_box_zyx = (full_output_box_zyx + 2**scale - 1) // (2**scale) # round up
    
            if scale == 0:
                scaled_instance_name = instance_name
            else:
                scaled_instance_name = f"{instance_name}_{scale}"
    
            if is_datainstance(server, uuid, scaled_instance_name):
                logger.info(f"'{scaled_instance_name}' already exists, skipping creation")
            else:
                create_rawarray8( server, uuid, scaled_instance_name, 3*(block_width,), compression )
    
            update_extents( server, uuid, scaled_instance_name, scaled_output_box_zyx )

        # Bottom level of pyramid is listed as neuroglancer-compatible
        extend_list_value(server, uuid, '.meta', 'neuroglancer', [instance_name])

        # We use node-local dvid servers when uploading to a gbucket backend,
        # and the gbucket backend needs to be explicitly reloaded
        # (TODO: Is this still true, or has it been fixed by now?)
        if server.startswith("http://127.0.0.1"):
            def reload_meta():
                reload_server_metadata(server)
            self.run_on_each_worker( reload_meta )


    def execute(self):
        self._init_services()
        self._validate_config()
        self._prepare_output()
        
        options = self.config_data["options"]
        input_bb_zyx = self.input_service.bounding_box_zyx

        num_scales = options["max-pyramid-scale"]
        for scale in range(num_scales+1):
            scaled_input_bb_zyx = np.zeros((2,3), dtype=int)
            scaled_input_bb_zyx[0] = input_bb_zyx[0] // 2**scale # round down
            
            # Proper downsampled bounding-box would round up here...
            #scaled_input_bb_zyx[1] = (input_bb_zyx[1] + 2**scale - 1) // 2**scale
            
            # ...but some some services probably don't do that, so we'll
            # round down to avoid out-of-bounds errors for higher scales. 
            scaled_input_bb_zyx[1] = input_bb_zyx[1] // 2**scale

            # Data is processed in Z-slabs
            # Auto-choose a depth that keeps all threads busy with at least one output brick
            output_brick_shape_zyx = self.output_service.preferred_message_shape
            output_brick_depth = output_brick_shape_zyx[0]
            assert output_brick_depth != -1
            
            slab_shape_zyx = scaled_input_bb_zyx[1] - scaled_input_bb_zyx[0]
            slab_shape_zyx[0] = options["slab-depth"]
    
            # This grid outlines the slabs -- each box in slab_grid is a full slab
            slab_grid = Grid(slab_shape_zyx, scaled_input_bb_zyx[0])
            slab_boxes = list(clipped_boxes_from_grid(scaled_input_bb_zyx, slab_grid))
    
            for slab_index, slab_box_zyx in enumerate(slab_boxes):
                self._convert_slab(scale, slab_box_zyx, slab_index, len(slab_boxes))
            logger.info(f"Done exporting {len(slab_boxes)} slabs for scale {scale}.", extra={'status': f"DONE with scale {scale}"})
        logger.info(f"DONE exporting {num_scales} scales")


    def _convert_slab(self, scale, slab_box_zyx, slab_index, num_slabs):
        # Contruct BrickWall from input bricks
        num_threads = num_worker_nodes() * cpus_per_worker()
        slab_voxels = np.prod(slab_box_zyx[1] - slab_box_zyx[0])
        voxels_per_thread = slab_voxels // num_threads

        bricked_slab_wall = BrickWall.from_volume_service(self.input_service, scale, slab_box_zyx, self.sc, voxels_per_thread // 2)

        # Force download
        bricked_slab_wall.persist_and_execute(f"Downloading slab {slab_index}/{num_slabs}: {slab_box_zyx[:,::-1]}", logger)
        
        # Remap to output bricks
        output_grid = Grid(self.output_service.preferred_message_shape, offset=slab_box_zyx[0])
        output_slab_wall = bricked_slab_wall.realign_to_new_grid( output_grid )
        
        padding_grid = Grid( 3*(self.output_service.block_width,), output_grid.offset )
        padded_slab_wall = output_slab_wall.fill_missing(lambda box: 0, padding_grid)
        padded_slab_wall.persist_and_execute(f"Assembling slab {slab_index}/{num_slabs} slices", logger)

        # Discard original bricks
        bricked_slab_wall.unpersist()
        del bricked_slab_wall

        with Timer() as timer:
            logger.info(f"Exporting slab {slab_index}/{num_slabs}", extra={"status": f"Exporting {slab_index}/{num_slabs}"})
            rt.foreach( partial(write_brick, self.output_service, scale), padded_slab_wall.bricks )

        logger.info(f"Exporting slab {slab_index}/{num_slabs} took {timer.timedelta}",
                    extra={"status": f"Done: {slab_index}/{num_slabs}"})
        
        # Discard output_bricks
        padded_slab_wall.unpersist()
        del padded_slab_wall


def write_brick(output_service, scale, brick):
    shape = np.array(brick.volume.shape)
    assert (shape[0:2] == output_service.block_width).all()
    assert shape[2] % output_service.block_width == 0
    
    # Omit leading/trailing empty blocks
    block_width = output_service.block_width
    assert (np.array(brick.volume.shape) % block_width).all() == 0
    blockwise_view = view_as_blocks( brick.volume, brick.volume.shape[0:2] + (block_width,) )
    
    # blockwise view has shape (1,1,X/bx, bz, by, bx)
    assert blockwise_view.shape[0:2] == (1,1)
    blockwise_view = blockwise_view[0,0] # drop singleton axes
    
    block_maxes = blockwise_view.max( axis=(1,2,3) )
    assert block_maxes.ndim == 1
    
    nonzero_block_indexes = np.nonzero(block_maxes)[0]
    if len(nonzero_block_indexes) == 0:
        return # brick is completely empty
    
    first_nonzero_block = nonzero_block_indexes[0]
    last_nonzero_block = nonzero_block_indexes[-1]
    
    nonzero_start = (0, 0, block_width*first_nonzero_block)
    nonzero_stop = ( brick.volume.shape[0:2] + (block_width*(last_nonzero_block+1),) )
    nonzero_subvol = brick.volume[box_to_slicing(nonzero_start, nonzero_stop)]

    output_service.write_subvolume(nonzero_subvol, brick.physical_box[0] + nonzero_start, scale)
