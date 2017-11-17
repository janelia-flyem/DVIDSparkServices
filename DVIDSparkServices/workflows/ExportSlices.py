import os
from os.path import isabs, dirname
import copy
import json
import logging

import numpy as np
import vigra

from dvid_resource_manager.client import ResourceManagerClient

from DVIDSparkServices import rddtools as rt
from DVIDSparkServices.io_util.brick import Grid, clipped_boxes_from_grid
from DVIDSparkServices.io_util.brickwall import BrickWall
from DVIDSparkServices.util import persist_and_execute, num_worker_nodes, cpus_per_worker
from DVIDSparkServices.json_util import flow_style
from DVIDSparkServices.workflow.workflow import Workflow

from .common_schemas import GrayscaleVolumeSchema, SliceFilesVolumeSchema

logger = logging.getLogger(__name__)

class ExportSlices(Workflow):
    OptionsSchema = copy.deepcopy(Workflow.OptionsSchema)
    OptionsSchema["additionalProperties"] = False
    OptionsSchema["properties"].update(
    {
        "slices-per-slab": {
            "description": "The volume is processed iteratively, in 'slabs' consisting of many contiguous Z-slices.\n"
                           "This setting determines the thickness of each slab. -1 means choose automatically from number of worker threads.\n"
                           "(Each worker thread processes a single Z-slice at a time.)",
            "type": "integer",
            "default": -1
        }
    })

    Schema = \
    {
        "$schema": "http://json-schema.org/schema#",
        "title": "Service to load raw and label data into DVID",
        "type": "object",
        "additionalProperties": False,
        "required": ["input", "output"],
        "properties": {
            "input": GrayscaleVolumeSchema,
            "output": copy.deepcopy(SliceFilesVolumeSchema),
            "options" : OptionsSchema
        }
    }
    
    # Adjust defaults for this workflow in particular
    Schema["properties"]["output"]\
            ["properties"]["geometry"]\
              ["properties"]["message-block-shape"]["default"] = flow_style([-1, -1, 1])

    Schema["properties"]["output"]\
            ["properties"]["geometry"]\
              ["properties"]["bounding-box"]["default"] = flow_style([[-1, -1, -1], [-1, -1, -1]])


    @classmethod
    def dumpschema(cls):
        return json.dumps(ExportSlices.Schema)

    @classmethod
    def schema(cls):
        return ExportSlices.Schema

    # name of application for DVID queries
    APPNAME = "ExportSlices".lower()

    def __init__(self, config_filename):
        super().__init__( config_filename, ExportSlices.dumpschema(), "Export Slices" )
        self._sanitize_config()

    def _sanitize_config(self):
        """
        Tidy up some config values, and fill in 'auto' values where needed.
        """
        options = self.config_data["options"]
        input_geometry = self.config_data["input"]["geometry"]
        output_specs = self.config_data["output"]["slice-files"]
        output_geometry = self.config_data["output"]["geometry"]
        
        assert output_specs["slice-xy-offset"] == [0,0], "Nonzero xy offset is meaningless for outputs."
        assert output_geometry["message-block-shape"] == [-1, -1, 1], "Output must be Z-slices, and complete XY planes"

        if options["slices-per-slab"] == -1:
            # Auto-choose a depth that keeps all threads busy with at least one slice
            brick_shape_zyx = input_geometry["message-block-shape"][::-1]
            brick_depth = brick_shape_zyx[0]
            num_threads = num_worker_nodes() * cpus_per_worker()
            threads_per_brick_layer = ((num_threads + brick_depth-1) // brick_depth) # round up
            options["slices-per-slab"] = brick_depth * threads_per_brick_layer

        # Convert relative path to absolute
        path = output_specs["slice-path-format"]
        if path.startswith('gs://'):
            assert False, "FIXME: Support gbuckets"
        elif not isabs(path):
            output_specs["slice-path-format"] = self.relpath_to_abspath(path)

        os.makedirs(dirname(output_specs["slice-path-format"]), exist_ok=True)

        # Enforce correct output bounding-box
        input_bb_zyx = np.array(input_geometry["bounding-box"])[:,::-1]
        assert -1 not in input_bb_zyx.flat[:], "Input bounding box must be completely specified."

        output_bb_zyx = np.array(output_geometry["bounding-box"])[:,::-1]
        ((output_bb_zyx == input_bb_zyx) | (output_bb_zyx == -1)).all(), \
            "Output bounding box must match the input bounding box exactly. (No translation permitted)."
        output_geometry["bounding-box"] = copy.deepcopy(input_geometry["bounding-box"])
        

    def execute(self):
        input_config = self.config_data["input"]
        input_geometry = input_config["geometry"]
        output_specs = self.config_data["output"]["slice-files"]
        options = self.config_data["options"]

        mgr_client = ResourceManagerClient( options["resource-server"],
                                            options["resource-port"] )

        # Data is processed in Z-slabs
        slab_depth = options["slices-per-slab"]

        input_bb_zyx = np.array(input_geometry["bounding-box"])[:,::-1]
        _, slice_start_y, slice_start_x = input_bb_zyx[0]

        slab_shape_zyx = input_bb_zyx[1] - input_bb_zyx[0]
        slab_shape_zyx[0] = slab_depth

        slice_shape_zyx = slab_shape_zyx.copy()
        slice_shape_zyx[0] = 1

        # This grid outlines the slabs -- each grid box is a full slab
        slab_grid = Grid(slab_shape_zyx, (0, slice_start_y, slice_start_x))
        slab_boxes = list(clipped_boxes_from_grid(input_bb_zyx, slab_grid))

        for slab_index, slab_box in enumerate(slab_boxes):
            # Contruct BrickWall from input bricks
            slab_config = copy.copy(input_config)
            slab_config["geometry"]["bounding-box"] = slab_box[:, ::-1]
            bricked_slab_wall = BrickWall.from_volume_config(slab_config, self.sc, resource_manager_client=mgr_client)
            
            # Force download
            persist_and_execute(bricked_slab_wall.bricks, f"Downloading slab {slab_index}/{len(slab_boxes)} bricks", logger)
            
            # Remap to slice-sized "bricks"
            sliced_grid = Grid(slice_shape_zyx, offset=slab_box[0])
            sliced_slab_wall = bricked_slab_wall.realign_to_new_grid( sliced_grid )
            persist_and_execute(sliced_slab_wall.bricks, f"Assembling slab {slab_index}/{len(slab_boxes)} slices", logger)

            # Discard original bricks
            bricked_slab_wall.bricks.unpersist()
            del bricked_slab_wall

            def export_slice(brick):
                assert (brick.physical_box == brick.logical_box).all()
                assert (brick.physical_box[1,0] - brick.physical_box[0,0]) == 1, "Expected a single slice"
                z_index = brick.physical_box[0,0]
                slice_data = vigra.taggedView( brick.volume, 'zyx' )
                vigra.impex.writeImage(slice_data[0], output_specs["slice-path-format"].format(z_index))

            # Export to PNG or TIFF, etc. (automatic via slice path extension)
            logger.info(f"Exporting slab {slab_index}/{len(slab_boxes)}", extra={"status": f"Exporting {slab_index}/{len(slab_boxes)}"})
            rt.foreach( export_slice, sliced_slab_wall.bricks )
            
            # Discard slice data
            sliced_slab_wall.bricks.unpersist()
            del sliced_slab_wall

        logger.info(f"DONE exporting {len(slab_boxes)} slabs.", extra={'status': "DONE"})



