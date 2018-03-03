import os
import time
import copy
import json
import logging
import socket
import sqlite3
from functools import partial

import numpy as np
import h5py

import pandas as pd

from dvid_resource_manager.client import ResourceManagerClient

from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.dvid.metadata import create_labelarray, is_datainstance
from DVIDSparkServices.util import Timer, runlength_encode, choose_pyramid_depth, nonconsecutive_bincount,\
    replace_default_entries, box_intersection, box_to_slicing

from DVIDSparkServices.io_util.brickwall import BrickWall, Grid
from DVIDSparkServices.io_util.volume_service import ( VolumeService, VolumeServiceWriter, SegmentationVolumeSchema,
                                                       SegmentationVolumeListSchema, TransposedVolumeService, ScaledVolumeService )
from DVIDSparkServices.io_util.volume_service.dvid_volume_service import DvidVolumeService
from DVIDSparkServices.io_util.brick import slabs_from_box, boxes_from_grid

logger = logging.getLogger(__name__)

SEGMENT_STATS_COLUMNS = ['segment', 'voxel_count', 'bounding_box_start', 'bounding_box_stop'] #, 'block_list']

class CopySegmentation(Workflow):
    """
    Workflow to copy segmentation from one source (e.g. a DVID segmentation
    instance or a BrainMaps volume) into a DVID segmentation instance.
    
    Notes:
    
    - The data is written to DVID in block-aligned 'bricks'.
      If the source data is not block-aligned at the edges,
      pre-existing data (if any) is read from the DVID destination
      to fill out ('pad') the bricks until they are completely block aligned.
    
    - The data is also downsampled into a multi-scale pyramid and uploaded.
    
    - The volume is processed in Z-slabs. To avoid complications during downsampling,
      the Z-slabs must be aligned to a multiple of the DVID block shape, which may be
      rather large, depending on the highest scale of the pyramid.
      (It is recommended that you don't set this explicitly in the config, so a
      suitable default can be chosen for you.)
      
    - This workflow uses DvidVolumeService to write the segmentation blocks,
      which is able to send them to DVID in the pre-encoded 'labelarray' block format.
      This saves CPU resources on the DVID server.
    
    - As a convenience, size of each label 'body' in the copied volume is also
      calculated and exported in an HDF5 file, sorted by body size.
    """
    
    BodySizesOptionsSchema = \
    {
        "type": "object",
        "additionalProperties": False,
        "default": {},
        "properties": {
            "compute": {
                "type": "boolean",
                "default": True
            },
            "minimum-size" : {
                "description": "Minimum size to include in the body size HDF5 output.\n"
                               "Smaller bodies are omitted.",
                "type": "integer",
                "default": 1000
            },
            "method" : {
                "description": "(For performance experiments)\n"
                               "Whether to perform the body size computation via reduceByKey.",
                "type": "string",
                "enum": ["reduce-by-key", "reduce-with-pandas"],
                "default": "reduce-with-pandas"
            }
        }
    }

    OptionsSchema = copy.deepcopy(Workflow.OptionsSchema)
    OptionsSchema["additionalProperties"] = False
    OptionsSchema["properties"].update(
    {
        "block-statistics-file": {
            "description": "Where to store block statistics for the INPUT segmentation\n"
                           "(but translated to output coordinates).\n"
                           "If the file already exists, it will be appended to (for restarting from a failed job).\n",
            "type": "string",
            "default": "block-statistics.sqlite"
        },

        "body-sizes": BodySizesOptionsSchema,
        
        "pyramid-depth": {
            "description": "Number of pyramid levels to generate \n"
                           "(-1 means choose automatically, 0 means no pyramid)",
            "type": "integer",
            "default": -1 # automatic by default
        },
        "permit-inconsistent-pyramid": {
            "description": "Normally overwriting a pre-existing data instance is\n"
                           "an error unless you rewrite ALL of its pyramid levels,\n"
                           "but this setting allows you to override that error.\n"
                           "(You had better know what you're doing...)\n",
            "type": "boolean",
            "default": False
        },
        "skip-scale-0-write": {
            "description": "Skip writing scale 0.  Useful if scale 0 is already downloaded and now\n"
                           "you just want to generate the rest of the pyramid to the same instance.\n",
            "type": "boolean",
            "default": False
        },        
        "slab-depth": {
            "description": "The data is downloaded and processed in Z-slabs.\n"
                           "This setting determines how thick each Z-slab is.\n"
                           "Should be a multiple of (block_width * 2**pyramid_depth) to ensure slabs\n"
                           "are completely independent, even after downsampling.\n",
            "type": "integer",
            "default": -1 # Choose automatically: block_width * 2**pyramid_depth
        },
        "delay-minutes-between-slabs": {
            "description": "Optionally introduce an artificial pause after finishing one slab before starting the next,\n"
                           "to give DVID time to index the blocks we've sent so far.",
            "type": "integer",
            "default": 0,
        },
        "enable-indexing": {
            "description": "Enable indexing on the new labelarray instance.\n"
                           "(Should normally be left as the default (true), except for benchmarking purposes.)",
            "type": "boolean",
            "default": True
        }
    })

    Schema = \
    {
        "$schema": "http://json-schema.org/schema#",
        "title": "Service to load raw and label data into DVID",
        "type": "object",
        "additionalProperties": False,
        "required": ["input", "outputs"],
        "properties": {
            "input": SegmentationVolumeSchema,
            "outputs": SegmentationVolumeListSchema,
            "options" : OptionsSchema
        }
    }

    @classmethod
    def schema(cls):
        return CopySegmentation.Schema

    # name of application for DVID queries
    APPNAME = "copysegmentation"


    def __init__(self, config_filename):
        super(CopySegmentation, self).__init__( config_filename,
                                                CopySegmentation.schema(),
                                                "Copy Segmentation" )


    def execute(self):
        self._init_services()
        self._create_output_instances_if_necessary()
        self._log_neuroglancer_links()
        self._sanitize_config()

        # Aim for 2 GB RDD partitions when loading segmentation
        GB = 2**30
        self.target_partition_size_voxels = 2 * GB // np.uint64().nbytes

        # (See note in _init_services() regarding output bounding boxes)
        input_bb_zyx = self.input_service.bounding_box_zyx
        output_bb_zyx = self.output_services[0].bounding_box_zyx
        self.translation_offset_zyx = output_bb_zyx[0] - input_bb_zyx[0]

        pyramid_depth = self.config_data["options"]["pyramid-depth"]
        slab_depth = self.config_data["options"]["slab-depth"]

        # Process data in Z-slabs
        output_slab_boxes = list( slabs_from_box(output_bb_zyx, slab_depth) )
        max_depth = max(map(lambda box: box[1][0] - box[0][0], output_slab_boxes))
        logger.info(f"Processing data in {len(output_slab_boxes)} slabs (max depth={max_depth}) for {pyramid_depth} pyramid levels")

        # Initialize the cumulative stats (from pre-existing file, if present).
        cumulative_block_stats_df = self._init_stats_file()

        try:
            # Read data and accumulate statistics, one slab at a time.
            for slab_index, output_slab_box in enumerate( output_slab_boxes ):
                with Timer() as timer:
                    slab_stats_df = self._process_slab(slab_index, output_slab_box)
                logger.info(f"Slab {slab_index}: Done copying to {len(self.config_data['outputs'])} destinations.")
                logger.info(f"Slab {slab_index}: Total processing time: {timer.timedelta}")
    
                with Timer(f"Slab {slab_index}: Appending stats and overwriting stats file"):
                    is_last_slab = (slab_index == len(output_slab_boxes)-1)
                    cumulative_block_stats_df = self._append_slab_statistics( cumulative_block_stats_df,
                                                                              slab_stats_df,
                                                                              drop_duplicates=is_last_slab )
    
                delay_minutes = self.config_data["options"]["delay-minutes-between-slabs"]
                if delay_minutes > 0 and slab_index != len(output_slab_boxes)-1:
                    logger.info(f"Delaying {delay_minutes} before continuing to next slab...")
                    time.sleep(delay_minutes * 60)
        except:
            
            raise

        logger.info(f"DONE copying/downsampling all slabs to {len(self.config_data['outputs'])} destinations.")

    def _init_services(self):
        """
        Initialize the input and output services,
        and fill in 'auto' config values as needed.
        
        Also check the service configurations for errors.
        """
        input_config = self.config_data["input"]
        output_configs = self.config_data["outputs"]
        options = self.config_data["options"]

        self.mgr_client = ResourceManagerClient( options["resource-server"], options["resource-port"] )
        self.input_service = VolumeService.create_from_config( input_config, self.config_dir, self.mgr_client )

        self.output_services = []
        for i, output_config in enumerate(output_configs):
            # Replace 'auto' dimensions with input bounding box
            replace_default_entries(output_config["geometry"]["bounding-box"], self.input_service.bounding_box_zyx[:, ::-1])
            output_service = VolumeService.create_from_config( output_config, self.config_dir, self.mgr_client )
            assert isinstance( output_service, VolumeServiceWriter )

            # These services aren't supported because we copied some geometry (bounding-box)
            # directly from the input service.
            assert not isinstance( output_service, TransposedVolumeService )
            assert not isinstance( output_service, ScaledVolumeService )

            assert output_service.base_service.disable_indexing, \
                "During ingestion, indexing should be disabled.\n" \
                "Please add 'disable-indexing':true to your output dvid config."

            logger.info(f"Output {i} bounding box (xyz) is: {output_service.bounding_box_zyx[:,::-1].tolist()}")
            self.output_services.append( output_service )

        first_output_service = self.output_services[0]
        
        input_shape = -np.subtract(*self.input_service.bounding_box_zyx)
        output_shape = -np.subtract(*first_output_service.bounding_box_zyx)
        
        assert not any(np.array(first_output_service.preferred_message_shape) % first_output_service.block_width), \
            "Output message-block-shape should be a multiple of the block size in all dimensions."
        assert (input_shape == output_shape).all(), \
            "Input bounding box and output bounding box do not have the same dimensions"

        # NOTE: For now, we require that all outputs use the same bounding box and grid scheme,
        #       to simplify the execute() function.
        #       (We avoid re-translating and re-downsampling the input data for every new output.)
        #       The only way in which the outputs may differ is their label mapping data.
        for output_service in self.output_services[1:]:
            bb_zyx = output_service.bounding_box_zyx
            bs = output_service.preferred_message_shape
            bw = output_service.block_width

            assert (first_output_service.bounding_box_zyx == bb_zyx).all(), \
                "For now, all output destinations must use the same bounding box and grid scheme"
            assert (first_output_service.preferred_message_shape == bs).all(), \
                "For now, all output destinations must use the same bounding box and grid scheme"
            assert (first_output_service.block_width == bw), \
                "For now, all output destinations must use the same bounding box and grid scheme"

        for output_config in output_configs:
            if ("apply-labelmap" in output_config) and (output_config["apply-labelmap"]["file-type"] != "__invalid__"):
                assert output_config["apply-labelmap"]["apply-when"] == "reading-and-writing", \
                    "Labelmap will be applied to voxels during pre-write and post-read (due to block padding).\n"\
                    "You cannot use this workflow with non-idempotent labelmaps, unless your data is already perfectly block aligned."


    def _create_output_instances_if_necessary(self):
        """
        If it doesn't exist yet, create it first with the user's specified
        pyramid-depth, or with an automatically chosen depth.
        """
        pyramid_depth = self.config_data["options"]["pyramid-depth"]
        permit_inconsistent_pyramids = self.config_data["options"]["permit-inconsistent-pyramid"]

        for output_service in self.output_services:
            base_service = output_service.base_service
            assert isinstance( base_service, DvidVolumeService )

            # Create new segmentation instance first if necessary
            if is_datainstance( base_service.server,
                                base_service.uuid,
                                base_service.instance_name ):

                existing_depth = self._read_pyramid_depth()
                if pyramid_depth not in (-1, existing_depth) and not permit_inconsistent_pyramids:
                    raise Exception(f"Can't set pyramid-depth to {pyramid_depth}: "
                                    f"Data instance '{base_service.instance_name}' already existed, with depth {existing_depth}")
            else:
                if pyramid_depth == -1:
                    # if no pyramid depth is specified, determine the max, based on bb size.
                    input_bb_zyx = self.input_service.bounding_box_zyx
                    pyramid_depth = choose_pyramid_depth(input_bb_zyx, 512)
        
                # create new label array with correct number of pyramid scales
                create_labelarray( base_service.server,
                                   base_service.uuid,
                                   base_service.instance_name,
                                   pyramid_depth,
                                   3*(base_service.block_width,),
                                   enable_index=self.config_data["options"]["enable-indexing"] )


    def _read_pyramid_depth(self):
        """
        Read the MaxDownresLevel from each output instance we'll be writing to,
        and verify that it matches our config for pyramid-depth.
        
        Return the max depth we found in the outputs.
        (They should all be the same...)
        """
        max_depth = -1
        for output_service in self.output_services:
            base_volume_service = output_service.base_service
            info = base_volume_service.node_service.get_typeinfo(base_volume_service.instance_name)
            existing_depth = int(info["Extended"]["MaxDownresLevel"])
            max_depth = max(max_depth, existing_depth)

        return max_depth


    def _log_neuroglancer_links(self):
        """
        Write a link to the log file for viewing the segmentation data after it is ingested.
        We assume that the output server is hosting neuroglancer at http://<server>:<port>/neuroglancer/
        """
        for index, output_service in enumerate(self.output_services):
            server = output_service.base_service.server # Note: Begins with http://
            uuid = output_service.base_service.uuid
            instance = output_service.base_service.instance_name
            
            output_box_xyz = np.array(output_service.bounding_box_zyx[:, :-1])
            output_center_xyz = (output_box_xyz[0] + output_box_xyz[1]) / 2
            
            link_prefix = f"{server}/neuroglancer/#!"
            link_json = \
            {
                "layers": {
                    "segmentation": {
                        "type": "segmentation",
                        "source": f"dvid://{server}/{uuid}/{instance}"
                    }
                },
                "navigation": {
                    "pose": {
                        "position": {
                            "voxelSize": [8,8,8],
                            "voxelCoordinates": output_center_xyz.tolist()
                        }
                    },
                    "zoomFactor": 8
                }
            }
            logger.info(f"Neuroglancer link to output {index}: {link_prefix}{json.dumps(link_json)}")


    def _sanitize_config(self):
        """
        Replace a few config values with reasonable defaults if necessary.
        (Note: Must be called AFTER services and output instances have been initialized.)
        """
        options = self.config_data["options"]

        # Overwrite pyramid depth in our config (in case the user specified -1, i.e. automatic)
        if options["pyramid-depth"] == -1:
            options["pyramid-depth"] = self._read_pyramid_depth()
        pyramid_depth = options["pyramid-depth"]

        block_width = self.output_services[0].block_width

        slab_depth = options["slab-depth"]
        if slab_depth == -1:
            slab_depth = block_width * 2**pyramid_depth
        elif slab_depth % ( block_width * 2**pyramid_depth ) != 0:
            raise RuntimeError(f"Slab depth ({slab_depth}) is not aligned properly.\n"
                               "Downsampled data blocks would span multiple slabs.")
        options["slab-depth"] = slab_depth

    def _init_stats_file(self):
        stats_path = self.relpath_to_abspath(self.config_data["options"]["block-statistics-file"])
        assert os.path.splitext(stats_path)[1] == '.sqlite'

        if os.path.exists(stats_path):
            with sqlite3.connect(stats_path) as conn:
                logger.info(f"Appending to pre-existing block statistics: {stats_path}")
        else:
            with sqlite3.connect(stats_path) as conn:
                c = conn.cursor()
                c.execute('CREATE TABLE block_stats (segment_id INTEGER, z INTEGER, y INTEGER, x INTEGER, count INTEGER)')

        with sqlite3.connect(stats_path) as conn:
            cumulative_block_stats_df = pd.read_sql('SELECT * from block_stats', conn)
        assert list(cumulative_block_stats_df.columns) == BLOCK_STATS_COLUMNS
        
        # Ensure proper (minimal width) dtypes
        # Sadly, pd.read_sql() will return everything as int64, which may induce significant RAM overhead.
        # We may need to revisit this, and read rows or columns in batches.
        cumulative_block_stats_df['segment_id'] = cumulative_block_stats_df['segment_id'].astype(np.uint64, copy=False)
        cumulative_block_stats_df['z'] = cumulative_block_stats_df['z'].astype(np.int32, copy=False)
        cumulative_block_stats_df['y'] = cumulative_block_stats_df['y'].astype(np.int32, copy=False)
        cumulative_block_stats_df['x'] = cumulative_block_stats_df['x'].astype(np.int32, copy=False)
        cumulative_block_stats_df['count'] = cumulative_block_stats_df['count'].astype(np.uint32, copy=False)

        conn.close()
        
        return cumulative_block_stats_df

    def _append_slab_statistics(self, cumulative_block_stats_df, slab_stats_df, drop_duplicates):
        """
        Append the rows of the given slab statistics DataFrame to the given cumulative
        statistics DataFrame, optionally dropping duplicate rows afterwards.
        Duplicates are defined by the segment and coordinates (not count).
        If duplicates are found, the all but the last are discarded, and a warning is logged.
        
        Args:
            cumulative_block_stats_df: DataFrame with rows ['segment_id', 'z', 'y', 'x']
            slab_stats_df: Similar DataFrame, to be appended
            write: If True, write to disk after concatenation
            drop_duplicates: If True, check for duplicate rows as described above.
        
        Returns:
            DataFrame after concatenation
        """
        stats_path = self.relpath_to_abspath(self.config_data["options"]["block-statistics-file"])

        # Append
        with sqlite3.connect(stats_path) as conn:
            slab_stats_df.to_sql('block_stats', conn, if_exists='append', index=False)

        cumulative_block_stats_df = pd.concat( (cumulative_block_stats_df, slab_stats_df), ignore_index=True )
        
        if drop_duplicates:
            # If we started from a pre-existing statistics file,
            # it's possible that the processed bounding box overlaps with the bounding box from previous runs.
            # Drop them.
            duplicate_rows = cumulative_block_stats_df.duplicated(['segment_id', 'z', 'y', 'x'], keep='last')
            if duplicate_rows.any():
                logger.warning(f"Duplicate rows found, implying bounding box overlaps previous runs! Deleting duplicates...")
                cumulative_block_stats_df = cumulative_block_stats_df[~duplicate_rows]

                # Overwrite
                with sqlite3.connect(stats_path) as conn:
                    cumulative_block_stats_df.to_sql('block_stats', conn, if_exists='replace')
            
        return cumulative_block_stats_df

    def _process_slab(self, slab_index, output_slab_box ):
        """
        (The main work of this file.)
        
        Process a large slab of voxels:
        
        1. Read a 'slab' of bricks from the input as a BrickWall
        2. Translate it to the output coordinates.
        3. Splice & group the bricks so that they are aligned to the optimal output grid
        4. 'Pad' the bricks on the edges of the wall by *reading* data from the output destination,
            so that all bricks are complete (i.e. they completely fill their grid block).
        5. Write all bricks to the output destination.
        6. Downsample the bricks and repeat steps 3-5 for the downsampled scale.
        """
        options = self.config_data["options"]
        pyramid_depth = options["pyramid-depth"]

        input_slab_box = output_slab_box - self.translation_offset_zyx
        input_wall = BrickWall.from_volume_service(self.input_service, 0, input_slab_box, self.sc, self.target_partition_size_voxels)
        input_wall.persist_and_execute(f"Slab {slab_index}: Reading ({input_slab_box[:,::-1].tolist()})", logger)

        # Translate coordinates from input to output
        # (which will leave the bricks in a new, offset grid)
        # This has no effect on the brick volumes themselves.
        if any(self.translation_offset_zyx):
            translated_wall = input_wall.translate(self.translation_offset_zyx)
        else:
            translated_wall = input_wall # no translation needed
        del input_wall

        # For now, all output_configs are required to have identical grid alignment settings
        # Therefore, we can save time in the loop below by aligning the input to the output grid in advance.
        aligned_input_wall = self._consolidate_and_pad(slab_index, translated_wall, 0, self.output_services[0], align=True, pad=False)
        del translated_wall

        with Timer(f"Slab {slab_index}: Computing slab block statistics", logger):
            block_shape = 3*[self.output_services[0].base_service.block_width]
            input_slab_block_stats_per_brick = aligned_input_wall.bricks.map( partial(block_stats_from_brick, block_shape) ).collect()
            input_slab_block_stats_df = pd.concat(input_slab_block_stats_per_brick, ignore_index=True)
            del input_slab_block_stats_per_brick

        for output_index, output_service in enumerate(self.output_services):
            if output_index < len(self.output_services) - 1:
                # Copy to a new RDD so the input can be re-used for subsequent outputs
                aligned_output_wall = aligned_input_wall.copy()
            else:
                # No copy needed for the last one
                aligned_output_wall = aligned_input_wall

            # Pad internally to block-align.
            # Here, we assume that any output labelmaps are idempotent,
            # so it's okay to read pre-existing output data that will ultimately get remapped.
            padded_wall = self._consolidate_and_pad(slab_index, aligned_output_wall, 0, output_service, align=False, pad=True)
            del aligned_output_wall
    
            # Compute body sizes and write to HDF5
            # FIXME: Needs to be fixed now that we copy data slab-by-slab
            #self._write_body_sizes( padded_wall, output_service )
    
            # Write scale 0 to DVID
            if not options["skip-scale-0-write"]:
                self._write_bricks( slab_index, padded_wall, 0, output_service )
    
            for new_scale in range(1, 1+pyramid_depth):
                # Compute downsampled (results in smaller bricks)
                downsampled_wall = padded_wall.label_downsample( (2,2,2) )
                downsampled_wall.persist_and_execute(f"Slab {slab_index}: Scale {new_scale}: Downsampling", logger)
                padded_wall.unpersist()
                del padded_wall

                # Consolidate to full-size bricks and pad internally to block-align
                consolidated_wall = self._consolidate_and_pad(slab_index, downsampled_wall, new_scale, output_service, align=True, pad=True)
                del downsampled_wall

                # Write to DVID
                self._write_bricks( slab_index, consolidated_wall, new_scale, output_service )

                padded_wall = consolidated_wall
                del consolidated_wall
            del padded_wall
        
        return input_slab_block_stats_df

    def _consolidate_and_pad(self, slab_index, input_wall, scale, output_service, align=True, pad=True):
        """
        Consolidate (align), and pad the given BrickWall

        Note: UNPERSISTS the input data and returns the new, downsampled data.

        Args:
            scale: The pyramid scale of the data.
            
            output_service: The output_service to align to and pad from
            
            align: If False, skip the alignment step.
                  (Only use this if the bricks are already aligned.)
            
            pad: If False, skip the padding step
        
        Returns a pre-executed and persisted BrickWall.
        """
        output_writing_grid = Grid(output_service.preferred_message_shape)

        if not align or output_writing_grid.equivalent_to(input_wall.grid):
            realigned_wall = input_wall
            realigned_wall.persist_and_execute(f"Slab {slab_index}: Scale {scale}: Persisting pre-aligned bricks", logger)
        else:
            # Consolidate bricks to full-size, aligned blocks (shuffles data)
            realigned_wall = input_wall.realign_to_new_grid( output_writing_grid )
            realigned_wall.persist_and_execute(f"Slab {slab_index}: Scale {scale}: Shuffling bricks into alignment", logger)

            # Discard original
            input_wall.unpersist()
        
        if not pad:
            return realigned_wall

        # Pad from previously-existing pyramid data until
        # we have full storage blocks, e.g. (64,64,64),
        # but not necessarily full bricks, e.g. (64,64,6400)
        storage_block_width = output_service.block_width
        output_padding_grid = Grid( (storage_block_width, storage_block_width, storage_block_width), output_writing_grid.offset )
        output_accessor_func = partial(output_service.get_subvolume, scale=scale)
        
        padded_wall = realigned_wall.fill_missing(output_accessor_func, output_padding_grid)
        padded_wall.persist_and_execute(f"Slab {slab_index}: Scale {scale}: Padding", logger)

        # Discard old
        realigned_wall.unpersist()

        return padded_wall


    def _write_bricks(self, slab_index, brick_wall, scale, output_service):
        """
        Writes partition to specified dvid.
        """
        instance_name = output_service.base_service.instance_name
        block_width = output_service.block_width
        EMPTY_VOXEL = 0
        
        @self.collect_log(lambda _: socket.gethostname() + '-write-blocks-' + str(scale))
        def write_brick(brick):
            logger = logging.getLogger(__name__)
            
            assert (brick.physical_box % block_width == 0).all(), \
                f"This function assumes each brick's physical data is already block-aligned: {brick}"
            
            x_size = brick.volume.shape[2]
            # Find all non-zero blocks (and record by block index)
            block_coords = []
            for block_index, block_x in enumerate(range(0, x_size, block_width)):
                if not (brick.volume[:, :, block_x:block_x+block_width] == EMPTY_VOXEL).all():
                    block_coords.append( (0, 0, block_index) ) # (Don't care about Z,Y indexes, just X-index)

            # Find *runs* of non-zero blocks
            block_runs = runlength_encode(block_coords, True) # returns [[Z,Y,X1,X2], [Z,Y,X1,X2], ...]
            
            # Convert stop indexes from inclusive to exclusive
            block_runs[:,-1] += 1
            
            # Discard Z,Y indexes and convert from indexes to pixels
            ranges = block_width * block_runs[:, 2:4]
            
            # iterate through contiguous blocks and write to DVID
            for (data_x_start, data_x_end) in ranges:
                datacrop = brick.volume[:,:,data_x_start:data_x_end].copy()
                data_offset_zyx = brick.physical_box[0] + (0,0,data_x_start)

                with Timer() as _put_timer:
                    output_service.write_subvolume(datacrop, data_offset_zyx, scale)

                # Note: This timing data doesn't reflect ideal throughput, since throttle
                #       and/or the resource manager muddy the numbers a bit...
                #megavoxels_per_second = datacrop.size / 1e6 / put_timer.seconds
                #logger.info(f"Put block {data_offset_zyx} in {put_timer.seconds:.3f} seconds ({megavoxels_per_second:.1f} Megavoxels/second)")

        logger.info(f"Slab {slab_index}: Scale {scale}: Writing bricks to {instance_name}...")
        with Timer() as timer:
            brick_wall.bricks.foreach(write_brick)
        logger.info(f"Slab {slab_index}: Scale {scale}: Writing bricks to {instance_name} took {timer.timedelta}")

    
    def _write_body_sizes( self, brick_wall, output_service ):
        """
        Calculate the size (in voxels) of all label bodies in the volume,
        and write the results to an HDF5 file.
        
        NOTE: For now, we implement two alternative methods of computing this result,
              for the sake of performance comparisons between the two methods.
              The method used is determined by the ['body-sizes]['method'] option.
        """
        if not self.config_data["options"]["body-sizes"]["compute"]:
            logger.info("Skipping body size calculation.")
            return

        logger.info("Computing body sizes...")
        with Timer() as timer:
            body_labels, body_sizes = self._compute_body_sizes(brick_wall)
        logger.info(f"Computing {len(body_labels)} body sizes took {timer.seconds} seconds")

        min_size = self.config_data["options"]["body-sizes"]["minimum-size"]

        nonzero_start = 0
        if body_labels[0] == 0:
            nonzero_start = 1
        nonzero_count = body_sizes[nonzero_start:].sum()
        logger.info(f"Final volume contains {nonzero_count} nonzero voxels")

        if min_size > 1:
            logger.info(f"Omitting body sizes below {min_size} voxels...")
            valid_rows = body_sizes >= min_size
            body_labels = body_labels[valid_rows]
            body_sizes = body_sizes[valid_rows]
        assert body_labels.shape == body_sizes.shape

        with Timer() as timer:
            logger.info(f"Sorting {len(body_labels)} bodies by size...")
            sort_indices = np.argsort(body_sizes)[::-1]
            body_sizes = body_sizes[sort_indices]
            body_labels = body_labels[sort_indices]
        logger.info(f"Sorting {len(body_labels)} bodies by size took {timer.seconds} seconds")

        suffix = output_service.base_service.instance_name
        output_path = self.relpath_to_abspath(f"body-sizes-{suffix}.h5")
        with Timer() as timer:
            logger.info(f"Writing {len(body_labels)} body sizes to {output_path}")
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('labels', data=body_labels, chunks=True)
                f.create_dataset('sizes', data=body_sizes, chunks=True)
                f['total_nonzero_voxels'] = nonzero_count
        logger.info(f"Writing {len(body_sizes)} body sizes took {timer.seconds} seconds")


    def _compute_body_sizes(self, brick_wall):
        if self.config_data["options"]["body-sizes"]["method"] == "reduce-by-key":
            body_labels, body_sizes = self._compute_body_sizes_via_reduce_by_key(brick_wall)
        else:
            body_labels, body_sizes = self._compute_body_sizes_via_pandas_dataframes(brick_wall)
        return body_labels, body_sizes
        

    def _compute_body_sizes_via_reduce_by_key(self, brick_wall):
        # Reduce using reduceByKey and simply concatenating the results
        from operator import add
        def transpose(labels_and_sizes):
            labels, sizes = labels_and_sizes
            return np.array( (labels, sizes) ).transpose()

        def reduce_to_array( pair_list_1, pair_list_2 ):
            return np.concatenate( (pair_list_1, pair_list_2) )

        labels_and_sizes = ( brick_wall.bricks.map( lambda br: br.volume )
                                              .map( nonconsecutive_bincount )
                                              .flatMap( transpose )
                                              .reduceByKey( add )
                                              .map( lambda pair: [pair] )
                                              .treeReduce( reduce_to_array, depth=4 ) )

        body_labels, body_sizes = np.transpose( labels_and_sizes )
        return body_labels, body_sizes
        

    def _compute_body_sizes_via_pandas_dataframes(self, brick_wall):
        import pandas as pd

        #@self.collect_log(lambda *_args: 'merge_label_counts')
        def merge_label_counts( labels_and_counts_A, labels_and_counts_B ):
            labels_A, counts_A = labels_and_counts_A
            labels_B, counts_B = labels_and_counts_B
            
            # Fast path
            if len(labels_A) == 0:
                return (labels_B, counts_B)
            if len(labels_B) == 0:
                return (labels_A, counts_A)
            
            series_A = pd.Series(index=labels_A, data=counts_A)
            series_B = pd.Series(index=labels_B, data=counts_B)
            combined = series_A.add(series_B, fill_value=0)
            
            #logger = logging.getLogger(__name__)
            #logger.info(f"Merging label count lists of sizes {len(labels_A)} + {len(labels_B)}"
            #            f" = {len(combined)} took {timer.seconds} seconds")

            return (combined.index, combined.values.astype(np.uint64))

        # Two-stage repartition/reduce, to avoid doing ALL the work on the driver.
        body_labels, body_sizes = ( brick_wall.bricks.map( lambda br: br.volume )
                                                     .map( nonconsecutive_bincount )
                                                     .treeReduce( merge_label_counts, depth=4 ) )
            
        return body_labels, body_sizes

BLOCK_STATS_COLUMNS = ['segment_id', 'z', 'y', 'x', 'count']
def block_stats_from_brick(block_shape, brick):
    """
    Get the count of voxels for each segment (excluding segment 0)
    in each block within the given brick, returned as a DataFrame.
    
    Returns a DataFrame with the following columns:
        ['segment_id', 'z', 'y', 'x', 'count']
        where z,y,z are the starting coordinates of each block.
    """
    block_grid = Grid(block_shape)
    
    block_dfs = []
    block_boxes = boxes_from_grid(brick.physical_box, block_grid)
    for box in block_boxes:
        clipped_box = box_intersection(box, brick.physical_box) - brick.physical_box[0]
        block_vol = brick.volume[box_to_slicing(*clipped_box)]
        counts = pd.Series(block_vol.reshape(-1)).value_counts(sort=False)
        segment_ids = counts.index.values

        block_df = pd.DataFrame( { 'segment_id': segment_ids, 'count': counts } )

        # Exclude segment 0 from output        
        block_df = block_df[block_df['segment_id'] != 0]

        block_df['z'] = 0
        block_df['y'] = 0
        block_df['x'] = 0
        block_df[['z', 'y', 'x']] = box[0].astype(np.int32)
        
        block_dfs.append(block_df)

    brick_df = pd.concat(block_dfs, ignore_index=True)
    brick_df = brick_df[['segment_id', 'z', 'y', 'x', 'count']]
    assert list(brick_df.columns) == BLOCK_STATS_COLUMNS
    return brick_df
