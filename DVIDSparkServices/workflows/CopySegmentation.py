import os
import copy
import json
import logging
import subprocess
import socket
from functools import partial

import numpy as np
import h5py

# Don't import pandas here; import it locally as needed
#import pandas as pd

from dvid_resource_manager.client import ResourceManagerClient

from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.dvid.metadata import create_labelarray, is_datainstance
from DVIDSparkServices.util import Timer, runlength_encode, choose_pyramid_depth, nonconsecutive_bincount,\
    replace_default_entries

from DVIDSparkServices.io_util.brickwall import BrickWall, Grid
from DVIDSparkServices.io_util.volume_service import VolumeService, SegmentationVolumeSchema, SegmentationVolumeListSchema

logger = logging.getLogger(__name__)

class CopySegmentation(Workflow):
    
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
        "body-sizes": BodySizesOptionsSchema,
        
        "pyramid-depth": {
            "description": "Number of pyramid levels to generate \n"
                           "(-1 means choose automatically, 0 means no pyramid)",
            "type": "integer",
            "default": -1 # automatic by default
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
            "input": SegmentationVolumeSchema,       # Labelmap, if any, is applied post-read
            
            "outputs": SegmentationVolumeListSchema, # LIST of output locations to write to.
                                                     # Labelmap, if any, is applied pre-write for each volume.
            "options" : OptionsSchema
        }
    }
    
    Schema["properties"]["input"]["description"] += "\n(Labelmap, if any, is applied post-read)"
    Schema["properties"]["outputs"]["description"] += "\n(Labelmaps, if any, are applied before writing for each volume.)"

    @staticmethod
    def dumpschema():
        return json.dumps(CopySegmentation.Schema)

    @classmethod
    def schema(cls):
        return CopySegmentation.Schema

    # name of application for DVID queries
    APPNAME = "copysegmentation"


    def __init__(self, config_filename):
        super(CopySegmentation, self).__init__( config_filename,
                                                CopySegmentation.dumpschema(),
                                                "Copy Segmentation" )

    def _sanitize_config(self):
        """
        Tidy up some config values.
        """
        input_config = self.config_data["input"]
        output_configs = self.config_data["outputs"]

        # Verify that the input config can be loaded,
        # and overwrite 'auto' values with real parameters.
        VolumeService.create_from_config(input_config, self.config_dir)
        
        for volume_config in [input_config] + output_configs:
            # Convert labelmap (if any) to absolute path (relative to config file)
            labelmap_file = volume_config["apply-labelmap"]["file"]
            if labelmap_file.startswith('gs://'):
                # Verify that gsutil is able to see the file before we start doing real work.
                subprocess.check_output(f'gsutil ls {labelmap_file}', shell=True)
            elif labelmap_file and not os.path.isabs(labelmap_file):
                volume_config["apply-labelmap"]["file"] = self.relpath_to_abspath(labelmap_file)

        ##
        ## Check input/output dimensions and grid schemes.
        ##
        input_bb_zyx = np.array(input_config["geometry"]["bounding-box"])[:,::-1]
        logger.info(f"Input bounding box (xyz) is: {input_config['geometry']['bounding-box']}")

        first_output_geometry = output_configs[0]["geometry"]
        output_bb_zyx = np.array(first_output_geometry["bounding-box"])[:,::-1]
        output_brick_shape = np.array(first_output_geometry["message-block-shape"])
        output_block_width = np.array(first_output_geometry["block-width"])

        # Replace 'auto' dimensions with input bounding box
        replace_default_entries(output_bb_zyx, input_bb_zyx)

        assert not any(np.array(output_brick_shape) % first_output_geometry["block-width"]), \
            "Output message-block-shape should be a multiple of the block size in all dimensions."
        assert ((input_bb_zyx[1] - input_bb_zyx[0]) == (output_bb_zyx[1] - output_bb_zyx[0])).all(), \
            "Input bounding box and output bounding box do not have the same dimensions"

        # NOTE: For now, we require that all outputs use the same bounding box and grid scheme,
        #       to simplify the execute() function.
        #       (We avoid re-translating and re-downsampling the input data for every new output.)
        #       The only way in which the outputs may differ is their label mapping data.
        for i, output_config in enumerate(output_configs):
            output_geometry = output_config["geometry"]

            bb_zyx = np.array(output_geometry["bounding-box"])[:,::-1]
            bs = output_geometry["message-block-shape"]
            bw = output_geometry["block-width"]

            # Replace 'auto' dimensions with input bounding box
            replace_default_entries(bb_zyx, input_bb_zyx)
            output_geometry["bounding-box"] = bb_zyx[:,::-1].tolist()

            assert (output_bb_zyx == bb_zyx).all(), \
                "For now, all output destinations must use the same bounding box and grid scheme"
            assert (output_brick_shape == bs).all(), \
                "For now, all output destinations must use the same bounding box and grid scheme"
            assert (output_block_width == bw).all(), \
                "For now, all output destinations must use the same bounding box and grid scheme"

            # Create a throw-away writer service, to verify that all
            # output configs are well-formed before we start the workflow.
            VolumeService.create_from_config(output_config, self.config_dir)
            logger.info(f"Output {i} bounding box (xyz) is: {output_geometry['bounding-box']}")


    def execute(self):
        self._sanitize_config()

        options = self.config_data["options"]
        input_config = self.config_data["input"]
        output_configs = self.config_data["outputs"]
        
        self.resource_mgr_client = ResourceManagerClient(options["resource-server"], options["resource-port"])

        # Aim for 2 GB RDD partitions
        GB = 2**30
        target_partition_size_voxels = 2 * GB // np.uint64().nbytes
        input_service = VolumeService.create_from_config(input_config, self.config_dir, self.resource_mgr_client)
        input_wall = BrickWall.from_volume_service(input_service, self.sc, target_partition_size_voxels)

        # See note in _sanitize_config()
        first_output_geometry = output_configs[0]["geometry"]

        input_bb_zyx = np.array(input_config["geometry"]["bounding-box"])[:,::-1]
        output_bb_zyx = np.array(first_output_geometry["bounding-box"])[:,::-1]
        translation_offset_zyx = output_bb_zyx[0] - input_bb_zyx[0]

        self._create_output_instances_if_necessary()

        # Overwrite pyramid depth in our config (in case the user specified -1, i.e. automatic)
        options["pyramid-depth"] = self._read_pyramid_depth()

        self._log_neuroglancer_links()

        input_wall.persist_and_execute(f"Reading entire volume", logger)

        # Apply post-input label map (if any)
        print(f"labelmap config: type: {type(input_config['apply-labelmap'])} / {repr(input_config['apply-labelmap'])}")
        remapped_input_wall = input_wall.apply_labelmap( input_config["apply-labelmap"],
                                                         self.config_dir,
                                                         unpersist_original=True )
        del input_wall

        # Translate coordinates from input to output
        # (which will leave the bricks in a new, offset grid)
        # This has no effect on the brick volumes themselves.
        translated_wall = remapped_input_wall.translate(translation_offset_zyx)
        del remapped_input_wall

        # For now, all output_configs are required to have identical grid alignment settings
        # Therefore, we can save time in the loop below by aligning the input to the output grid in advance.
        aligned_input_wall = self._consolidate_and_pad(translated_wall, 0, output_configs[0], align=True, pad=False)
        del translated_wall

        for output_config in self.config_data["outputs"]:
            # Apply pre-output label map (if any)
            # Don't delete input, because we need to reuse it for each iteration of this loop
            remapped_output_wall = aligned_input_wall.apply_labelmap( output_config["apply-labelmap"],
                                                                      self.config_dir,
                                                                      unpersist_original=False )

            # Pad internally to block-align.
            padded_wall = self._consolidate_and_pad(remapped_output_wall, 0, output_config, align=False, pad=True)
            del remapped_output_wall
    
            # Compute body sizes and write to HDF5
            self._write_body_sizes( padded_wall, output_config )
    
            # Write scale 0 to DVID
            self._write_bricks( padded_wall, 0, output_config )
    
            for new_scale in range(1, 1+options["pyramid-depth"]):
                # Compute downsampled (results in smaller bricks)
                downsampled_wall = padded_wall.label_downsample( (2,2,2) )
                downsampled_wall.persist_and_execute(f"Scale {new_scale}: Downsampling", logger)
                padded_wall.unpersist()
                del padded_wall

                # Consolidate to full-size bricks and pad internally to block-align
                consolidated_wall = self._consolidate_and_pad( downsampled_wall, new_scale, output_config, align=True, pad=True )
                del downsampled_wall

                # Write to DVID
                self._write_bricks( consolidated_wall, new_scale, output_config )
                padded_wall = consolidated_wall
                del consolidated_wall
            del padded_wall

        logger.info(f"DONE copying segmentation to {len(self.config_data['outputs'])} destinations.")

    
    def _create_output_instances_if_necessary(self):
        """
        If it doesn't exist yet, create it first with the user's specified
        pyramid-depth, or with an automatically chosen depth.
        """
        options = self.config_data["options"]

        for output_config in self.config_data["outputs"]:
            # Create new segmentation instance first if necessary
            if not is_datainstance( output_config["dvid"]["server"],
                                    output_config["dvid"]["uuid"],
                                    output_config["dvid"]["segmentation-name"] ):

                depth = options["pyramid-depth"]
                if depth == -1:
                    # if no pyramid depth is specified, determine the max, based on bb size.
                    input_bb_zyx = np.array(self.config_data["input"]["geometry"]["bounding-box"])[:,::-1]
                    depth = choose_pyramid_depth(input_bb_zyx, 512)
        
                # create new label array with correct number of pyramid scales
                create_labelarray( output_config["dvid"]["server"],
                                   output_config["dvid"]["uuid"],
                                   output_config["dvid"]["segmentation-name"],
                                   depth,
                                   3*(output_config["geometry"]["block-width"],) )


    def _log_neuroglancer_links(self):
        """
        Write a link to the log file for viewing the segmentation data after it is ingested.
        We assume that the output server is hosting neuroglancer at http://<server>:<port>/neuroglancer/
        """
        for index, output_config in enumerate(self.config_data["outputs"]):
            server = output_config["dvid"]["server"] # Note: Begins with http://
            uuid = output_config["dvid"]["uuid"]
            instance = output_config["dvid"]["segmentation-name"]
            
            output_box_xyz = np.array(output_config["geometry"]["bounding-box"])
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


    def _read_pyramid_depth(self):
        """
        Read the MaxDownresLevel from each output instance we'll be writing to,
        and verify that it matches our config for pyramid-depth.
        
        Return the max depth we found in the outputs.
        (They should all be the same...)
        """
        max_depth = -1
        for output_config in self.config_data["outputs"]:
            output_source = output_config["dvid"]
            options = self.config_data["options"]
    
            node_service = retrieve_node_service( output_source["server"],
                                                  output_source["uuid"], 
                                                  self.resource_server,
                                                  self.resource_port,
                                                  self.APPNAME )
    
            info = node_service.get_typeinfo(output_source["segmentation-name"])
    
            existing_depth = int(info["Extended"]["MaxDownresLevel"])
            if options["pyramid-depth"] not in (-1, existing_depth):
                raise Exception(f"Can't set pyramid-depth to {options['pyramid-depth']}: "
                                f"Data instance '{output_source['segmentation-name']}' already existed, with depth {existing_depth}")

            max_depth = max(max_depth, existing_depth)

        return existing_depth


    def _consolidate_and_pad(self, input_wall, scale, output_config, align=True, pad=True):
        """
        Consolidate (align), and pad the given BrickWall

        Note: UNPERSISTS the input data and returns the new, downsampled data.

        Args:
            scale: The pyramid scale of the data.
            
            output_config: The config settings for the output volume to align to and pad from
            
            align: If False, skip the alignment step.
                  (Only use this if the bricks are already aligned.)
            
            pad: If False, skip the padding step
        
        Returns a pre-executed and persisted BrickWall.
        """
        output_writing_grid = Grid(output_config["geometry"]["message-block-shape"][::-1])

        if not align or output_writing_grid.equivalent_to(input_wall.grid):
            realigned_wall = input_wall
            realigned_wall.persist_and_execute(f"Scale {scale}: Persisting pre-aligned bricks", logger)
        else:
            # Consolidate bricks to full-size, aligned blocks (shuffles data)
            realigned_wall = input_wall.realign_to_new_grid( output_writing_grid )
            realigned_wall.persist_and_execute(f"Scale {scale}: Shuffling bricks into alignment", logger)

            # Discard original
            input_wall.unpersist()
        
        if not pad:
            return realigned_wall

        output_reader = VolumeService.create_from_config(output_config, self.config_dir, self.resource_mgr_client)

        # Pad from previously-existing pyramid data until
        # we have full storage blocks, e.g. (64,64,64),
        # but not necessarily full bricks, e.g. (64,64,6400)
        storage_block_width = output_config["geometry"]["block-width"]
        output_padding_grid = Grid( (storage_block_width, storage_block_width, storage_block_width), output_writing_grid.offset )
        output_accessor_func = partial(output_reader.get_subvolume, scale=scale)
        
        padded_wall = realigned_wall.fill_missing(output_accessor_func, output_padding_grid)
        padded_wall.persist_and_execute(f"Scale {scale}: Padding", logger)

        # Discard old
        realigned_wall.unpersist()

        return padded_wall


    def _write_bricks(self, brick_wall, scale, output_config):
        """
        Writes partition to specified dvid.
        """
        output_writer = VolumeService.create_from_config(output_config, self.config_dir, self.resource_mgr_client)
        instance_name = output_config["dvid"]["segmentation-name"]
        block_width = output_config["geometry"]["block-width"]
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

                with Timer() as put_timer:
                    output_writer.write_subvolume(datacrop, data_offset_zyx, scale)

                # Note: This timing data doesn't reflect ideal throughput, since throttle
                #       and/or the resource manager muddy the numbers a bit...
                megavoxels_per_second = datacrop.size / 1e6 / put_timer.seconds
                logger.info(f"Put block {data_offset_zyx} in {put_timer.seconds:.3f} seconds ({megavoxels_per_second:.1f} Megavoxels/second)")

        logger.info(f"Scale {scale}: Writing bricks to {instance_name}...")
        with Timer() as timer:
            brick_wall.bricks.foreach(write_brick)
        logger.info(f"Scale {scale}: Writing bricks to {instance_name} took {timer.timedelta}")

    
    def _write_body_sizes( self, brick_wall, output_config ):
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

        suffix = output_config["dvid"]["segmentation-name"]
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
            