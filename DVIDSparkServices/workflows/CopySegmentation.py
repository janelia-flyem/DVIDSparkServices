from __future__ import print_function, absolute_import
from __future__ import division
import copy
import json
import logging
import socket
from functools import reduce, partial

import numpy as np
import h5py
import pandas as pd

from pyspark import StorageLevel

from DVIDSparkServices.io_util.brick import Grid, Brick, remap_bricks_to_new_grid, pad_brick_data_from_volume_source
from DVIDSparkServices.sparkdvid import sparkdvid
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 
from DVIDSparkServices.dvid.metadata import create_labelarray, is_datainstance
from DVIDSparkServices.reconutils.downsample import downsample_labels_3d_suppress_zero
from DVIDSparkServices.util import Timer, runlength_encode, choose_pyramid_depth, nonconsecutive_bincount

logger = logging.getLogger(__name__)

class CopySegmentation(Workflow):
    
    DvidSegmentationSourceSchema = \
    {
        "description": "Parameters to use DVID as a source of voxel data",
        "type": "object",
        "required": ["service-type", "server", "uuid", "segmentation-name"],
        "properties": {
            "service-type": { "type": "string", "enum": ["dvid"] },
            "server": {
                "description": "location of DVID server to READ.  Either IP:PORT or the special word 'node-local'.",
                "type": "string",
            },
            "uuid": {
                "description": "version node for READING segmentation",
                "type": "string"
            },
            "segmentation-name": {
                "description": "The labels instance to READ from. Instance may be either googlevoxels, labelblk, or labelarray.",
                "type": "string",
                "minLength": 1
            },
        }
    }

    BoundingBoxSchema = \
    {
        "description": "The bounding box [[x0,y0,z0],[x1,y1,z1]], "
                       "where [x1,y1,z1] == maxcoord+1 (i.e. Python conventions)",
        "type": "array",
        "minItems": 2,
        "maxItems": 2,
        "items": {
            "type": "array",
            "items": { "type": "integer" },
            "minItems": 3,
            "maxItems": 3
        }
    }

    SegmentationVolumeSchema = \
    {
        "description": "Describes a segmentation volume source, extents, and preferred access pattern",
        "type": "object",
        "required": ["bounding-box", "message-block-shape"],
        "oneOf": [
            DvidSegmentationSourceSchema
        ],
        "properties": {
            "bounding-box": BoundingBoxSchema,
            "message-block-shape": {
                "description": "The block shape (XYZ) for the initial tasks that fetch segmentation from DVID.",
                "type": "array",
                "items": { "type": "integer" },
                "minItems": 3,
                "maxItems": 3,
                "default": [6400,64,64],
            },
            "block-width": {
                "description": "The block size of the underlying volume storage.",
                "type": "integer",
                "default": 64
            }
        }
    }

    OptionsSchema = copy.deepcopy(Workflow.OptionsSchema)
    OptionsSchema["properties"].update(
    {
        "body-size-output-path" : {
            "description": "A file name to write the body size HDF5 output. "
                           "Relative paths are interpreted as relative to this config file. "
                           "An empty string forces body size calculation to be skipped.",
            "type": "string",
            "default": "./body-sizes.h5"
        },
        "body-size-minimum" : {
            "description": "Minimum size to include in the body size HDF5 output.  Smaller bodies are omitted.",
            "type": "integer",
            "default": 0
        },
        "body-size-reduce-by-key" : {
            "description": "(For performance experiments) Whether to perform the body size computation via reduceByKey.",
            "type": "boolean",
            "default": False
        },
        "pyramid-depth": {
            "description": "Number of pyramid levels to generate (-1 means choose automatically, 0 means no pyramid)",
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
        "required": ["input", "output"],
        "properties": {
            "input": SegmentationVolumeSchema,
            "output": SegmentationVolumeSchema,
            "options" : OptionsSchema
        }
    }

    @staticmethod
    def dumpschema():
        return json.dumps(CopySegmentation.Schema)

    # name of application for DVID queries
    APPNAME = "copysegmentation"

    def __init__(self, config_filename):
        super(CopySegmentation, self).__init__( config_filename,
                                                CopySegmentation.dumpschema(),
                                                "Copy Segmentation" )
        self._sanitize_config()

        input_config = self.config_data["input"]
        output_config = self.config_data["output"]

        # create spark dvid contexts
        self.sparkdvid_input_context = sparkdvid.sparkdvid(self.sc, input_config["server"], input_config["uuid"], self)
        self.sparkdvid_output_context = sparkdvid.sparkdvid(self.sc, output_config["server"], output_config["uuid"], self)


    def _sanitize_config(self):
        """
        Tidy up some config values.
        """
        for cfg in (self.config_data["input"], self.config_data["output"]):
            # Prepend 'http://' to the server if necessary.
            if "server" in cfg and not cfg["server"].startswith('http'):
                cfg["server"] = 'http://' + cfg["server"]


    def execute(self):
        input_config = self.config_data["input"]
        output_config = self.config_data["output"]
        options = self.config_data["options"]

        input_bb_zyx = np.array(input_config["bounding-box"])[:,::-1]
        output_bb_zyx = np.array(output_config["bounding-box"])[:,::-1]

        assert ((input_bb_zyx[1] - input_bb_zyx[0]) == (output_bb_zyx[1] - output_bb_zyx[0])).all(), \
            "Input bounding box and output bounding box do not have the same dimensions"
        assert not any(np.array(output_config["message-block-shape"]) % output_config["block-width"]), \
            "Output message-block-shape should be a multiple of the block size in all dimensions."


        input_bricks, bounding_box, _input_grid = self._partition_input()
        self._create_output_instance_if_necessary(bounding_box)

        # Overwrite pyramid depth in our config (in case the user specified -1, i.e. automatic)
        options["pyramid-depth"] = self._read_pyramid_depth()

        persist_and_execute(input_bricks, f"Reading entire volume")

        def translate_brick(offset, brick):
            return Brick( brick.logical_box + offset,
                          brick.physical_box + offset,
                          brick.volume )

        # Translate coordinates from input to output
        # (which will leave the bricks in a new, offset grid)
        translation_offset_zyx = output_bb_zyx[0] - input_bb_zyx[0]
        output_bricks = input_bricks.map( partial(translate_brick, translation_offset_zyx) )
        del input_bricks

        # Re-align to output grid, pad internally to block-align.
        aligned_bricks = self._consolidate_and_pad(output_bricks, 0)
        del output_bricks

        # Compute body sizes and write to HDf5
        self._write_body_sizes( aligned_bricks )

        # Write scale 0 to DVID
        self._write_bricks( aligned_bricks, output_config["segmentation-name"], 0 )

        # Downsample and write the rest of the pyramid scales
        for new_scale in range(1, 1+options["pyramid-depth"]):
            # Compute downsampled (results in small bricks)
            downsampled_bricks = self._downsample_bricks(aligned_bricks, new_scale)

            # Consolidate to full-size bricks and pad internally to block-align
            aligned_bricks = self._consolidate_and_pad(downsampled_bricks, new_scale)

            # Write to DVID
            self._write_bricks(aligned_bricks, output_config["segmentation-name"], new_scale)
            del downsampled_bricks


    def _partition_input(self):
        """
        Map the input segmentation
        volume from DVID into an RDD of (volumePartition, data),
        using the config's bounding-box setting for the full volume region,
        using the input 'message-block-shape' as the partition size.

        Returns: (RDD, bounding_box_zyx, partition_shape_zyx)
            where:
                - RDD is (volumePartition, data)
                - bounding box is tuple (start_zyx, stop_zyx)
                - partition_shape_zyx is a tuple
            
        """
        input_config = self.config_data["input"]
        input_bb = input_config["bounding-box"]

        # repartition to be z=blksize, y=blksize, x=runlength
        brick_shape_zyx = input_config["message-block-shape"][::-1]
        input_grid = Grid(brick_shape_zyx, (0,0,0))
        
        input_bb_zyx = np.array(input_bb)[:,::-1]

        # Aim for 2 GB RDD partitions
        GB = 2**30
        target_partition_size_voxels = 2 * GB // np.uint64().nbytes
        bricks = self.sparkdvid_input_context.parallelize_bounding_box( input_config["segmentation-name"],
                                                                        input_bb_zyx,
                                                                        input_grid,
                                                                        target_partition_size_voxels )

        return bricks, input_bb_zyx, input_grid


    def _create_output_instance_if_necessary(self, bounding_box):
        """
        If it doesn't exist yet, create it first with the user's specified
        pyramid-depth, or with an automatically chosen depth.
        """
        output_config = self.config_data["output"]
        options = self.config_data["options"]

        # Create new segmentation instance first if necessary
        if is_datainstance( output_config["server"],
                            output_config["uuid"],
                            output_config["segmentation-name"] ):
            return

        depth = options["pyramid-depth"]
        if depth == -1:
            # if no pyramid depth is specified, determine the max
            depth = choose_pyramid_depth(bounding_box, 512)

        # create new label array with correct number of pyramid scales
        create_labelarray( output_config["server"],
                           output_config["uuid"],
                           output_config["segmentation-name"],
                           depth,
                           3*(output_config["block-width"],) )


    def _read_pyramid_depth(self):
        """
        Read the MaxDownresLevel from it and verify that it matches our config for pyramid-depth.
        Return the MaxDownresLevel.
        """
        output_config = self.config_data["output"]
        options = self.config_data["options"]

        node_service = retrieve_node_service( output_config["server"],
                                              output_config["uuid"], 
                                              self.resource_server,
                                              self.resource_port,
                                              self.APPNAME )

        info = node_service.get_typeinfo(output_config["segmentation-name"])

        existing_depth = int(info["Extended"]["MaxDownresLevel"])
        if options["pyramid-depth"] not in (-1, existing_depth):
            raise Exception("Can't set pyramid-depth to {}. Data instance '{}' already existed, with depth {}"
                            .format(options["pyramid-depth"], output_config["segmentation-name"], existing_depth))
        return existing_depth


    def _downsample_bricks(self, bricks, new_scale):
        """
        Immediately downsample the given RDD of Bricks by a factor of 2 and persist the results.
        Also, unpersist the input.
        """
        # Downsampling effectively divides grid by half (i.e. 32x32x32)
        downsampled_bricks = bricks.map(downsample_brick)
        persist_and_execute(downsampled_bricks, f"Scale {new_scale}: Downsampling")
        bricks.unpersist()
        del bricks

        # Bricks are now half-size
        return downsampled_bricks


    def _consolidate_and_pad(self, bricks, scale):
        """
        Consolidate (align), and pad the given RDD of Bricks.

        scale: The pyramid scale of the data.
        
        Note: UNPERSISTS the input data and returns the new, downsampled data.
        """
        output_config = self.config_data["output"]

        # Consolidate bricks to full size, aligned blocks (shuffles data)
        # FIXME: We should skip this if the grids happen to be aligned already.
        #        This shuffle takes ~15 minutes per tab.
        output_writing_grid = Grid(output_config["message-block-shape"], (0,0,0))
        remapped_bricks = remap_bricks_to_new_grid( output_writing_grid, bricks ).values()
        persist_and_execute(remapped_bricks, f"Scale {scale}: Shuffling bricks into alignment")

        # Discard original
        bricks.unpersist()
        del bricks

        # Pad from previously-existing pyramid data.
        output_padding_grid = Grid(output_config["block-width"], (0,0,0))
        output_accessor = self.sparkdvid_output_context.get_volume_accessor(output_config["segmentation-name"], scale)
        padded_bricks = remapped_bricks.map( partial(pad_brick_data_from_volume_source, output_padding_grid, output_accessor) )
        persist_and_execute(padded_bricks, f"Scale {scale}: Padding")

        # Discard
        remapped_bricks.unpersist()
        del remapped_bricks

        return padded_bricks


    def _write_bricks(self, bricks, dataname, scale):
        """
        Writes partition to specified dvid.
        """
        output_config = self.config_data["output"]
        appname = self.APPNAME

        server = output_config["server"]
        uuid = output_config["uuid"]
        block_width = output_config["block-width"]
        
        resource_server = self.resource_server 
        resource_port = self.resource_port 
        
        # default delimiter
        delimiter = 0
    
        @self.collect_log(lambda i: socket.gethostname() + '-write-blocks-' + str(scale))
        def write_brick(brick):
            logger = logging.getLogger(__name__)
            
            assert (brick.physical_box % block_width == 0).all(), \
                f"This function assumes each brick's physical data is already block-aligned: {brick}"
            
            node_service = retrieve_node_service(server, uuid, resource_server, resource_port, appname)

            x_size = brick.volume.shape[2]
            # Find all non-zero blocks (and record by block index)
            block_coords = []
            for block_index, block_x in enumerate(range(0, x_size, block_width)):
                if not (brick.volume[:, :, block_x:block_x+block_width] == delimiter).all():
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

                throttle = (resource_server == "" and not server.startswith("http://127.0.0.1"))
                with Timer() as put_timer:
                    node_service.put_labelblocks3D( str(dataname), datacrop, data_offset_zyx, throttle, scale)

                # Note: This timing data doesn't reflect ideal throughput, since throttle
                #       and/or the resource manager muddy the numbers a bit...
                voxels_per_second = datacrop.size / put_timer.seconds
                logger.info("Put block {} in {:.3f} seconds ({:.1f} Megavoxels/second)"
                            .format(data_offset_zyx, put_timer.seconds, voxels_per_second / 1e6))

        with Timer() as timer:
            bricks.foreach(write_brick)
        logger.info(f"Scale {scale}: Writing to DVID took {timer.timedelta}")

    
    def _write_body_sizes( self, bricks ):
        """
        Calculate the size (in voxels) of all label bodies in the volume,
        and write the results to an HDF5 file.
        
        NOTE: For now, we implement two alternative methods of computing this result,
              for the sake of performance comparisons between the two methods.
              The method used is determined by the 'body-size-reduce-by-key' setting.
        """
        if not self.config_data["options"]["body-size-output-path"]:
            logger.info("Skipping body size calculation.")
            return

        if self.config_data["options"]["body-size-reduce-by-key"]:
            # Reduce using reduceByKey and simply concatenating the results
            from operator import add
            def transpose(labels_and_sizes):
                labels, sizes = labels_and_sizes
                return np.array( (labels, sizes) ).transpose()
    
            def reduce_to_array( pair_list_1, pair_list_2 ):
                return np.concatenate( (pair_list_1, pair_list_2) )
    
            with Timer() as timer:
                labels_and_sizes = ( bricks.map( lambda br: br.volume )
                                           .map( nonconsecutive_bincount )
                                           .flatMap( transpose )
                                           .reduceByKey( add )
                                           .map( lambda pair: [pair] )
                                           .treeReduce( reduce_to_array, depth=4 ) )
    
                body_labels, body_sizes = np.transpose( labels_and_sizes )
            logger.info("Computing {} body sizes took {} seconds".format(len(body_labels), timer.seconds))
        
        else: # Reduce by merging pandas dataframes
            
            @self.collect_log(lambda *args: 'merge_label_counts')
            def merge_label_counts( labels_and_counts_A, labels_and_counts_B ):
                labels_A, counts_A = labels_and_counts_A
                labels_B, counts_B = labels_and_counts_B
                
                # Fast path
                if len(labels_A) == 0:
                    return (labels_B, counts_B)
                if len(labels_B) == 0:
                    return (labels_A, counts_A)
                
                with Timer() as timer:
                    series_A = pd.Series(index=labels_A, data=counts_A)
                    series_B = pd.Series(index=labels_B, data=counts_B)
                    combined = series_A.add(series_B, fill_value=0)
                
                logger = logging.getLogger(__name__)
                logger.info("Merging label count lists of sizes {} + {} = {} took {} seconds"
                            .format(len(labels_A), len(labels_B), len(combined), timer.seconds))
    
                return (combined.index, combined.values.astype(np.uint64))

            def reduce_partition( partition_elements ):
                # Almost like the builtin reduce() function, but wraps the result in a list,
                # which is what RDD.mapPartitions() expects to see.
                return [reduce(merge_label_counts, partition_elements, [(), ()])]
    
            with Timer() as timer:
                logger.info("Computing body sizes...")
     
                # Two-stage repartition/reduce, to avoid doing ALL the work on the driver.
                body_labels, body_sizes = ( bricks.map( lambda br: br.volume )
                                                  .map( nonconsecutive_bincount )
                                                  .treeReduce( merge_label_counts, depth=4 ) )
            logger.info("Computing {} body sizes took {} seconds".format(len(body_labels), timer.seconds))

        min_size = self.config_data["options"]["body-size-minimum"]
        if min_size > 1:
            logger.info("Omitting body sizes below {} voxels...".format(min_size))
            valid_rows = body_sizes >= min_size
            body_labels = body_labels[valid_rows]
            body_sizes = body_sizes[valid_rows]
        assert body_labels.shape == body_sizes.shape

        with Timer() as timer:
            logger.info("Sorting {} bodies by size...".format(len(body_labels)))
            sort_indices = np.argsort(body_sizes)[::-1]
            body_sizes = body_sizes[sort_indices]
            body_labels = body_labels[sort_indices]
        logger.info("Sorting {} bodies by size took {} seconds".format(len(body_labels), timer.seconds))

        output_path = self.relpath_to_abspath(self.config_data["options"]["body-size-output-path"])
        with Timer() as timer:
            logger.info("Writing {} body sizes to {}".format(len(body_labels), output_path))
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('labels', data=body_labels, chunks=True)
                f.create_dataset('sizes', data=body_sizes, chunks=True)
        logger.info("Writing {} body sizes took {} seconds".format(len(body_sizes), timer.seconds))


def downsample_brick(brick):
    assert (brick.physical_box % 2 == 0).all()
    assert (brick.logical_box % 2 == 0).all()

    # For consistency with DVID's on-demand downsampling, we suppress 0 pixels.
    downsampled_volume, _ = \
        downsample_labels_3d_suppress_zero(brick.volume, (2,2,2), brick.physical_box)

    downsampled_logical_box = brick.logical_box // 2
    downsampled_physical_box = brick.physical_box // 2
    
    return Brick(downsampled_logical_box, downsampled_physical_box, downsampled_volume)


def persist_and_execute(rdd, description):
    with Timer() as timer:
        rdd.persist(StorageLevel.MEMORY_AND_DISK)
        rdd.count()
    logger.info(f"{description} took {timer.timedelta}")
