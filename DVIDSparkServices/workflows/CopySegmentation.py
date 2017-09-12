from __future__ import print_function, absolute_import
from __future__ import division
import copy
import json
import logging
import socket
from functools import reduce

import numpy as np
import h5py
import pandas as pd

from pyspark import StorageLevel

from DVIDSparkServices.io_util.partitionSchema import volumePartition, VolumeOffset, partitionSchema
from DVIDSparkServices.sparkdvid import sparkdvid
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 
from DVIDSparkServices.dvid.metadata import create_labelarray, is_datainstance
from DVIDSparkServices.reconutils.downsample import downsample_labels_3d
from DVIDSparkServices.util import Timer, runlength_encode, choose_pyramid_depth, blockwise_boxes, nonconsecutive_bincount

logger = logging.getLogger(__name__)

class CopySegmentation(Workflow):
    
    DataInfoSchema = \
    {
        "type": "object",
        "default": {},
        "required": ["input", "output", "bounding-box"],
        "additionalProperties": False,
        "properties": {
            #
            # INPUT
            #
            "input": {
                "type": "object",
                "default": {},
                "additionalProperties": False,
                "required": ["server", "uuid", "segmentation-name"],
                "properties": {
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
                    }
                }
            },
    
            #
            # OUTPUT
            #
            "output": {
                "type": "object",
                "default": {},
                "additionalProperties": False,
                "required": ["server", "uuid", "segmentation-name"],
                "properties": {
                    "server": {
                        "description": "location of DVID server to WRITE",
                        "type": "string",
                    },
                    "uuid": {
                        "description": "version node for WRITING segmentation",
                        "type": "string"
                    },
                    "segmentation-name": {
                        "description": "The labels instance to WRITE to.  If necessary, will be created (as labelarray).",
                        "type": "string",
                        "minLength": 1
                    },
                    "block-size": {
                        "description": "The DVID blocksize for new segmentation instances. Ignored if the output segmentation instance already exists.",
                        "type": "integer",
                        "default": 64
                    }
                }
            },

            #
            # BOUNDING BOX (alternative to ROI)
            #
            "bounding-box": {
                "type": "object",
                "default": {"start": [0,0,0], "stop": [0,0,0]},
                "required": ["start", "stop"],
                "start": {
                    "description": "The bounding box lower coordinate in XYZ order",
                    "type": "array",
                    "items": { "type": "integer" },
                    "minItems": 3,
                    "maxItems": 3,
                    "default": [0,0,0]
                },
                "stop": {
                    "description": "The bounding box upper coordinate in XYZ order.  (numpy-conventions, i.e. maxcoord+1)",
                    "type": "array",
                    "items": { "type": "integer" },
                    "minItems": 3,
                    "maxItems": 3,
                    "default": [0,0,0] # Default is start == stop, so we can easily detect whether the bounding box was provided in the config.
                },
            }
        }
    }

    OptionsSchema = copy.copy(Workflow.OptionsSchema)
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
        },
       "fetch-block-shape": {
            "description": "The block shape (XYZ) for the initial tasks that fetch segmentation from DVID.",
            "type": "array",
            "items": { "type": "integer" },
            "minItems": 3,
            "maxItems": 3,
            "default": [6400,64,64]
        },
    })

    Schema = \
    {
      "$schema": "http://json-schema.org/schema#",
      "title": "Service to load raw and label data into DVID",
      "type": "object",
      "properties": {
        "data-info": DataInfoSchema,
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

        input_config = self.config_data["data-info"]["input"]
        output_config = self.config_data["data-info"]["output"]

        for cfg in (input_config, output_config):
            # Prepend 'http://' if necessary.
            if not cfg['server'].startswith('http'):
                cfg['server'] = 'http://' + cfg['server']

        # Convert from unicode for easier C++ calls
            cfg["server"] = str(cfg["server"])
            cfg["uuid"] = str(cfg["uuid"])
            cfg["segmentation-name"] = str(cfg["segmentation-name"])

        # create spark dvid contexts
        self.sparkdvid_input_context = sparkdvid.sparkdvid(self.sc, input_config["server"], input_config["uuid"], self)
        self.sparkdvid_output_context = sparkdvid.sparkdvid(self.sc, output_config["server"], output_config["uuid"], self)


    def execute(self):
        output_config = self.config_data["data-info"]["output"]
        options = self.config_data["options"]

        # RDD: (volumePartition, data)
        seg_chunks_partitioned, bounding_box, partition_shape_zyx = self._partition_input()
        self._create_output_instance_if_necessary(bounding_box)

        # Overwrite pyramid depth in our config (in case the user specified -1 == 'automatic')
        options["pyramid-depth"] = self._read_pyramid_depth()

        # data must exist after writing to dvid for downsampling
        seg_chunks_partitioned.persist(StorageLevel.MEMORY_AND_DISK)

        # FIXME: Instead of interleaving read and write operations,
        #        let's force the entire read first, then write, for easier benchmarking of those two steps.
        with Timer() as timer:
            seg_chunks_partitioned.count()
        logger.info(f"Reading entire volume took {timer.timedelta}")

        # write level 0
        with Timer() as timer:
            self._write_blocks(seg_chunks_partitioned, output_config["segmentation-name"], 0)
        logger.info(f"Writing entire volume at scale 0 took {timer.timedelta}")

        # Forcibly update DVID's extents metadata
        # TODO: Soon DVID will do this for us, and we can remove this step...
        self._set_output_extents(bounding_box)

        # Write body sizes to JSON
        self._write_body_sizes( seg_chunks_partitioned )
        
        # write pyramid levels for >=1 
        for level in range(1, options["pyramid-depth"] + 1):
            # downsample seg partition
            def downsample(part_vol):
                part, vol = part_vol
                vol = downsample_labels_3d(vol, (2,2,2))
                return (part, vol)
            downsampled_array = seg_chunks_partitioned.map(downsample)

            # prepare for repartition
            # (!!assume vol and offset will always be power of two because of padding)
            def repartition_down(part_volume):
                part, volume = part_volume
                downsampled_offset = np.array(part.get_offset()) // 2
                downsampled_reloffset = np.array(part.get_reloffset()) // 2
                offsetnew = VolumeOffset(*downsampled_offset)
                reloffsetnew = VolumeOffset(*downsampled_reloffset)
                partnew = volumePartition((offsetnew.z, offsetnew.y, offsetnew.x), offsetnew, reloffset=reloffsetnew)
                return partnew, volume
            downsampled_array = downsampled_array.map(repartition_down)
            
            # repartition downsampled data
            schema = partitionSchema(partition_shape_zyx, padding=output_config["block-size"])
            downsampled_chunks_partitioned = schema.partition_data(downsampled_array)

            # persist for next level
            downsampled_chunks_partitioned.persist(StorageLevel.MEMORY_AND_DISK)

            # FIXME: Instead of interleaving compute and write operations,
            #        let's force the entire compute first, then write, for easier benchmarking of those two steps.
            with Timer() as timer:
                downsampled_chunks_partitioned.count()
            logger.info(f"Computing scale {level} took {timer.timedelta}")
            
            # Unpersist previous level
            seg_chunks_partitioned.unpersist()
            seg_chunks_partitioned = downsampled_chunks_partitioned
            
            #  write data new level
            with Timer() as timer:
                self._write_blocks(seg_chunks_partitioned, output_config["segmentation-name"], level)
            logger.info(f"Writing scale {level} took {timer.timedelta}")


    def _partition_input(self):
        """
        Map the input segmentation
        volume from DVID into an RDD of (volumePartition, data),
        using the config's bounding-box setting for the full volume region,
        using the 'fetch-block-shape' as the partition size.

        Returns: (RDD, bounding_box_zyx, partition_shape_zyx)
            where:
                - RDD is (volumePartition, data)
                - bounding box is tuple (start_zyx, stop_zyx)
                - partition_shape_zyx is a tuple
            
        """
        input_config = self.config_data["data-info"]["input"]
        output_config = self.config_data["data-info"]["output"]
        bb_config = self.config_data["data-info"]["bounding-box"]
        options = self.config_data["options"]

        # repartition to be z=blksize, y=blksize, x=runlength (x=0 is unlimited)
        partition_shape_zyx = options["fetch-block-shape"][::-1]
        
        assert not any(np.array(partition_shape_zyx) % output_config["block-size"]), \
            "fetch-block-shape should be a multiple of the block size in all dimensions."
        
        bounding_box_xyz = np.array([bb_config["start"], bb_config["stop"]])
        bounding_box_zyx = bounding_box_xyz[:,::-1]

        # Aim for 2 GB RDD partitions
        GB = 2**30
        target_partition_size_voxels = 2 * GB // np.uint64().nbytes
        seg_chunks_partitioned = \
            self.sparkdvid_input_context.parallelize_bounding_box( input_config['segmentation-name'],
                                                                   bounding_box_zyx,
                                                                   partition_shape_zyx,
                                                                   target_partition_size_voxels )

        return seg_chunks_partitioned, bounding_box_zyx, partition_shape_zyx


    def _create_output_instance_if_necessary(self, bounding_box):
        """
        If it doesn't exist yet, create it first with the user's specified
        pyramid-depth, or with an automatically chosen depth.
        """
        output_config = self.config_data["data-info"]["output"]
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

        # create new label array with correct number of pyramid levels
        create_labelarray( output_config["server"],
                           output_config["uuid"],
                           output_config["segmentation-name"],
                           depth,
                           3*(output_config["block-size"],) )

    def _read_pyramid_depth(self):
        """
        Read the MaxDownresLevel from it and verify that it matches our config for pyramid-depth.
        Return the MaxDownresLevel.
        """
        output_config = self.config_data["data-info"]["output"]
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

    def _set_output_extents(self, bounding_box):
        # For now, the only way to set extents is by fetching the min and max
        # blocks and re-writing them with the non-block get_labels method (the /raw endpoint).
        output_config = self.config_data["data-info"]["output"]
        node_service = retrieve_node_service( output_config["server"],
                                              output_config["uuid"], 
                                              self.resource_server,
                                              self.resource_port,
                                              self.APPNAME )

        min_block_start = bounding_box[0]
        max_block_start = bounding_box[1] - (64,64,64)

        min_block = node_service.get_labels3D( output_config["segmentation-name"],
                                               (64,64,64),
                                               min_block_start )

        max_block = node_service.get_labels3D( output_config["segmentation-name"],
                                               (64,64,64),
                                               max_block_start )

        node_service.put_labels3D( output_config["segmentation-name"],
                                   min_block,
                                   min_block_start )

        node_service.put_labels3D( output_config["segmentation-name"],
                                   max_block,
                                   max_block_start )

    def _write_blocks(self, seg_chunks_partitioned, dataname, level):
        """Writes partition to specified dvid.
        """
        output_config = self.config_data["data-info"]["output"]
        appname = self.APPNAME

        server = output_config["server"]
        uuid = output_config["uuid"]
        blksize = output_config["block-size"]
        
        resource_server = self.resource_server 
        resource_port = self.resource_port 
        
        # default delimiter
        delimiter = 0
    
        @self.collect_log(lambda i: socket.gethostname() + '-write-blocks-' + str(level))
        def write_blocks(part_vol):
            logger = logging.getLogger(__name__)
            part, data = part_vol
            offset = part.get_offset()
            reloffset = part.get_reloffset()
            _, _, x_size = data.shape
            if x_size % blksize != 0:
                # check if padded
                raise ValueError("Data is not block aligned")

            shiftedoffset = (offset.z+reloffset.z, offset.y+reloffset.y, offset.x+reloffset.x)
            node_service = retrieve_node_service(server, uuid, resource_server, resource_port, appname)

            # Find all non-zero blocks (and record by block index)
            block_coords = []
            for block_index, block_x in enumerate(range(0, x_size, blksize)):
                if not (data[:, :, block_x:block_x+blksize] == delimiter).all():
                    block_coords.append( (0, 0, block_index) ) # (Don't care about Z,Y indexes, just X-index)

            # Find *runs* of non-zero blocks
            block_runs = runlength_encode(block_coords, True) # returns [[Z,Y,X1,X2], [Z,Y,X1,X2], ...]
            
            # Convert stop indexes from inclusive to exclusive
            block_runs[:,-1] += 1
            
            # Discard Z,Y indexes and convert from indexes to pixels
            ranges = blksize * block_runs[:, 2:4]
            
            # iterate through contiguous blocks and write to DVID
            for (data_x_start, data_x_end) in ranges:
                datacrop = data[:,:,data_x_start:data_x_end].copy()
                data_offset_zyx = (shiftedoffset[0], shiftedoffset[1], shiftedoffset[2] + data_x_start)

                throttle = (resource_server == "" and not server.startswith("http://127.0.0.1"))
                with Timer() as put_timer:
                    node_service.put_labelblocks3D( str(dataname), datacrop, data_offset_zyx, throttle, level)

                # Note: This timing data doesn't measure ideal throughput, since throttle
                #       and/or the resource manager muddy the numbers a bit...
                voxels_per_second = datacrop.size / put_timer.seconds
                logger.info("Put block {} in {:.3f} seconds ({:.1f} Megavoxels/second)"
                            .format(data_offset_zyx, put_timer.seconds, voxels_per_second / 1e6))

        seg_chunks_partitioned.foreach(write_blocks)
       
    
    def _write_body_sizes( self, seg_chunks_partitioned ):
        logger = logging.getLogger(__name__)
        
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
                labels_and_sizes = ( seg_chunks_partitioned
                                        .values()
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
                body_labels, body_sizes = ( seg_chunks_partitioned
                                                .values()
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
