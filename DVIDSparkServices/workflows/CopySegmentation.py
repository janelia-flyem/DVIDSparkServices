from __future__ import print_function, absolute_import
from __future__ import division
import os
import csv
import copy
import json
import logging
import socket
from itertools import chain
from functools import partial

import numpy as np
import h5py

# Don't import pandas here; import it locally as needed
#import pandas as pd


from dvid_resource_manager.client import ResourceManagerClient

from dvidutils import downsample_labels

from DVIDSparkServices.io_util.brick import Grid, Brick, generate_bricks_from_volume_source, realign_bricks_to_new_grid, pad_brick_data_from_volume_source
from DVIDSparkServices.sparkdvid.sparkdvid import sparkdvid, retrieve_node_service
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.dvid.metadata import create_labelarray, is_datainstance
from DVIDSparkServices.reconutils.downsample import downsample_labels_3d_suppress_zero
from DVIDSparkServices.util import Timer, runlength_encode, choose_pyramid_depth, nonconsecutive_bincount, cpus_per_worker, num_worker_nodes, persist_and_execute
from DVIDSparkServices.io_util.brainmaps import BrainMapsVolume 
from DVIDSparkServices.auto_retry import auto_retry

from .common_schemas import SegmentationVolumeSchema, SegmentationVolumeListSchema

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
                "description": "Minimum size to include in the body size HDF5 output.  Smaller bodies are omitted.",
                "type": "integer",
                "default": 1000
            },
            "method" : {
                "description": "(For performance experiments) Whether to perform the body size computation via reduceByKey.",
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
        "required": ["input", "outputs"],
        "properties": {
            "input": SegmentationVolumeSchema,       # Labelmap, if any, is applied post-read
            "outputs": SegmentationVolumeListSchema, # LIST of output locations to write to.
                                                     # Labelmap, if any, is applied pre-write for each volume.
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


    def _sanitize_config(self):
        """
        Tidy up some config values.
        """
        input_config = self.config_data["input"]
        output_configs = self.config_data["outputs"]
        
        for cfg in [input_config] + output_configs:
            # Prepend 'http://' to the server if necessary.
            if "server" in cfg and not cfg["server"].startswith('http'):
                cfg["server"] = 'http://' + cfg["server"]

            # Convert labelmap (if any) to absolute path (relative to config file)
            labelmap_file = cfg["apply-labelmap"]["file"]
            if labelmap_file and not os.path.isabs(labelmap_file):
                cfg["apply-labelmap"]["file"] = self.relpath_to_abspath(labelmap_file)

        ##
        ## Check input/output dimensions and grid schemes.
        ##
        input_bb_zyx = np.array(input_config["bounding-box"])[:,::-1]

        first_output_config = output_configs[0]
        output_bb_zyx = np.array(first_output_config["bounding-box"])[:,::-1]
        output_brick_shape = np.array(first_output_config["message-block-shape"])
        output_block_width = np.array(first_output_config["block-width"])

        assert not any(np.array(output_brick_shape) % output_configs[0]["block-width"]), \
            "Output message-block-shape should be a multiple of the block size in all dimensions."
        assert ((input_bb_zyx[1] - input_bb_zyx[0]) == (output_bb_zyx[1] - output_bb_zyx[0])).all(), \
            "Input bounding box and output bounding box do not have the same dimensions"

        # NOTE: For now, we require that all outputs use the same bounding box and grid scheme,
        #       to simplify the execute() function.
        #       (We avoid re-translating and re-downsampling the input data for every new output.)
        #       The only way in which the outputs may differ is their label mapping data.
        for output_config in output_configs:
            bb = np.array(output_config["bounding-box"])[:,::-1]
            bs = output_config["message-block-shape"]
            bw = output_config["block-width"]
            
            assert (output_bb_zyx == bb).all(), \
                "For now, all output destinations must use the same bounding box and grid scheme"
            assert (output_brick_shape == bs).all(), \
                "For now, all output destinations must use the same bounding box and grid scheme"
            assert (output_block_width == bw).all(), \
                "For now, all output destinations must use the same bounding box and grid scheme"

    def execute(self):
        options = self.config_data["options"]

        input_config = self.config_data["input"]
        output_configs = self.config_data["outputs"]

        # See note in _sanitize_config()
        first_output_config = output_configs[0]

        input_bb_zyx = np.array(input_config["bounding-box"])[:,::-1]
        output_bb_zyx = np.array(first_output_config["bounding-box"])[:,::-1]

        input_bricks, bounding_box, _input_grid = self._partition_input()
        self._create_output_instances_if_necessary(bounding_box)
        self._log_neuroglancer_links()

        # Overwrite pyramid depth in our config (in case the user specified -1, i.e. automatic)
        options["pyramid-depth"] = self._read_pyramid_depth()

        persist_and_execute(input_bricks, f"Reading entire volume", logger)

        # Apply post-input label map (if any)
        remapped_input_bricks = self._remap_bricks(input_bricks, input_config["apply-labelmap"])
        del input_bricks
        
        def translate_brick(offset, brick):
            return Brick( brick.logical_box + offset,
                          brick.physical_box + offset,
                          brick.volume )

        # Translate coordinates from input to output
        # (which will leave the bricks in a new, offset grid)
        # This has no effect on the brick volumes themselves.
        translation_offset_zyx = output_bb_zyx[0] - input_bb_zyx[0]
        translated_bricks = remapped_input_bricks.map( partial(translate_brick, translation_offset_zyx) )
        remapped_input_bricks.unpersist()
        del remapped_input_bricks

        for output_config in self.config_data["outputs"]:
            # Re-align to output grid, pad internally to block-align.
            aligned_bricks = self._consolidate_and_pad(translated_bricks, 0, output_config)
    
            # Apply pre-output label map (if any)
            remapped_output_bricks = self._remap_bricks(aligned_bricks, output_config["apply-labelmap"])
    
            # Compute body sizes and write to HDF5
            self._write_body_sizes( remapped_output_bricks, output_config )
    
            # Write scale 0 to DVID
            self._write_bricks( remapped_output_bricks, 0, output_config )
            remapped_output_bricks.unpersist()
            del remapped_output_bricks
    
        # Downsample and write the rest of the pyramid scales
        for new_scale in range(1, 1+options["pyramid-depth"]):
            # Compute downsampled (results in small bricks)
            downsampled_bricks = self._downsample_bricks(aligned_bricks, new_scale)
            aligned_bricks.unpersist()
            del aligned_bricks

            for output_config in self.config_data["outputs"]:
                # Consolidate to full-size bricks and pad internally to block-align
                consolidated_input_bricks = self._consolidate_and_pad(downsampled_bricks, new_scale, output_config)

                # Remap the downsampled bricks.
                remapped_consolidated_bricks = self._remap_bricks(consolidated_input_bricks, output_config["apply-labelmap"])
                
                # Write to DVID
                self._write_bricks( remapped_consolidated_bricks, new_scale, output_config )

            aligned_bricks = downsampled_bricks
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
        options = self.config_data["options"]

        # repartition to be z=blksize, y=blksize, x=runlength
        brick_shape_zyx = input_config["message-block-shape"][::-1]
        input_grid = Grid(brick_shape_zyx, (0,0,0))
        
        input_bb_zyx = np.array(input_config["bounding-box"])[:,::-1]

        # Aim for 2 GB RDD partitions
        GB = 2**30
        target_partition_size_voxels = 2 * GB // np.uint64().nbytes

        if input_config["service-type"] == "dvid":
            sparkdvid_input_context = sparkdvid(self.sc, input_config["server"], input_config["uuid"], self)
            bricks = sparkdvid_input_context.parallelize_bounding_box( input_config["segmentation-name"],
                                                                       input_bb_zyx,
                                                                       input_grid,
                                                                       target_partition_size_voxels )
        elif input_config["service-type"] == "brainmaps":

            # Instantiate this outside of get_brainmaps_subvolume,
            # so it can be shared across an entire partition.
            vol = BrainMapsVolume( input_config["project"],
                                   input_config["dataset"],
                                   input_config["volume-id"],
                                   input_config["change-stack-id"],
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
                if not options["resource-server"]:
                    return vol.get_subvolume(box)

                req_bytes = 8 * np.prod(box[1] - box[0])
                client = ResourceManagerClient(options["resource-server"], options["resource-port"])
                with client.access_context('brainmaps', True, 1, req_bytes):
                    return vol.get_subvolume(box)
                
            block_size_voxels = np.prod(input_grid.block_shape)
            rdd_partition_length = target_partition_size_voxels // block_size_voxels

            bricks = generate_bricks_from_volume_source( input_bb_zyx,
                                                         input_grid,
                                                         get_brainmaps_subvolume,
                                                         self.sc,
                                                         rdd_partition_length )

            # If we're working with a tiny volume (e.g. testing),
            # make sure we at least parallelize across all cores.
            if bricks.getNumPartitions() < cpus_per_worker() * num_worker_nodes():
                bricks.repartition( cpus_per_worker() * num_worker_nodes() )
        else:
            raise RuntimeError(f'Unknown service-type: {input_config["service-type"]}')

        return bricks, input_bb_zyx, input_grid

    
    def _create_output_instances_if_necessary(self, bounding_box):
        """
        If it doesn't exist yet, create it first with the user's specified
        pyramid-depth, or with an automatically chosen depth.
        """
        options = self.config_data["options"]
        for output_config in self.config_data["outputs"]:
            # Create new segmentation instance first if necessary
            if not is_datainstance( output_config["server"],
                                output_config["uuid"],
                                output_config["segmentation-name"] ):

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

    def _log_neuroglancer_links(self):
        """
        Write a link to the log file for viewing the segmentation data after it is ingested.
        We assume that the output server is hosting neuroglancer at http://<server>:<port>/neuroglancer/
        """
        for index, output_config in enumerate(self.config_data["outputs"]):
            server = output_config["server"] # Note: Begins with http://
            uuid = output_config["uuid"]
            instance = output_config["segmentation-name"]
            
            output_box_xyz = np.array(output_config["bounding-box"])
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
            options = self.config_data["options"]
    
            node_service = retrieve_node_service( output_config["server"],
                                                  output_config["uuid"], 
                                                  self.resource_server,
                                                  self.resource_port,
                                                  self.APPNAME )
    
            info = node_service.get_typeinfo(output_config["segmentation-name"])
    
            existing_depth = int(info["Extended"]["MaxDownresLevel"])
            if options["pyramid-depth"] not in (-1, existing_depth):
                raise Exception(f"Can't set pyramid-depth to {options['pyramid-depth']}: "
                                f"Data instance '{output_config['segmentation-name']}' already existed, with depth {existing_depth}")

            max_depth = max(max_depth, existing_depth)

        return existing_depth

    def _remap_bricks(self, bricks, labelmap_config):
        if not labelmap_config["file"]:
            return bricks

        from dvidutils import LabelMapper

        # Mapping is loaded once, in driver
        if labelmap_config["file-type"] == "label-to-body":
            with open(labelmap_config["file"], 'r') as csv_file:
                rows = csv.reader(csv_file)
                all_items = chain.from_iterable(rows)
                mapping_pairs = np.fromiter(all_items, np.uint64).reshape(-1,2)
        elif labelmap_config["file-type"] == "equivalence-edges":
            mapping_pairs = BrainMapsVolume.equivalence_mapping_from_edge_csv(labelmap_config["file"])

        def remap_bricks(partition_bricks):
            domain, codomain = mapping_pairs.transpose()
            mapper = LabelMapper(domain, codomain)
            
            partition_bricks = list(partition_bricks)
            for brick in partition_bricks:
                mapper.apply_inplace(brick.volume)
            return partition_bricks
        
        # Use mapPartitions (instead of map) so LabelMapper can be constructed just once per partition
        remapped_bricks = bricks.mapPartitions(remap_bricks)
        persist_and_execute(remapped_bricks, f"Remapping bricks", logger)
        return remapped_bricks

    def _downsample_bricks(self, bricks, new_scale):
        """
        Immediately downsample the given RDD of Bricks by a factor of 2 and persist the results.
        Also, unpersist the input.
        """
        # Downsampling effectively divides grid by half (i.e. 32x32x32)
        downsampled_bricks = bricks.map(downsample_brick)
        persist_and_execute(downsampled_bricks, f"Scale {new_scale}: Downsampling", logger)
        bricks.unpersist()
        del bricks

        # Bricks are now half-size
        return downsampled_bricks


    def _consolidate_and_pad(self, bricks, scale, output_config):
        """
        Consolidate (align), and pad the given RDD of Bricks.

        scale: The pyramid scale of the data.
        
        Note: UNPERSISTS the input data and returns the new, downsampled data.
        """
        # Consolidate bricks to full size, aligned blocks (shuffles data)
        # FIXME: We should skip this if the grids happen to be aligned already.
        #        This shuffle takes ~15 minutes per tab.
        output_writing_grid = Grid(output_config["message-block-shape"], (0,0,0))
        realigned_bricks = realign_bricks_to_new_grid( output_writing_grid, bricks ).values()
        persist_and_execute(realigned_bricks, f"Scale {scale}: Shuffling bricks into alignment", logger)

        # Discard original
        bricks.unpersist()
        del bricks

        # Pad from previously-existing pyramid data.
        output_padding_grid = Grid(output_config["block-width"], (0,0,0))

        output_context = sparkdvid( self.sc, output_config["server"], output_config["uuid"], self )
        output_accessor = output_context.get_volume_accessor(output_config["segmentation-name"], scale)
        padded_bricks = realigned_bricks.map( partial(pad_brick_data_from_volume_source, output_padding_grid, output_accessor) )
        persist_and_execute(padded_bricks, f"Scale {scale}: Padding", logger)

        # Discard
        realigned_bricks.unpersist()
        del realigned_bricks

        return padded_bricks


    def _write_bricks(self, bricks, scale, output_config):
        """
        Writes partition to specified dvid.
        """
        appname = self.APPNAME

        server = output_config["server"]
        uuid = output_config["uuid"]
        block_width = output_config["block-width"]
        dataname = output_config["segmentation-name"]
        
        resource_server = self.resource_server 
        resource_port = self.resource_port 
        
        # default delimiter
        delimiter = 0
    
        @self.collect_log(lambda _: socket.gethostname() + '-write-blocks-' + str(scale))
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
                megavoxels_per_second = datacrop.size / 1e6 / put_timer.seconds
                logger.info(f"Put block {data_offset_zyx} in {put_timer.seconds:.3f} seconds ({megavoxels_per_second:.1f} Megavoxels/second)")

        with Timer() as timer:
            bricks.foreach(write_brick)
        logger.info(f"Scale {scale}: Writing to DVID took {timer.timedelta}")

    
    def _write_body_sizes( self, bricks, output_config ):
        """
        Calculate the size (in voxels) of all label bodies in the volume,
        and write the results to an HDF5 file.
        
        NOTE: For now, we implement two alternative methods of computing this result,
              for the sake of performance comparisons between the two methods.
              The method used is determined by the ['body-sizes]['method'] option.
        """
        import pandas as pd
        if not self.config_data["options"]["body-sizes"]["compute"]:
            logger.info("Skipping body size calculation.")
            return

        if self.config_data["options"]["body-sizes"]["method"] == "reduce-by-key":
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
            logger.info(f"Computing {len(body_labels)} body sizes took {timer.seconds} seconds")
        
        else: # Reduce by merging pandas dataframes
            
            #@self.collect_log(lambda *_args: 'merge_label_counts')
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
                
                #logger = logging.getLogger(__name__)
                #logger.info(f"Merging label count lists of sizes {len(labels_A)} + {len(labels_B)}"
                #            f" = {len(combined)} took {timer.seconds} seconds")
    
                return (combined.index, combined.values.astype(np.uint64))

            logger.info("Computing body sizes...")
            with Timer() as timer:
                # Two-stage repartition/reduce, to avoid doing ALL the work on the driver.
                body_labels, body_sizes = ( bricks.map( lambda br: br.volume )
                                                  .map( nonconsecutive_bincount )
                                                  .treeReduce( merge_label_counts, depth=4 ) )
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

        suffix = output_config["segmentation-name"]
        output_path = self.relpath_to_abspath(f"body-sizes-{suffix}.h5")
        with Timer() as timer:
            logger.info(f"Writing {len(body_labels)} body sizes to {output_path}")
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('labels', data=body_labels, chunks=True)
                f.create_dataset('sizes', data=body_sizes, chunks=True)
                f['total_nonzero_voxels'] = nonzero_count
        logger.info(f"Writing {len(body_sizes)} body sizes took {timer.seconds} seconds")


def downsample_brick(brick):
    # For consistency with DVID's on-demand downsampling, we suppress 0 pixels.
    assert (brick.physical_box % 2 == 0).all()
    assert (brick.logical_box % 2 == 0).all()

    # Old: Python downsampling
    # downsample_3Dlabels(brick.volume)

    # Newer: Numba downsampling
    #downsampled_volume, _ = downsample_labels_3d_suppress_zero(brick.volume, (2,2,2), brick.physical_box)

    # Even Newer: C++ downsampling (note: only works on aligned data.)
    downsampled_volume = downsample_labels(brick.volume, 2, suppress_zero=True)

    downsampled_logical_box = brick.logical_box // 2
    downsampled_physical_box = brick.physical_box // 2
    
    return Brick(downsampled_logical_box, downsampled_physical_box, downsampled_volume)
