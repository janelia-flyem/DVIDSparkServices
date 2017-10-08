import os
import copy
import json
import signal
import datetime
import logging
from functools import partial

import numpy as np
import requests

from dvid_resource_manager.client import ResourceManagerClient

from DVIDSparkServices.util import MemoryWatcher, Timer, persist_and_execute
from DVIDSparkServices.io_util.brick import Grid
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 
from DVIDSparkServices.skeletonize_array import SkeletonConfigSchema, skeletonize_array
from DVIDSparkServices.reconutils.morpho import object_masks_for_labels, assemble_masks
from DVIDSparkServices.sparkdvid import sparkdvid
from DVIDSparkServices.dvid.metadata import is_node_locked

from .common_schemas import SegmentationVolumeSchema

from multiprocessing import Pool, TimeoutError

logger = logging.getLogger(__name__)

class CreateSkeletons(Workflow):
    DvidInfoSchema = copy.deepcopy(SegmentationVolumeSchema)
    DvidInfoSchema["properties"].update(
    {
        "skeletons-destination": {
            "description": "Name of key-value instance to store the skeletons. "
                           "By convention, this should usually be {segmentation-name}_skeletons, "
                           "which will be used by default if you don't provide this setting.",
            "type": "string",
            "default": ""
        }
    })

    SkeletonWorkflowOptionsSchema = copy.copy(Workflow.OptionsSchema)
    SkeletonWorkflowOptionsSchema["properties"].update(
    {
        "minimum-segment-size": {
            "description": "Segments smaller than this voxel count will not be skeletonized",
            "type": "number",
            "default": 1e6
        },
        "downsample-factor": {
            "description": "Minimum factor by which to downsample bodies before skeletonization. "
                           "NOTE: If the object is larger than max-skeletonization-volume, even after "
                           "downsampling, then it will be downsampled even further before skeletonization. "
                           "The comments in the generated SWC file will indicate the final-downsample-factor.",
            "type": "integer",
            "default": 1
        },
        "max-skeletonization-volume": {
            "description": "The above downsample-factor will be overridden if the body would still "
                           "be too large to skeletonize, as defined by this setting.",
            "type": "number",
            "default": 1e9 # 1 GB max
        }
    })
    
    Schema = \
    {
      "$schema": "http://json-schema.org/schema#",
      "title": "Service to create skeletons from segmentation",
      "type": "object",
      "required": ["dvid-info"],
      "properties": {
        "dvid-info": SegmentationVolumeSchema,
        "skeleton-config": SkeletonConfigSchema,
        "options" : SkeletonWorkflowOptionsSchema
      }
    }

    @staticmethod
    def dumpschema():
        return json.dumps(CreateSkeletons.Schema)

    def __init__(self, config_filename):
        super(CreateSkeletons, self).__init__(config_filename, CreateSkeletons.dumpschema(), "CreateSkeletons")

        # Prepend 'http://' if necessary.
        dvid_info = self.config_data['dvid-info']
        if not dvid_info['server'].startswith('http'):
            dvid_info['server'] = 'http://' + dvid_info['server']

        # create spark dvid context
        self.sparkdvid_context = sparkdvid.sparkdvid(self.sc,
                self.config_data["dvid-info"]["server"],
                self.config_data["dvid-info"]["uuid"], self)

    def execute(self):
        self._init_skeletons_instance()

        bricks, _bounding_box_zyx, _input_grid = self._partition_input()
        persist_and_execute(bricks, "Downloading segmentation", logger)
        
        # brick -> (body_id, (box, mask, count))
        body_ids_and_masks = bricks.flatMap( partial(body_masks, self.config_data) )
        persist_and_execute(body_ids_and_masks, "Computing brick-local masks", logger)
        bricks.unpersist()
        del bricks

        # (body_id, (box, mask, count))
        #   --> (body_id, [(box, mask, count), (box, mask, count), (box, mask, count), ...])
        grouped_body_ids_and_masks = body_ids_and_masks.groupByKey()
        persist_and_execute(grouped_body_ids_and_masks, "Grouping masks by body id", logger)
        body_ids_and_masks.unpersist()
        del body_ids_and_masks

        # (Same RDD contents, but without small bodies)
        grouped_large_body_ids_and_masks = grouped_body_ids_and_masks.filter( partial(is_combined_object_large_enough, self.config_data) )
        persist_and_execute(grouped_large_body_ids_and_masks, "Filtering masks by size", logger)
        grouped_body_ids_and_masks.unpersist()
        del grouped_body_ids_and_masks

        #     --> (body_id, swc_contents)
        body_ids_and_skeletons = grouped_large_body_ids_and_masks.map( partial(combine_and_skeletonize, self.config_data) )
        persist_and_execute(body_ids_and_skeletons, "Aggregating masks and computing skeletons", logger)
        grouped_large_body_ids_and_masks.unpersist()
        del grouped_large_body_ids_and_masks

        # Write
        with Timer() as timer:
            body_ids_and_skeletons.foreach( partial(post_swc_to_dvid, self.config_data) )
        logger.info(f"Writing skeletons to DVID took {timer.seconds}")

    def _init_skeletons_instance(self):
        dvid_info = self.config_data["dvid-info"]
        options = self.config_data["options"]
        if is_node_locked(dvid_info["server"], dvid_info["uuid"]):
            raise RuntimeError(f"Can't write skeletons: The node you specified ({dvid_info['server']} / {dvid_info['uuid']}) is locked.")

        if not dvid_info["skeletons-destination"]:
            # Edit the config to supply a default key/value instance name
            dvid_info["skeletons-destination"] = dvid_info["segmentation-name"]+ '_skeletons'

        node_service = retrieve_node_service( dvid_info["server"],
                                              dvid_info["uuid"],
                                              options["resource-server"],
                                              options["resource-port"] )

        node_service.create_keyvalue(dvid_info["skeletons-destination"])

    def _partition_input(self):
        """
        Map the input segmentation
        volume from DVID into an RDD of Bricks,
        using the config's bounding-box setting for the full volume region,
        using the 'message-block-shape' as the partition size.

        Returns: (RDD, bounding_box_zyx, partition_shape_zyx)
            where:
                - RDD is (volumePartition, data)
                - bounding box is tuple (start_zyx, stop_zyx)
                - partition_shape_zyx is a tuple
            
        """
        dvid_info = self.config_data["dvid-info"]
        assert dvid_info["service-type"] == "dvid", \
            "Only DVID sources are supported right now."

        # repartition to be z=blksize, y=blksize, x=runlength
        brick_shape_zyx = dvid_info["message-block-shape"][::-1]
        input_grid = Grid(brick_shape_zyx, (0,0,0))

        bounding_box_xyz = np.array(dvid_info["bounding-box"])
        bounding_box_zyx = bounding_box_xyz[:,::-1]

        # Aim for 2 GB RDD partitions
        target_partition_size_voxels = (2 * 2**30 // np.uint64().nbytes)
        bricks = self.sparkdvid_context.parallelize_bounding_box( dvid_info['segmentation-name'],
                                                                  bounding_box_zyx,
                                                                  input_grid,
                                                                  target_partition_size_voxels )

        return bricks, bounding_box_zyx, input_grid


def body_masks(config, brick):
    """
    Produce a binary label mask for each object (except label 0).
    Return a list of those masks, along with their bounding boxes (expressed in global coordinates).
    
    For more details, see documentation for object_masks_for_labels()

    Returns:
        List of tuples: [(label_id, (mask_bounding_box, mask)), 
                         (label_id, (mask_bounding_box, mask)), ...]
    """
    return object_masks_for_labels( brick.volume,
                                    brick.physical_box,
                                    config["options"]["minimum-segment-size"],
                                    always_keep_border_objects=True,
                                    compress_masks=True )


def is_combined_object_large_enough(config, ids_and_boxes_and_compressed_masks):
    """
    Given a tuple of the form:
    
        ( body_id,
          [(box, mask, count), (box, mask, count), (box, mask, count), ...] )
    
    Return True if the combined counts is large enough (according to the config).
    """
    _body_id, boxes_and_compressed_masks = ids_and_boxes_and_compressed_masks
    _boxes, _compressed_masks, counts = zip(*boxes_and_compressed_masks)
    return (sum(counts) >= config["options"]["minimum-segment-size"])


def combine_masks(config, body_id, boxes_and_compressed_masks ):
    """
    Given a list of binary masks and corresponding bounding
    boxes, assemble them all into a combined binary mask.
    
    To save RAM, the data can be downsampled while it is added to the combined mask,
    resulting in a downsampled final mask.

    For more details, see documentation for assemble_masks().

    Returns: (combined_bounding_box, combined_mask, downsample_factor)

        where:
            combined_bounding_box:
                the bounding box of the returned mask,
                in NON-downsampled coordinates: ((z0,y0,x0), (z1,y1,x1))
            
            combined_mask:
                the full downsampled combined mask,
            
            downsample_factor:
                The chosen downsampling factor if using 'auto' downsampling,
                otherwise equal to the downsample_factor you passed in.

    """
    boxes, compressed_masks, _counts = zip(*boxes_and_compressed_masks)

    boxes = np.asarray(boxes)
    assert boxes.shape == (len(boxes_and_compressed_masks), 2,3)
    
    # Important to use a generator expression here (not a list comprehension)
    # to avoid excess RAM usage from many uncompressed masks.
    masks = ( compressed.deserialize() for compressed in compressed_masks )

    # In theory this can return 'None' for the combined mask if the object is too small,
    # but we already filtered out 
    combined_box, combined_mask_downsampled, chosen_downsample_factor = \
        assemble_masks( boxes,
                        masks,
                        config["options"]["downsample-factor"],
                        config["options"]["minimum-segment-size"],
                        config["options"]["max-skeletonization-volume"] )

    return (combined_box, combined_mask_downsampled, chosen_downsample_factor)


def execute_in_subprocess(timeout=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # TODO: Use initializer to handle logging?
            pool = Pool(1)
            pid = pool.apply(os.getpid)
            fut = pool.apply_async(func, args, kwargs)
            try:
                return fut.get(timeout)
            except TimeoutError:
                # Make sure it's dead
                os.kill(pid, signal.SIGTERM)
                os.kill(pid, signal.SIGKILL)
                raise
        return wrapper
    return decorator

                
def combine_and_skeletonize(config, ids_and_boxes_and_compressed_masks):
    """
    Executes _combine_and_skeletonize() in a subprocess,
    but times out after 60 seconds.
    """
    f = execute_in_subprocess(60.0)(_combine_and_skeletonize)
    try:
        return f(config, ids_and_boxes_and_compressed_masks)
    except TimeoutError:
        body_id, boxes_and_compressed_masks = ids_and_boxes_and_compressed_masks
        boxes, _compressed_masks, counts = zip(*boxes_and_compressed_masks)

        total_count = sum(counts)
        boxes = np.asarray(boxes)
        combined_box = np.zeros((2,3), dtype=np.int64)
        combined_box[0] = boxes[:, 0, :].min(axis=0)
        combined_box[1] = boxes[:, 1, :].max(axis=0)
        
        logger.error(f"Failed to skeletonize body: id={body_id} size={total_count} box={combined_box.tolist()}")
        return (body_id, None)

def _combine_and_skeletonize(config, ids_and_boxes_and_compressed_masks):
    """
    Given a list of binary masks and corresponding bounding
    boxes, assemble them all into a combined binary mask.
    
    Then convert that combined mask into a skeleton (SWC string).
    """
    logger = logging.getLogger(__name__ + '.combine_and_skeletonize')

    with MemoryWatcher() as memory_watcher:
        body_id, boxes_and_compressed_masks = ids_and_boxes_and_compressed_masks
        (combined_box_start, _combined_box_stop), combined_mask, downsample_factor = combine_masks( config, body_id, boxes_and_compressed_masks )

        if combined_mask is None:
            return (body_id, None)

        memory_watcher.log_increase(logger, logging.DEBUG,
                                    'After mask assembly (combined_mask.shape: {} downsample_factor: {})'
                                    .format(combined_mask.shape, downsample_factor))
        
        tree = skeletonize_array(combined_mask, config["skeleton-config"])
        tree.rescale(downsample_factor, downsample_factor, downsample_factor, True)
        tree.translate(*combined_box_start.astype(np.float64)[::-1]) # Pass x,y,z, not z,y,x

        memory_watcher.log_increase(logger, logging.DEBUG, 'After skeletonization')

        del combined_mask
        memory_watcher.log_increase(logger, logging.DEBUG, 'After mask deletion')
        
        # Also show which downsample factor was actually chosen
        config_copy = copy.deepcopy(config)
        config_copy["options"]["(final-downsample-factor)"] = downsample_factor
        
        config_comment = json.dumps(config_copy, sort_keys=True, indent=4, separators=(',', ': '))
        config_comment = "\n".join( "# " + line for line in config_comment.split("\n") )
        config_comment += "\n\n"
        
        swc_contents =  "# {:%Y-%m-%d %H:%M:%S}\n".format(datetime.datetime.now())
        swc_contents += "# Generated by the 'CreateSkeletons' workflow,\n"
        swc_contents += "# using the following configuration:\n"
        swc_contents += "# \n"
        swc_contents += config_comment + tree.toString()

        del tree
        memory_watcher.log_increase(logger, logging.DEBUG, 'After tree deletion')

        return (body_id, swc_contents)

def post_swc_to_dvid(config, body_id_and_swc_contents):
    """
    Send the given SWC as a key/value pair to DVID.
    
    TODO: There is a tiny bit of extra overhead here due to the fact
          that we are creating a new requests.Session() and ResourceManagerClient
          for every SWC we write.
          If we want, we could refactor this to be used with mapPartitions(),
          which would allow those objects to be initialized only once per partition,
          and then shared for all SWCs in the partition.
    """
    body_id, swc_contents = body_id_and_swc_contents
    if swc_contents is None:
        return

    swc_contents = swc_contents.encode('utf-8')

    dvid_server = config["dvid-info"]["server"]
    uuid = config["dvid-info"]["uuid"]
    instance = config["dvid-info"]["skeletons-destination"]

    if not config["options"]["resource-server"]:
        # No throttling.
        requests.post(f'{dvid_server}/api/node/{uuid}/{instance}/key/{body_id}_swc', swc_contents)
    else:
        resource_client = ResourceManagerClient( config["options"]["resource-server"],
                                                 config["options"]["resource-port"] )
    
        with resource_client.access_context(dvid_server, False, 1, len(swc_contents)):
            requests.post(f'{dvid_server}/api/node/{uuid}/{instance}/key/{body_id}_swc', swc_contents)
