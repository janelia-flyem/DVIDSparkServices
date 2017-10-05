import copy
import json
import datetime
import logging
from functools import partial

import numpy as np

from DVIDSparkServices.util import MemoryWatcher
from DVIDSparkServices.io_util.brick import Grid
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 
from DVIDSparkServices.skeletonize_array import SkeletonConfigSchema, skeletonize_array
from DVIDSparkServices.reconutils.morpho import object_masks_for_labels, assemble_masks
from DVIDSparkServices.sparkdvid import sparkdvid
from DVIDSparkServices.dvid.metadata import is_node_locked

from .common_schemas import SegmentationVolumeSchema

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
            "default": 1 # 1 means "no downsampling, if possible"
                         # 2 means "downsample only by 2x, if possible"
                         # 3 means "downsample only by 3x, if possible [note: NOT (2^3)x]"
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
        # ?! set number of cpus per task to 2 (make dynamic?)
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
        d = self.config_data["dvid-info"]
        if is_node_locked(d["server"], d["uuid"]):
            raise RuntimeError(f"Can't write skeletons: The node you specified ({d['server']} / {d['uuid']}) is locked.")
        
        bricks, _bounding_box_zyx, _input_grid = self._partition_input()
        
        # (vol_part, seg) -> (body_id, (box, mask))
        body_ids_and_masks = bricks.flatMap( partial(body_masks, self.config_data) )

        # (body_id, (box, mask))
        #   --> (body_id, [(box, mask), (box, mask), (box, mask), ...])
        grouped_body_ids_and_masks = body_ids_and_masks.groupByKey()

        #     --> (body_id, swc_contents)
        body_ids_and_skeletons = grouped_body_ids_and_masks.map( partial(combine_and_skeletonize, self.config_data) )

        # Write
        body_ids_and_skeletons.foreach( partial(post_swc_to_dvid, self.config_data) )

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
    boxes, compressed_masks = zip(*boxes_and_compressed_masks)
    boxes = np.asarray(boxes)
    assert boxes.shape == (len(boxes_and_compressed_masks), 2,3)
    
    # Important to use a generator expression here (not a list comprehension)
    # to avoid excess RAM usage from many uncompressed masks.
    masks = ( compressed.deserialize() for compressed in compressed_masks )

    # Note that combined_mask_downsampled may be 'None', which is handled below.
    combined_box, combined_mask_downsampled, chosen_downsample_factor = \
        assemble_masks( boxes,
                        masks,
                        config["options"]["downsample-factor"],
                        config["options"]["minimum-segment-size"],
                        config["options"]["max-skeletonization-volume"] )

    return (combined_box, combined_mask_downsampled, chosen_downsample_factor)


def combine_and_skeletonize(config, ids_and_boxes_and_compressed_masks):
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

        memory_watcher.log_increase(logger, logging.INFO,
                                    'After mask assembly (combined_mask.shape: {} downsample_factor: {})'
                                    .format(combined_mask.shape, downsample_factor))
        
        tree = skeletonize_array(combined_mask, config["skeleton-config"])
        tree.rescale(downsample_factor, downsample_factor, downsample_factor, True)
        tree.translate(*combined_box_start.astype(np.float64)[::-1]) # Pass x,y,z, not z,y,x

        memory_watcher.log_increase(logger, logging.INFO, 'After skeletonization')

        del combined_mask
        memory_watcher.log_increase(logger, logging.INFO, 'After mask deletion')
        
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
        memory_watcher.log_increase(logger, logging.INFO, 'After tree deletion')

        return (body_id, swc_contents)

def post_swc_to_dvid(config, body_id_and_swc_contents ):
    body_id, swc_contents = body_id_and_swc_contents
    if swc_contents is None:
        return

    node_service = retrieve_node_service(config["dvid-info"]["server"],
                                         config["dvid-info"]["uuid"],
                                         config["options"]["resource-server"],
                                         config["options"]["resource-port"])

    skeletons_kv_instance = config["dvid-info"]["skeletons-destination"]
    if not skeletons_kv_instance:
        skeletons_kv_instance = config["dvid-info"]["segmentation-name"] + '_skeletons'
    node_service.create_keyvalue(skeletons_kv_instance)
    node_service.put(skeletons_kv_instance, "{}_swc".format(body_id), swc_contents.encode('utf-8'))

