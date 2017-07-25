import copy
import json
import datetime
import logging

from DVIDSparkServices.util import MemoryWatcher
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.workflow.dvidworkflow import DVIDWorkflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 
from DVIDSparkServices.skeletonize_array import SkeletonConfigSchema, skeletonize_array
from DVIDSparkServices.reconutils.morpho import object_masks_for_labels, assemble_masks

class CreateSkeletons(DVIDWorkflow):
    DvidInfoSchema = \
    {
      "type": "object",
      "properties": {
        "dvid-server": {
          "description": "location of DVID server",
          "type": "string",
          "minLength": 1,
          "property": "dvid-server"
        },
        "uuid": {
          "description": "version node to retrieve the segmentation from",
          "type": "string",
          "minLength": 1
        },
        "roi": {
          "description": "region of interest to skeletonize",
          "type": "string",
          "minLength": 1
        },
        "segmentation": {
          "description": "location of segmentation",
          "type": "string",
          "minLength": 1
        },
        "skeletons-destination": {
            "description": "name of key-value instance to store the skeletons",
            "type": "string",
            "minLength": 1
        }
      },
      "required": ["dvid-server", "uuid", "roi", "segmentation"],
      "additionalProperties": False
    }
    
    SkeletonWorkflowOptionsSchema = copy.copy(Workflow.OptionsSchema)
    SkeletonWorkflowOptionsSchema["properties"].update(
    {
      "chunk-size": {
        "description": "Size of blocks to process independently (and then stitched together).",
        "type": "integer",
        "default": 512
      },
      "minimum-segment-size": {
        "description": "Size of blocks to process independently (and then stitched together).",
        "type": "integer",
        "default": 1000
      },
      "downsample-factor": {
        "description": "Factor by which to downsample bodies before skeletonization. (0 means 'choose automatically')",
        "type": "integer",
        "default": 0 # 0 means "auto", based on RAM.
      },
      "max-skeletonization-volume": {
        "description": "The above downsample-factor will be overridden if the body would still be too large to skeletonize, as defined by this setting.",
        "type": "number",
        "default": 1e9 # 1 GB
      }
    })
    
    Schema = \
    {
      "$schema": "http://json-schema.org/schema#",
      "title": "Service to create skeletons from segmentation",
      "type": "object",
      "properties": {
        "dvid-info": DvidInfoSchema,
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

    def execute(self):
        from pyspark import StorageLevel

        config = self.config_data
        chunksize = config["options"]["chunk-size"]

        # grab ROI subvolumes and find neighbors
        distsubvolumes = self.sparkdvid_context.parallelize_roi( config["dvid-info"]["roi"],
                                                                 chunksize,
                                                                 0,
                                                                 True )
        distsubvolumes.persist(StorageLevel.MEMORY_AND_DISK_SER)

        # grab seg chunks: (sv_id, seg)
        seg_chunks = self.sparkdvid_context.map_labels64( distsubvolumes,
                                                          config["dvid-info"]["segmentation"],
                                                          0,
                                                          config["dvid-info"]["roi"])

        # (sv_id, seg) -> (seg) ...that is, drop sv_id
        seg_chunks = seg_chunks.values()

        # (sv, segmentation)
        sv_and_seg_chunks = distsubvolumes.values().zip(seg_chunks)
        distsubvolumes.unpersist()

        def body_masks(sv_and_seg):
            """
            Produce a binary label mask for each object (except label 0).
            Return a list of those masks, along with their bounding boxes (expressed in global coordinates).
            
            For more details, see documentation for object_masks_for_labels()

            Returns:
                List of tuples: [(label_id, (mask_bounding_box, mask)), 
                                 (label_id, (mask_bounding_box, mask)), ...]
            """
            subvolume, segmentation = sv_and_seg
            z1, y1, x1, z2, y2, x2 = subvolume.box_with_border
            sv_start, sv_stop = (z1, y1, x1), (z2, y2, x2)

            return object_masks_for_labels( segmentation,
                                            (sv_start, sv_stop),
                                            config["options"]["minimum-segment-size"], 
                                            always_keep_border_objects=True,
                                            compress_masks=True )

        # (sv, seg) -> (body_id, (box, mask))
        body_ids_and_masks = sv_and_seg_chunks.flatMap( body_masks )


        def combine_masks( body_id, boxes_and_compressed_masks ):
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
            import numpy as np
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


        @self.collect_log(lambda arg: 'Body-{:07}'.format(arg[0]))
        def combine_and_skeletonize(ids_and_boxes_and_compressed_masks):
            """
            Given a list of binary masks and corresponding bounding
            boxes, assemble them all into a combined binary mask.
            
            Then convert that combined mask into a skeleton (SWC string).
            """
            logger = logging.getLogger(__name__ + '.combine_and_skeletonize')
            with MemoryWatcher() as memory_watcher:
                body_id, boxes_and_compressed_masks = ids_and_boxes_and_compressed_masks
                (combined_box_start, _combined_box_stop), combined_mask, downsample_factor = combine_masks( body_id, boxes_and_compressed_masks )

                if combined_mask is None:
                    return (body_id, None)
    
                memory_watcher.log_increase(logger, logging.INFO,
                                            'After mask assembly (combined_mask.shape: {} downsample_factor: {})'
                                            .format(combined_mask.shape, downsample_factor))
                
                tree = skeletonize_array(combined_mask, config["skeleton-config"])
                tree.rescale(downsample_factor, downsample_factor, downsample_factor, True)
                tree.translate(*combined_box_start[::-1]) # Pass x,y,z, not z,y,x

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

        
        # (body_id, (box, mask))
        #   --> (body_id, [(box, mask), (box, mask), (box, mask), ...])
        #     --> (body_id, swc_contents)
        grouped_body_ids_and_masks = body_ids_and_masks.groupByKey()
        body_ids_and_skeletons = grouped_body_ids_and_masks.map(combine_and_skeletonize)

        def post_swc_to_dvid( body_id_and_swc_contents ):
            body_id, swc_contents = body_id_and_swc_contents
            if swc_contents is None:
                return
        
            node_service = retrieve_node_service(config["dvid-info"]["dvid-server"],
                                                 config["dvid-info"]["uuid"],
                                                 config["options"]["resource-server"],
                                                 config["options"]["resource-port"])

            skeletons_kv_instance = config["dvid-info"]["skeletons-destination"]
            node_service.create_keyvalue(skeletons_kv_instance)
            node_service.put(skeletons_kv_instance, "{}_swc".format(body_id), swc_contents)

        body_ids_and_skeletons.foreach(post_swc_to_dvid)
