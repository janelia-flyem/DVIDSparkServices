import copy
import json

from DVIDSparkServices.util import bb_to_slicing, bb_as_tuple
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.workflow.dvidworkflow import DVIDWorkflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 
from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray
from DVIDSparkServices.skeletonize_array import SkeletonConfigSchema, skeletonize_array

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

        # (sv_id, seg) -> (seg), that is, drop sv_id
        seg_chunks = seg_chunks.values()

        # (sv, segmentation)
        sv_and_seg_chunks = distsubvolumes.values().zip(seg_chunks)
        distsubvolumes.unpersist()

        def body_masks(sv_and_seg):
            import numpy as np
            import vigra
            subvolume, segmentation = sv_and_seg
            z1, y1, x1, z2, y2, x2 = subvolume.box_with_border
            sv_start, sv_stop = (z1, y1, x1), (z2, y2, x2)
            
            segmentation = vigra.taggedView(segmentation, 'zyx')
            consecutive_seg = np.empty_like(segmentation, dtype=np.uint32)
            _, maxlabel, bodies_to_consecutive = vigra.analysis.relabelConsecutive(segmentation, out=consecutive_seg)
            consecutive_to_bodies = { v:k for k,v in bodies_to_consecutive.items() }
            del segmentation
            
            # We don't care what the 'image' parameter is, but we have to give something
            image = consecutive_seg.view(np.float32)
            acc = vigra.analysis.extractRegionFeatures(image, consecutive_seg, features=['Coord<Minimum >', 'Coord<Maximum >', 'Count'])

            body_ids_and_masks = []
            for label in xrange(1, maxlabel+1): # Skip 0
                count = acc['Count'][label]
                min_coord = acc['Coord<Minimum >'][label].astype(int)
                max_coord = acc['Coord<Maximum >'][label].astype(int)
                box_local = np.array((min_coord, 1+max_coord))
                
                mask = (consecutive_seg[bb_to_slicing(*box_local)] == label).view(np.uint8)
                compressed_mask = CompressedNumpyArray(mask)

                body_id = consecutive_to_bodies[label]
                box_global = box_local + sv_start

                # Only keep segments that are big enough OR touch the subvolume border.
                if count >= config["options"]["minimum-segment-size"] \
                or (box_global[0] == sv_start).any() \
                or (box_global[1] == sv_stop).any():
                    body_ids_and_masks.append( (body_id, (bb_as_tuple(box_global), compressed_mask)) )
            
            return body_ids_and_masks


        # (sv, seg) -> (body_id, (box, mask))
        body_ids_and_masks = sv_and_seg_chunks.flatMap( body_masks )


        def combine_masks( boxes_and_compressed_masks ):
            import numpy as np
            boxes, _compressed_masks = zip(*boxes_and_compressed_masks)
            boxes = np.asarray(boxes)
            assert boxes.shape == (len(boxes_and_compressed_masks), 2,3)
            
            combined_box_start = boxes[:, 0, :].min(axis=0)
            combined_box_stop  = boxes[:, 1, :].max(axis=0)
            
            combined_shape = combined_box_stop - combined_box_start
            combined_mask = np.zeros( combined_shape, dtype=np.uint8 )
            
            for (box_global, compressed_mask) in boxes_and_compressed_masks:
                mask = compressed_mask.deserialize()
                box_combined = box_global - combined_box_start
                combined_mask[bb_to_slicing(*box_combined)] |= mask

            if combined_mask.sum() < config["options"]["minimum-segment-size"]:
                # 'None' results will be filtered out. See below.
                combined_mask = None

            return ( (combined_box_start, combined_box_stop), combined_mask )


        def combine_and_skeletonize(boxes_and_compressed_masks):
            (combined_box_start, _combined_box_stop), combined_mask = combine_masks( boxes_and_compressed_masks )
            if combined_mask is None:
                return None
            tree = skeletonize_array(combined_mask, config["skeleton-config"])
            tree.translate(*combined_box_start[::-1]) # Pass x,y,z, not z,y,x
            swc_contents = tree.toString()
            return swc_contents

        
        # (body_id, (box, mask))
        #   --> (body_id, [(box, mask), (box, mask), (box, mask), ...])
        #     --> (body_id, swc_contents)
        grouped_body_ids_and_masks = body_ids_and_masks.groupByKey()
        body_ids_and_skeletons = grouped_body_ids_and_masks.mapValues(combine_and_skeletonize)

        def post_swc_to_dvid( body_id_and_swc_contents ):
            body_id, swc_contents = body_id_and_swc_contents
            if swc_contents is None:
                return
        
            node_service = retrieve_node_service(config["dvid-info"]["dvid-server"],
                                                 config["dvid-info"]["uuid"],
                                                 config["options"]["resource-server"],
                                                 config["options"]["resource-port"])

            node_service.create_keyvalue("skeletons")
            node_service.put("skeletons", "{}_swc".format(body_id), swc_contents)

        body_ids_and_skeletons.foreach(post_swc_to_dvid)
