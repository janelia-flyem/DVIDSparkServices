"""Defines base class for segmentation plugins."""
import os
import json
import importlib
import textwrap
from functools import partial
import numpy as np

from quilted.h5blockstore import H5BlockStore

from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray
from DVIDSparkServices.json_util import validate_and_inject_defaults
from DVIDSparkServices.auto_retry import auto_retry
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 

class Segmentor(object):
    """
    Contains functionality for segmenting large datasets.
    
    It implements a very crude watershed algorithm by default.

    This class's segment() functionality can be customized in one of two ways:
    
    1. Override segment() directly in a subclass.
    2. Instead of implementing a subclass, use the default segment() 
        implementation, which breaks the segmentation problem into separate
        steps, each of which can be overridden via the config file: 
        - "background-mask"
        - "predict-voxels"
        - "create-supervoxels"
        - "agglomerate-supervoxels"

    The other functions involve stitching the
    subvolumes and performing other RDD and DVID manipulations.

    Plugins such as this class (or subclasses of it) must reside in DVIDSparkServices.reconutils.plugins.
    """

    SegmentorSchema = textwrap.dedent("""\
        {
          "$schema": "http://json-schema.org/schema#",
          "title": "Plugin configuration options for the Segmentor base class",
          "type": "object",
        """
        
        # This subschema is for referencing custom functions for various steps in the pipeline.
        # Example: { "function": "skimage.filters.threshold_otsu",
        #            "parameters": {"nbins": 256 } }
        """          
          "definitions" : {
            "custom-function" : {
              "description": "Configuration for a custom function to replace a segmentation step.",
              "type": "object",
              "properties": {
                "function": {
                  "description": "Name of a python function.",
                  "type": "string",
                  "minLength": 1
                },
                "parameters" : {
                  "description": "Arbitrary dict of parameters.",
                  "type": "object",
                  "default" : {}
                }
              },
              "additionalProperties": false
            }
          },
        """

        # Functions for each segmentation step are provided directly in the json config.
        # If not, these defaults from this schema are used.
        """
          "properties": {
            "background-mask"         : { "$ref": "#/definitions/custom-function",
                                          "default": { "function": "DVIDSparkServices.reconutils.misc.find_large_empty_regions" } },
            "predict-voxels"          : { "$ref": "#/definitions/custom-function",
                                          "default": { "function": "DVIDSparkServices.reconutils.misc.naive_membrane_predictions" } },
            "create-supervoxels"      : { "$ref": "#/definitions/custom-function",
                                          "default": { "function": "DVIDSparkServices.reconutils.misc.seeded_watershed" } },
            "agglomerate-supervoxels" : { "$ref": "#/definitions/custom-function",
                                          "default": { "function": "DVIDSparkServices.reconutils.misc.noop_aggolmeration" } },
            "preserve-bodies": {
              "description": "Configuration to describe which bodies to preserve instead of overwriting with new segmentation.",
              "type": "object",
              "properties": {
                "dvid-server": {
                  "description": "DVID server from which to extract preserved bodies",
                  "type": "string",
                  "minLength": 1,
                  "property": "dvid-server"
                },
                "uuid": {
                  "description": "version node from which to extract preserved bodies",
                  "type": "string",
                  "minLength": 1
                },
                "segmentation-name": {
                  "description": "labels instance from which to extract preserved bodies",
                  "type": "string",
                  "minLength": 1
                },
                "bodies": {
                  "type": "array",
                  "items": { "type": "number" },
                  "minItems": 0,
                  "uniqueItems": true,
                  "default": []
                }
              },
              "additionalProperties": false,
              "default": {}
            }
          },
          "additionalProperties": false,
          "default": {}
        }
        """)

    def __init__(self, context, workflow_config):
        self.context = context
        self.segmentor_config = workflow_config["options"]["segmentor"]["configuration"]

        segmentor_schema = json.loads(Segmentor.SegmentorSchema)
        validate_and_inject_defaults(self.segmentor_config, segmentor_schema)        

        stitch_modes = { "none" : 0, "conservative" : 1, "medium" : 2, "aggressive" : 3 }
        self.stitch_mode = stitch_modes[ workflow_config["options"]["stitch-algorithm"] ]
        self.labeloffset = 0
        if "label-offset" in workflow_config["options"]:
            self.labeloffset = int(workflow_config["options"]["label-offset"])


        # save masked bodies
        self.pdconf = None
        self.preserve_bodies = None
        if self.segmentor_config["preserve-bodies"]["bodies"]:
            self.pdconf = self.segmentor_config["preserve-bodies"]
            self.preserve_bodies = set(self.pdconf["bodies"])


    def segment(self, gray_chunks, pred_checkpoint_dir="", enable_pred_rollback=False, seg_checkpoint_dir="", enable_seg_rollback=False):
        """Top-level pipeline (can overwrite) -- gray RDD => label RDD.

        Defines a segmentation workflow consisting of voxel prediction,
        watershed, and agglomeration.  One can overwrite specific functions
        or the entire workflow as long as RDD input and output constraints
        are statisfied.  RDD transforms should preserve the partitioner -- 
        subvolume id is the key.

        Args:
            gray_chunks (RDD) = (subvolume key, (subvolume, numpy grayscale))
            checkpoint_dir (str) = enables checkpointing for boundary prediction if not empty
            enable_pred_rollback (boolean) = enables rollback of prediction if available
        Returns:
            segmentation (RDD) as (subvolume key, (subvolume, numpy compressed array))

        """
        # Compute mask of background area that can be skipped (if any)
        gray_mask_chunks = self.compute_background_mask(gray_chunks)
        
        # run voxel prediction (default: grayscale is boundary)
        pred_chunks = self.predict_voxels(gray_mask_chunks, pred_checkpoint_dir, enable_pred_rollback)
        
        # run watershed from voxel prediction (default: seeded watershed)
        sp_chunks = self.create_supervoxels(pred_chunks)

        # run agglomeration (default: none)
        segmentation = self.agglomerate_supervoxels(sp_chunks, seg_checkpoint_dir, enable_seg_rollback)

        return segmentation 

    def _get_segmentation_function(self, segmentation_step):
        """
        Read the user's config and return the image processing
        function specified for the given segmentation step.

        If the user provided a dict of parameters, then they will be
        bound into the returned function as keyword args.
        """
        full_function_name = self.segmentor_config[segmentation_step]["function"]
        module_name = '.'.join(full_function_name.split('.')[:-1])
        function_name = full_function_name.split('.')[-1]
        module = importlib.import_module(module_name)
        
        parameters = self.segmentor_config[segmentation_step]["parameters"]
        return partial( getattr(module, function_name), **parameters )
        
    def compute_background_mask(self, gray_chunks):
        """
        Detect large 'background' regions that lie outside the area of interest for segmentation.
        """
        mask_function = self._get_segmentation_function('background-mask')
        def _execute_for_chunk(gray_chunk):
            (subvolume, gray) = gray_chunk

            # Call the (custom) function
            mask = mask_function(gray)
            
            if mask is None:
                return (subvolume, gray, None)
            else:
                assert mask.dtype == np.bool, "Mask array should be boolean"
                assert mask.ndim == 3
            return (subvolume, gray, CompressedNumpyArray(mask))

        return gray_chunks.mapValues(_execute_for_chunk)

    def predict_voxels(self, gray_mask_chunks, checkpoint_dir=None, enable_pred_rollback=False):
        """Create a dummy placeholder boundary channel from grayscale.

        Takes an RDD of grayscale numpy volumes and produces
        an RDD of predictions (z,y,x).
        """
        prediction_function = self._get_segmentation_function('predict-voxels')
        def _execute_for_chunk(gray_mask_chunk):

            (subvolume, gray, mask_compressed) = gray_mask_chunk
            roi = subvolume.roi
            border = subvolume.border
            block_bounds_zyx = ( (roi.z1 - border, roi.y1 - border, roi.x1 - border),
                                 (roi.z2 + border, roi.y2 + border, roi.x2 + border) )

            # Can we load the predictions from on-disk cache?
            if enable_pred_rollback and checkpoint_dir:
                try:
                    block_store = H5BlockStore(checkpoint_dir, mode='r')
                except H5BlockStore.StoreDoesNotExistError:
                    pass
                else:
                    try:
                        # We don't actually know how many channels are in the predictions,
                        # but we need to figure it out in order to get the right block.
                        # Get the number of channels by inspecting the first block
                        block_bounds_list = block_store.get_block_bounds_list()
                        num_channels = block_bounds_list[0][1][-1] # first block, upper bound, last index
                    except IndexError:
                        pass
                    else:
                        try:
                            # Return the predictions we found on-disk (if it's there)
                            block_bounds_zyxc = ( block_bounds_zyx[0] + (0,),
                                                  block_bounds_zyx[1] + (num_channels,) )
                            prediction_block = block_store.get_block( block_bounds_zyxc )
                            return ( subvolume, gray, CompressedNumpyArray(prediction_block[:]), mask_compressed )
                        except H5BlockStore.MissingBlockError:
                            pass
            
            # mask can be None
            assert mask_compressed is None or isinstance( mask_compressed, CompressedNumpyArray )
            mask = mask_compressed and mask_compressed.deserialize()

            # Call the (custom) function
            predictions = prediction_function(gray, mask)
            assert predictions.ndim == 4, "Predictions volume should be 4D: z-y-x-c"
            assert predictions.dtype == np.float32, "Predictions should be float32"
            assert predictions.shape[:3] == tuple(np.array(block_bounds_zyx[1]) - block_bounds_zyx[0]), \
                "predictions have unexpected shape: {}, expected block_bounds: {}".format( predictions.shape, block_bounds_zyx )

            if checkpoint_dir:
                num_channels = predictions.shape[-1]
                block_bounds_zyxc = ( block_bounds_zyx[0] + (0,),
                                      block_bounds_zyx[1] + (num_channels,) )
                block_store = H5BlockStore(checkpoint_dir, mode='a', axes='zyxc', dtype=np.float32)
                block = block_store.get_block( block_bounds_zyxc )
                block[:] = predictions

            return ( subvolume, gray, CompressedNumpyArray(predictions), mask_compressed )

        return gray_mask_chunks.mapValues(_execute_for_chunk)

    def create_supervoxels(self, prediction_chunks):
        """Performs watershed based on voxel prediction.

        Takes an RDD of numpy volumes with multiple prediction
        channels and produces an RDD of label volumes.  A mask must
        be provided indicating which parts of the volume should
        have a supervoxels (true to keep, false to ignore).  Currently,
        this is a seeded watershed, an option to use the distance transform
        for the watershed is forthcoming.  There are 3 hidden options
        that can be specified:
        
        Args:
            prediction_chunks (RDD) = (subvolume key, (subvolume, 
                compressed numpy predictions, compressed numpy mask))
        Returns:
            watershed+predictions (RDD) as (subvolume key, (subvolume, 
                (numpy compressed array, numpy compressed array)))
        """
        supervoxel_function = self._get_segmentation_function('create-supervoxels')

        pdconf = self.pdconf
        preserve_bodies = self.preserve_bodies

        def _execute_for_chunk(prediction_chunks):
            (subvolume, gray, prediction_compressed, mask_compressed) = prediction_chunks
            # mask can be None
            assert mask_compressed is None or isinstance( mask_compressed, CompressedNumpyArray )
            mask = mask_compressed and mask_compressed.deserialize()
            prediction = prediction_compressed.deserialize()
            if mask is None:
                mask = np.ones(shape=prediction.shape[:-1], dtype=np.uint8)

            # add body mask
            preserve_seg = None
            mask_bodies = None
            if pdconf is not None:
                # extract labels 64
                border = subvolume.border
                # get sizes of roi
                size1 = subvolume.roi[3]+2*border-subvolume.roi[0]
                size2 = subvolume.roi[4]+2*border-subvolume.roi[1]
                size3 = subvolume.roi[5]+2*border-subvolume.roi[2]
                 
                # retrieve data from roi start position considering border
                @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
                def get_segmask():
                    node_service = retrieve_node_service(pdconf["dvid-server"], 
                            pdconf["uuid"])
                    # retrieve data from roi start position
                    # Note: libdvid uses zyx order for python functions
                    return node_service.get_labels3D(str(pdconf["segmentation-name"]),
                        (size3,size2,size1),
                        (subvolume.roi[2]-border, subvolume.roi[1]-border, subvolume.roi[0]-border))
                preserve_seg = get_segmask()

                orig_bodies = set(np.unique(preserve_seg))

                mask_bodies = preserve_bodies & orig_bodies

                for body in mask_bodies:
                    mask[preserve_seg == body] = False

            # Call the (custom) function
            supervoxels = supervoxel_function(prediction, mask)
            subvolume.set_max_id( supervoxels.max() )
            
            # insert bodies back and avoid conflicts with pre-existing bodies
            if mask_bodies is not None:
                curr_id = supervoxels.max() + 1
                new_bodies = set(np.unique(supervoxels))
                conf_bodies = new_bodies & preserve_bodies
                for body in conf_bodies:
                    # replace value with an unused id
                    while curr_id in preserve_bodies:
                        curr_id += 1
                    supervoxels[supervoxels == body] = curr_id
                    curr_id += 1
                for body in mask_bodies:
                    supervoxels[preserve_seg == body] = body
            
            assert supervoxels.ndim == 3, "Supervoxels should be 3D (no channel dimension)"
            assert supervoxels.dtype == np.uint32, "Supervoxels for a single chunk should be uint32"
            
            supervoxels_compressed = CompressedNumpyArray(supervoxels)
            return (subvolume, gray, prediction_compressed, supervoxels_compressed)

        return prediction_chunks.mapValues(_execute_for_chunk)

    def agglomerate_supervoxels(self, sp_chunks, checkpoint_dir, enable_seg_rollback):
        """Agglomerate supervoxels

        Note: agglomeration should contain a subset of supervoxel
        body ids.

        Args:
            seg_chunks (RDD) = (subvolume key, (subvolume, numpy compressed array, 
                numpy compressed array))
        Returns:
            segmentation (RDD) = (subvolume key, (subvolume, numpy compressed array))
        """
        
        agglomeration_function = self._get_segmentation_function('agglomerate-supervoxels')

        pdconf = self.pdconf
        preserve_bodies = self.preserve_bodies

        def _execute_for_chunk(sp_chunk):
            (subvolume, gray, prediction_compressed, supervoxels_compressed) = sp_chunk
            roi = subvolume.roi
            border = subvolume.border
            block_bounds_zyx = ( (roi.z1 - border, roi.y1 - border, roi.x1 - border),
                                 (roi.z2 + border, roi.y2 + border, roi.x2 + border) )
            
            # Can we load the segmentation from on-disk cache?
            if enable_seg_rollback and checkpoint_dir:
                try:
                    block_store = H5BlockStore(checkpoint_dir, mode='r')
                except H5BlockStore.StoreDoesNotExistError:
                    pass
                else:
                    try:
                        # Return the segmentation we found on-disk
                        segmentation_block = block_store.get_block( block_bounds_zyx )
                        return ( subvolume, CompressedNumpyArray(segmentation_block[:]) )
                    except H5BlockStore.MissingBlockError:
                        pass
            
            supervoxels = supervoxels_compressed.deserialize()
            predictions = prediction_compressed.deserialize()
            
            # remove preserved bodies to ignore for agglomeration
            curr_seg = None
            mask_bodies = None
            if pdconf is not None:
                curr_seg = supervoxels.copy()
                
                curr_bodies = set(np.unique(supervoxels))
                mask_bodies = preserve_bodies & curr_bodies
            
                # 0'd bodies will be ignored
                for body in mask_bodies:
                    supervoxels[curr_seg == body] = 0


            # Call the (custom) function
            agglomerated = agglomeration_function(gray, predictions, supervoxels)
            assert agglomerated.ndim == 3, "Agglomerated supervoxels should be 3D (no channel dimension)"
            assert agglomerated.dtype == np.uint32, "Agglomerated supervoxels for a single chunk should be uint32"
           
            # ?! assumes that agglomeration function reuses label ids
            # reinsert bodies
            if mask_bodies is not None:
                for body in mask_bodies:
                    agglomerated[curr_seg == body] = body

            if checkpoint_dir:
                block_store = H5BlockStore(checkpoint_dir, mode='a', axes='zyx', dtype=np.uint32)
                block = block_store.get_block( block_bounds_zyx )
                block[:] = agglomerated
            
            agglomerated_compressed = CompressedNumpyArray(agglomerated)
            return (subvolume, agglomerated_compressed)

        # preserver partitioner
        return sp_chunks.mapValues(_execute_for_chunk)
    

    # label volumes to label volumes remapped, preserves partitioner 
    def stitch(self, label_chunks):
        def example(stuff):
            return stuff[1][0]
        subvolumes = label_chunks.map(example).collect()

        # return all subvolumes back to the driver
        # create offset map (substack id => offset) and broadcast
        offsets = {}
        offset = self.labeloffset

        pdconf = self.pdconf
        preserve_bodies = self.preserve_bodies
        
        num_preserve = 0
        if pdconf is not None:
            num_preserve = len(pdconf["bodies"])
        
        for subvolume in subvolumes:
            offsets[subvolume.roi_id] = offset
            offset += subvolume.max_id
            offset += num_preserve
        subvolume_offsets = self.context.sc.broadcast(offsets)

        # (key, subvolume, label chunk)=> (new key, (subvolume, boundary))
        def extract_boundaries(key_labels):
            # compute overlap -- assume first point is less than second
            def intersects(pt1, pt2, pt1_2, pt2_2):
                assert pt1 <= pt2, "point 1 greater than point 2: {} > {}".format( pt1, pt2 )
                assert pt1_2 <= pt2_2, "point 1_2 greater than point 2_2: {} > {}".format( pt1_2, pt2_2 )

                val1 = max(pt1, pt1_2)
                val2 = min(pt2, pt2_2)
                size = val2-val1
                npt1 = val1 - pt1 
                npt1_2 = val1 - pt1_2

                return npt1, npt1+size, npt1_2, npt1_2+size

            import numpy

            oldkey, (subvolume, labelsc) = key_labels
            labels = labelsc.deserialize()

            boundary_array = []
            
            # iterate through all ROI partners
            for partner in subvolume.local_regions:
                key1 = subvolume.roi_id
                key2 = partner[0]
                roi2 = partner[1]
                if key2 < key1:
                    key1, key2 = key2, key1
                
                # create key for boundary pair
                newkey = (key1, key2)

                # crop volume to overlap
                offx1, offx2, offx1_2, offx2_2 = intersects(
                                subvolume.roi.x1-subvolume.border,
                                subvolume.roi.x2+subvolume.border,
                                roi2.x1-subvolume.border,
                                roi2.x2+subvolume.border
                            )
                offy1, offy2, offy1_2, offy2_2 = intersects(
                                subvolume.roi.y1-subvolume.border,
                                subvolume.roi.y2+subvolume.border,
                                roi2.y1-subvolume.border,
                                roi2.y2+subvolume.border
                            )
                offz1, offz2, offz1_2, offz2_2 = intersects(
                                subvolume.roi.z1-subvolume.border,
                                subvolume.roi.z2+subvolume.border,
                                roi2.z1-subvolume.border,
                                roi2.z2+subvolume.border
                            )
                            
                labels_cropped = numpy.copy(labels[offz1:offz2, offy1:offy2, offx1:offx2])

                labels_cropped_c = CompressedNumpyArray(labels_cropped)
                # add to flat map
                boundary_array.append((newkey, (subvolume, labels_cropped_c)))

            return boundary_array


        # return compressed boundaries (id1-id2, boundary)
        mapped_boundaries = label_chunks.flatMap(extract_boundaries) 

        # shuffle the hopefully smallish boundaries into their proper spot
        # groupby is not a big deal here since same keys will not be in the same partition
        grouped_boundaries = mapped_boundaries.groupByKey()

        stitch_mode = self.stitch_mode

        # mappings to one partition (larger/second id keeps orig labels)
        # (new key, list<2>(subvolume, boundary compressed)) =>
        # (key, (subvolume, mappings))
        def stitcher(key_boundary):
            import numpy
            key, (boundary_list) = key_boundary

            # should be only two values
            if len(boundary_list) != 2:
                raise Exception("Expects exactly two subvolumes per boundary")
            # extract iterables
            boundary_list_list = []
            for item1 in boundary_list:
                boundary_list_list.append(item1)

            # order subvolume regions (they should be the same shape)
            subvolume1, boundary1_c = boundary_list_list[0] 
            subvolume2, boundary2_c = boundary_list_list[1] 

            if subvolume1.roi_id > subvolume2.roi_id:
                subvolume1, subvolume2 = subvolume2, subvolume1
                boundary1_c, boundary2_c = boundary2_c, boundary1_c

            boundary1 = boundary1_c.deserialize()
            boundary2 = boundary2_c.deserialize()

            if boundary1.shape != boundary2.shape:
                raise Exception("Extracted boundaries are different shapes")
            
            # determine list of bodies in play
            z2, y2, x2 = boundary1.shape
            z1 = y1 = x1 = 0 

            # determine which interface there is touching between subvolumes 
            if subvolume1.touches(subvolume1.roi.x1, subvolume1.roi.x2,
                                subvolume2.roi.x1, subvolume2.roi.x2):
                x1 = x2/2 
                x2 = x1 + 1
            if subvolume1.touches(subvolume1.roi.y1, subvolume1.roi.y2,
                                subvolume2.roi.y1, subvolume2.roi.y2):
                y1 = y2/2 
                y2 = y1 + 1
            
            if subvolume1.touches(subvolume1.roi.z1, subvolume1.roi.z2,
                                subvolume2.roi.z1, subvolume2.roi.z2):
                z1 = z2/2 
                z2 = z1 + 1

            eligible_bodies = set(numpy.unique(boundary2[z1:z2, y1:y2, x1:x2]))
            body2body = {}

            label2_bodies = numpy.unique(boundary2)

            # 0 is off,
            # 1 is very conservative (high percentages and no bridging),
            # 2 is less conservative (no bridging),
            # 3 is the most liberal (some bridging allowed if overlap
            # greater than X and overlap threshold)
            hard_lb = 50
            liberal_lb = 1000
            conservative_overlap = 0.90

            if stitch_mode > 0:
                for body in label2_bodies:
                    if body == 0:
                        continue
                    body2body[body] = {}

                # traverse volume to find maximum overlap
                for (z,y,x), body1 in numpy.ndenumerate(boundary1):
                    body2 = boundary2[z,y,x]
                    if body2 == 0 or body1 == 0:
                        continue
                    
                    if body1 not in body2body[body2]:
                        body2body[body2][body1] = 0
                    body2body[body2][body1] += 1


            # create merge list 
            merge_list = []
            mutual_list = {}
            retired_list = set()

            small_overlap_prune = 0
            conservative_prune = 0
            aggressive_add = 0
            not_mutual = 0

            for body2, bodydict in body2body.items():
                if body2 in eligible_bodies:
                    bodysave = -1
                    max_val = hard_lb
                    total_val = 0
                    for body1, val in bodydict.items():
                        total_val += val
                        if val > max_val:
                            bodysave = body1
                            max_val = val
                    if bodysave == -1:
                        small_overlap_prune += 1
                    elif (stitch_mode == 1) and (max_val / float(total_val) < conservative_overlap):
                        conservative_prune += 1
                    elif (stitch_mode == 3) and (max_val / float(total_val) > conservative_overlap) and (max_val > liberal_lb):
                        merge_list.append([int(bodysave), int(body2)])
                        # do not add
                        retired_list.add((int(bodysave), int(body2))) 
                        aggressive_add += 1
                    else:
                        if int(bodysave) not in mutual_list:
                            mutual_list[int(bodysave)] = {}
                        mutual_list[int(bodysave)][int(body2)] = max_val
                       

            eligible_bodies = set(numpy.unique(boundary1[z1:z2, y1:y2, x1:x2]))
            body2body = {}
            
            if stitch_mode > 0:
                label1_bodies = numpy.unique(boundary1)
                for body in label1_bodies:
                    if body == 0:
                        continue
                    body2body[body] = {}

                # traverse volume to find maximum overlap
                for (z,y,x), body1 in numpy.ndenumerate(boundary1):
                    body2 = boundary2[z,y,x]
                    if body2 == 0 or body1 == 0:
                        continue
                    if body2 not in body2body[body1]:
                        body2body[body1][body2] = 0
                    body2body[body1][body2] += 1
            
            # add to merge list 
            for body1, bodydict in body2body.items():
                if body1 in eligible_bodies:
                    bodysave = -1
                    max_val = hard_lb
                    total_val = 0
                    for body2, val in bodydict.items():
                        total_val += val
                        if val > max_val:
                            bodysave = body2
                            max_val = val

                    if (int(body1), int(bodysave)) in retired_list:
                        # already in list
                        pass
                    elif bodysave == -1:
                        small_overlap_prune += 1
                    elif (stitch_mode == 1) and (max_val / float(total_val) < conservative_overlap):
                        conservative_prune += 1
                    elif (stitch_mode == 3) and (max_val / float(total_val) > conservative_overlap) and (max_val > liberal_lb):
                        merge_list.append([int(body1), int(bodysave)])
                        aggressive_add += 1
                    elif int(body1) in mutual_list:
                        partners = mutual_list[int(body1)]
                        if int(bodysave) in partners:
                            merge_list.append([int(body1), int(bodysave)])
                        else:
                            not_mutual += 1
                    else:
                        not_mutual += 1
            
            # remove mergers that involve preserve bodies
            if pdconf is not None: 
                merge_list_temp = []
 
                for merger in merge_list:
                    if merger[0] not in preserve_bodies and merger[1] not in preserve_bodies:
                        merge_list_temp.append(merger)
                merge_list = merge_list_temp

            
            # handle offsets in mergelist
            offset1 = subvolume_offsets.value[subvolume1.roi_id] 
            offset2 = subvolume_offsets.value[subvolume2.roi_id] 
            for merger in merge_list:
                merger[0] = merger[0]+offset1
                merger[1] = merger[1]+offset2

            # return id and mappings, only relevant for stack one
            return (subvolume1.roi_id, merge_list)

        # key, mapping1; key mapping2 => key, mapping1+mapping2
        def reduce_mappings(b1, b2):
            b1.extend(b2)
            return b1

        # map from grouped boundary to substack id, mappings
        subvolume_mappings = grouped_boundaries.map(stitcher).reduceByKey(reduce_mappings)

        # reconcile all the mappings by sending them to the driver
        # (not a lot of data and compression will help but not sure if there is a better way)
        merge_list = []
        all_mappings = subvolume_mappings.collect()
        for (substack_id, mapping) in all_mappings:
            merge_list.extend(mapping)

        # make a body2body map
        body1body2 = {}
        body2body1 = {}

        for merger in merge_list:
            # body1 -> body2
            body1 = merger[0]
            if merger[0] in body1body2:
                body1 = body1body2[merger[0]]
            body2 = merger[1]
            if merger[1] in body1body2:
                body2 = body1body2[merger[1]]

            if body2 not in body2body1:
                body2body1[body2] = set()
            
            # add body1 to body2 map
            body2body1[body2].add(body1)
            # add body1 -> body2 mapping
            body1body2[body1] = body2

            if body1 in body2body1:
                for tbody in body2body1[body1]:
                    body2body1[body2].add(tbody)
                    body1body2[tbody] = body2

        # avoid renumbering bodies that are to be preserved from previous segmentation
        if self.preserve_bodies is not None:
            # changing mappings to avoid map-to conflicts
            relabel_confs = {}
            body2body_tmp = body1body2.copy()

            for key, val in body2body_tmp.items():
                if val in self.preserve_bodies:
                    assert False, "FIXME!"
                    if val not in relabelconfs:
                        newval = val + 1
                        while newval in self.preserve_bodies:
                            newval += 1
                        relabelconfs[val] = newval
                        self.preserve_bodies.add(newval)
                    body1body2[key] = relabelconfs[val]

        body2body = zip(body1body2.keys(), body1body2.values())
       
        # potentially costly broadcast
        # (possible to split into substack to make more efficient but compression should help)
        master_merge_list = self.context.sc.broadcast(body2body)

        # use offset and mappings to relabel volume
        def relabel(key_label_mapping):
            import numpy

            (subvolume, label_chunk_c) = key_label_mapping
            labels = label_chunk_c.deserialize()

            # grab broadcast offset
            offset = subvolume_offsets.value[subvolume.roi_id]

            # check for body mask labels and protect from renumber
            mask_bodies =  None
            fix_bodies = []
            
            if pdconf is not None:
                curr_bodies = set(np.unique(labels))
                mask_bodies = preserve_bodies & curr_bodies
                # see if offset will cause new conflicts 
                for body in curr_bodies:
                    if (body + offset) in preserve_bodies and body not in preserve_bodies:
                        fix_bodies.append(body+offset)


            labels = labels + offset 
            
            # make sure 0 is 0
            labels[labels == offset] = 0

            # replace preserved body removing offset
            if mask_bodies is not None:
                for body in mask_bodies:
                    labels[labels == (body+offset)] = body

            # check for new body conflicts and remap
            relabeled_bodies = {}
            if pdconf is not None:
                curr_id = labels.max() + 1
                for body in fix_bodies:
                    while curr_id in preserve_bodies:
                        curr_id +=1
                    labels[labels == body] = curr_id
                    relabeled_bodies[body] = curr_id
                    curr_id += 1

            # create default map 
            mapping_col = numpy.unique(labels)
            label_mappings = dict(zip(mapping_col, mapping_col))
           
            # create maps from merge list
            for mapping in master_merge_list.value:
                if mapping[0] in label_mappings:
                    label_mappings[mapping[0]] = mapping[1]
                elif mapping[0] in relabeled_bodies:
                    label_mappings[relabeled_bodies[mapping[0]]] = mapping[1]

            # apply maps
            vectorized_relabel = numpy.frompyfunc(label_mappings.__getitem__, 1, 1)
            labels = vectorized_relabel(labels).astype(numpy.uint64)
       
            return (subvolume, CompressedNumpyArray(labels))

        # just map values with broadcast map
        # Potential TODO: consider fast join with partitioned map (not broadcast)
        return label_chunks.mapValues(relabel)


