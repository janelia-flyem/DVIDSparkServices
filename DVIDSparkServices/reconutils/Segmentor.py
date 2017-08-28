"""Defines base class for segmentation plugins."""
from __future__ import print_function, absolute_import
import sys
import json
import importlib
import textwrap
from functools import partial, wraps
import logging
import numpy as np
import vigra

from quilted.h5blockstore import H5BlockStore

import DVIDSparkServices
from DVIDSparkServices.json_util import validate_and_inject_defaults
from DVIDSparkServices.auto_retry import auto_retry
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service
from DVIDSparkServices.util import zip_many, select_item, dense_roi_mask_for_subvolume
from DVIDSparkServices.sparkdvid.Subvolume import Subvolume
from DVIDSparkServices.subprocess_decorator import execute_in_subprocess

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
                },
                "use-subprocess" : {
                  "description": "Run this segmentation step in a subprocess, to enable logging of plain stdout.",
                  "type": "boolean",
                  "default": false
                },
                "subprocess-timeout" : {
                  "description": "Automatically kill the subprocess after this timeout and raise an error.",
                  "type": "integer",
                  "default": 0
                }
              },
              "additionalProperties": true
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
                                          "default": { "function": "DVIDSparkServices.reconutils.misc.noop_agglomeration" } },
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
              "additionalProperties": true,
              "default": {}
            }
          },
          "additionalProperties": true,
          "default": {}
        }
        """)

    def __init__(self, context, workflow):
        self.context = context
        self.workflow = workflow
        workflow_config = workflow.config_data
        self.segmentor_config = workflow_config["options"]["segmentor"]["configuration"]

        segmentor_schema = json.loads(Segmentor.SegmentorSchema)
        validate_and_inject_defaults(self.segmentor_config, segmentor_schema)        

        stitch_modes = { "none" : 0, "conservative" : 1, "medium" : 2, "aggressive" : 3 }
        self.stitch_mode = stitch_modes[ workflow_config["options"]["stitch-algorithm"] ]
        self.stitch_constraints = workflow_config["options"]["stitch-constraints"]
        self.labeloffset = 0
        if "label-offset" in workflow_config["options"]:
            self.labeloffset = int(workflow_config["options"]["label-offset"])


        # save masked bodies
        self.pdconf = None
        self.preserve_bodies = None
        if self.segmentor_config["preserve-bodies"]["bodies"]:
            self.pdconf = self.segmentor_config["preserve-bodies"]
            self.preserve_bodies = set(self.pdconf["bodies"])


    def segment(self, subvols_rdd, gray_blocks,
                gray_checkpoint_dir, mask_checkpoint_dir, pred_checkpoint_dir, sp_checkpoint_dir, seg_checkpoint_dir,
                allow_pred_rollback, allow_sp_rollback, allow_seg_rollback):
        """Top-level pipeline (can overwrite) -- gray RDD => label RDD.

        Defines a segmentation workflow consisting of voxel prediction,
        watershed, and agglomeration.  One can overwrite specific functions
        or the entire workflow as long as RDD input and output constraints
        are statisfied.  RDD transforms should preserve the partitioner -- 
        subvolume id is the key.

        Args:
            gray_chunks, cached_voxel_prediction_chunks, cached_supervoxel_chunks, cached_segmentation_chunks (RDD) = (subvolume key, (subvolume, numpy grayscale))
        Returns:
            segmentation (RDD) as (subvolume key, (subvolume, numpy compressed array))

        """
        # Developers might set gray_checkpoint_dir while debugging.
        # In that case, force the grayscale data to get written to cache.
        if gray_checkpoint_dir:
            gray_blocks = self.cache_grayscale(subvols_rdd, gray_blocks, gray_checkpoint_dir)
            gray_blocks.persist()
        
        # Compute mask of background area that can be skipped (if any)
        mask_blocks = self.compute_background_mask(subvols_rdd, gray_blocks, mask_checkpoint_dir)
        mask_blocks.persist()

        # run voxel prediction (default: grayscale is boundary)
        pred_blocks = self.predict_voxels(subvols_rdd, gray_blocks, mask_blocks, pred_checkpoint_dir, allow_pred_rollback)
        pred_blocks.persist()

        # run watershed from voxel prediction (default: seeded watershed)
        sp_blocks = self.create_supervoxels(subvols_rdd, pred_blocks, mask_blocks, sp_checkpoint_dir, allow_sp_rollback)
        sp_blocks.persist()

        # run agglomeration (default: none)
        seg_blocks = self.agglomerate_supervoxels(subvols_rdd, gray_blocks, pred_blocks, sp_blocks, seg_checkpoint_dir, allow_seg_rollback)
        
        return seg_blocks

    @classmethod
    def use_block_cache(cls, blockstore_dir, allow_read=True, allow_write=True, dset_options={'compression': 'gzip', 'shuffle': True}):
        """
        Returns a decorator, intended to decorate functions that execute in spark workers.
        Before performing the work, check the block cache in the given directory and return the data from the cache if possible.
        If the data isn't there, execute the function as usual and store the result in the cache before returning.
        """
        def decorator(f):
            if not blockstore_dir:
                return f

            # If the store does exist, reset it now (in the driver)
            # to clean up after any failed runs.
            try:
                H5BlockStore(blockstore_dir, mode='r', reset_access=True)
            except H5BlockStore.StoreDoesNotExistError:
                pass

            @wraps(f)
            def wrapped(item):
                subvol = item[0]
                z1, y1, x1, z2, y2, x2 = subvol.box_with_border
                assert isinstance(subvol, Subvolume), "Key must be a Subvolume object"
        
                if allow_read:
                    try:
                        block_store = H5BlockStore(blockstore_dir, mode='r', default_timeout=15*60)
                        if block_store.axes[-1] == 'c':
                            block_bounds = ((z1, y1, x1, 0), (z2, y2, x2, None))
                        else:
                            block_bounds = ((z1, y1, x1), (z2, y2, x2))
                        h5_block = block_store.get_block( block_bounds )
                        return h5_block[:]
                    except H5BlockStore.StoreDoesNotExistError:
                        pass
                    except H5BlockStore.MissingBlockError:
                        pass

                block_data = f(item)

                if allow_write and block_data is not None:
                    assert isinstance(block_data, np.ndarray), \
                        "Return type can't be stored in the block cache: {}".format( type(block_data) )
                    
                    axes = 'zyxc'[:block_data.ndim]
                    if axes[-1] == 'c':
                        c2 = block_data.shape[3]
                        block_bounds = ((z1, y1, x1, 0), (z2, y2, x2, c2))
                    else:
                        block_bounds = ((z1, y1, x1), (z2, y2, x2))

                    block_store = H5BlockStore(blockstore_dir, mode='a', axes=axes, dtype=block_data.dtype,
                                               dset_options=dset_options,
                                               default_timeout=15*60)
                    h5_block = block_store.get_block( block_bounds )
                    h5_block[:] = block_data

                return block_data
            
            return wrapped
        return decorator


    def _get_segmentation_function(self, segmentation_step):
        """
        Read the user's config and return the image processing
        function specified for the given segmentation step.

        If the user provided a dict of parameters, then they will be
        bound into the returned function as keyword args.
        """
        full_function_name = self.segmentor_config[segmentation_step]["function"]
        module_name = '.'.join(full_function_name.split('.')[:-1])
        module = importlib.import_module(module_name)
        function_name = full_function_name.split('.')[-1]
        func = getattr(module, function_name)
        
        if self.segmentor_config[segmentation_step]["use-subprocess"]:
            timeout = self.segmentor_config[segmentation_step]["subprocess-timeout"]
            def log_msg(msg):
                logger = logging.getLogger(full_function_name)
                logger.info(msg.rstrip())
            func = execute_in_subprocess(log_msg, timeout)(func)
        else:
            assert self.segmentor_config[segmentation_step]["subprocess-timeout"] == 0, \
                "Can't use subprocess-timeout without use-subprocess: True"
        
        parameters = self.segmentor_config[segmentation_step]["parameters"]
        return partial( func, **parameters )

    def cache_grayscale(self, subvols, gray_vols, gray_checkpoint_dir):
        """
        This is a dumb pass-through function just to cache the downloaded grayscale data to disk.
        The cached data isn't actually used by this pipeline, can be useful for viewing later
        (for debugging purposes).
        """
        @self.workflow.collect_log(lambda sv_g: str(sv_g[0]))
        @Segmentor.use_block_cache(gray_checkpoint_dir, allow_read=False, dset_options={})
        def _execute_for_chunk(xxx_todo_changeme ):
            (_subvolume, gray) = xxx_todo_changeme
            logging.getLogger(__name__).debug("Caching grayscale")
            return gray

        return subvols.zip(gray_vols).map(_execute_for_chunk, True)

    def compute_background_mask(self, subvols, gray_vols, mask_checkpoint_dir):
        """
        Construct a mask to distinguish between foreground (which will be segmented)
        and background (which will be excluded from the segmentation).
        
        The mask is a combination of two sources:
        
          1. Call a custom plugin function to generate a mask
             of foreground pixels, based on grayscale values.

          2. Also, determine which pixels (if any) of each subvolue 
             fall outside of the ROI, and remove them from the mask.
        
        In the returned mask, 0 means 'background and 1 means foreground.
        
        Optimization: If all mask pixels are 1, then we may return 'None'
                      which, by convention, means "everything is foreground".
                      (This saves RAM in the common case.)
        """
        mask_function = self._get_segmentation_function('background-mask')

        @self.workflow.collect_log(lambda sv_g1: str(sv_g1[0]))
        @Segmentor.use_block_cache(mask_checkpoint_dir, allow_read=False)
        def _execute_for_chunk(args):
            import DVIDSparkServices
            subvolume, gray = args

            # Call the plugin function
            data_mask = mask_function(gray)
            if data_mask is not None:
                assert data_mask.dtype == np.bool, "Mask array should be boolean"
                assert data_mask.ndim == 3
            
            # Determine how the ROI intersects this subvolume
            roi_mask = dense_roi_mask_for_subvolume(subvolume)

            # Combine
            if data_mask is None:
                data_mask = roi_mask
            else:
                data_mask[:] &= roi_mask

            if data_mask.all():
                # By convention, None means "everything"
                return None

            return data_mask

        return subvols.zip(gray_vols).map(_execute_for_chunk, True)

    def predict_voxels(self, subvols, gray_blocks, mask_blocks, pred_checkpoint_dir, allow_pred_rollback):
        """Create a dummy placeholder boundary channel from grayscale.

        Takes an RDD of grayscale numpy volumes and produces
        an RDD of predictions (z,y,x).
        """
        prediction_function = self._get_segmentation_function('predict-voxels')

        @self.workflow.collect_log(lambda sv_g_mc: str(sv_g_mc[0]))
        @Segmentor.use_block_cache(pred_checkpoint_dir, allow_read=allow_pred_rollback)
        def _execute_for_chunk(args):
            import DVIDSparkServices

            subvolume, (gray, mask) = args
            box = subvolume.box_with_border
            block_bounds_zyx = ( (box.z1, box.y1, box.x1), (box.z2, box.y2, box.x2) )

            # Call the (custom) function
            predictions = prediction_function(gray, mask)
            assert predictions.ndim == 4, "Predictions volume should be 4D: z-y-x-c"
            assert predictions.dtype == np.float32, "Predictions should be float32"
            assert predictions.shape[:3] == tuple(np.array(block_bounds_zyx[1]) - block_bounds_zyx[0]), \
                "predictions have unexpected shape: {}, expected block_bounds: {}"\
                .format( predictions.shape, block_bounds_zyx )

            #import numpy
            #predictions = predictions * 100
            #predictions = predictions.astype(numpy.uint8)
            return predictions
             
        return subvols.zip( gray_blocks.zip(mask_blocks) ).map(_execute_for_chunk, True)

    def create_supervoxels(self, subvols, pred_blocks, mask_blocks, sp_checkpoint_dir, allow_sp_rollback):
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
        resource_server = self.context.workflow.resource_server
        resource_port = self.context.workflow.resource_port

        @self.workflow.collect_log(lambda sv_pc_mc: str(sv_pc_mc[0]))
        @Segmentor.use_block_cache(sp_checkpoint_dir, allow_read=allow_sp_rollback)
        def _execute_for_chunk(args):
            import DVIDSparkServices
            subvolume, (prediction, mask) = args
            box = subvolume.box_with_border
            block_bounds_zyx = ( (box.z1, box.y1, box.x1), (box.z2, box.y2, box.x2) )
            if mask is None:
                mask = np.ones(shape=prediction.shape[:-1], dtype=np.uint8)

            # add body mask
            preserve_seg = None
            mask_bodies = None
            if pdconf is not None:
                # extract labels 64
                border = subvolume.border
                # get sizes of sv box
                size_z = subvolume.box.z2 + 2*border - subvolume.box.z1
                size_y = subvolume.box.y2 + 2*border - subvolume.box.y1
                size_x = subvolume.box.x2 + 2*border - subvolume.box.x1
                 
                # retrieve data from box start position considering border
                @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
                def get_segmask():
                    node_service = retrieve_node_service(pdconf["dvid-server"], 
                            pdconf["uuid"], resource_server, resource_port)
                    # retrieve data from box start position
                    # Note: libdvid uses zyx order for python functions
                    if resource_server != "":  
                        return node_service.get_labels3D(str(pdconf["segmentation-name"]),
                                (size_z, size_y, size_x),
                                (subvolume.box.z1-border, subvolume.box.y1-border, subvolume.box.x1-border), throttle=False)
                    else:   
                        return node_service.get_labels3D(str(pdconf["segmentation-name"]),
                                (size_z, size_y, size_x),
                                (subvolume.box.z1-border, subvolume.box.y1-border, subvolume.box.x1-border))
                preserve_seg = get_segmask()

                orig_bodies = set(np.unique(preserve_seg))

                mask_bodies = preserve_bodies & orig_bodies

                for body in mask_bodies:
                    mask[preserve_seg == body] = False

            # Call the (custom) function
            #import numpy
            #prediction = prediction.astype(numpy.float32)
            #prediction = prediction / 100

            supervoxels = supervoxel_function(prediction, mask)
            
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
            assert supervoxels.shape == tuple(np.array(block_bounds_zyx[1]) - block_bounds_zyx[0]), \
                "segmentation block has unexpected shape: {}, expected block_bounds: {}"\
                .format( supervoxels.shape, block_bounds_zyx )
            
            return supervoxels

        return subvols.zip( pred_blocks.zip(mask_blocks) ).map(_execute_for_chunk, True)

    def agglomerate_supervoxels(self, subvols, gray_blocks, pred_blocks, sp_blocks, seg_checkpoint_dir, allow_seg_rollback):
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

        @self.workflow.collect_log(lambda sv_g_pc_sc: str(sv_g_pc_sc[0]))
        @Segmentor.use_block_cache(seg_checkpoint_dir, allow_read=allow_seg_rollback)
        def _execute_for_chunk(args):
            import DVIDSparkServices
            subvolume, (gray, predictions, supervoxels) = args
            #import numpy
            #predictions = predictions.astype(numpy.float32)
            #predictions = predictions / 100
            box = subvolume.box_with_border
            block_bounds_zyx = ( (box.z1, box.y1, box.x1), (box.z2, box.y2, box.x2) )
            
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
            assert agglomerated.shape == tuple(np.array(block_bounds_zyx[1]) - block_bounds_zyx[0]), \
                "segmentation block has unexpected shape: {}, expected block_bounds: {}"\
                .format( agglomerated.shape, block_bounds_zyx )

            # ?! assumes that agglomeration function reuses label ids
            # reinsert bodies
            if mask_bodies is not None:
                for body in mask_bodies:
                    agglomerated[curr_seg == body] = body

            return agglomerated

        # preserve partitioner
        return subvols.zip( zip_many(gray_blocks, pred_blocks, sp_blocks) ).map(_execute_for_chunk, True)
    

    # label volumes to label volumes remapped, preserves partitioner 
    def stitch(self, label_chunks):
        """
        label_chunks (RDD): [ (subvol, (seg_vol, max_id)),
                              (subvol, (seg_vol, max_id)),
                              ... ]

        Note: This function requires that label_chunks is already persist()ed in memory.
        """
        assert label_chunks.is_cached, "You must persist() label_chunks before calling this function."
        subvolumes_rdd = select_item(label_chunks, 0)
        subvolumes = subvolumes_rdd.collect()
        max_ids = select_item(label_chunks, 1, 1).collect()

        # return all subvolumes back to the driver
        # create offset map (substack id => offset) and broadcast
        offsets = {}
        offset = self.labeloffset

        pdconf = self.pdconf
        preserve_bodies = self.preserve_bodies
        
        num_preserve = 0
        if pdconf is not None:
            num_preserve = len(pdconf["bodies"])
        
        for subvolume, max_id in zip(subvolumes, max_ids):
            offsets[subvolume.sv_index] = offset
            offset += max_id
            offset += num_preserve
        subvolume_offsets = self.context.sc.broadcast(offsets)

        stitch_constraints = self.stitch_constraints
        # (subvol, label_vol) => [ (sv_index_1, sv_index_2), (subvol, boundary_labels)), 
        #                          (sv_index_1, sv_index_2), (subvol, boundary_labels)), ...] 
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
            subvolume, labels = key_labels

            boundary_array = []
            
            # if optioned: extract graph, apply offset, and add to specific boundary in for loop 
            graph_edges = None
            if stitch_constraints:
                graph_edges = set()
                from DVIDSparkServices.reconutils import SimpleGraph
                tempoptions = {}
                tempoptions["graph-builder-exe"] = "neuroproof_graph_build_stream"
                sg = SimpleGraph.SimpleGraph(tempoptions) 
                elements = sg.build_graph(key_labels)                
                # extract graph edges (n1 should be greater than n2)
                for element in elements:
                    (n1, n2), weight = element
                    if n2 != -1:
                        n1 +=  subvolume_offsets.value[subvolume.sv_index]
                        n2 +=  subvolume_offsets.value[subvolume.sv_index]
                        if n1 < n2:
                            n1, n2 = n2, n1
                        graph_edges.add((n1,n2)) 

            # iterate through all box partners
            for partner in subvolume.local_regions:
                key1 = subvolume.sv_index
                key2 = partner[0]
                box2 = partner[1]
                if key2 < key1:
                    key1, key2 = key2, key1
                
                # crop volume to overlap
                offx1, offx2, offx1_2, offx2_2 = intersects(
                                subvolume.box.x1-subvolume.border,
                                subvolume.box.x2+subvolume.border,
                                box2.x1-subvolume.border,
                                box2.x2+subvolume.border
                            )
                offy1, offy2, offy1_2, offy2_2 = intersects(
                                subvolume.box.y1-subvolume.border,
                                subvolume.box.y2+subvolume.border,
                                box2.y1-subvolume.border,
                                box2.y2+subvolume.border
                            )
                offz1, offz2, offz1_2, offz2_2 = intersects(
                                subvolume.box.z1-subvolume.border,
                                subvolume.box.z2+subvolume.border,
                                box2.z1-subvolume.border,
                                box2.z2+subvolume.border
                            )
                            
                labels_cropped = numpy.copy(labels[offz1:offz2, offy1:offy2, offx1:offx2])

                # extract constraint graph
                graph_edges_sub = None
                if graph_edges is not None:
                    graph_edges_sub = set()
                    bound_labels = set(numpy.unique(labels_cropped))
                    for (n1,n2) in graph_edges:
                        if n1 in bound_labels and n2 in bound_labels:
                            graph_edges_sub.add((n1,n2))

                # create key for boundary pair
                newkey = (key1, key2)

                # add to flat map
                boundary_array.append((newkey, (subvolume, labels_cropped, graph_edges_sub)))

            return boundary_array


        # return compressed boundaries (id1-id2, boundary)
        # (subvol, labels) -> [ ( (k1, k2), (subvol, boundary_labels_1) ),
        #                       ( (k1, k2), (subvol, boundary_labels_1) ),
        #                       ( (k1, k2), (subvol, boundary_labels_1) ), ... ]
        label_vols_rdd = select_item(label_chunks, 1, 0)
        mapped_boundaries = subvolumes_rdd.zip(label_vols_rdd).flatMap(extract_boundaries) 

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
            subvolume1, boundary1, graphedge1 = boundary_list_list[0] 
            subvolume2, boundary2, graphedge2 = boundary_list_list[1] 

            if subvolume1.sv_index > subvolume2.sv_index:
                subvolume1, subvolume2 = subvolume2, subvolume1
                boundary1, boundary2 = boundary2, boundary1

            if boundary1.shape != boundary2.shape:
                raise Exception("Extracted boundaries are different shapes")
            
            # determine list of bodies in play
            z2, y2, x2 = boundary1.shape
            z1 = y1 = x1 = 0 

            # determine which interface there is touching between subvolumes 
            if subvolume1.touches(subvolume1.box.x1, subvolume1.box.x2,
                                  subvolume2.box.x1, subvolume2.box.x2):
                x1 = x2/2 
                x2 = x1 + 1
            if subvolume1.touches(subvolume1.box.y1, subvolume1.box.y2,
                                  subvolume2.box.y1, subvolume2.box.y2):
                y1 = y2/2 
                y2 = y1 + 1
            
            if subvolume1.touches(subvolume1.box.z1, subvolume1.box.z2,
                                  subvolume2.box.z1, subvolume2.box.z2):
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
            offset1 = subvolume_offsets.value[subvolume1.sv_index] 
            offset2 = subvolume_offsets.value[subvolume2.sv_index] 
            for merger in merge_list:
                merger[0] = merger[0]+offset1
                merger[1] = merger[1]+offset2

            # if stitch constraint: prune mergers that lead to split violation 
            destroyset1 = set()
            if graphedge2 is not None:
                # check for self-touch in boundary2, suggests that body1 has false merge
                merge_temp = {}
                for merger in merge_list:
                    if merger[0] not in merge_temp:
                        merge_temp[merger[0]] = [merger[1]]
                    else:
                        merge_temp[merger[0]].append(merger[1])

                for b1, b2list in merge_temp.items():
                    if len(b2list) > 1:
                        finis = False
                        for iter1 in range(0, len(b2list)-1):
                            if finis:
                                break
                            for iter2 in range(iter1, len(b2list)):
                                n1 = b2list[iter1]
                                n2 = b2list[iter2]
                                if n1 < n2:
                                    n1, n2 = n2, n1
                                if (n1,n2) in graphedge2:
                                    destroyset1.add(b1)
                                    finis = True
                                    break
            destroyset2 = set()
            if graphedge1 is not None:
                # check for self-touch in boundary1, suggests that body2 has false merge
                merge_temp = {}
                for merger in merge_list:
                    if merger[1] not in merge_temp:
                        merge_temp[merger[1]] = [merger[0]]
                    else:
                        merge_temp[merger[1]].append(merger[0])

                for b2, b1list in merge_temp.items():
                    if len(b1list) > 1:
                        finis = False
                        for iter1 in range(0, len(b1list)-1):
                            if finis:
                                break
                            for iter2 in range(iter1, len(b1list)):
                                n1 = b1list[iter1]
                                n2 = b1list[iter2]
                                if n1 < n2:
                                    n1, n2 = n2, n1
                                if (n1,n2) in graphedge1:
                                    destroyset2.add(b2)
                                    finis = True
                                    break
            if len(destroyset1) > 0 or len(destroyset2) > 0:
                merge_list2 = []
                for merger in merge_list:
                    if merger[0] not in destroyset1 and merger[1] not in destroyset2:
                        merge_list2.append(merger)
                merge_list = merge_list2

            # dump merge decisions for each body 
            if stitch_constraints:
                # group all mergers together
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

                # create list of all stitch decision for each body
                body_decisions = {}
                for merger in merge_list:
                    if merger[0] not in body_decisions:
                        body_decisions[merger[0]] = set([merger[1]])
                    else:
                        body_decisions[merger[0]].add(merger[1])
                    
                    if merger[1] not in body_decisions:
                        body_decisions[merger[1]] = set([merger[0]])
                    else:
                        body_decisions[merger[1]].add(merger[0])
              
                # create list of edges per body
                graphedges = graphedge2.union(graphedge1)
                vertexedges = {}
                for (n1, n2) in graphedges:
                    if n1 not in vertexedges:
                        vertexedges[n1] = set([n2])
                    else:
                        vertexedges[n1].add(n2)
                    if n2 not in vertexedges:
                        vertexedges[n2] = set([n1])
                    else:
                        vertexedges[n2].add(n1)

                # return flattened data: (body id, (is_head, replist(ptr), merger list, graph list))
                bodydata = []
                for body, bodylist in body2body1.items():
                    maxid = body
                    for bodytemp in bodylist:
                        if bodytemp > maxid:
                            maxid = bodytemp
                    replist = [maxid]
                    declist = [body_decisions[maxid]]
                    if maxid not in vertexedges:
                        vertexedges[maxid] = set()
                    graphlist = [vertexedges[maxid]] # possibly no constraints
                    if body != maxid:
                        replist.append(body)
                        declist.append(body_decisions[body])
                        if body not in vertexedges:
                            vertexedges[body] = set()
                        graphlist.append(vertexedges[body])
                    for bodytemp in bodylist:
                        if maxid != bodytemp:
                            replist.append(bodytemp)
                            declist.append(body_decisions[bodytemp])
                            if bodytemp not in vertexedges:
                                vertexedges[bodytemp] = set()
                            graphlist.append(vertexedges[bodytemp])
                    bodydata.append((maxid, (True, replist, declist, graphlist, [])))
                    for iter1 in range(1, len(replist)):
                        bodydata.append((replist[iter1], (False, [maxid], [body_decisions[maxid]], [vertexedges[maxid]], [])))
            
                return bodydata
            else:               
                # return id and mappings, only relevant for stack one
                return (subvolume1.sv_index, merge_list)

        merge_list = []
        if stitch_constraints:
            current_decisions = grouped_boundaries.flatMap(stitcher)
            
            def combine_decisions(dec1, dec2):
                dec1_ishead, dec1_reps, dec1_decisions, dec1_graph, readjust1 = dec1
                dec2_ishead, dec2_reps, dec2_decisions, dec2_graph, readjust2 = dec2
                
                dec_ishead = dec1_ishead and dec2_ishead
                dec_reps = dec1_reps
                dec_decisions = dec1_decisions
                dec_graph = dec1_graph
                readjust = readjust1

                alt_reps = dec2_reps
                alt_decisions = dec2_decisions
                alt_graph = dec2_graph

                # first element the same if it is the representative element
                if dec_ishead:
                    assert dec1_reps[0] == dec2_reps[0], "first element should be the same"
                
                if dec2_reps[0] > dec1_reps[0]: # shift everything to new master
                    dec_reps = dec2_reps
                    dec_decisions = dec2_decisions
                    dec_graph = dec2_graph
                    alt_reps = dec1_reps
                    alt_decisions = dec1_decisions
                    alt_graph = dec1_graph
                    
                    # make a list of ids that need to be notified about a new head
                    if not dec1_ishead:
                        readjust.append(dec1_reps[0])
                    readjust.extend(readjust1)
                else:
                    if dec2_reps[0] != dec1_reps[0] and not dec2_ishead:
                        readjust.append(dec2_reps[0])
                    readjust.extend(readjust2)

                prev_reps = {}
                for iter1, dec_rep in enumerate(dec_reps):
                    prev_reps[dec_rep] = iter1

                for iter1 in range(0, len(alt_reps)):
                    if alt_reps[iter1] not in prev_reps:
                        dec_reps.append(alt_reps[iter1])
                        dec_decisions.append(alt_decisions[iter1])
                        dec_graph.append(alt_graph[iter1])
                    else:
                        # combine dec1 decisions
                        previd = prev_reps[alt_reps[iter1]]
                        dec_decisions[previd] = dec_decisions[previd].union(alt_decisions[iter1])
                return (dec_ishead, dec_reps, dec_decisions, dec_graph, readjust)

            # propage any new masters, propagate any to current head, otherwise identity
            def prop_decisions(body_decs):
                body, decs = body_decs
                dec1_ishead, dec1_reps, dec1_decisions, dec1_graph, readjust1 = decs
                
                # propagate decisions if there are any
                if dec1_ishead:
                    return [(body,decs)] # no-op
               
                props = []
                # add pointer to head
                head_node = dec1_reps[0]
                props.append((body, (False, [head_node], [dec1_decisions[0]], [dec1_graph[0]], []))) 
                # if there are other elements then propagate updates
                if len(dec1_reps) > 1:
                    props.append((head_node, (True, dec1_reps, dec1_decisions, dec1_graph, [])))

                # iterate through readjust list
                for bodyreadjust in readjust1:
                    props.append((bodyreadjust, (False, [head_node], [dec1_decisions[0]], [dec1_graph[0]], []))) 
                return props

            num_dec1 = -1
            num_dec2 = 0
            current_decisions2 = None
            # iterate until flatmap produces nothing new to propagate
            counter = 0
            while num_dec1 != num_dec2: 
                # group by body
                current_decisions = current_decisions.reduceByKey(combine_decisions)
                if num_dec1 == -1:
                    current_decisions.persist()
                    num_dec1 = current_decisions.count()
                    current_decisions.unpersist()
                
                # propagate changes (?? is data being recomputed)
                current_decisions2 = current_decisions.flatMap(prop_decisions)
                current_decisions2.persist()
                num_dec2 = current_decisions2.count()
                current_decisions2.unpersist()
                current_decisions = current_decisions2
                counter += 1
                print("Convergence check:", counter, num_dec1, num_dec2)


            # return merge list array and handle any constraint violations
            def solve_constraints(body_decs):
                body, decs = body_decs
                dec1_ishead, dec1_reps, dec1_decisions, dec1_graph, readjust1 = decs
                if not dec1_ishead:
                    assert len(dec1_reps) == 1, "more than one representative remains in pointer"
                    return []

                # create master merge list 
                merge_list = set()
                for iter1, curr_decs in enumerate(dec1_decisions):
                    head_node = dec1_reps[iter1]
                    for dec in curr_decs:
                        if head_node < dec:
                            head_node, dec = dec, head_node
                        merge_list.add((head_node, dec))
               
                # show all constraints
                constraints = set()
                for iter1, curr_graph in enumerate(dec1_graph):
                    head_node = dec1_reps[iter1]
                    for graphnode in curr_graph:
                        if head_node < graphnode:
                            head_node, graphnode = graphnode, head_node
                        constraints.add((head_node, graphnode))

                # check for violations
                def find_violations(bodylist, constraints):
                    violations = set()
                    for iter1 in range(0, len(bodylist)-1):
                        for iter2 in range(iter1, len(bodylist)):
                            n1 = bodylist[iter1]
                            n2 = bodylist[iter2]
                            if n1 < n2:
                                n1, n2 = n2, n1
                            if (n1, n2) in constraints:
                                violations.add((n1, n2))
                    return violations
                master_violations = find_violations(dec1_reps, constraints)

                def find_violations2(bodylist, constraints):
                    for (n1, n2) in constraints:
                        if n1 in bodylist and n2 in bodylist:
                            return True
                    return False

                # handle constraints
                if len(master_violations) > 0:
                    node_decisions = {}
                    node_mapping = {}
                    node_groupings = {} # rep body and set
                    fanout_order = []
                    for iter1, node in enumerate(dec1_reps):
                        node_mapping[node] = node
                        node_groupings[node] = set([node])
                        node_decisions[node] = dec1_decisions[iter1]
                        fanout_order.append((len(dec1_decisions[iter1]), node))
                    fanout_order.sort()
                    fanout_order.reverse()

                    # remove decisions from merge list as needed to prevent violations
                    while len(fanout_order) > 0:
                        dummy, head_node = fanout_order[0]
                        local_decisions = node_decisions[head_node]
                        curr_rep = node_mapping[head_node]
                        curr_set = node_groupings[curr_rep]

                        for dec in local_decisions:
                            curr_dec = node_mapping[dec]
                            # violation occurs remove decision
                            curr_set_temp = curr_set.union(node_groupings[curr_dec])
                            if find_violations2(curr_set_temp, master_violations):
                                # eliminate all pairs between two groups
                                for n1 in node_groupings[curr_dec]:
                                    for n2 in curr_set:
                                        if n1 < n2:
                                            n1, n2 = n2, n1
                                        if (n1, n2) in merge_list:
                                            merge_list.remove((n1,n2))
                                            #print "violation"
                            else:
                                # merge occurs combine sets
                                curr_set = curr_set_temp
                                max_id = max(curr_set)
                                node_groupings[max_id] = curr_set
                                for tempnode in curr_set:
                                    node_mapping[tempnode] = max_id

                            # remove decision from other node decision list
                            node_decisions[dec].remove(head_node)

                        del node_decisions[head_node]

                        fanout_order = []
                        for (node, decs) in node_decisions.items():
                            fanout_order.append((len(decs), node))
                        fanout_order.sort()
                        fanout_order.reverse()
                return list(merge_list)
              
            # map to convert each node to assignments (handle constraints)
            final_mappings = current_decisions.map(solve_constraints)

            # produce a new merge_list
            all_mappings = final_mappings.collect()
            for mapping in all_mappings:
                merge_list.extend(mapping)
        else:
            # key, mapping1; key mapping2 => key, mapping1+mapping2
            def reduce_mappings(b1, b2):
                b1.extend(b2)
                return b1

            # map from grouped boundary to substack id, mappings
            subvolume_mappings = grouped_boundaries.map(stitcher).reduceByKey(reduce_mappings)

            # reconcile all the mappings by sending them to the driver
            # (not a lot of data and compression will help but not sure if there is a better way)
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

        body2body = list(zip(body1body2.keys(), body1body2.values()))
       
        # potentially costly broadcast
        # (possible to split into substack to make more efficient but compression should help)
        master_merge_list = self.context.sc.broadcast(body2body)

        # use offset and mappings to relabel volume
        def relabel(key_label_mapping):
            import numpy

            (subvolume, labels) = key_label_mapping

            # grab broadcast offset
            offset = subvolume_offsets.value[subvolume.sv_index]

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
            labels_view = vigra.taggedView(labels.astype(numpy.uint64), 'zyx')
            mapping_col = numpy.sort( vigra.analysis.unique(labels_view) )
            label_mappings = dict(zip(mapping_col, mapping_col))
           
            # create maps from merge list
            for mapping in master_merge_list.value:
                if mapping[0] in label_mappings:
                    label_mappings[mapping[0]] = mapping[1]
                elif mapping[0] in relabeled_bodies:
                    label_mappings[relabeled_bodies[mapping[0]]] = mapping[1]

            # apply maps
            new_labels = numpy.empty_like( labels, dtype=numpy.uint64 )
            new_labels_view = vigra.taggedView(new_labels, 'zyx')
            vigra.analysis.applyMapping(labels_view, label_mappings, allow_incomplete_mapping=True, out=new_labels_view)
            return (subvolume, new_labels)

        # just map values with broadcast map
        # Potential TODO: consider fast join with partitioned map (not broadcast)
        # (subvol, labels) -> (subvol, labels)
        label_vols_rdd = select_item(label_chunks, 1, 0)
        return subvolumes_rdd.zip(label_vols_rdd).map(relabel)


