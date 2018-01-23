import os
import copy
import json
import tarfile
from datetime import datetime
import logging
from functools import partial
from io import BytesIO
from contextlib import closing

import numpy as np
import requests

from vol2mesh.mesh_from_array import mesh_from_array

from dvid_resource_manager.client import ResourceManagerClient

from DVIDSparkServices.util import Timer, persist_and_execute, unpersist, num_worker_nodes, cpus_per_worker
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 
from DVIDSparkServices.skeletonize_array import SkeletonConfigSchema, skeletonize_array
from DVIDSparkServices.reconutils.morpho import object_masks_for_labels, assemble_masks
from DVIDSparkServices.dvid.metadata import is_node_locked
from DVIDSparkServices.subprocess_decorator import execute_in_subprocess
from DVIDSparkServices.sparkdvid.CompressedNumpyArray import CompressedNumpyArray

from DVIDSparkServices.io_util.volume_service import DvidSegmentationVolumeSchema, LabelMapSchema
from DVIDSparkServices.io_util.labelmap_utils import load_labelmap

from multiprocessing import TimeoutError
from DVIDSparkServices.io_util.brickwall import BrickWall
from DVIDSparkServices.io_util.volume_service.volume_service import VolumeService

logger = logging.getLogger(__name__)

class CreateSkeletons(Workflow):
    SkeletonDvidInfoSchema = copy.deepcopy(DvidSegmentationVolumeSchema)
    SkeletonDvidInfoSchema["properties"]["dvid"]["properties"].update(
    {
        "skeletons-destination": {
            "description": "Name of key-value instance to store the skeletons. \n"
                           "By convention, this should usually be {segmentation-name}_skeletons, \n"
                           "which will be used by default if you don't provide this setting.\n",
            "type": "string",
            "default": ""
        },
        "meshes-destination": {
            "description": "Name of key-value instance to store the meshes. \n"
                           "By convention, this should usually be {segmentation-name}_meshes_tars for 'grouped' meshes,\n"
                           "or {segmentation-name}_meshes, if using 'no-groups',\n"
                           "which are the default names to be used if you don't provide this setting.\n",
            "type": "string",
            "default": ""
        }
    })

    MeshGenerationSchema = \
    {
        "type": "object",
        "description": "Mesh generation settings",
        "default": {},
        "properties": {
            "simplify-ratio": {
                "description": "Mesh simplification aims to reduce the number of \n"
                               "mesh vertices in the mesh to a fraction of the original mesh. \n"
                               "This ratio is the fraction to aim for.  To disable simplification, use 1.0.\n",
                "type": "number",
                "minimum": 0.0000001,
                "maximum": 1.0,
                "default": 0.2 # Set to 1.0 to disable.
            },
            "step-size": {
                "description": "Passed to skimage.measure.marching_cubes_lewiner().\n"
                               "Larger values result in coarser results via faster computation.\n",
                "type": "integer",
                "default": 1
            },
            "use-subprocesses": {
                "description": "Whether or not to generate meshes in a subprocess, \n"
                               "to protect against timeouts and failures.\n",
                "type": "boolean",
                "default": False
            },
            "storage": {
                "description": "Options to group meshes in tarballs, if desired",
                "type": "object",
                "default": {},
                "properties": {
                    "grouping-scheme": {
                        "description": "If/how to group meshes into tarballs before uploading them to DVID.\n"
                                       "Choices:\n"
                                       "- no-groups: No tarballs. Each mesh is written as a separate key.\n"
                                       "- singletons: Each mesh is written as a separate key, but it is wrapped in a 1-item tarball.\n"
                                       "- hundreds: Group meshes in groups of up to 100, such that ids xxx00 through xxx99 end up in the same group.\n"
                                       "- labelmap: Use the labelmap setting below to determine the grouping.\n",
                        "type": "string",
                        "enum": ["no-groups", "singletons", "hundreds", "labelmap"],
                        "default": "no-groups"
                    },
                    "naming-scheme": {
                        "description": "How to name mesh keys (and internal files, if grouped)",
                        "type": "string",
                        "enum": ["trivial", # Used for testing, and 'no-groups' mode.
                                 "neu3-level-0",  # Used for tarballs of supervoxels (one tarball per body)
                                 "neu3-level-1"], # Used for "tarballs" of pre-agglomerated bodies (one tarball per body, but only one file per tarball).
                        "default": "trivial"
                    },
                    "format": {
                        "description": "Format to save the meshes in. ",
                        "type": "string",
                        "enum": ["obj",    # Wavefront OBJ (.obj)
                                 "drc"],   # Draco (compressed) (.drc)
                        "default": "obj"
                    },
                    "labelmap": copy.copy(LabelMapSchema) # Only used by the 'labelmap' grouping-scheme
                }
            }
        }
    }
    
    MeshGenerationSchema\
        ["properties"]["storage"]\
            ["properties"]["labelmap"]\
                ["description"] = ("A labelmap file to determine mesh groupings.\n"
                                   "Only used by the 'labelmap' grouping-scheme.\n"
                                   "Will be applied AFTER any labelmap you specified in the segmentation volume info.\n")

    SkeletonWorkflowOptionsSchema = copy.copy(Workflow.OptionsSchema)
    SkeletonWorkflowOptionsSchema["additionalProperties"] = False
    SkeletonWorkflowOptionsSchema["properties"].update(
    {
        "output-types": {
            "description": "Either skeletons or meshes can be generated, or both.",
            "type": "array",
            "items": { "type": "string",
                       "enum": ["neutube-skeleton", "mesh"] },
            "minItems": 1,
            "default": ["neutube-skeleton"]
        },
        "minimum-segment-size": {
            "description": "Segments smaller than this voxel count will not be skeletonized",
            "type": "number",
            "default": 1e6
        },
        "downsample-factor": {
            "description": "Minimum factor by which to downsample bodies before skeletonization. \n"
                           "NOTE: If the object is larger than max-analysis-volume, even after \n"
                           "downsampling, then it will be downsampled even further before skeletonization. \n"
                           "The comments in the generated SWC file will indicate the final-downsample-factor. \n",
            "type": "integer",
            "default": 1
        },
        "max-analysis-volume": {
            "description": "The above downsample-factor will be overridden if the body would still \n"
                           "be too large to skeletonize, as defined by this setting.\n",
            "type": "number",
            "default": 1e9 # 1 GB max
        },
        "downsample-timeout": {
            "description": "The maximum time to wait for an object to be downsampled before skeletonization. \n"
                           "If timeout is exceeded, the an error is logged and the object is skipped.\n"
                           "IGNORED IF downsample-in-subprocess is False\n",
            "type": "number",
            "default": 600.0 # 10 minutes max
        },
        "downsample-in-subprocess":  {
            "description": "Collect and downsample each object in a subprocess, to protect against timeouts and failures.\n"
                           "Must be True for downsample-timeout to have any effect.\n",
            "type": "boolean",
            "default": True
        },
        "analysis-timeout": {
            "description": "The maximum time to wait for an object to be skeletonized or meshified. \n"
                           "If timeout is exceeded, the an error is logged and the object is skipped.\n",
            "type": "number",
            "default": 600.0 # 10 minutes max
        },
        "failed-mask-dir": {
            "description": "Volumes that fail to skeletonize (due to timeout) will \n"
                           "be written out as h5 files to this directory.",
            "type": "string",
            "default": "./failed-masks"
        },
        "rescale-before-write": {
            "description": "How much to rescale the skeletons/meshes before writing to DVID.\n"
                           "Specified as a multiplier, not power-of-2 'scale'.\n",
            "type": "number",
            "default": 1.0
        },
        "write-mask-stats":  {
            "description": "Debugging feature.  Writes a CSV file containing \n"
                           "information about the body masks computed during the job.",
            "type": "boolean",
            "default": False
        }
    })
    
    Schema = \
    {
      "$schema": "http://json-schema.org/schema#",
      "title": "Service to create skeletons from segmentation",
      "type": "object",
      "required": ["dvid-info"],
      "properties": {
        "dvid-info": SkeletonDvidInfoSchema,
        "skeleton-config": SkeletonConfigSchema,
        "mesh-config": MeshGenerationSchema,
        "options" : SkeletonWorkflowOptionsSchema
      }
    }

    @classmethod
    def schema(cls):
        return CreateSkeletons.Schema

    def __init__(self, config_filename):
        super(CreateSkeletons, self).__init__(config_filename, CreateSkeletons.schema(), "CreateSkeletons")
        
    def _sanitize_config(self):
        dvid_info = self.config_data['dvid-info']
        mesh_config = self.config_data["mesh-config"]
        options = self.config_data['options']
        
        # Convert failed-mask-dir to absolute path
        failed_skeleton_dir = options['failed-mask-dir']
        if failed_skeleton_dir and not os.path.isabs(failed_skeleton_dir):
            options['failed-mask-dir'] = self.relpath_to_abspath(failed_skeleton_dir)

        # Provide default skeletons instance name if needed
        if not dvid_info["dvid"]["skeletons-destination"]:
            dvid_info["dvid"]["skeletons-destination"] = dvid_info["dvid"]["segmentation-name"] + '_skeletons'

        # Provide default meshes instance name if needed
        if not dvid_info["dvid"]["meshes-destination"]:
            if mesh_config["storage"]["grouping-scheme"] == 'no-groups':
                suffix = "_meshes"
            else:
                suffix = "_meshes_tars"
            dvid_info["dvid"]["meshes-destination"] = dvid_info["dvid"]["segmentation-name"] + suffix

    
    def execute(self):
        self._sanitize_config()

        config = self.config_data
        options = config["options"]
        
        resource_mgr_client = ResourceManagerClient(options["resource-server"], options["resource-port"])
        volume_service = VolumeService.create_from_config(config["dvid-info"], self.config_dir, resource_mgr_client)

        self._init_skeletons_instance()

        # Aim for 2 GB RDD partitions
        GB = 2**30
        target_partition_size_voxels = 2 * GB // np.uint64().nbytes

        brick_wall = BrickWall.from_volume_service(volume_service, 0, None, self.sc, target_partition_size_voxels)
        brick_wall.persist_and_execute("Downloading segmentation", logger)
        
        # brick -> (body_id, (box, mask, count))
        body_ids_and_masks = brick_wall.bricks.flatMap( partial(body_masks, config) )
        persist_and_execute(body_ids_and_masks, "Computing brick-local masks", logger)
        brick_wall.unpersist()
        del brick_wall

        # In the case of catastrophic merges, some bodies may be too big to handle.
        # Skeletonizing them would probably time out anyway.
        bad_bodies = self.list_unmanageable_bodies(body_ids_and_masks)
        body_ids_and_masks = body_ids_and_masks.filter(lambda k_v: k_v[0] not in bad_bodies)

        # (body_id, (box, mask, count))
        #   --> (body_id, [(box, mask, count), (box, mask, count), (box, mask, count), ...])
        grouped_body_ids_and_masks = body_ids_and_masks.groupByKey()
        persist_and_execute(grouped_body_ids_and_masks, "Grouping masks by body id", logger)
        body_ids_and_masks.unpersist()
        del body_ids_and_masks

        # (Same RDD contents, but without small bodies)
        grouped_large_body_ids_and_masks = grouped_body_ids_and_masks.filter( partial(is_combined_object_large_enough, config) )
        persist_and_execute(grouped_large_body_ids_and_masks, "Filtering masks by size", logger)
        grouped_body_ids_and_masks.unpersist()
        del grouped_body_ids_and_masks

        @self.collect_log(lambda _: '_AGGREGATION_ERRORS')
        def logged_combine(arg):
            return combine_masks_in_subprocess(config, arg)

        #  --> (body_id, combined_box, mask, downsample_factor)
        id_box_mask_factor_err = grouped_large_body_ids_and_masks.map( logged_combine )
        persist_and_execute(id_box_mask_factor_err, "Downsampling and aggregating masks", logger)
        grouped_large_body_ids_and_masks.unpersist()
        del grouped_large_body_ids_and_masks

        # Errors were already written to a separate file, but let's duplicate them in the master log. 
        errors = id_box_mask_factor_err.map(lambda i_b_m_f_e: i_b_m_f_e[-1]).filter(bool).collect()
        for error in errors:
            logger.error(error)

        # Small bodies (or those with errors) were not processed,
        # and 'None' was returned instead of a mask. Remove them.
        def mask_is_not_none(i_b_m_f_e):
            _body_id, _combined_box, combined_mask, _downsample_factor, _error_msg = i_b_m_f_e
            return combined_mask is not None

        large_id_box_mask_factor_err = id_box_mask_factor_err.filter( mask_is_not_none )

        if "neutube-skeleton" in config["options"]["output-types"]:
            self._execute_skeletonization(large_id_box_mask_factor_err)

        if "mesh" in config["options"]["output-types"]:
            self._execute_mesh_generation(large_id_box_mask_factor_err)

    def list_unmanageable_bodies(self, body_ids_and_masks):
        """
        Return a set of body IDs whose aggregate compressed mask data is larger than 2GB, 
        and therefore unmanageable by Spark's groupByKey() operation.
        
        body_ids_and_masks: RDD of (body_id, (box, mask, count))
        """
        def mask_stats(element):
            (box, mask, count) = element
            assert isinstance(mask, CompressedNumpyArray)
            box = np.asarray(box)
            box_voxel_count = np.prod(box[1] - box[0])
            return (box_voxel_count, mask.compressed_nbytes, count)

        # (body_id, (box, mask, count))
        #   --> (body_id, [(box_voxel_count, mask_nbytes, mask_voxel_count), ...])
        logger.info("Collecting mask stats...")
        body_ids_and_stats = body_ids_and_masks.mapValues( mask_stats )
        grouped_body_ids_and_stats = body_ids_and_stats.groupByKey().collect()

        # Were any bodies too big?
        bad_body_ids = set()
        for body_id, elements in grouped_body_ids_and_stats:
            total_mask_nbytes = sum(mask_nbytes for (_, mask_nbytes, _) in elements)
            if total_mask_nbytes > 2e9:
                bad_body_ids.add(body_id)

        if not bad_body_ids and not self.config_data["options"]["write-mask-stats"]:
            return set()

        if bad_body_ids:
            logger.warn(f"Warning: {len(bad_body_ids)} bodies were too large to aggregate,"
                        f" and will be skipped: {list(bad_body_ids)}")

        # Write the body block statistics to a file.
        csv_path = self.relpath_to_abspath('.') + '/mask-stats.csv'
        logger.info(f"Writing mask stats to {csv_path}...")
        with open(csv_path, 'w') as f:
            f.write('body_id,total_box_voxel_count,total_mask_nbytes,total_mask_voxel_count,block_count\n')
            for body_id, elements in grouped_body_ids_and_stats:
                total_box_voxel_count = total_mask_nbytes = total_mask_voxel_count = 0
                for box_voxel_count, mask_nbytes, mask_voxel_count in elements:
                    total_box_voxel_count += box_voxel_count
                    total_mask_nbytes += mask_nbytes
                    total_mask_voxel_count += mask_voxel_count
                f.write(f'{body_id},{total_box_voxel_count},{total_mask_nbytes},{total_mask_voxel_count},{len(elements)}\n')
        logger.info(f"Done writing mask stats")
        return bad_body_ids

    def _execute_skeletonization(self, large_id_box_mask_factor_err):
        config = self.config_data
        @self.collect_log(lambda _: '_SKELETONIZATION_ERRORS')
        def logged_skeletonize(arg):
            return skeletonize_in_subprocess(config, arg)
        
        #     --> (body_id, swc_contents, error_msg)
        body_ids_and_skeletons = large_id_box_mask_factor_err.map( logged_skeletonize )
        persist_and_execute(body_ids_and_skeletons, "Computing skeletons", logger)

        # Errors were already written to a separate file, but let's duplicate them in the master log. 
        errors = body_ids_and_skeletons.map(lambda id_swc_err: id_swc_err[-1]).filter(bool).collect()
        for error in errors:
            logger.error(error)

        # Write
        with Timer() as timer:
            body_ids_and_skeletons.foreachPartition( partial(post_swcs_to_dvid, config) )
        logger.info(f"Writing skeletons to DVID took {timer.seconds}")


    def _execute_mesh_generation(self, large_id_box_mask_factor_err):
        config = self.config_data
        @self.collect_log(lambda _: '_MESH_GENERATION_ERRORS')
        def logged_generate_mesh(arg):
            return generate_mesh_in_subprocess(config, arg)
        
        #     --> (body_id, mesh_bytes, error_msg)
        body_ids_and_meshes_with_err = large_id_box_mask_factor_err.map( logged_generate_mesh )
        persist_and_execute(body_ids_and_meshes_with_err, "Computing meshes", logger)

        # Errors were already written to a separate file, but let's duplicate them in the master log. 
        errors = body_ids_and_meshes_with_err.map(lambda id_mesh_err: id_mesh_err[-1]).filter(bool).collect()
        for error in errors:
            logger.error(error)

        # Filter out error cases
        body_ids_and_meshes = body_ids_and_meshes_with_err.filter(lambda id_mesh_err: id_mesh_err[-1] is None) \
                                                          .map( lambda id_mesh_err: id_mesh_err[:2] )
                                                          
        # Group according to scheme
        grouping_scheme = config["mesh-config"]["storage"]["grouping-scheme"]
        n_partitions = num_worker_nodes() * cpus_per_worker()

        if grouping_scheme in "hundreds":
            def last_six_digits( id_mesh ):
                body_id, _mesh = id_mesh
                group_id = body_id - (body_id % 100)
                return group_id
            grouped_body_ids_and_meshes = body_ids_and_meshes.groupBy(last_six_digits, numPartitions=n_partitions)

        elif grouping_scheme == "labelmap":
            import pandas as pd
            mapping_pairs = load_labelmap( config["mesh-config"]["storage"]["labelmap"], self.config_dir )

            def prepend_mapped_group_id( id_mesh_partition ):
                df = pd.DataFrame( mapping_pairs, columns=["body_id", "group_id"] )

                new_partition = []
                for id_mesh in id_mesh_partition:
                    body_id, mesh = id_mesh
                    rows = df.loc[df.body_id == body_id]
                    if len(rows) == 0:
                        # If missing from labelmap,
                        # we assume an implicit identity mapping
                        group_id = body_id
                    else:
                        group_id = rows['group_id'].iloc[0]
                    new_partition.append( (group_id, (body_id, mesh)) )
                return new_partition
            
            # We do this via mapPartitions().groupByKey() instead of a simple groupBy()
            # to save time constructing the DataFrame inside the closure above.
            # (TODO: Figure out why the dataframe isn't pickling properly...)
            grouped_body_ids_and_meshes = body_ids_and_meshes.mapPartitions( prepend_mapped_group_id ) \
                                                             .groupByKey(numPartitions=n_partitions)
        elif grouping_scheme in ("singletons", "no-groups"):
            # Create 'groups' of one item each, re-using the body ID as the group id.
            # (The difference between 'singletons', and 'no-groups' is in how the mesh is stored, below.)
            grouped_body_ids_and_meshes = body_ids_and_meshes.map( lambda id_mesh: (id_mesh[0], [(id_mesh[0], id_mesh[1])]) )

        persist_and_execute(grouped_body_ids_and_meshes, f"Grouping meshes with scheme: '{grouping_scheme}'", logger)
        unpersist(body_ids_and_meshes)
        del body_ids_and_meshes
        
        with Timer() as timer:
            grouped_body_ids_and_meshes.foreachPartition( partial(post_meshes_to_dvid, config) )
        logger.info(f"Writing meshes to DVID took {timer.seconds}")


    def _init_skeletons_instance(self):
        dvid_info = self.config_data["dvid-info"]
        options = self.config_data["options"]
        if is_node_locked(dvid_info["dvid"]["server"], dvid_info["dvid"]["uuid"]):
            raise RuntimeError(f"Can't write skeletons/meshes: The node you specified ({dvid_info['dvid']['server']} / {dvid_info['dvid']['uuid']}) is locked.")

        node_service = retrieve_node_service( dvid_info["dvid"]["server"],
                                              dvid_info["dvid"]["uuid"],
                                              options["resource-server"],
                                              options["resource-port"] )

        if "neutube-skeleton" in options["output-types"]:
            node_service.create_keyvalue(dvid_info["dvid"]["skeletons-destination"])

        if "mesh" in options["output-types"]:
            node_service.create_keyvalue(dvid_info["dvid"]["meshes-destination"])


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


def combine_masks_in_subprocess(config, ids_and_boxes_and_compressed_masks):
    """
    Execute combine_masks() in a subprocess, and handle TimeoutErrors.
    """
    logger = logging.getLogger(__name__ + '.combine_masks')
    logger.setLevel(logging.WARN)
    timeout = config['options']['downsample-timeout']

    body_id, boxes_and_compressed_masks = ids_and_boxes_and_compressed_masks

    if config["options"]["downsample-in-subprocess"]:
        func = execute_in_subprocess(timeout, logger)(combine_masks)
    else:
        func = combine_masks
        
    try:
        body_id, combined_box, combined_mask_downsampled, chosen_downsample_factor = func(config, body_id, boxes_and_compressed_masks)
        return (body_id, combined_box, combined_mask_downsampled, chosen_downsample_factor, None)
    
    except TimeoutError: # fyi: can't be raised unless a subprocessed is used
        boxes, _compressed_masks, counts = zip(*boxes_and_compressed_masks)

        total_count = sum(counts)
        boxes = np.asarray(boxes)
        combined_box = np.zeros((2,3), dtype=np.int64)
        combined_box[0] = boxes[:, 0, :].min(axis=0)
        combined_box[1] = boxes[:, 1, :].max(axis=0)

        err_msg = f"Timeout ({timeout}) while downsampling/assembling body: id={body_id} size={total_count} box={combined_box.tolist()}"
        logger.error(err_msg)
        return (body_id, combined_box, None, None, err_msg)


def combine_masks(config, body_id, boxes_and_compressed_masks ):
    """
    Given a list of binary masks and corresponding bounding
    boxes, assemble them all into a combined binary mask.
    
    To save RAM, the data can be downsampled while it is added to the combined mask,
    resulting in a downsampled final mask.

    For more details, see documentation for assemble_masks().

    Returns: (body_id, combined_bounding_box, combined_mask, downsample_factor)

        where:
            body_id:
                As passed in
            
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
                        config["options"]["max-analysis-volume"],
                        suppress_zero=True,
                        pad=1 ) # mesh generation requires 1-px halo of zeros

    return (body_id, combined_box, combined_mask_downsampled, chosen_downsample_factor)


def skeletonize_in_subprocess(config, id_box_mask_factor_err):
    """
    Execute skeletonize() in a subprocess, and handle TimeoutErrors.
    """
    logger = logging.getLogger(__name__ + '.skeletonize')
    logger.setLevel(logging.WARN)
    timeout = config['options']['analysis-timeout']

    body_id, combined_box, combined_mask, downsample_factor, _err_msg = id_box_mask_factor_err

    try:
        func = execute_in_subprocess(timeout, logger)(skeletonize)
        body_id, swc = func(config, body_id, combined_box, combined_mask, downsample_factor)
        return (body_id, swc, None)
    except TimeoutError:
        err_msg = f"Timeout ({timeout}) while skeletonizing body: id={body_id} box={combined_box.tolist()}"     
        logger.error(err_msg)

        output_dir = config['options']['failed-mask-dir']
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            output_path = output_dir + f'/failed-skeleton-{body_id}.h5'
            logger.error(f"Writing mask to {output_path}")

            import h5py
            with h5py.File(output_path, 'w') as f:
                f["downsample_factor"] = downsample_factor
                f["box"] = combined_box
                f.create_dataset("mask", data=combined_mask, chunks=True)
        
        return (body_id, None, err_msg)


def skeletonize(config, body_id, combined_box, combined_mask, downsample_factor):
    (combined_box_start, _combined_box_stop) = combined_box

    # This config factor is an option to artificially scale the meshes up before
    # writing them, on top of whatever amount the data was downsampled.
    rescale_factor = config["options"]["rescale-before-write"]
    downsample_factor *= rescale_factor
    combined_box = combined_box * rescale_factor

    with Timer() as timer:
        # FIXME: Should the skeleton-config be tweaked in any way based on the downsample_factor??
        tree = skeletonize_array(combined_mask, config["skeleton-config"])
        tree.rescale(downsample_factor, downsample_factor, downsample_factor, True)
        tree.translate(*combined_box_start.astype(np.float64)[::-1]) # Pass x,y,z, not z,y,x

    del combined_mask
    
    swc_contents =   "# {:%Y-%m-%d %H:%M:%S}\n".format(datetime.now())
    swc_contents +=  "# Generated by the DVIDSparkServices 'CreateSkeletons' workflow.\n"
    swc_contents += f"# (Skeletonization time: {timer.seconds}):\n"
    swc_contents +=  "# Workflow configuration:\n"
    swc_contents +=  "# \n"

    # Also show which downsample factor was actually chosen
    config_copy = copy.deepcopy(config)
    config_copy["options"]["(final-downsample-factor)"] = downsample_factor
    
    config_comment = json.dumps(config_copy, sort_keys=True, indent=4, separators=(',', ': '))
    config_comment = "\n".join( "# " + line for line in config_comment.split("\n") )
    config_comment += "\n\n"

    swc_contents += config_comment + tree.toString()

    del tree

    return (body_id, swc_contents) # No error


def post_swcs_to_dvid(config, items):
    """
    Send the given SWC files as key/value pairs to DVID.
    
    Args:
        config: The CreateSkeletons workflow config data
    
        items: list-of-tuples (body_id, swc_text, error_text)
               If swc_text is None or error_text is NOT None, then nothing is posted.
               (We could have filtered out such items upstream, but it's convenient to just handle it here.)
    """
    # Re-use session for connection pooling.
    session = requests.Session()

    # Re-use resource manager client connections, too.
    # (If resource-server is empty, this will return a "dummy client")    
    resource_client = ResourceManagerClient( config["options"]["resource-server"],
                                             config["options"]["resource-port"] )

    dvid_server = config["dvid-info"]["dvid"]["server"]
    uuid = config["dvid-info"]["dvid"]["uuid"]
    instance = config["dvid-info"]["dvid"]["skeletons-destination"]

    for (body_id, swc_contents, err) in items:
        if swc_contents is None or err is not None:
            continue
    
        swc_contents = swc_contents.encode('utf-8')
        with resource_client.access_context(dvid_server, False, 1, len(swc_contents)):
            session.post(f'{dvid_server}/api/node/{uuid}/{instance}/key/{body_id}_swc', swc_contents)


def generate_mesh_in_subprocess(config, id_box_mask_factor_err):
    """
    If use_subprocess is True, execute generate_mesh() in a subprocess, and handle TimeoutErrors.
    Otherwise, just call generate_mesh() directly, with the appropriate parameters
    """
    logger = logging.getLogger(__name__ + '.generate_mesh')
    logger.setLevel(logging.WARN)
    timeout = config['options']['analysis-timeout']

    body_id, combined_box, combined_mask, downsample_factor, _err_msg = id_box_mask_factor_err

    if config["mesh-config"]["use-subprocesses"]:
        func = execute_in_subprocess(timeout, logger)(generate_mesh)
    else:
        func = generate_mesh

    try:
        _body_id, mesh_obj = func(config, body_id, combined_box, combined_mask, downsample_factor)
        return (body_id, mesh_obj, None)

    except TimeoutError: # fyi: can't be raised unless a subprocessed is used
        err_msg = f"Timeout ({timeout}) while meshifying body: id={body_id} box={combined_box.tolist()}"     
        logger.error(err_msg)

        output_dir = config['options']['failed-mask-dir']
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            output_path = output_dir + f'/failed-mesh-{body_id}.h5'
            logger.error(f"Writing mask to {output_path}")

            import h5py
            with h5py.File(output_path, 'w') as f:
                f["downsample_factor"] = downsample_factor
                f["box"] = combined_box
                f.create_dataset("mask", data=combined_mask, chunks=True)
        
        return (body_id, None, err_msg)


def generate_mesh(config, body_id, combined_box, combined_mask, downsample_factor):
    # This config factor is an option to artificially scale the meshes up before
    # writing them, on top of whatever amount the data was downsampled.
    rescale_factor = config["options"]["rescale-before-write"]
    downsample_factor *= rescale_factor
    combined_box = combined_box * rescale_factor

    mesh_bytes = mesh_from_array( combined_mask,
                                  combined_box[0],
                                  downsample_factor,
                                  config["mesh-config"]["simplify-ratio"],
                                  config["mesh-config"]["step-size"],
                                  config["mesh-config"]["storage"]["format"])
    return body_id, mesh_bytes


def post_meshes_to_dvid(config, partition_items):
    """
    Send the given meshes (either .obj or .drc) as key/value pairs to DVID.
    
    Args:
        config: The CreateSkeletons workflow config data
            
        items: tuple (body_id, mesh_data, error_text)
                      If mesh_data is None or error_text is NOT None, then nothing is posted.
                      (We could have filtered out such items upstream, but it's convenient to just handle it here.)

        session: A requests.Session object to re-use for posting data.                      
    """
    # Re-use session for connection pooling.
    session = requests.Session()

    # Re-use resource manager client connections, too.
    # (If resource-server is empty, this will return a "dummy client")    
    resource_client = ResourceManagerClient( config["options"]["resource-server"],
                                             config["options"]["resource-port"] )

    dvid_server = config["dvid-info"]["dvid"]["server"]
    uuid = config["dvid-info"]["dvid"]["uuid"]
    instance = config["dvid-info"]["dvid"]["meshes-destination"]
    
    grouping_scheme = config["mesh-config"]["storage"]["grouping-scheme"]

    if grouping_scheme == "no-groups":
        for group_id, body_ids_and_meshes in partition_items:
            for (body_id, mesh_data) in body_ids_and_meshes:
                with resource_client.access_context(dvid_server, False, 1, len(mesh_data)):
                    session.post(f'{dvid_server}/api/node/{uuid}/{instance}/key/{body_id}', mesh_data)
    else:
        # All other grouping schemes, including 'singletons' write tarballs.
        # (In the 'singletons' case, there is just one tarball per body.)
        for group_id, body_ids_and_meshes in partition_items:
            tar_name = _get_group_name(config, group_id)
            tar_stream = BytesIO()
            with closing(tarfile.open(tar_name, 'w', tar_stream)) as tf:
                for (body_id, mesh_data) in body_ids_and_meshes:
                    mesh_name = _get_mesh_name(config, body_id)
                    f_info = tarfile.TarInfo(mesh_name)
                    f_info.size = len(mesh_data)
                    tf.addfile(f_info, BytesIO(mesh_data))
    
            tar_bytes = tar_stream.getbuffer()
            with resource_client.access_context(dvid_server, False, 1, len(tar_bytes)):
                session.post(f'{dvid_server}/api/node/{uuid}/{instance}/key/{tar_name}', tar_bytes)

def _get_group_name(config, group_id):
    """
    Encode the given group name (e.g. a 'body' in neu3)
    into a suitable key name for the group tarball.
    """
    grouping_scheme = config["mesh-config"]["storage"]["grouping-scheme"]
    naming_scheme = config["mesh-config"]["storage"]["naming-scheme"]

    if naming_scheme == "trivial":
        group_name = str(group_id) # no special encoding
    elif naming_scheme == "neu3-level-0":
        keyEncodeLevel0 = 10000000000000
        group_name = str(group_id + keyEncodeLevel0)
    elif naming_scheme == "neu3-level-1":
        keyEncodeLevel1 = 10100000000000
        group_name = str(group_id + keyEncodeLevel1)
    else:
        raise RuntimeError(f"Unknown naming scheme: {naming_scheme}")
    
    if grouping_scheme != 'no-groups':
        group_name += '.tar'
    
    return group_name

def _get_mesh_name(config, mesh_id):
    """
    Encode the given mesh id (e.g. a 'supervoxel ID' in neu3)
    into a suitable filename for inclusion in a mesh tarball.
    """
    naming_scheme = config["mesh-config"]["storage"]["naming-scheme"]
    mesh_format = config["mesh-config"]["storage"]["format"]

    if naming_scheme == "trivial":
        mesh_name = str(mesh_id) # no special encoding
    elif naming_scheme == "neu3-level-0":
        fileEncodeLevel0 = 0 # identity (supervoxel names remain unchanged)
        mesh_name = str(mesh_id + fileEncodeLevel0)
    elif naming_scheme == "neu3-level-1":
        fileEncodeLevel1 = 100000000000
        mesh_name = str(mesh_id + fileEncodeLevel1)
    else:
        raise RuntimeError(f"Unknown naming scheme: {naming_scheme}")

    mesh_name += '.' + mesh_format
    return mesh_name

