import os
import copy
import tarfile
import socket
import logging
from functools import partial
from io import BytesIO
from contextlib import closing

import numpy as np
import pandas as pd

from vol2mesh.mesh import Mesh, concatenate_meshes

from dvid_resource_manager.client import ResourceManagerClient

import DVIDSparkServices.rddtools as rt
from DVIDSparkServices.auto_retry import auto_retry
from DVIDSparkServices.util import Timer, persist_and_execute, num_worker_nodes, cpus_per_worker, default_dvid_session
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.sparkdvid.sparkdvid import sparkdvid, retrieve_node_service 
from DVIDSparkServices.reconutils.morpho import object_masks_for_labels
from DVIDSparkServices.dvid.metadata import is_node_locked

from DVIDSparkServices.io_util.volume_service import DvidSegmentationVolumeSchema, LabelMapSchema, LabelmappedVolumeService
from DVIDSparkServices.io_util.labelmap_utils import load_labelmap

from DVIDSparkServices.io_util.brick import Grid, SparseBlockMask
from DVIDSparkServices.io_util.brickwall import BrickWall
from DVIDSparkServices.io_util.volume_service.volume_service import VolumeService
from DVIDSparkServices.io_util.volume_service.dvid_volume_service import DvidVolumeService

from DVIDSparkServices.segstats import aggregate_segment_stats_from_bricks

from DVIDSparkServices.subprocess_decorator import execute_in_subprocess
from multiprocessing import TimeoutError

logger = logging.getLogger(__name__)

class CreateStitchedMeshes(Workflow):
    MeshDvidInfoSchema = copy.deepcopy(DvidSegmentationVolumeSchema)
    MeshDvidInfoSchema["properties"]["dvid"]["properties"].update(
    {
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
        "additionalProperties": False,
        "default": {},
        "properties": {
            "task-block-shape": {
                "description": "Meshes will be generated in blocks before stitching. This specifies the block shape.\n"
                               "By default, the input's preferred-message-shape is used.\n",
                "type": "array",
                "items": { "type": "integer" },
                "minItems": 3,
                "maxItems": 3,
                "default": [-1,-1,-1] # use preferred-message-shape
            },
            "task-block-halo": {
                "description": "Meshes will be generated in blocks before stitching.\n"
                               "This setting specifies the width of the halo around each block,\n"
                               "to ensure overlapping coverage of the computed meshes.\n"
                               "A halo of 1 pixel suffices if no decimation will be applied.\n"
                               "When using smoothing and/or decimation, a halo of 2 or more is better to avoid artifacts.",
                "type": "integer",
                "default": 1
            },
            "pre-stitch-smoothing-iterations": {
                "description": "How many iterations of smoothing to apply to the mesh BEFORE the block meshes are stitched",
                "type": "integer",
                "default": 0
            },

            "post-stitch-smoothing-iterations": {
                "description": "How many iterations of smoothing to apply to the mesh AFTER the block meshes are stitched together",
                "type": "integer",
                "default": 3
            },
            
            "pre-stitch-decimation": {
                "description": "Mesh decimation aims to reduce the number of \n"
                               "mesh vertices in the mesh to a fraction of the original mesh. \n"
                               "This setting is the fraction to aim for BEFORE stitching block meshes together.\n"
                               "To disable decimation, use 1.0.\n",
                "type": "number",
                "minimum": 0.0000001,
                "maximum": 1.0, # 1.0 == disable
                "default": 0.1
            },

            "post-stitch-decimation": {
                "description": "Mesh decimation aims to reduce the number of \n"
                               "mesh vertices in the mesh to a fraction of the original mesh. \n"
                               "This setting is the fraction to aim for AFTER stitching block meshes together.\n"
                               "To disable decimation, use 1.0.\n",
                "type": "number",
                "minimum": 0.0000001,
                "maximum": 1.0, # 1.0 == disable
                "default": 0.1
            },
            
            "post-stitch-max-vertices": {
                "description": "After stitching, further decimate the mesh (if necessary) "
                               "to have no more than this vertex count in the ENTIRE BODY.\n"
                               "For 'no max', set to -1",
                "type": "number",
                "default": -1 # No max
            },
            
            "compute-normals": {
                "description": "Compute vertex normals and include them in the uploaded results.",
                "type": "boolean",
                "default": False # Default is false for now because Neu3 doesn't read them properly.
            },
    

            "stitch-method": {
                "description": "How to combine each segment's blockwise meshes into a single file.",
                "type": "string",
                "enum": ["simple-concatenate", # Just dump the vertices and faces into the same file
                                               # (renumber the faces to match the vertices, but don't unify identical vertices.)
                                               # If using this setting it is important to use a task-block-halo of > 2 to hide
                                               # the seams, even if smoothing is used.

                         "stitch",             # Search for duplicate vertices and remap the corresponding face corners,
                                               # so that the duplicate entries are not used. Topologically stitches adjacent faces.
                                               # Will be ineffective unless you used a task-block-halo of at least 1, and no
                                               # pre-stitch smoothing or decimation.

                         "stitch-and-filter"], # Same as above, but also filter out duplicate vertices and deduplicate faces.
                
                "default": "simple-concatenate",
            },

            "rescale-before-write": {
                "description": "How much to rescale the meshes before writing to DVID.\n"
                               "Specified as a multiplier, not power-of-2 'scale'.\n",
                "type": "number",
                "default": 1.0
            },
            "storage": {
                "description": "Options to group meshes in tarballs, if desired",
                "type": "object",
                "default": {},
                "additionalProperties": False,
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
                    
                    "labelmap": copy.copy(LabelMapSchema), # Only used by the 'labelmap' grouping-scheme

                    "subset-bodies": {
                        "description": "(Optional.) Instead of generating meshes for all meshes in the volume,\n"
                                       "only generate meshes for a subset of the bodies in the volume.\n",
                        "type": "array",
                        "default": []
                    },
                    "input-is-mapped-supervoxels": {
                        "description": "When using 'subset-bodies' option, we need to know how to fetch sparse blocks from dvid.\n"
                                       "If the input is a pre-mapped supervoxel volume, we'll have to use the supervoxel IDs (not body ids) when fetching from DVID.\n"
                                       "This option will probably become unnecessary once dvid's native 'labelmap' datatype is ready, since we'll always have a 'materialized' segmentation to use.",
                        "type": "boolean",
                        "default": False
                    }
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

    MeshWorkflowOptionsSchema = copy.copy(Workflow.OptionsSchema)
    MeshWorkflowOptionsSchema["additionalProperties"] = False
    MeshWorkflowOptionsSchema["properties"].update(
    {
        "minimum-segment-size": {
            "description": "Segments smaller than this voxel count will not be processed.",
            "type": "number",
            "default": 1
        },
        "maximum-segment-size": {
            "description": "Segments larger than this voxel count will not be processed. (Useful for avoiding processing errors.)",
            "type": "number",
            "default": 1e100 # unbounded by default
        },
        "minimum-agglomerated-size": {
            "description": "Agglomerated groups smaller than this voxel count will not be processed.",
            "type": "number",
            "default": 1e6 # 1 Megavoxel
        },
        "maximum-agglomerated-size": {
            "description": "Agglomerated groups larger than this voxel count will not be processed.",
            "type": "number",
            "default": 10e9 # 10 Gigavoxels
        },
        "minimum-agglomerated-segment-count": {
            "description": "Don't bother with any meshes that are part of a body with fewer segments than this setting.",
            "type": "integer",
            "default": 0
        },
        
        "force-evaluation-checkpoints": {
            "description": "Debugging feature. Force persistence of RDDs after every map step, for better logging.",
            "type": "boolean",
            "default": True
        }
    })
    

    Schema = \
    {
      "$schema": "http://json-schema.org/schema#",
      "title": "Service to create meshes from segmentation",
      "type": "object",
      "required": ["dvid-info"],
      "properties": {
        "dvid-info": MeshDvidInfoSchema,
        "mesh-config": MeshGenerationSchema,
        "options" : MeshWorkflowOptionsSchema
      }
    }


    @classmethod
    def schema(cls):
        return CreateStitchedMeshes.Schema


    def __init__(self, config_filename):
        super(CreateStitchedMeshes, self).__init__(config_filename, CreateStitchedMeshes.schema(), "CreateStitchedMeshes")
        self._labelmap = None


    def _sanitize_config(self):
        dvid_info = self.config_data['dvid-info']
        mesh_config = self.config_data["mesh-config"]
        
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
        bad_mesh_dir = f"{self.config_dir}/bad-meshes"
        os.makedirs(bad_mesh_dir, exist_ok=True)

        
        resource_mgr_client = ResourceManagerClient(options["resource-server"], options["resource-port"])
        volume_service = VolumeService.create_from_config(config["dvid-info"], self.config_dir, resource_mgr_client)

        self._init_meshes_instance()

        # Aim for 10 GB RDD partitions -- too many partitions causes a crash on the DRIVER because Spark is not good at its job.
        GB = 2**30
        target_partition_size_voxels = 10 * GB // np.uint64().nbytes
        
        # This will return None if we're not using sparse blocks
        sparse_block_mask = self._get_sparse_block_mask(volume_service, config["mesh-config"]["storage"]["input-is-mapped-supervoxels"])
        
        # Bricks have a halo of 1 to ensure that there will be no gaps between meshes from different blocks
        brick_wall = BrickWall.from_volume_service(volume_service, 0, None, self.sc, target_partition_size_voxels, sparse_block_mask)
        brick_wall.drop_empty()
        brick_wall.persist_and_execute("Downloading segmentation", logger)
        
        mesh_task_shape = np.array(config["mesh-config"]["task-block-shape"])
        if (mesh_task_shape < 1).any():
            assert (mesh_task_shape < 1).all()
            mesh_task_shape = volume_service.preferred_message_shape
        
        mesh_task_grid = Grid( mesh_task_shape, halo=config["mesh-config"]["task-block-halo"] )
        if not brick_wall.grid.equivalent_to( mesh_task_grid ):
            aligned_wall = brick_wall.realign_to_new_grid(mesh_task_grid)
            aligned_wall.persist_and_execute("Aligning bricks to mesh task grid...", logger)
            brick_wall.unpersist()
            brick_wall = aligned_wall

        full_stats_df = self.compute_segment_and_body_stats(brick_wall.bricks)
        keep_col = full_stats_df['keep_segment'] & full_stats_df['keep_body']
        if keep_col.all():
            segments_to_keep = None # keep everything
        else:
            # Note: This array will be broadcasted to the workers.
            #       It will be potentially quite large if we're keeping most (but not all) segments.
            #       Broadcast expense should be minimal thanks to lz4 compression,
            #       but RAM usage will be high.
            segments_to_keep = full_stats_df['segment'][keep_col].values
            if segments_to_keep.max() < np.iinfo(np.uint32).max:
                segments_to_keep = segments_to_keep.astype(np.uint32) # Save some RAM
        
        def generate_meshes_for_brick( brick ):
            import DVIDSparkServices # Ensure faulthandler logging is active.
            if segments_to_keep is None:
                filtered_volume = brick.volume
            else:
                # Mask out segments we don't want to process
                filtered_volume = brick.volume.copy('C')
                filtered_flat = filtered_volume.reshape(-1)
                s = pd.Series(filtered_flat)
                filter_mask = ~s.isin(segments_to_keep).values
                filtered_flat[filter_mask] = 0

            ids_and_mesh_datas = []
            for (segment_id, (box, mask, _count)) in object_masks_for_labels(filtered_volume, brick.physical_box):
                mesh = Mesh.from_binary_vol(mask, box)
                mesh.normals_zyx = np.zeros((0,3), np.float32) # discard normals now; they would be discarded later, anyway
                mesh_compressed_size = mesh.compress()
                ids_and_mesh_datas.append( (segment_id, (mesh, mesh_compressed_size)) )

            # We're done with the segmentation at this point. Save some RAM.
            brick.destroy()
            return ids_and_mesh_datas

        # Compute meshes per-block
        # --> (segment_id, (mesh_for_one_block, compressed_size))
        segment_ids_and_mesh_blocks = brick_wall.bricks.flatMap( generate_meshes_for_brick )
        rt.persist_and_execute(segment_ids_and_mesh_blocks, "Computing block segment meshes", logger)
        
        segments_and_counts_and_size = segment_ids_and_mesh_blocks \
                                       .map( lambda seg_mesh_size: (seg_mesh_size[0], (len(seg_mesh_size[1][0].vertices_zyx), seg_mesh_size[1][1]) ) ) \
                                       .groupByKey() \
                                       .map( lambda seg_counts_size: (seg_counts_size[0], *np.array(list(seg_counts_size[1])).sum(axis=0) ) ) \
                                       .collect()

        counts_df = pd.DataFrame( segments_and_counts_and_size, columns=['segment', 'prestitch_vertex_count', 'total_compressed_size'] )
        del segments_and_counts_and_size
        counts_df.to_csv(self.relpath_to_abspath('prestitch-vertex-counts.csv'), index=False)
        with Timer("Merging prestitch_vertex_count onto segment stats", logger):
            full_stats_df = full_stats_df.merge(counts_df, 'left', on='segment')
        
        # Drop size
        # --> (segment_id, mesh_for_one_block)
        segment_ids_and_mesh_blocks = segment_ids_and_mesh_blocks.map(lambda a_bc: (a_bc[0], a_bc[1][0]))
        
        # Pre-stitch smoothing
        # --> (segment_id, mesh_for_one_block)
        smoothing_iterations = config["mesh-config"]["pre-stitch-smoothing-iterations"]
        if smoothing_iterations > 0:
            def smooth(mesh):
                import DVIDSparkServices # Ensure faulthandler logging is active.
                mesh.laplacian_smooth(smoothing_iterations)
                return mesh
            segment_id_and_smoothed_mesh = segment_ids_and_mesh_blocks.mapValues( smooth )
    
            rt.persist_and_execute(segment_id_and_smoothed_mesh, "Smoothing block meshes", logger)
            rt.unpersist(segment_ids_and_mesh_blocks)
            segment_ids_and_mesh_blocks = segment_id_and_smoothed_mesh
            del segment_id_and_smoothed_mesh

        # Pre-stitch decimation
        # --> (segment_id, mesh_for_one_block)
        decimation_fraction = config["mesh-config"]["pre-stitch-decimation"]
        if decimation_fraction < 1.0:
            @self.collect_log(lambda _: socket.gethostname() + '-mesh-decimation')
            def decimate(id_mesh):
                import DVIDSparkServices # Ensure faulthandler logging is active.
                segment_id, mesh = id_mesh
                timeout = 300.0 # 5 minutes
                logger = logging.getLogger(__name__)
                subproc_decimator = execute_in_subprocess(timeout, logger)(decimate_mesh)
                try:
                    mesh = subproc_decimator(decimation_fraction, mesh)
                    return (segment_id, mesh)
                except TimeoutError:
                    bad_mesh_export_path = f'{bad_mesh_dir}/failed-decimation-{decimation_fraction:.2f}-{segment_id}.obj'
                    mesh.serialize(f'{bad_mesh_export_path}')
                    logger.error(f"Timed out while decimating a block mesh! Skipped decimation and wrote bad mesh to {bad_mesh_export_path}")
                    return (segment_id, mesh)

            segment_id_and_decimated_mesh = segment_ids_and_mesh_blocks.map(decimate)

            rt.persist_and_execute(segment_id_and_decimated_mesh, "Decimating block meshes", logger)
            rt.unpersist(segment_ids_and_mesh_blocks)
            segment_ids_and_mesh_blocks = segment_id_and_decimated_mesh
            del segment_id_and_decimated_mesh
        
        if (smoothing_iterations > 0 or decimation_fraction < 1.0) and config["mesh-config"]["compute-normals"]:
            # Compute normals
            def recompute_normals(mesh):
                import DVIDSparkServices # Ensure faulthandler logging is active.
                mesh.recompute_normals()
                return mesh
            
            segment_id_and_mesh_with_normals = segment_ids_and_mesh_blocks.map(decimate)

            rt.persist_and_execute(segment_id_and_mesh_with_normals, "Computing block mesh normals", logger)
            rt.unpersist(segment_ids_and_mesh_blocks)
            segment_ids_and_mesh_blocks = segment_id_and_mesh_with_normals
            del segment_id_and_mesh_with_normals
        
        # Group by segment ID
        # --> (segment_id, [mesh_for_block, mesh_for_block, ...])
        mesh_blocks_grouped_by_segment = segment_ids_and_mesh_blocks.groupByKey()
        rt.persist_and_execute(mesh_blocks_grouped_by_segment, "Grouping block segment meshes", logger)
        rt.unpersist(segment_ids_and_mesh_blocks)
        del segment_ids_and_mesh_blocks
        
        # Concatenate into a single mesh per segment
        # --> (segment_id, mesh)
        stitch_method = config["mesh-config"]["stitch-method"]
        @self.collect_log()
        def concatentate_and_stitch(meshes):
            import DVIDSparkServices # Ensure faulthandler logging is active.
            def _impl():
                concatenated_mesh = concatenate_meshes(meshes)
                for mesh in meshes:
                    mesh.destroy() # Save RAM -- we're done with the block meshes at this point
    
                if stitch_method == "stitch":
                    concatenated_mesh.stitch_adjacent_faces(False, False)
                elif stitch_method == "stitch-and-filter":
                    concatenated_mesh.stitch_adjacent_faces(True, True)
    
                concatenated_mesh.compress()
                return concatenated_mesh
            
            total_vertices = sum(len(mesh.vertices_zyx) for mesh in meshes)
            if (total_vertices) < 10e6:
                return _impl()
            with Timer(f"Concatenating a big mesh ({total_vertices} vertices)", logging.getLogger(__name__)):
                return _impl()
            
        segment_id_and_mesh = mesh_blocks_grouped_by_segment.mapValues(concatentate_and_stitch)
        
        rt.persist_and_execute(segment_id_and_mesh, "Stitching block segment meshes", logger)
        rt.unpersist(mesh_blocks_grouped_by_segment)
        del mesh_blocks_grouped_by_segment

        # Post-stitch Smoothing
        # --> (segment_id, mesh)
        smoothing_iterations = config["mesh-config"]["post-stitch-smoothing-iterations"]
        if smoothing_iterations > 0:
            def smooth(mesh):
                import DVIDSparkServices # Ensure faulthandler logging is active.
                mesh.laplacian_smooth(smoothing_iterations)
                return mesh
            segment_id_and_smoothed_mesh = segment_id_and_mesh.mapValues( smooth )
    
            rt.persist_and_execute(segment_id_and_smoothed_mesh, "Smoothing stitched meshes", logger)
            rt.unpersist(segment_id_and_mesh)
            segment_id_and_mesh = segment_id_and_smoothed_mesh
            del segment_id_and_smoothed_mesh

        # Post-stitch decimation
        # --> (segment_id, mesh)
        decimation_fraction = config["mesh-config"]["post-stitch-decimation"]
        max_vertices = config["mesh-config"]["post-stitch-max-vertices"]
        
        # body is the INDEX
        body_vertex_counts = full_stats_df[['body', 'prestitch_vertex_count']].groupby('body').sum()
        body_vertex_counts.columns = ['body_prestitch_vertex_count']
        full_stats_df = full_stats_df.merge(body_vertex_counts, 'inner', left_on='body', right_index=True, copy=False)
        
        # segment is the INDEX
        body_prestitch_vertex_counts_df = full_stats_df[['segment', 'body_prestitch_vertex_count']].set_index('segment')
        
        if decimation_fraction < 1.0 or max_vertices > 0:
            @self.collect_log(lambda *_a, **_kw: 'post-stitch-decimation', logging.WARNING)
            def decimate(seg_and_mesh):
                import DVIDSparkServices # Ensure faulthandler logging is active.
                segment_id, mesh = seg_and_mesh
                final_decimation = decimation_fraction

                # If the total vertex count of all segments in this segment's
                # body would be too large, apply further decimation.                
                body_prestitch_vertex_count = body_prestitch_vertex_counts_df[segment_id]
                if final_decimation * body_prestitch_vertex_count > max_vertices:
                    final_decimation = max_vertices / body_prestitch_vertex_count
                mesh.simplify(final_decimation)
                return (segment_id, mesh)
            segment_id_and_decimated_mesh = segment_id_and_mesh.map(decimate)

            rt.persist_and_execute(segment_id_and_decimated_mesh, "Decimating stitched meshes", logger)
            rt.unpersist(segment_id_and_mesh)
            segment_id_and_mesh = segment_id_and_decimated_mesh
            del segment_id_and_decimated_mesh

        # Get post-decimation vertex count and ovewrite stats file
        segments_and_counts = segment_id_and_mesh.map(lambda id_mesh: (id_mesh[0], len(id_mesh[1].vertices_zyx))).collect()
        counts_df = pd.DataFrame( segments_and_counts, columns=['segment', 'post_stitch_decimated_vertex_count'] )
        counts_df.to_csv(self.relpath_to_abspath('poststitch-vertex-counts.csv'),  index=False)

        # Post-stitch normals
        # --> (segment_id, mesh)
        if (smoothing_iterations > 0 or decimation_fraction < 1.0) and config["mesh-config"]["compute-normals"]:
            def decimate(mesh):
                import DVIDSparkServices # Ensure faulthandler logging is active.
                mesh.recompute_normals()
                return mesh
            segment_id_and_mesh_with_normals = segment_id_and_mesh.mapValues(decimate)

            rt.persist_and_execute(segment_id_and_mesh_with_normals, "Computing stitched mesh normals", logger)
            rt.unpersist(segment_id_and_mesh)
            segment_id_and_mesh = segment_id_and_mesh_with_normals
            del segment_id_and_mesh_with_normals

        rescale_factor = config["mesh-config"]["rescale-before-write"]
        if rescale_factor != 1.0:
            def rescale(mesh):
                mesh.vertices_zyx *= rescale_factor
                return mesh
            segment_id_and_rescaled_mesh = segment_id_and_mesh.mapValues(rescale)
            
            rt.persist_and_execute(segment_id_and_rescaled_mesh, "Rescaling segment meshes", logger)
            rt.unpersist(segment_id_and_mesh)
            segment_id_and_mesh = segment_id_and_rescaled_mesh
            del segment_id_and_rescaled_mesh

        # Serialize
        # --> (segment_id, mesh_bytes)
        fmt = config["mesh-config"]["storage"]["format"]
        @self.collect_log(echo_threshold=logging.INFO)
        def serialize(id_mesh):
            import DVIDSparkServices # Ensure faulthandler logging is active.
            segment_id, mesh = id_mesh
            try:
                if len(mesh.vertices_zyx) < 10e6:
                    return (segment_id, mesh.serialize(fmt=fmt))
            
                with Timer(f"Serializing a big mesh ({len(mesh.vertices_zyx)} vertices)", logging.getLogger(__name__)):
                    return (segment_id, mesh.serialize(fmt=fmt))
            except:
                # This assumes that we can still it serialize in obj format...
                output_path = f'{bad_mesh_dir}/failed-serialization-{segment_id}.obj'
                mesh.serialize(output_path)
                logger = logger.getLogger(__name__)
                logger.error(f"Failed to serialize mesh.  Wrote to {output_path}")
                return (segment_id, b'')

        segment_id_and_mesh_bytes = segment_id_and_mesh.map( serialize ) \
                                                       .filter(lambda mesh_bytes: len(mesh_bytes) > 0)

        rt.persist_and_execute(segment_id_and_mesh_bytes, "Serializing segment meshes", logger)
        rt.unpersist(segment_id_and_mesh)
        del segment_id_and_mesh

        # Group by body ID
        # --> ( body_id, [( segment_id, mesh_bytes ), ( segment_id, mesh_bytes ), ...] )
        segment_id_and_mesh_bytes_grouped_by_body = self.group_by_body(segment_id_and_mesh_bytes)

        instance_name = config["dvid-info"]["dvid"]["meshes-destination"]
        with Timer("Writing meshes to DVID", logger):
            segment_id_and_mesh_bytes_grouped_by_body.foreachPartition( partial(post_meshes_to_dvid, config, instance_name) )

            
    def _init_meshes_instance(self):
        dvid_info = self.config_data["dvid-info"]
        options = self.config_data["options"]
        if is_node_locked(dvid_info["dvid"]["server"], dvid_info["dvid"]["uuid"]):
            raise RuntimeError(f"Can't write meshes: The node you specified ({dvid_info['dvid']['server']} / {dvid_info['dvid']['uuid']}) is locked.")

        node_service = retrieve_node_service( dvid_info["dvid"]["server"],
                                              dvid_info["dvid"]["uuid"],
                                              options["resource-server"],
                                              options["resource-port"] )

        instance_name = dvid_info["dvid"]["meshes-destination"]
        node_service.create_keyvalue( instance_name )


    def _get_sparse_block_mask(self, volume_service, use_service_labelmap=False):
        """
        If the user's config specified a sparse subset of bodies to process,
        Return a SparseBlockMask object indicating where those bodies reside.
        
        If the user did not specify a 'subset-bodies' list, returns None, indicating
        that all segmentation blocks in the volume should be read.
        
        Also, if the input volume is not from a DvidVolumeService, return None.
        (In that case, the 'subset-bodies' feature can be used, but it isn't as efficient.)
        """
        config = self.config_data
        
        sparse_body_ids = config["mesh-config"]["storage"]["subset-bodies"]
        if not sparse_body_ids:
            return None

        assert isinstance(volume_service.base_service, DvidVolumeService), \
            "Can't use subset-bodies feature for non-DVID sources" 

        assert (volume_service.base_service.bounding_box_zyx == volume_service.bounding_box_zyx).all(), \
            "Can't use subset-bodies feature with transposed or rescaled services"
        
        grouping_scheme = config["mesh-config"]["storage"]["grouping-scheme"]
        assert grouping_scheme in ('no-groups', 'singletons', 'labelmap'), \
            f"Not allowed to use 'subset-bodies' setting for grouping scheme: {grouping_scheme}"

        mapping_pairs = None
        if use_service_labelmap:
            assert isinstance(volume_service, LabelmappedVolumeService), \
                "Cant' use service labelmap: The input isn't a LabelmappedVolumeService"
            mapping_pairs = volume_service.mapping_pairs
        elif grouping_scheme == 'labelmap':
            mapping_pairs = self.load_labelmap()

        if mapping_pairs is not None:
            segments, bodies = mapping_pairs.transpose()
            
            # pandas.Series permits duplicate index values,
            # which is convenient for this reverse lookup
            reverse_lookup = pd.Series(index=bodies, data=segments)
            sparse_segment_ids = reverse_lookup.loc[sparse_body_ids].values
        else:
            # No labelmap: The 'body ids' are identical to segment ids
            sparse_segment_ids = sparse_body_ids

        logger.info("Reading sparse block mask for body subset...")
        # Fetch the sparse mask of blocks that the sparse segments belong to
        dvid_service = volume_service.base_service
        block_mask, lowres_box, block_shape = \
            sparkdvid.get_union_block_mask_for_bodies( dvid_service.server,
                                                       dvid_service.uuid,
                                                       dvid_service.instance_name,
                                                       sparse_segment_ids )

        fullres_box = lowres_box * block_shape
        return SparseBlockMask(block_mask, fullres_box, block_shape)


    def load_labelmap(self):
        """
        Load the labelmap for aggregating segments into bodies.
        Note that this is NOT the same as the labelmap (if any) that may
        be involved in the original segment source.
        """
        if self._labelmap is None:
            config = self.config_data
            grouping_scheme = config["mesh-config"]["storage"]["grouping-scheme"]
            assert grouping_scheme == 'labelmap'
            self._labelmap = load_labelmap( config["mesh-config"]["storage"]["labelmap"], self.config_dir )
        return self._labelmap


    def compute_segment_and_body_stats(self, bricks):
        """
        For the given RDD of Brick objects, compute the statistics for all
        segments and their associated bodies (if using the labelmap grouping scheme).
        
        The returned DataFrame will contain the following columns:
        
        segment, segment_voxel_count, body, body_voxel_count, keep_segment, keep_body.
        
        Note that "keep_segment" and "keep_body" are computed independently.
        A segment should only be kept if BOTH of those columns are True for
        that segment's row in the DataFrame.
        
        Before returning, the DataFrame is also written to disk.
        """
        config = self.config_data
        
        with Timer(f"Computing segment statistics", logger):
            full_stats_df = aggregate_segment_stats_from_bricks( bricks, ['segment', 'voxel_count'] )
            full_stats_df.columns = ['segment', 'segment_voxel_count']
        
        ##
        ## If grouping segments into bodies (for tarballs),
        ## also append body stats
        ##
        grouping_scheme = config["mesh-config"]["storage"]["grouping-scheme"]
        if grouping_scheme != "labelmap":
            # Not grouping -- Just duplicate segment stats into body columns
            full_stats_df['body'] = full_stats_df['segment']
            full_stats_df['body_voxel_count'] = full_stats_df['segment_voxel_count']
        else:
            # Add body column
            segment_to_body_df = pd.DataFrame( self.load_labelmap(), columns=['segment', 'body'] )
            full_stats_df = full_stats_df.merge(segment_to_body_df, 'left', on='segment', copy=False)

            # Missing segments in the labelmap are assumed to be identity-mapped
            full_stats_df['body'].fillna( full_stats_df['segment'], inplace=True )
            full_stats_df['body'] = full_stats_df['body'].astype(np.uint64)

            # Calculate body voxel sizes
            body_stats_df = full_stats_df[['body', 'segment_voxel_count']].groupby('body').agg(['size', 'sum'])
            body_stats_df.columns = ['body_segment_count', 'body_voxel_count']
            body_stats_df['body'] = body_stats_df.index

            full_stats_df = full_stats_df.merge(body_stats_df, 'left', on='body', copy=False)

            # For offline analysis, write body stats to a file
            output_path = self.config_dir + '/body-stats.csv'
            logger.info(f"Saving body statistics to {output_path}")
            body_stats_df = body_stats_df[['body', 'body_segment_count', 'body_voxel_count']] # Set col order
            body_stats_df.columns = ['body', 'segment_count', 'voxel_count'] # rename columns for csv
            body_stats_df.sort_values('voxel_count', ascending=False, inplace=True)
            body_stats_df.to_csv(output_path, header=True, index=False)
            
        full_stats_df['keep_segment'] = ((full_stats_df['segment_voxel_count'] >= config['options']['minimum-segment-size']) &
                                         (full_stats_df['segment_voxel_count'] <= config['options']['maximum-segment-size']) )

        full_stats_df['keep_body'] = ((full_stats_df['body_voxel_count'] >= config['options']['minimum-agglomerated-size']) &
                                      (full_stats_df['body_voxel_count'] <= config['options']['maximum-agglomerated-size']) &
                                      (full_stats_df['body_segment_count'] >= config['options']['minimum-agglomerated-segment-count']))

        # If subset-bodies were given, exclude all others.
        sparse_body_ids = config["mesh-config"]["storage"]["subset-bodies"]
        if sparse_body_ids:
            for body_id in sparse_body_ids:
                if not full_stats_df[full_stats_df['body'] == body_id]['keep_body'].all():
                    logger.error(f"You explicitly listed body {body_id} in subset-bodies, "
                                 "but it will be excluded due to your other config settings.")
            full_stats_df['keep_body'] &= full_stats_df.eval('body in @sparse_body_ids')

        # Sort for convenience of viewing output
        with Timer("Sorting segment stats", logger):
            full_stats_df.sort_values(['body_voxel_count', 'segment_voxel_count'], ascending=False, inplace=True)

        #import pandas as pd
        #pd.set_option('expand_frame_repr', False)
        #logger.info(f"FULL_STATS:\n{full_stats_df}")
                
        # Write the Stats DataFrame to a file for offline analysis.
        output_path = self.config_dir + '/segment-stats-dataframe.csv'
        logger.info(f"Saving segment statistics to {output_path}")
        full_stats_df.to_csv(output_path, index=False)
        full_stats_df.to_pickle(output_path)
        
        return full_stats_df


    def group_by_body(self, segment_id_and_mesh_bytes):
        """
        For the RDD with items: (segment_id, mesh_bytes),
        group the items by the body (a.k.a. group) that each segment is in.
        
        Returns: RDD with items: (body_id, [(segment_id, mesh_bytes), (segment_id, mesh_bytes), ...]
        """
        config = self.config_data

        # Group according to scheme
        grouping_scheme = config["mesh-config"]["storage"]["grouping-scheme"]
        n_partitions = num_worker_nodes() * cpus_per_worker()

        if grouping_scheme in "hundreds":
            def last_six_digits( id_mesh ):
                segment_id, _mesh = id_mesh
                body_id = segment_id - (segment_id % 100)
                return body_id
            grouped_segment_ids_and_meshes = segment_id_and_mesh_bytes.groupBy(last_six_digits, numPartitions=n_partitions)

        elif grouping_scheme == "labelmap":
            mapping_pairs = self.load_labelmap()

            df = pd.DataFrame( mapping_pairs, columns=["segment_id", "body_id"] )
            def body_id_from_segment_id( id_mesh ):
                segment_id, _mesh = id_mesh
                rows = df.loc[df.segment_id == segment_id]
                if len(rows) == 0:
                    # If missing from labelmap,
                    # we assume an implicit identity mapping
                    return segment_id
                return rows['body_id'].iloc[0]

            grouped_segment_ids_and_meshes = segment_id_and_mesh_bytes.groupBy( body_id_from_segment_id )

        elif grouping_scheme in ("singletons", "no-groups"):
            # Create 'groups' of one item each, re-using the body ID as the group id.
            # (The difference between 'singletons', and 'no-groups' is in how the mesh is stored, below.)
            grouped_segment_ids_and_meshes = segment_id_and_mesh_bytes.map( lambda id_mesh: (id_mesh[0], [id_mesh]) )

        persist_and_execute(grouped_segment_ids_and_meshes, f"Grouping meshes with scheme: '{grouping_scheme}'", logger)
        return grouped_segment_ids_and_meshes

def decimate_mesh(decimation_fraction, mesh):
    mesh.simplify(decimation_fraction)
    return mesh

def post_meshes_to_dvid(config, instance_name, partition_items):
    """
    Send the given meshes (either .obj or .drc) as key/value pairs to DVID.
    
    Args:
        config: The CreateStitchedMeshes workflow config data
        
        instance_name: key-value instance to post to
            
        partition_items: tuple (group_id, [(segment_id, mesh_data), (segment_id, mesh_data)])
    """
    # Re-use session for connection pooling.
    session = default_dvid_session()

    # Re-use resource manager client connections, too.
    # (If resource-server is empty, this will return a "dummy client")    
    resource_client = ResourceManagerClient( config["options"]["resource-server"],
                                             config["options"]["resource-port"] )

    dvid_server = config["dvid-info"]["dvid"]["server"]
    uuid = config["dvid-info"]["dvid"]["uuid"]
    
    grouping_scheme = config["mesh-config"]["storage"]["grouping-scheme"]
    mesh_format = config["mesh-config"]["storage"]["format"]

    if grouping_scheme == "no-groups":
        for group_id, segment_ids_and_meshes in partition_items:
            for (segment_id, mesh_data) in segment_ids_and_meshes:

                @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
                def write_mesh():
                    with resource_client.access_context(dvid_server, False, 2, len(mesh_data)):
                        session.post(f'{dvid_server}/api/node/{uuid}/{instance_name}/key/{segment_id}', mesh_data)
                        session.post(f'{dvid_server}/api/node/{uuid}/{instance_name}/key/{segment_id}_info', json={ 'format': mesh_format })
                
                write_mesh()
    else:
        # All other grouping schemes, including 'singletons' write tarballs.
        # (In the 'singletons' case, there is just one tarball per body.)
        for group_id, segment_ids_and_meshes in partition_items:
            tar_name = _get_group_name(config, group_id)
            tar_stream = BytesIO()
            with closing(tarfile.open(tar_name, 'w', tar_stream)) as tf:
                for (segment_id, mesh_data) in segment_ids_and_meshes:
                    mesh_name = _get_mesh_name(config, segment_id)
                    f_info = tarfile.TarInfo(mesh_name)
                    f_info.size = len(mesh_data)
                    tf.addfile(f_info, BytesIO(mesh_data))
    
            tar_bytes = tar_stream.getbuffer()

            @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
            def write_tar():
                with resource_client.access_context(dvid_server, False, 1, len(tar_bytes)):
                    session.post(f'{dvid_server}/api/node/{uuid}/{instance_name}/key/{tar_name}', tar_bytes)
            
            write_tar()

def _get_group_name(config, group_id):
    """
    Encode the given group name (e.g. a 'body' in neu3)
    into a suitable key name for the group tarball.
    """
    grouping_scheme = config["mesh-config"]["storage"]["grouping-scheme"]
    naming_scheme = config["mesh-config"]["storage"]["naming-scheme"]

    # Must not allow np.uint64, which uses a custom __str__()
    group_id = int(group_id)

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

    # Must not allow np.uint64, which uses a custom __str__()
    mesh_id = int(mesh_id)

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

