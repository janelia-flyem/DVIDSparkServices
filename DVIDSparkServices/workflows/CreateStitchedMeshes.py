import os
import csv
import copy
import tarfile
import socket
import logging
from math import ceil
from functools import partial
from io import BytesIO
from contextlib import closing
from multiprocessing import TimeoutError

import numpy as np
import pandas as pd
import requests

from vol2mesh.mesh import Mesh, concatenate_meshes
from neuclease.logging_setup import PrefixedLogger
from neuclease.dvid import fetch_complete_mappings
from neuclease.dvid import create_instance, create_tarsupervoxel_instance, fetch_full_instance_info

from dvid_resource_manager.client import ResourceManagerClient

import DVIDSparkServices.rddtools as rt
from DVIDSparkServices.auto_retry import auto_retry
from DVIDSparkServices.util import Timer, persist_and_execute, num_worker_nodes, cpus_per_worker, default_dvid_session
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.sparkdvid.sparkdvid import sparkdvid 
from DVIDSparkServices.reconutils.morpho import object_masks_for_labels
from DVIDSparkServices.dvid.metadata import is_node_locked

from DVIDSparkServices.io_util.volume_service import DvidSegmentationVolumeSchema, LabelMapSchema, LabelmappedVolumeService, ScaledVolumeService, TransposedVolumeService
from DVIDSparkServices.io_util.labelmap_utils import load_labelmap

from DVIDSparkServices.io_util.brick import Grid, SparseBlockMask
from DVIDSparkServices.io_util.brickwall import BrickWall
from DVIDSparkServices.io_util.volume_service.volume_service import VolumeService
from DVIDSparkServices.io_util.volume_service.dvid_volume_service import DvidVolumeService

from DVIDSparkServices.segstats import aggregate_segment_stats_from_bricks

from DVIDSparkServices.subprocess_decorator import execute_in_subprocess

logger = logging.getLogger(__name__)

class CreateStitchedMeshes(Workflow):
    InputDvidSchema = copy.deepcopy(DvidSegmentationVolumeSchema)
    OutputDvidSchema = copy.deepcopy(DvidSegmentationVolumeSchema)
    OutputDvidSchema["properties"]["dvid"]["properties"].update(
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

            "batch-count": {
                "description": "After segmentation is loaded, the meshes will be computed in batches.\n"
                               "Specify the number of batches.",
                "type": "integer",
                "minimum": 1,
                "default": 1
            },
            
            "skip-batches": {
                "description": "Skip the given batch indexes.  Useful for resuming a failed job or skipping a problematic batch.\n",
                "type": "array",
                "items": { "type": "integer", "minimum": 0 },
                "default": []
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

            "pre-stitch-max-vertices": {
                "description": "Before stitching, further decimate the mesh (if necessary) "
                               "to have no more than this vertex count in the ENTIRE BODY.\n"
                               "For 'no max', set to -1",
                "type": "number",
                "default": -1 # No max
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
            
#             "post-stitch-max-vertices": {
#                 "description": "After stitching, further decimate the mesh (if necessary) "
#                                "to have no more than this vertex count in the ENTIRE BODY.\n"
#                                "For 'no max', set to -1",
#                 "type": "number",
#                 "default": -1 # No max
#             },
#             
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
                        ],
                
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
                                       "- hundreds: Group meshes in groups of up to 100, such that ids xxxx00 through xxxx99 end up in the same group.\n"
                                       "- thousands: Group meshes in groups of up to 1000, such that ids xxx000 through xxx999 end up in the same group.\n"
                                       "- labelmap: Use the labelmap setting below to determine the grouping.\n"
                                       "- dvid-labelmap: Use the dvid-labelmap setting below to determine the grouping.\n",
                        "type": "string",
                        "enum": ["no-groups", "singletons", "hundreds", "thousands", "labelmap", "dvid-labelmap"],
                        "default": "no-groups"
                    },
                    "naming-scheme": {
                        "description": "How to name mesh keys (and internal files, if grouped)",
                        "type": "string",
                        "enum": ["trivial", # Used for testing, and 'no-groups' mode.
                                 "neu3-level-0",  # Used for tarballs of supervoxels (one tarball per body)
                                 "neu3-level-1",
                                 "tarsupervoxels"], # Used for storage into a dvid 'tarsupervoxels' instance.
                        "default": "trivial"
                    },
                    "format": {
                        "description": "Format to save the meshes in. ",
                        "type": "string",
                        "enum": ["obj",    # Wavefront OBJ (.obj)
                                 "drc"],   # Draco (compressed) (.drc)
                        "default": "obj"
                    },
                    
                    # Only used by the 'labelmap' grouping-scheme
                    "labelmap": copy.copy(LabelMapSchema),
                    "dvid-labelmap": {
                        # Only used by the 'dvid-labelmap' grouping-scheme
                        "description": "Parameters specify a DVID labelmap instance from which mappings can be queried",
                        "type": "object",
                        "default": {},
                        #"additionalProperties": False, # Can't use this in conjunction with 'oneOf' schema feature
                        "properties": {
                            "server": {
                                "description": "location of DVID server.",
                                "type": "string",
                            },
                            "uuid": {
                                "description": "version node",
                                "type": "string"
                            },
                            "segmentation-name": {
                                "description": "The labels instance to read mappings from\n",
                                "type": "string",
                                "minLength": 1
                            }
                        }
                    },

                    "subset-bodies": {
                        "description": "(Optional.) Instead of generating meshes for all meshes in the volume,\n"
                                       "only generate meshes for a subset of the bodies in the volume.\n",
                        "oneOf": [
                            {
                                "description": "A list of body IDs to generate meshes for.",
                                "type": "array",
                                "default": []
                            },
                            {
                                "description": "A CSV file containing a single column of body IDs to generate meshes for.",
                                "type": "string",
                                "default": ""
                            }
                        ],
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
    MeshGenerationSchema\
        ["properties"]["storage"]\
            ["properties"]["dvid-labelmap"]\
                ["description"] = ("A labelmap file to determine mesh groupings.\n"
                                   "Only used by the 'dvid-labelmap' grouping-scheme.\n"
                                   "Will be applied AFTER any labelmap you specified in the segmentation volume info.\n")


    MeshWorkflowOptionsSchema = copy.copy(Workflow.OptionsSchema)
    MeshWorkflowOptionsSchema["additionalProperties"] = False
    MeshWorkflowOptionsSchema["properties"].update(
    {
        "initial-partition-size": {
            "description": "Set the partition size for downloading the initial segmentation bricks, in bytes.\n"
                           "Be careful: Spark sucks. A too-small size results in a ton of partitions to keep track of,\n"
                           "which somehow causes an out-of-memory crash on the DRIVER.\n"
                           "Yes, the friggin' DRIVER, which does **almost nothing**, but apparently needs a crap-ton of RAM to do it.\n"
                           "The crash occurs when using more than, say, 200k-1M partitions. (I know, WTF, right?).\n"
                           "But a too-large size results in fewer, large partitions, at which point you run the risk of exceeding Java's 2GB size limit for each partition.\n"
                           "(Yes, the idea of using a language with a 2GB limitation for 'Big Data' workflows is laughable.  But here we are.)\n",
            "type": "number",
            "default": 4 * (2**30) # 4 GB
        },
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
            "default": 1
        },
        "maximum-agglomerated-size": {
            "description": "Agglomerated groups larger than this voxel count will not be processed.",
            "type": "number",
            "default": 100e9 # 100 Gigavoxels (HUGE)
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
        },
        "skip-stats-export": {
            "description": "Segment stats must be computed for grouping and dynamic decimation.  But SAVING them to a file can be disabled, as a debugging feature.",
            "type": "boolean",
            "default": False
        }
    })
    

    Schema = \
    {
      "$schema": "http://json-schema.org/schema#",
      "title": "Service to create meshes from segmentation",
      "type": "object",
      "required": ["input", "output"],
      "properties": {
        "input": InputDvidSchema,
        "output": OutputDvidSchema,
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
        output_info = self.config_data['output']
        mesh_config = self.config_data["mesh-config"]
        
        if not output_info["dvid"]["server"].startswith('http://'):
            output_info["dvid"]["server"] = 'http://' + output_info["dvid"]["server"]
        
        # Provide default meshes instance name if needed
        if not output_info["dvid"]["meshes-destination"]:
            if mesh_config["storage"]["grouping-scheme"] == 'no-groups':
                suffix = "_meshes"
            else:
                if mesh_config["storage"]["naming-scheme"] == "tarsupervoxels":
                    suffix = "_sv_meshes"
                else:
                    suffix = "_meshes_tars"

            output_info["dvid"]["meshes-destination"] = output_info["dvid"]["segmentation-name"] + suffix

        if isinstance(mesh_config["storage"]["subset-bodies"], str):
            csv_path = self.relpath_to_abspath(mesh_config["storage"]["subset-bodies"])
            with open(csv_path, 'r') as csv_file:
                first_line = csv_file.readline()
                csv_file.seek(0)
                if ',' not in first_line:
                    # csv.Sniffer doesn't work if there's only one column in the file
                    try:
                        int(first_line)
                        has_header = False
                    except:
                        has_header = True
                else:
                    has_header = csv.Sniffer().has_header(csv_file.read(1024))
                    csv_file.seek(0)
                rows = iter(csv.reader(csv_file))
                if has_header:
                    _header = next(rows) # Skip header
                
                # File is permitted to have multiple columns,
                # but body ID must be first column
                subset_bodies = [int(row[0]) for row in rows]
            
            # Overwrite config with bodies list from the csv file
            mesh_config["storage"]["subset-bodies"] = subset_bodies

    def execute(self):
        self._sanitize_config()

        config = self.config_data
        options = config["options"]
        bad_mesh_dir = f"{self.config_dir}/bad-meshes"
        os.makedirs(bad_mesh_dir, exist_ok=True)

        
        resource_mgr_client = ResourceManagerClient(options["resource-server"], options["resource-port"])
        volume_service = VolumeService.create_from_config(config["input"], self.config_dir, resource_mgr_client)

        self._init_meshes_instance()

        # See notes in config for description of this setting.
        partition_bytes = options["initial-partition-size"]
        target_partition_size_voxels = partition_bytes // np.uint64().nbytes
        
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
        
        batch_count = config["mesh-config"]['batch-count']
        if keep_col.all() and batch_count == 1:
            segments_to_keep = None # keep everything
        else:
            # Note: This array will be broadcasted to the workers.
            #       It will be potentially quite large if we're keeping most (but not all) segments.
            #       Broadcast expense should be minimal thanks to lz4 compression,
            #       but RAM usage will be high.
            segments_to_keep = full_stats_df['segment'][keep_col].values
            
            if len(segments_to_keep) == 0:
                raise RuntimeError("Based on your config settings, no meshes will be generated at all. See segment stats.")

            if segments_to_keep.max() < np.iinfo(np.uint32).max:
                segments_to_keep = segments_to_keep.astype(np.uint32) # Save some RAM

        logger.info(f"Processing {keep_col.sum()} meshes in {batch_count} batches")
        if segments_to_keep is None:
            self.process_batch(brick_wall, full_stats_df, None, 0)
        else:
            batch_size = ceil(len(segments_to_keep) / batch_count)
            batch_bounds = list(range(0, batch_size*batch_count+1, batch_size))
            for batch_index, (start, stop) in enumerate(zip(batch_bounds[:-1], batch_bounds[1:])):
                if batch_index in config["mesh-config"]["skip-batches"]:
                    logger.info(f"Skipping batch {batch_index}")
                    continue

                batch_logger = PrefixedLogger(logger, f"Batch {batch_index}: ")
                batch_segments = segments_to_keep[start:stop]
                with Timer(f"Processing {len(batch_segments)} meshes", batch_logger):
                    self.process_batch(brick_wall, full_stats_df, batch_segments, batch_index, batch_logger)


    def process_batch(self, brick_wall, full_stats_df, segments_to_keep, batch_index, batch_logger):
        from pyspark import StorageLevel
        config = self.config_data
        bad_mesh_dir = f"{self.config_dir}/bad-meshes"


        def generate_meshes_for_brick( brick ):
            import DVIDSparkServices # Ensure faulthandler logging is active. # @UnusedImport
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
        rt.persist_and_execute(segment_ids_and_mesh_blocks, "Computing block segment meshes", batch_logger, StorageLevel.MEMORY_AND_DISK) # @UndefinedVariable
        
        segments_and_counts_and_size = segment_ids_and_mesh_blocks \
                                       .map( lambda seg_mesh_size: (seg_mesh_size[0], (len(seg_mesh_size[1][0].vertices_zyx), seg_mesh_size[1][1]) ) ) \
                                       .groupByKey() \
                                       .map( lambda seg_counts_size: (seg_counts_size[0], *np.array(list(seg_counts_size[1])).sum(axis=0) ) ) \
                                       .collect()

        
        counts_df = pd.DataFrame( segments_and_counts_and_size, columns=['segment', 'initial_vertex_count', 'total_compressed_size'] )
        del segments_and_counts_and_size
        with Timer("Merging initial_vertex_count onto segment stats", batch_logger):
            full_stats_df = full_stats_df.merge(counts_df, 'left', on='segment')
        if not config["options"]["skip-stats-export"]:
            counts_df.to_csv(self.relpath_to_abspath(f'batch-{batch_index}-initial-vertex-counts.csv'), index=False)
        
        # Drop size
        # --> (segment_id, mesh_for_one_block)
        segment_ids_and_mesh_blocks = segment_ids_and_mesh_blocks.map(lambda a_bc: (a_bc[0], a_bc[1][0]))

        # append column for body vertex counts (body is the INDEX)
        body_vertex_counts = full_stats_df[['body', 'initial_vertex_count']].groupby('body').sum()
        body_vertex_counts.columns = ['body_initial_vertex_count']
        full_stats_df = full_stats_df.merge(body_vertex_counts, 'inner', left_on='body', right_index=True, copy=False)
        
        # Pre-stitch smoothing
        # --> (segment_id, mesh_for_one_block)
        smoothing_iterations = config["mesh-config"]["pre-stitch-smoothing-iterations"]
        if smoothing_iterations > 0:
            def smooth(mesh):
                import DVIDSparkServices # Ensure faulthandler logging is active. # @UnusedImport
                mesh.laplacian_smooth(smoothing_iterations)
                mesh.drop_normals()
                mesh.compress()
                return mesh
            segment_id_and_smoothed_mesh = segment_ids_and_mesh_blocks.mapValues( smooth )
    
            rt.persist_and_execute(segment_id_and_smoothed_mesh, "Smoothing block meshes", batch_logger, StorageLevel.MEMORY_AND_DISK) # @UndefinedVariable
            rt.unpersist(segment_ids_and_mesh_blocks)
            segment_ids_and_mesh_blocks = segment_id_and_smoothed_mesh
            del segment_id_and_smoothed_mesh

        # per-body vertex counts
        body_initial_vertex_counts_df = full_stats_df.query('keep_segment and keep_body')[['segment', 'body_initial_vertex_count']]
        
        # Join mesh blocks with corresponding body vertex counts
        body_initial_vertex_counts = self.sc.parallelize(body_initial_vertex_counts_df.itertuples(index=False))
        segment_ids_and_mesh_blocks_and_body_counts = segment_ids_and_mesh_blocks.join(body_initial_vertex_counts)
        rt.persist_and_execute(segment_ids_and_mesh_blocks_and_body_counts, "Joining mesh blocks with body vertex counts", batch_logger)
        rt.unpersist(segment_ids_and_mesh_blocks)
        
        max_vertices = config["mesh-config"]["pre-stitch-max-vertices"]

        # Pre-stitch decimation
        # --> (segment_id, mesh_for_one_block)
        decimation_fraction = config["mesh-config"]["pre-stitch-decimation"]
        if decimation_fraction < 1.0:
            @self.collect_log(lambda _: socket.gethostname() + '-mesh-decimation')
            def decimate(id_mesh_bcount):
                import DVIDSparkServices # Ensure faulthandler logging is active. # @UnusedImport
                segment_id, (mesh, body_vertex_count) = id_mesh_bcount
                try:
                    final_decimation = decimation_fraction
    
                    # If the total vertex count of all segments in this segment's
                    # body would be too large, apply further decimation.
                    if final_decimation * body_vertex_count > max_vertices:
                        final_decimation = max_vertices / body_vertex_count
                    
                    mesh.simplify(final_decimation, in_memory=False, timeout=600) # 10 minutes
                    mesh.drop_normals()
                    mesh.compress()
                    return (segment_id, mesh)
                except TimeoutError:
                    bad_mesh_export_path = f'{bad_mesh_dir}/failed-decimation-{final_decimation:.2f}-{segment_id}.obj'
                    mesh.serialize(f'{bad_mesh_export_path}')
                    logger = logging.getLogger(__name__)
                    logger.error(f"Timed out while decimating a block mesh! Skipped decimation and wrote bad mesh to {bad_mesh_export_path}")
                    return (segment_id, mesh)

            segment_id_and_decimated_mesh = segment_ids_and_mesh_blocks_and_body_counts.map(decimate)

            rt.persist_and_execute(segment_id_and_decimated_mesh, "Decimating block meshes", batch_logger)
            rt.unpersist(segment_ids_and_mesh_blocks_and_body_counts)
            segment_ids_and_mesh_blocks = segment_id_and_decimated_mesh
            del segment_id_and_decimated_mesh
        
        if (smoothing_iterations > 0 or decimation_fraction < 1.0) and config["mesh-config"]["compute-normals"]:
            # Compute normals
            def recompute_normals(mesh):
                import DVIDSparkServices # Ensure faulthandler logging is active. # @UnusedImport
                mesh.recompute_normals()
                return mesh
            
            segment_id_and_mesh_with_normals = segment_ids_and_mesh_blocks.map(recompute_normals)

            rt.persist_and_execute(segment_id_and_mesh_with_normals, "Computing block mesh normals", batch_logger)
            rt.unpersist(segment_ids_and_mesh_blocks)
            segment_ids_and_mesh_blocks = segment_id_and_mesh_with_normals
            del segment_id_and_mesh_with_normals
        
        # Group by segment ID
        # --> (segment_id, [mesh_for_block, mesh_for_block, ...])
        mesh_blocks_grouped_by_segment = segment_ids_and_mesh_blocks.groupByKey()
        rt.persist_and_execute(mesh_blocks_grouped_by_segment, "Grouping block segment meshes", batch_logger)
        rt.unpersist(segment_ids_and_mesh_blocks)
        del segment_ids_and_mesh_blocks
        
        # Concatenate into a single mesh per segment
        # --> (segment_id, mesh)
        stitch_method = config["mesh-config"]["stitch-method"]
        @self.collect_log()
        def concatentate_and_stitch(meshes):
            import DVIDSparkServices # Ensure faulthandler logging is active. # @UnusedImport
            def _impl():
                concatenated_mesh = concatenate_meshes(meshes)
                for mesh in meshes:
                    mesh.destroy() # Save RAM -- we're done with the block meshes at this point
    
                if stitch_method == "simple-concatenate":
                    # This is required for proper draco encoding
                    concatenated_mesh.drop_unused_vertices()
                elif stitch_method == "stitch":
                    concatenated_mesh.stitch_adjacent_faces(True, True)
    
                concatenated_mesh.compress()
                return concatenated_mesh
            
            total_vertices = sum(len(mesh.vertices_zyx) for mesh in meshes)
            if (total_vertices) < 10e6:
                return _impl()
            with Timer(f"Concatenating a big mesh ({total_vertices} vertices)", logging.getLogger(__name__)):
                return _impl()
            
        segment_id_and_mesh = mesh_blocks_grouped_by_segment.mapValues(concatentate_and_stitch)
        
        rt.persist_and_execute(segment_id_and_mesh, "Stitching block segment meshes", batch_logger)
        rt.unpersist(mesh_blocks_grouped_by_segment)
        del mesh_blocks_grouped_by_segment

        # Post-stitch Smoothing
        # --> (segment_id, mesh)
        smoothing_iterations = config["mesh-config"]["post-stitch-smoothing-iterations"]
        if smoothing_iterations > 0:
            def smooth(mesh):
                import DVIDSparkServices # Ensure faulthandler logging is active. # @UnusedImport
                mesh.laplacian_smooth(smoothing_iterations)
                return mesh
            segment_id_and_smoothed_mesh = segment_id_and_mesh.mapValues( smooth )
    
            rt.persist_and_execute(segment_id_and_smoothed_mesh, "Smoothing stitched meshes", batch_logger)
            rt.unpersist(segment_id_and_mesh)
            segment_id_and_mesh = segment_id_and_smoothed_mesh
            del segment_id_and_smoothed_mesh

        # Post-stitch decimation
        # --> (segment_id, mesh)
        decimation_fraction = config["mesh-config"]["post-stitch-decimation"]
        #max_vertices = config["mesh-config"]["post-stitch-max-vertices"]
        
        if decimation_fraction < 1.0 or max_vertices > 0:
            @self.collect_log(lambda *_a, **_kw: 'post-stitch-decimation', logging.WARNING)
            def decimate(seg_and_mesh):
                import DVIDSparkServices # Ensure faulthandler logging is active. # @UnusedImport
                segment_id, mesh = seg_and_mesh
                final_decimation = decimation_fraction

                # If the total vertex count of all segments in this segment's
                # body would be too large, apply further decimation.                
                ##body_prestitch_vertex_count = body_prestitch_vertex_counts_df[segment_id]
                ##if final_decimation * body_prestitch_vertex_count > max_vertices:
                ##    final_decimation = max_vertices / body_prestitch_vertex_count
                mesh.simplify(final_decimation, in_memory=False, timeout=600)
                return (segment_id, mesh)
            segment_id_and_decimated_mesh = segment_id_and_mesh.map(decimate)

            rt.persist_and_execute(segment_id_and_decimated_mesh, "Decimating stitched meshes", batch_logger)
            rt.unpersist(segment_id_and_mesh)
            segment_id_and_mesh = segment_id_and_decimated_mesh
            del segment_id_and_decimated_mesh

        # Get post-decimation vertex count and ovewrite stats file
        if not config["options"]["skip-stats-export"]:
            segments_and_counts = segment_id_and_mesh.map(lambda id_mesh: (id_mesh[0], len(id_mesh[1].vertices_zyx))).collect()
            counts_df = pd.DataFrame( segments_and_counts, columns=['segment', 'post_stitch_decimated_vertex_count'] )
            counts_df.to_csv(self.relpath_to_abspath(f'batch-{batch_index}-poststitch-vertex-counts.csv'),  index=False)

        # Post-stitch normals
        # --> (segment_id, mesh)
        if (smoothing_iterations > 0 or decimation_fraction < 1.0) and config["mesh-config"]["compute-normals"]:
            def compute_normals(mesh):
                import DVIDSparkServices # Ensure faulthandler logging is active. # @UnusedImport
                mesh.recompute_normals()
                return mesh
            segment_id_and_mesh_with_normals = segment_id_and_mesh.mapValues(compute_normals)

            rt.persist_and_execute(segment_id_and_mesh_with_normals, "Computing stitched mesh normals", batch_logger)
            rt.unpersist(segment_id_and_mesh)
            segment_id_and_mesh = segment_id_and_mesh_with_normals
            del segment_id_and_mesh_with_normals

        rescale_factor = config["mesh-config"]["rescale-before-write"]
        if rescale_factor != 1.0:
            def rescale(mesh):
                mesh.vertices_zyx *= rescale_factor
                return mesh
            segment_id_and_rescaled_mesh = segment_id_and_mesh.mapValues(rescale)
            
            rt.persist_and_execute(segment_id_and_rescaled_mesh, "Rescaling segment meshes", batch_logger)
            rt.unpersist(segment_id_and_mesh)
            segment_id_and_mesh = segment_id_and_rescaled_mesh
            del segment_id_and_rescaled_mesh

        # Serialize
        # --> (segment_id, mesh_bytes)
        fmt = config["mesh-config"]["storage"]["format"]
        @self.collect_log(echo_threshold=logging.INFO)
        def serialize(id_mesh):
            import DVIDSparkServices # Ensure faulthandler logging is active. # @UnusedImport
            segment_id, mesh = id_mesh
            try:
                mesh_serialize = execute_in_subprocess(600, logging.getLogger(__name__))(mesh.serialize)
                return (segment_id, mesh_serialize(fmt=fmt))
            except:
                # This assumes that we can still it serialize in obj format...
                output_path = f'{bad_mesh_dir}/failed-serialization-{segment_id}.obj'
                mesh.serialize(output_path)
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to serialize mesh.  Wrote to {output_path}")
                return (segment_id, b'')

        segment_id_and_mesh_bytes = segment_id_and_mesh.map( serialize ) \
                                                       .filter(lambda mesh_bytes: len(mesh_bytes) > 0)

        rt.persist_and_execute(segment_id_and_mesh_bytes, "Serializing segment meshes", batch_logger)
        rt.unpersist(segment_id_and_mesh)
        del segment_id_and_mesh

        if not config["options"]["skip-stats-export"]:
            # Record final mesh file sizes
            # (Will need to be grouped by body to compute tarball size        
            mesh_file_sizes = segment_id_and_mesh_bytes.map(lambda id_mesh: (id_mesh[0], len(id_mesh[1])) ).collect()
            mesh_file_sizes_df = pd.DataFrame(mesh_file_sizes, columns=['segment', 'file_size'])
            del mesh_file_sizes
            mesh_file_sizes_df.to_csv(self.relpath_to_abspath(f'batch-{batch_index}-file-sizes.csv'), index=False)
            del mesh_file_sizes_df

        # Group by body ID
        # --> ( body_id, [( segment_id, mesh_bytes ), ( segment_id, mesh_bytes ), ...] )
        segment_id_and_mesh_bytes_grouped_by_body = self.group_by_body(segment_id_and_mesh_bytes, batch_logger)

        instance_name = config["output"]["dvid"]["meshes-destination"]
        with Timer("Writing meshes to DVID", batch_logger):
            keys_written = segment_id_and_mesh_bytes_grouped_by_body.mapPartitions( partial(post_meshes_to_dvid, config, instance_name) ).collect()
            keys_written = list(keys_written)

        keys_written = pd.Series(keys_written, name='key')
        keys_written.to_csv(self.relpath_to_abspath(f'batch-{batch_index}-keys-uploaded.csv'), index=False, header=False)

            
    def _init_meshes_instance(self):
        output_info = self.config_data["output"]
        server = output_info["dvid"]["server"]
        uuid = output_info["dvid"]["uuid"]
        seg_instance = output_info["dvid"]["segmentation-name"]
        mesh_instance = output_info["dvid"]["meshes-destination"]
        naming_scheme = self.config_data["mesh-config"]["storage"]["naming-scheme"]
        extension = self.config_data["mesh-config"]["storage"]["format"]

        if is_node_locked(server, uuid):
            raise RuntimeError(f"Can't write meshes: The node you specified ({server} / {uuid}) is locked.")

        try:
            mesh_info = fetch_full_instance_info((server, uuid, mesh_instance))
        except requests.HTTPError:
            # Doesn't exist yet; must create.
            # (No compression -- we'll send pre-compressed files)
            if naming_scheme == "tarsupervoxels":
                create_tarsupervoxel_instance( (server, uuid, mesh_instance), seg_instance, extension )
            else:
                create_instance( (server, uuid, mesh_instance), "keyvalue", versioned=True, compression='none', tags=["type=meshes"] )
        else:
            if naming_scheme == "tarsupervoxels" and mesh_info["Base"]["TypeName"] != "tarsupervoxels":
                raise RuntimeError("You are attempting to use the 'tarsupervoxels' scheme with the wrong instance type.")


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

        service_scale = 0
        for service in volume_service.service_chain:
            if isinstance(service, ScaledVolumeService):
                service_scale += service.scale_delta
            if isinstance(service, TransposedVolumeService):
                # Can't fetch sparse bodies with transposed services, so the entire volume will
                # be read and subset-bodies will be selected after seg is downloaded.
                #
                # (With some work, we could add support for transposed services,
                # but it would probably be easier to add a feature to just transpose the meshes
                # after they're computed.)
                logger.warning("You are using subset-bodies with transposed volume. Sparsevol will not be fetched. "
                               "Instead, dense segmentation will be fetched, and your subset will be selected from it.")
                return None
        
        grouping_scheme = config["mesh-config"]["storage"]["grouping-scheme"]
        assert grouping_scheme in ('no-groups', 'singletons', 'labelmap', 'dvid-labelmap'), \
            f"Not allowed to use 'subset-bodies' setting for grouping scheme: {grouping_scheme}"

        mapping_pairs = None
        if use_service_labelmap:
            assert isinstance(volume_service, LabelmappedVolumeService), \
                "Can't use service labelmap: The input isn't a LabelmappedVolumeService"
            mapping_pairs = volume_service.mapping_pairs
        elif grouping_scheme in ("labelmap", "dvid-labelmap"):
            mapping_pairs = self.load_labelmap()

        if mapping_pairs is not None:
            segments, bodies = mapping_pairs.transpose()
            
            # pandas.Series permits duplicate index values,
            # which is convenient for this reverse lookup
            reverse_lookup = pd.Series(index=bodies, data=segments)
            sparse_segment_ids = reverse_lookup.loc[sparse_body_ids]
            
            # Single-supervoxel bodies are not present in the mapping,
            # and thus result in NaN entries.  Replace them with identity mappings.
            missing_entries = sparse_segment_ids.isnull()
            sparse_segment_ids[missing_entries] = sparse_segment_ids.index[missing_entries]
            sparse_segment_ids = sparse_segment_ids.astype(np.uint64).values
        else:
            # No labelmap: The 'body ids' are identical to segment ids
            sparse_segment_ids = sparse_body_ids

        logger.info("Reading sparse block mask for body subset...")
        # Fetch the sparse mask of blocks that the sparse segments belong to
        dvid_service = volume_service.base_service
        block_mask, lowres_box, dvid_block_shape = \
            sparkdvid.get_union_block_mask_for_bodies( dvid_service.server,
                                                       dvid_service.uuid,
                                                       dvid_service.instance_name,
                                                       sparse_segment_ids,
                                                       dvid_service.supervoxels )

        # None of the bodies on the list could be found in the sparsevol data.
        # Something is very wrong.
        if block_mask is None:
            logger.error("Could not retrieve a sparse block mask!  Fetching all segmentation.")
            return None

        # Box in full-res DVID coordinates
        fullres_box = lowres_box * dvid_block_shape
        
        # Box in service coordinates (if using ScaledVolumeService, need to shrink the box accordingly) 
        service_res_box = fullres_box // (2**service_scale)
        service_res_block_shape = np.array(dvid_block_shape) // (2**service_scale)
        
        return SparseBlockMask(block_mask, service_res_box, service_res_block_shape)


    def load_labelmap(self):
        """
        Load the labelmap for aggregating segments into bodies.
        Note that this is NOT the same as the labelmap (if any) that may
        be involved in the original segment source.
        """
        if self._labelmap is None:
            config = self.config_data
            grouping_scheme = config["mesh-config"]["storage"]["grouping-scheme"]
            assert grouping_scheme in ('labelmap', 'dvid-labelmap')
            labelmap_config = config["mesh-config"]["storage"]["labelmap"]
            dvid_labelmap_config = config["mesh-config"]["storage"]["dvid-labelmap"]
            
            assert "server" not in dvid_labelmap_config or labelmap_config["file-type"] == "__invalid__", \
                "Can't supply both labelmap and dvid-labelmap grouping parameters.  Pick one."
            
            if "server" in dvid_labelmap_config:
                sui = ( dvid_labelmap_config["server"], dvid_labelmap_config["uuid"], dvid_labelmap_config["segmentation-name"] )
                mapping_series = fetch_complete_mappings(sui)
                mapping_array = np.array((mapping_series.index.values, mapping_series.values))
                self._labelmap = np.transpose(mapping_array).astype(np.uint64, copy=False)
            else:
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

            MAX_SAFE_INT = 2**53    # Above this number, double-precision floats may not exactly represent integers
                                    # The pandas manipulations below temporarily store body IDs as doubles during the
                                    # merge() step.
                                    # (FWIW, this is the same reason JavaScript integers aren't safe above 2**53.)
            if full_stats_df['segment'].max() > MAX_SAFE_INT:
                logger.error("Some segments in the volume have higher IDs than 2**53. "
                             "Those will not be mapped to the correct bodies, even if they are the only segment in the body.")

            #logger.info("debugging: Saving BRICK stats...")
            #full_stats_df.to_csv(self.relpath_to_abspath('brick-stats.csv'), index=False)
        
        ##
        ## If grouping segments into bodies (for tarballs),
        ## also append body stats
        ##
        grouping_scheme = config["mesh-config"]["storage"]["grouping-scheme"]
        if grouping_scheme not in ("labelmap", "dvid-labelmap"):
            # Not grouping -- Just duplicate segment stats into body columns
            full_stats_df['body'] = full_stats_df['segment']
            body_stats_df = pd.DataFrame({ 'body': full_stats_df['segment'] })
            body_stats_df['body_voxel_count'] = full_stats_df['segment_voxel_count']
            body_stats_df['body_segment_count'] = np.uint8(1)
        else:
            # Load agglomeration mapping
            segment_to_body_df = pd.DataFrame( self.load_labelmap(), columns=['segment', 'body'] )
            if (segment_to_body_df['segment'].max() > MAX_SAFE_INT) or (segment_to_body_df['body'].max() > MAX_SAFE_INT):
                # See comment above regarding MAX_SAFE_INT.
                logger.error("Some segments or bodies in the label-to-body mapping have higher IDs than 2**53. "
                             "Those will not be mapped to the correct bodies, even if they are the only segment in the body.")

            # Add body column via merge of agglomeration mapping
            # Missing segments in the labelmap are assumed to be identity-mapped
            full_stats_df = full_stats_df.merge(segment_to_body_df, 'left', on='segment', copy=False)
            full_stats_df['body'].fillna( full_stats_df['segment'], inplace=True )
            full_stats_df['body'] = full_stats_df['body'].astype(np.uint64)

            with Timer("Computing body statistics", logger=logger):
                body_stats_df = full_stats_df[['body', 'segment_voxel_count']].groupby('body').agg(['size', 'sum'])
                body_stats_df.columns = ['body_segment_count', 'body_voxel_count']
                body_stats_df['body'] = body_stats_df.index

        body_stats_df['keep_body'] = ((body_stats_df['body_voxel_count'] >= config['options']['minimum-agglomerated-size']) &
                                      (body_stats_df['body_voxel_count'] <= config['options']['maximum-agglomerated-size']) &
                                      (body_stats_df['body_segment_count'] >= config['options']['minimum-agglomerated-segment-count']))

        # If subset-bodies were given, exclude all others.
        sparse_body_ids = config["mesh-config"]["storage"]["subset-bodies"]
        if sparse_body_ids:
            sparse_body_ids = set(sparse_body_ids)
            sparse_body_stats_df = body_stats_df.query('body in @sparse_body_ids')
            excluded_bodies_df = sparse_body_stats_df[~sparse_body_stats_df['keep_body']]
            if len(excluded_bodies_df) > 0:
                output_path = self.config_dir + '/excluded-body-stats.csv'
                logger.error(f"You explicitly listed {len(excluded_bodies_df)} bodies in subset-bodies, "
                             f"but they will be excluded due to your other config settings.  See {output_path}.")
                excluded_bodies_df.to_csv(output_path, header=True, index=False)
            body_stats_df['keep_body'] &= body_stats_df.eval('body in @sparse_body_ids')

        full_stats_df = full_stats_df.merge(body_stats_df, 'left', on='body', copy=False)
        full_stats_df['keep_segment'] = ((full_stats_df['segment_voxel_count'] >= config['options']['minimum-segment-size']) &
                                         (full_stats_df['segment_voxel_count'] <= config['options']['maximum-segment-size']) )


        #import pandas as pd
        #pd.set_option('expand_frame_repr', False)
        #logger.info(f"FULL_STATS:\n{full_stats_df}")
                
        if not config["options"]["skip-stats-export"]:

            if body_stats_df is not None:
                output_path = self.config_dir + '/body-stats.csv'
                logger.info(f"Saving body statistics to {output_path}")
                # For offline analysis, write body stats to a file
                body_stats_df = body_stats_df[['body', 'body_segment_count', 'body_voxel_count', 'keep_body']] # Set col order
                body_stats_df.columns = ['body', 'segment_count', 'voxel_count', 'keep_body'] # rename columns for csv
                body_stats_df.sort_values('voxel_count', ascending=False, inplace=True)
    
                body_stats_df.to_csv(output_path, header=True, index=False)

            # Sort for convenience of viewing output
            with Timer("Sorting segment stats", logger):
                full_stats_df.sort_values(['body_voxel_count', 'segment_voxel_count'], ascending=False, inplace=True)
    
            # Write the Stats DataFrame to a file for offline analysis.
            output_path = self.config_dir + '/segment-stats-dataframe.csv'
            logger.info(f"Saving segment statistics to {output_path}")
            full_stats_df.to_csv(output_path, index=False)
        
        return full_stats_df


    def group_by_body(self, segment_id_and_mesh_bytes, logger):
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
            def hundreds( id_mesh ):
                segment_id, _mesh = id_mesh
                group_id = segment_id // 100 * 100
                return group_id
            grouped_segment_ids_and_meshes = segment_id_and_mesh_bytes.groupBy(hundreds, numPartitions=n_partitions)

        elif grouping_scheme in "thousands":
            def thousands( id_mesh ):
                segment_id, _mesh = id_mesh
                group_id = segment_id // 1000 * 1000
                return group_id
            grouped_segment_ids_and_meshes = segment_id_and_mesh_bytes.groupBy(thousands, numPartitions=n_partitions)

        elif grouping_scheme in ("labelmap", "dvid-labelmap"):
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


def post_meshes_to_dvid(config, instance_name, partition_items):
    """
    Send the given meshes (either .obj or .drc) as key/value pairs to DVID.
    
    Args:
        config: The CreateStitchedMeshes workflow config data
        
        instance_name: key-value instance to post to
            
        partition_items: tuple (group_id, [(segment_id, mesh_data), (segment_id, mesh_data)])
    
    Returns:
        The list of written keys.
    """
    keys_written = []
    
    # Re-use session for connection pooling.
    session = default_dvid_session()

    # Re-use resource manager client connections, too.
    # (If resource-server is empty, this will return a "dummy client")    
    resource_client = ResourceManagerClient( config["options"]["resource-server"],
                                             config["options"]["resource-port"] )

    dvid_server = config["output"]["dvid"]["server"]
    uuid = config["output"]["dvid"]["uuid"]
    
    grouping_scheme = config["mesh-config"]["storage"]["grouping-scheme"]
    mesh_format = config["mesh-config"]["storage"]["format"]

    if grouping_scheme == "no-groups":
        for group_id, segment_ids_and_meshes in partition_items:
            for (segment_id, mesh_data) in segment_ids_and_meshes:

                @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
                def write_mesh():
                    if len(mesh_data) == 0:
                        # Empty meshes may be 'serialized' as an empty buffer.
                        # Don't upload such files.
                        return
                    with resource_client.access_context(dvid_server, False, 2, len(mesh_data)):
                        session.post(f'{dvid_server}/api/node/{uuid}/{instance_name}/key/{segment_id}', mesh_data)
                        session.post(f'{dvid_server}/api/node/{uuid}/{instance_name}/key/{segment_id}_info', json={ 'format': mesh_format })
                
                write_mesh()
                keys_written.append(str(segment_id))
    else:
        # All other grouping schemes, including 'singletons' write tarballs.
        # (In the 'singletons' case, there is just one tarball per body.)
        for group_id, segment_ids_and_meshes in partition_items:
            tar_name = _get_group_name(config, group_id)
            tar_stream = BytesIO()
            nonempty_mesh_count = 0
            with closing(tarfile.open(tar_name, 'w', tar_stream)) as tf:
                for (segment_id, mesh_data) in segment_ids_and_meshes:
                    if len(mesh_data) == 0:
                        # Empty meshes may be 'serialized' as an empty buffer.
                        # Don't upload such files.
                        continue
                    nonempty_mesh_count += 1
                    mesh_name = _get_mesh_name(config, segment_id)
                    f_info = tarfile.TarInfo(mesh_name)
                    f_info.size = len(mesh_data)
                    tf.addfile(f_info, BytesIO(mesh_data))
    
            if nonempty_mesh_count == 0:
                # Tarball has no content -- all meshes were empty.
                continue
            
            tar_bytes = tar_stream.getbuffer()

            @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
            def write_tar():
                with resource_client.access_context(dvid_server, False, 1, len(tar_bytes)):
                    if config["mesh-config"]["storage"]["naming-scheme"] == "tarsupervoxels":
                        session.post(f'{dvid_server}/api/node/{uuid}/{instance_name}/load', tar_bytes)
                    else:
                        session.post(f'{dvid_server}/api/node/{uuid}/{instance_name}/key/{tar_name}', tar_bytes)
            
            write_tar()
            keys_written.append(tar_name)
    return keys_written

def _get_group_name(config, group_id):
    """
    Encode the given group name (e.g. a 'body' in neu3)
    into a suitable key name for the group tarball.
    """
    grouping_scheme = config["mesh-config"]["storage"]["grouping-scheme"]
    naming_scheme = config["mesh-config"]["storage"]["naming-scheme"]

    # Must not allow np.uint64, which uses a custom __str__()
    group_id = int(group_id)

    if naming_scheme in ("trivial", "tarsupervoxels"):
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

    if naming_scheme in ("trivial", "tarsupervoxels"):
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

