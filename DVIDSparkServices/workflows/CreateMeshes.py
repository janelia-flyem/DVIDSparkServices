import os
import copy
import tarfile
import logging
from functools import partial
from io import BytesIO
from contextlib import closing

import numpy as np

from vol2mesh.mesh_from_array import mesh_from_array

from dvid_resource_manager.client import ResourceManagerClient

from DVIDSparkServices.auto_retry import auto_retry
from DVIDSparkServices.util import Timer, persist_and_execute, unpersist, num_worker_nodes, cpus_per_worker, default_dvid_session
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.sparkdvid.sparkdvid import sparkdvid, retrieve_node_service 
from DVIDSparkServices.reconutils.morpho import object_masks_for_labels, assemble_masks
from DVIDSparkServices.dvid.metadata import is_node_locked

from DVIDSparkServices.io_util.volume_service import DvidSegmentationVolumeSchema, LabelMapSchema
from DVIDSparkServices.io_util.labelmap_utils import load_labelmap

from DVIDSparkServices.io_util.brick import SparseBlockMask
from DVIDSparkServices.io_util.brickwall import BrickWall
from DVIDSparkServices.io_util.volume_service.volume_service import VolumeService
from DVIDSparkServices.io_util.volume_service.dvid_volume_service import DvidVolumeService

logger = logging.getLogger(__name__)

class CreateMeshes(Workflow):
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
            "simplify-ratios": {
                "description": "Meshes will be generated multiple times, at different simplification settings, based on this list.",
                "type": "array",
                "minItems": 1,
                "default": [0.2],
                "items": {
                    "description": "Mesh simplification aims to reduce the number of \n"
                                   "mesh vertices in the mesh to a fraction of the original mesh. \n"
                                   "This ratio is the fraction to aim for.  To disable simplification, use 1.0.\n",
                    "type": "number",
                    "minimum": 0.0000001,
                    "maximum": 1.0,
                    "default": 0.2 # Set to 1.0 to disable.
                }
            },
            "step-size": {
                "description": "Passed to skimage.measure.marching_cubes_lewiner().\n"
                               "Larger values result in coarser results via faster computation.\n",
                "type": "integer",
                "default": 1
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
                    
                    "skip-groups": {
                        "description": "For 'labelmap grouping scheme, optionally skip writing of this list of groups (tarballs).'",
                        "type": "array",
                        "items": { "type": "integer" },
                        "default": []
                    },
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
        "minimum-downsample-factor": {
            "description": "Minimum factor by which to downsample bodies before processing.\n"
                           "NOTE: If the object is larger than max-analysis-volume, even after \n"
                           "downsampling, then it will be downsampled even further before processing. \n"
                           "The comments in the generated SWC file will indicate the final-downsample-factor. \n",
            "type": "integer",
            "default": 1
        },
        "force-uniform-downsampling": {
            "description": "If true, force all segments in each group to be downsampled at the same level before meshification.\n"
                           "That is, small supervoxels will be downsampled to the same resolution as the largest supervoxel in the body.\n",
            "type": "boolean",
            "default": False
        },

        "max-analysis-volume": {
            "description": "The above downsample-factor will be overridden if the body would still \n"
                           "be too large to process, as defined by this setting.\n",
            "type": "number",
            "default": 1e9 # 1 GB max
        },
        "rescale-before-write": {
            "description": "How much to rescale the meshes before writing to DVID.\n"
                           "Specified as a multiplier, not power-of-2 'scale'.\n",
            "type": "number",
            "default": 1.0
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
        return CreateMeshes.Schema

    def __init__(self, config_filename):
        super(CreateMeshes, self).__init__(config_filename, CreateMeshes.schema(), "CreateMeshes")
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
        import pandas as pd
        self._sanitize_config()

        config = self.config_data
        options = config["options"]
        
        resource_mgr_client = ResourceManagerClient(options["resource-server"], options["resource-port"])
        volume_service = VolumeService.create_from_config(config["dvid-info"], self.config_dir, resource_mgr_client)

        self._init_meshes_instances()

        # Aim for 2 GB RDD partitions
        GB = 2**30
        target_partition_size_voxels = 2 * GB // np.uint64().nbytes
        
        # This will return None if we're not using sparse blocks
        sparse_block_mask = self._get_sparse_block_mask(volume_service)
        
        brick_wall = BrickWall.from_volume_service(volume_service, 0, None, self.sc, target_partition_size_voxels, sparse_block_mask)
        brick_wall.persist_and_execute("Downloading segmentation", logger)

        # brick -> [ (segment_label, (box, mask, count)),
        #            (segment_label, (box, mask, count)), ... ]
        segments_and_masks = brick_wall.bricks.map( partial(compute_segment_masks, config) )
        persist_and_execute(segments_and_masks, "Computing brick-local segment masks", logger)
        brick_wall.unpersist()
        del brick_wall

        with Timer("Computing segment statistics", logger):
            mask_stats_df = self.compute_mask_stats(segments_and_masks)

        # Flatten now, AFTER stats have been computed
        # (compute_mask_stats() requires that the RDDs not have duplicate labels in them.)
        # While we're at it, drop the count (not needed any more)
        # --> (segment_label, (box, mask))
        def drop_count(items):
            new_items = []
            for item in items:
                segment_label, (box, mask, _count) = item
                new_items.append( (segment_label, (box, mask)) )
            return new_items
        segments_and_masks = segments_and_masks.flatMap( drop_count )

        bad_segments = mask_stats_df[['segment', 'compressed_bytes']].query('compressed_bytes > 1.9e9')['segment']
        if len(bad_segments) > 0:
            logger.error(f"SOME SEGMENTS (N={len(bad_segments)}) ARE TOO BIG TO PROCESS.  Skipping segments: {list(bad_segments)}.")
            segments_and_masks = segments_and_masks.filter( lambda seg_mask: seg_mask[0] not in bad_segments.values )
        
        # (segment, (box, mask))
        #   --> (segment, boxes_and_masks)
        #   === (segment, [(box, mask), (box, mask), (box, mask), ...])
        masks_by_segment_id = segments_and_masks.groupByKey()
        persist_and_execute(masks_by_segment_id, "Grouping segment masks by segment label ID", logger)
        segments_and_masks.unpersist()
        del segments_and_masks

        # Insert chosen downsample_factor (a.k.a. dsf)
        #   --> (segment, dsf_and_boxes_and_masks)
        #   === (segment, (downsample_factor, [(box, mask), (box, mask), (box, mask), ...]))
        downsample_df = pd.Series( mask_stats_df['downsample_factor'].values, # Must use '.values' here, otherwise
                                   index=mask_stats_df['segment'].values )    # index is used to read initial data.
        def insert_dsf(item):
            segment, boxes_and_masks = item
            downsample_factor = downsample_df[segment]
            return (segment, (downsample_factor, boxes_and_masks))
        masks_by_segment_id = masks_by_segment_id.map( insert_dsf )

        ##
        ## Filter out small segments and/or small bodies
        ##
        keep_col = mask_stats_df['keep_segment'] & mask_stats_df['keep_body']
        if not keep_col.all():
            # Note: This array will be broadcasted to the workers.
            #       It will be potentially quite large if we're keeping most (but not all) segments.
            #       Broadcast expense should be minimal thanks to lz4 compression,
            #       but RAM usage will be high.
            segments_to_keep = mask_stats_df['segment'][keep_col].values
            filtered_masks_by_segment_id = masks_by_segment_id.filter( lambda key_and_value: key_and_value[0] in segments_to_keep )
            persist_and_execute(filtered_masks_by_segment_id, "Filtering masks by segment and size", logger)
            del masks_by_segment_id
            masks_by_segment_id = filtered_masks_by_segment_id

        # Aggregate
        # --> (segment_label, (box, mask, downsample_factor))
        segment_box_mask_factor = masks_by_segment_id.mapValues( partial(combine_masks, config) )
        persist_and_execute(segment_box_mask_factor, "Assembling masks", logger)

        #
        # Re-compute meshes once for every simplification ratio in the config
        #
        for instance_name, simplification_ratio in zip(self.mesh_instances, config["mesh-config"]["simplify-ratios"]):
            def _generate_mesh(box_mask_factor):
                box, mask, factor = box_mask_factor
                return generate_mesh(config, simplification_ratio, box, mask, factor)
    
            # --> (segment_label, (mesh_bytes, vertex_count))
            segments_meshes_counts = segment_box_mask_factor.mapValues( _generate_mesh )
            persist_and_execute(segments_meshes_counts, f"Computing meshes at decimation {simplification_ratio:.2f}", logger)
    
            with Timer("Computing mesh statistics", logger):
                mask_and_mesh_stats_df = self.append_mesh_stats( mask_stats_df, segments_meshes_counts, f'{simplification_ratio:.2f}' )
    
            # Update the 'keep_body' column: Skip meshes that are too big.
            huge_bodies = (mask_and_mesh_stats_df['body_mesh_bytes'] > 1.9e9)
            if huge_bodies.any():
                logger.error("SOME BODY MESH GROUPS ARE TOO BIG TO PROCESS.  See dumped DataFrame for details.")
                mask_and_mesh_stats_df['keep_body'] &= ~huge_bodies
    
                # Drop them from the processing list
                segments_in_huge_bodies = mask_and_mesh_stats_df['segment'][huge_bodies].values
                segments_meshes_counts = segments_meshes_counts.filter(lambda seg_and_values: not (seg_and_values[0] in segments_in_huge_bodies))
    
            # --> (segment_label, mesh_bytes)
            def drop_vcount(item):
                segment_label, (mesh_bytes, _vertex_count) = item
                return (segment_label, mesh_bytes)
            segments_and_meshes = segments_meshes_counts.map(drop_vcount)
    
            # Group by body ID
            # --> ( body_id ( segment_label, mesh_bytes ) )
            grouped_body_ids_segments_meshes = self.group_by_body(segments_and_meshes)
            unpersist(segments_and_meshes)
            del segments_and_meshes
    
            unpersist(segments_meshes_counts)
            del segments_meshes_counts

            with Timer("Writing meshes to DVID", logger):
                grouped_body_ids_segments_meshes.foreachPartition( partial(post_meshes_to_dvid, config, instance_name) )
            
            unpersist(grouped_body_ids_segments_meshes)
            del grouped_body_ids_segments_meshes
            
    def _get_sparse_block_mask(self, volume_service):
        """
        If the user's config specified a sparse subset of bodies to process,
        Return a SparseBlockMask object indicating where those bodies reside.
        
        If the user did not specify a 'subset-bodies' list, returns None, indicating
        that all segmentation blocks in the volume should be read.
        
        Also, if the input volume is not from a DvidVolumeService, return None.
        (In that case, the 'subset-bodies' feature can be used, but it isn't as efficient.)
        """
        import pandas as pd
        config = self.config_data
        
        sparse_body_ids = config["mesh-config"]["storage"]["subset-bodies"]
        if not sparse_body_ids:
            return None

        if not isinstance(volume_service.base_service, DvidVolumeService):
            # We only know how to retrieve sparse blocks for DVID volumes.
            # For other volume sources, we'll just have to fetch everything and filter
            # out the unwanted bodies at the mask aggregation step.
            return None
        
        grouping_scheme = config["mesh-config"]["storage"]["grouping-scheme"]
        assert grouping_scheme in ('no-groups', 'singletons', 'labelmap'), \
            f"Not allowed to use 'subset-bodies' setting for grouping scheme: {grouping_scheme}"
        
        if grouping_scheme in ('no-groups', 'singletons'):
            # The 'body ids' are identical to segment ids
            sparse_segment_ids = sparse_body_ids
        elif grouping_scheme == 'labelmap':
            # We need to convert the body ids into sparse segment ids        
            mapping_pairs = self.load_labelmap()
            segments, bodies = mapping_pairs.transpose()
            
            # pandas.Series permits duplicate index values,
            # which is convenient for this reverse lookup
            reverse_lookup = pd.Series(index=bodies, data=segments)
            sparse_segment_ids = reverse_lookup.loc[sparse_body_ids].values

        # Fetch the sparse mask of blocks that the sparse segments belong to        
        dvid_service = volume_service.base_service
        block_mask, lowres_box, block_shape = \
            sparkdvid.get_union_block_mask_for_bodies( dvid_service.server,
                                                       dvid_service.uuid,
                                                       dvid_service.instance_name,
                                                       sparse_segment_ids )

        fullres_box = lowres_box * block_shape
        return SparseBlockMask(block_mask, fullres_box, block_shape)

    def compute_mask_stats(self, segments_and_masks):
        """
        segments_and_masks: RDD wher each element is of the form:
                            (label, (box, mask, count))
                             AND labels within a partition are UNIQUE.
        """
        config = self.config_data
        
        # In DataFrames, bounding box is stored as 6 int columns instead 
        # of 1 'object' column for easier joins, combines, serialization, etc.
        BB_COLS = ['z0', 'y0', 'x0', 'z1', 'y1', 'x1']
        STATS_COLUMNS = ['segment', 'segment_voxel_count', 'compressed_bytes'] + BB_COLS

        def stats_df_for_masks(segments_and_masks):
            """
            Convert the list of elements, each in the form: (segment, (box, compressed_mask, count))
            into a pandas DataFrame.
            
            Note: This function assumes that there are no duplicate segments in the list.
                  Therefore, it must be called only with the list of masks from a single 'brick'.
            """
            import pandas as pd
            pd.set_option('expand_frame_repr', False)
            
            # Each item is (segment, (box, compressed_mask, count))
            bounding_boxes = [object_info[1][0] for object_info in segments_and_masks]

            item_df = pd.DataFrame(columns=STATS_COLUMNS)
            item_df['segment'] = [object_info[0] for object_info in segments_and_masks]
            item_df['compressed_bytes'] = [object_info[1][1].compressed_nbytes for object_info in segments_and_masks]
            item_df['segment_voxel_count'] = [object_info[1][2] for object_info in segments_and_masks]
            item_df[BB_COLS] = np.array(bounding_boxes).reshape(-1, 6)

            return item_df

        def merge_stats( left, right ):
            import pandas as pd
            pd.set_option('expand_frame_repr', False)

            # Join the two DFs and replace missing values with appropriate defaults
            joined = left.merge(right, 'outer', on='segment', suffixes=('_left', '_right'), copy=False)
            fillna_inplace(joined, np.inf, ['z0_left', 'y0_left', 'x0_left'])
            fillna_inplace(joined, np.inf, ['z0_right', 'y0_right', 'x0_right'])
            fillna_inplace(joined, -np.inf, ['z1_left', 'y1_left', 'x1_left'])
            fillna_inplace(joined, -np.inf, ['z1_right', 'y1_right', 'x1_right'])
            fillna_inplace(joined, 0, ['segment_voxel_count_left', 'segment_voxel_count_right'])
            fillna_inplace(joined, 0, ['compressed_bytes_left', 'compressed_bytes_right'])

            # Now that the data is aligned by segment label, combine corresponding columns
            result = pd.DataFrame({ 'segment': joined['segment'] })
            result['segment_voxel_count'] = joined['segment_voxel_count_left'] + joined['segment_voxel_count_right']
            result['compressed_bytes'] = joined['compressed_bytes_left'] + joined['compressed_bytes_right']
            result[['z0', 'y0', 'x0']] = np.minimum(joined[['z0_left', 'y0_left', 'x0_left']], joined[['z0_right', 'y0_right', 'x0_right']])
            result[['z1', 'y1', 'x1']] = np.maximum(joined[['z1_left', 'y1_left', 'x1_left']], joined[['z1_right', 'y1_right', 'x1_right']])
            assert set(result.columns) == set(STATS_COLUMNS)
            
            return result

        # Calculate segment (a.k.a. supervoxel) stats
        full_stats_df = segments_and_masks.map(stats_df_for_masks).treeReduce(merge_stats, depth=4)

        # Convert column types (float64 was used above to handle NaNs, but now we can convert back to int)
        convert_dtype_inplace(full_stats_df, np.uint64, ['segment_voxel_count', 'compressed_bytes'])
        convert_dtype_inplace(full_stats_df, np.int64, BB_COLS) # int32 is dangerous because multiplying them together quickly overflows


        full_stats_df['box_size'] = full_stats_df.eval('(z1 - z0)*(y1 - y0)*(x1 - x0)')
        full_stats_df['keep_segment'] = (full_stats_df['segment_voxel_count'] >= config['options']['minimum-segment-size'])
        full_stats_df['keep_segment'] &= (full_stats_df['segment_voxel_count'] <= config['options']['maximum-segment-size'])

        max_analysis_voxels = config['options']['max-analysis-volume']

        # Chosen dowsnsample factor is max of user's minimum and auto-minimum
        full_stats_df['downsample_factor'] = 1 + np.power(full_stats_df['box_size'].values / max_analysis_voxels, (1./3)).astype(np.int16)
        full_stats_df['downsample_factor'] = np.maximum( full_stats_df['downsample_factor'],
                                                         config['options']['minimum-downsample-factor'] )

        # Convert to uint8 to save RAM (will be broadcasted to workers)
        assert full_stats_df['downsample_factor'].max() < 256
        full_stats_df['downsample_factor'] = full_stats_df['downsample_factor'].astype(np.uint8)
        assert full_stats_df['downsample_factor'].dtype == np.uint8

        ##
        ## If grouping segments into bodies (for tarballs),
        ## also append body stats
        ##
        grouping_scheme = config["mesh-config"]["storage"]["grouping-scheme"]
        if grouping_scheme == "labelmap":
            import pandas as pd
            mapping_pairs = self.load_labelmap()

            # Add body column
            segment_to_body_df = pd.DataFrame( mapping_pairs, columns=['segment', 'body'] )
            full_stats_df = full_stats_df.merge(segment_to_body_df, 'left', on='segment', copy=False)

            # Missing segments in the labelmap are assumed to be identity-mapped
            full_stats_df['body'].fillna( full_stats_df['segment'], inplace=True )
            full_stats_df['body'] = full_stats_df['body'].astype(np.uint64)

            # Calculate body voxel sizes
            body_stats_df = full_stats_df[['body', 'segment_voxel_count']].groupby('body').agg(['size', 'sum'])
            body_stats_df.columns = ['body_segment_count', 'body_voxel_count']
            body_stats_df['body'] = body_stats_df.index

            full_stats_df = full_stats_df.merge(body_stats_df, 'left', on='body', copy=False)

            if config["options"]["force-uniform-downsampling"]:
                body_downsample_factors = full_stats_df[['body', 'downsample_factor']].groupby('body', as_index=False).max()
                adjusted_downsample_factors = full_stats_df[['body']].merge(body_downsample_factors, 'left', on='body')
                full_stats_df['downsample_factor'] = adjusted_downsample_factors['downsample_factor'].astype(np.uint8)

            # For offline analysis, write body stats to a file
            output_path = self.config_dir + '/body-stats.csv'
            logger.info(f"Saving body statistics to {output_path}")
            body_stats_df = body_stats_df[['body', 'body_segment_count', 'body_voxel_count']] # Set col order
            body_stats_df.columns = ['body', 'segment_count', 'voxel_count'] # rename columns for csv
            body_stats_df.sort_values('voxel_count', ascending=False, inplace=True)
            body_stats_df.to_csv(output_path, header=True, index=False)
            
        else:
            # Not grouping -- Just duplicate segment stats into body columns
            full_stats_df['body'] = full_stats_df['segment']
            full_stats_df['body_voxel_count'] = full_stats_df['segment_voxel_count']
        
        full_stats_df['keep_body'] = ((full_stats_df['body_voxel_count'] >= config['options']['minimum-agglomerated-size']) &
                                      (full_stats_df['body_voxel_count'] <= config['options']['maximum-agglomerated-size']) )

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
        
        stats_bytes = full_stats_df.memory_usage().sum()
        stats_gb = stats_bytes / 1e9
        
        # Write the Stats DataFrame to a file for offline analysis.
        output_path = self.config_dir + '/segment-stats-dataframe.pkl.xz'
        logger.info(f"Saving segment statistics ({stats_gb:.3f} GB) to {output_path}")
        full_stats_df.to_pickle(output_path)
        
        return full_stats_df


    def append_mesh_stats(self, mask_stats_df, segments_meshes_counts, tag=''):
        # Add mesh sizes to stats columns
        def mesh_stat_row(item):
            segment_label, (mesh_bytes, vertex_count) = item
            return (segment_label, len(mesh_bytes), vertex_count)

        import pandas as pd
        mesh_stats_df = pd.DataFrame(segments_meshes_counts.map( mesh_stat_row ).collect(),
                                      columns=['segment', 'mesh_bytes', 'vertexes'])
                    
        mask_and_mesh_stats_df = mask_stats_df.merge(mesh_stats_df, 'left', on='segment')

        # Calculate body mesh size (in total bytes, not including tar overhead)
        body_mesh_bytes = mask_and_mesh_stats_df[['body', 'mesh_bytes']].groupby('body').sum()
        body_mesh_bytes_df = pd.DataFrame({ 'body': body_mesh_bytes.index,
                                            'body_mesh_bytes': body_mesh_bytes['mesh_bytes'] })
        
        # Add body mesh bytes column
        mask_and_mesh_stats_df = mask_and_mesh_stats_df.merge(body_mesh_bytes_df, 'left', on='body', copy=False)

        stats_gb = mask_and_mesh_stats_df.memory_usage().sum() / 1e9

        # Write the Stats DataFrame to a file for offline analysis.
        if tag:
            output_path = self.config_dir + f'/mesh-stats-{tag}-dataframe.pkl.xz'
        else:
            output_path = self.config_dir + '/mesh-stats-dataframe.pkl.xz'
            
        logger.info(f"Saving mesh statistics ({stats_gb:.3f} GB) to {output_path}")
        mask_and_mesh_stats_df.to_pickle(output_path)

        return mask_and_mesh_stats_df

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


    def group_by_body(self, segments_and_meshes):
        config = self.config_data

        # Group according to scheme
        grouping_scheme = config["mesh-config"]["storage"]["grouping-scheme"]
        n_partitions = num_worker_nodes() * cpus_per_worker()

        if grouping_scheme in "hundreds":
            def last_six_digits( id_mesh ):
                body_id, _mesh = id_mesh
                group_id = body_id - (body_id % 100)
                return group_id
            grouped_body_ids_and_meshes = segments_and_meshes.groupBy(last_six_digits, numPartitions=n_partitions)

        elif grouping_scheme == "labelmap":
            import pandas as pd
            mapping_pairs = self.load_labelmap()

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
            skip_groups = set(config["mesh-config"]["storage"]["skip-groups"])
            grouped_body_ids_and_meshes = segments_and_meshes.mapPartitions( prepend_mapped_group_id ) \
                                                             .filter(lambda item: item[0] not in skip_groups) \
                                                             .groupByKey(numPartitions=n_partitions)
        elif grouping_scheme in ("singletons", "no-groups"):
            # Create 'groups' of one item each, re-using the body ID as the group id.
            # (The difference between 'singletons', and 'no-groups' is in how the mesh is stored, below.)
            grouped_body_ids_and_meshes = segments_and_meshes.map( lambda id_mesh: (id_mesh[0], [(id_mesh[0], id_mesh[1])]) )

        persist_and_execute(grouped_body_ids_and_meshes, f"Grouping meshes with scheme: '{grouping_scheme}'", logger)
        return grouped_body_ids_and_meshes
        

    def _init_meshes_instances(self):
        dvid_info = self.config_data["dvid-info"]
        options = self.config_data["options"]
        if is_node_locked(dvid_info["dvid"]["server"], dvid_info["dvid"]["uuid"]):
            raise RuntimeError(f"Can't write meshes: The node you specified ({dvid_info['dvid']['server']} / {dvid_info['dvid']['uuid']}) is locked.")

        node_service = retrieve_node_service( dvid_info["dvid"]["server"],
                                              dvid_info["dvid"]["uuid"],
                                              options["resource-server"],
                                              options["resource-port"] )

        self.mesh_instances = []
        for simplification_ratio in self.config_data["mesh-config"]["simplify-ratios"]:
            instance_name = dvid_info["dvid"]["meshes-destination"]
            if len(self.config_data["mesh-config"]["simplify-ratios"]) > 1:
                instance_name += f"_dec{simplification_ratio:.2f}"

            node_service.create_keyvalue( instance_name )
            self.mesh_instances.append( instance_name )


def compute_segment_masks(config, brick):
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


def combine_masks(config, dsf_and_boxes_and_masks ):
    """
    Given a list of binary masks and corresponding bounding
    boxes, assemble them all into a combined binary mask.
    
    To save RAM, the data can be downsampled while it is added to the combined mask,
    resulting in a downsampled final mask.

    For more details, see documentation for assemble_masks().

    Arg:
    
        boxes_masks_dsfs: tuple(boxes, compressed_masks, downsample_factor)

            where:
                boxes:
                    list of boxes [(z0,y0,x0), (z1,y1,x1)]
                
                compressed_masks:
                    corresponding list of boolean masks, as CompressedNumpyArrays
                
                downsample_factor:
                    The MINIMUM downsample_factor to use.

    Returns: (combined_bounding_box, combined_mask, downsample_factor)

        where:
            combined_bounding_box:
                the bounding box of the returned mask,
                in NON-downsampled coordinates: ((z0,y0,x0), (z1,y1,x1))
            
            combined_mask:
                the full downsampled combined mask,

            downsample_factor:
                The same downsample_factor as passed in.
                (For convenience of chaining with downstream operations.)
    """
    downsample_factor, boxes_and_masks = dsf_and_boxes_and_masks
    boxes, compressed_masks = zip(*boxes_and_masks)

    boxes = np.asarray(boxes)
    assert boxes.shape == (len(boxes_and_masks), 2,3)
    
    
    # Important to use a generator expression here (not a list comprehension)
    # to avoid excess RAM usage from many uncompressed masks.
    masks = ( compressed.deserialize() for compressed in compressed_masks )

    # In theory this can return 'None' for the combined mask if the object is too small,
    # but we already filtered out 
    combined_box, combined_mask_downsampled, auto_chosen_downsample_factor = \
        assemble_masks( boxes,
                        masks,
                        downsample_factor,
                        config["options"]["minimum-segment-size"],
                        config["options"]["max-analysis-volume"],
                        suppress_zero=True,
                        pad=1 ) # mesh generation requires 1-px halo of zeros

    assert auto_chosen_downsample_factor == downsample_factor, \
        f"The config/driver chose a downsampling factor {downsample_factor} that was "\
        f"different than the one chosen by assemble_masks ({auto_chosen_downsample_factor})!"
        
    return (combined_box, combined_mask_downsampled, downsample_factor)


def generate_mesh(config, simplification_ratio, combined_box, combined_mask, downsample_factor):
    # This config factor is an option to artificially scale the meshes up before
    # writing them, on top of whatever amount the data was downsampled.
    rescale_factor = config["options"]["rescale-before-write"]
    downsample_factor *= rescale_factor
    combined_box = combined_box * rescale_factor

    mesh_bytes, vertex_count = mesh_from_array( combined_mask,
                                                combined_box[0],
                                                downsample_factor,
                                                simplification_ratio,
                                                config["mesh-config"]["step-size"],
                                                config["mesh-config"]["storage"]["format"],
                                                return_vertex_count=True)
    return mesh_bytes, vertex_count


def post_meshes_to_dvid(config, instance_name, partition_items):
    """
    Send the given meshes (either .obj or .drc) as key/value pairs to DVID.
    
    Args:
        config: The CreateMeshes workflow config data
        
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

def fillna_inplace(df, value=0, columns=None):
    """
    Apparently there is no convenient and safe way to call fillna in-place
    with multiple dataframe columns, so here's a utility function for that.
    """
    if columns is None:
        columns = df.columns
    for col in columns:
        df[col].fillna(value, inplace=True)

def convert_dtype_inplace(df, dtype, columns=None):
    if columns is None:
        columns = df.columns
    for col in columns:
        df[col] = df[col].astype(dtype)
