import os
import copy
import tarfile
import logging
from functools import partial
from io import BytesIO
from contextlib import closing

import numpy as np
import requests

from vol2mesh.mesh_from_array import mesh_from_array

from dvid_resource_manager.client import ResourceManagerClient

from DVIDSparkServices.auto_retry import auto_retry
from DVIDSparkServices.util import Timer, persist_and_execute, unpersist, num_worker_nodes, cpus_per_worker
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 
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
                    
                    "labelmap": copy.copy(LabelMapSchema), # Only used by the 'labelmap' grouping-scheme
                    
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
            "default": 1.9e9
        },
        "minimum-agglomerated-size": {
            "description": "Agglomerated groups smaller than this voxel count will not be processed.",
            "type": "number",
            "default": 10e6
        },
        "downsample-factor": {
            "description": "Minimum factor by which to downsample bodies before processing.n"
                           "NOTE: If the object is larger than max-analysis-volume, even after \n"
                           "downsampling, then it will be downsampled even further before processing. \n"
                           "The comments in the generated SWC file will indicate the final-downsample-factor. \n",
            "type": "integer",
            "default": 1
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
        self._sanitize_config()

        config = self.config_data
        options = config["options"]
        
        resource_mgr_client = ResourceManagerClient(options["resource-server"], options["resource-port"])
        volume_service = VolumeService.create_from_config(config["dvid-info"], self.config_dir, resource_mgr_client)

        self._init_meshes_instance()

        # Aim for 2 GB RDD partitions
        GB = 2**30
        target_partition_size_voxels = 2 * GB // np.uint64().nbytes

        brick_wall = BrickWall.from_volume_service(volume_service, 0, None, self.sc, target_partition_size_voxels)
        brick_wall.persist_and_execute("Downloading segmentation", logger)

        # brick -> (segment_label, (box, mask, count))
        segments_and_masks = brick_wall.bricks.map( partial(compute_segment_masks, config) )
        persist_and_execute(segments_and_masks, "Computing brick-local segment masks", logger)
        brick_wall.unpersist()
        del brick_wall

        with Timer("Computing segment statistics", logger):
            mask_stats_df = self.compute_mask_stats(segments_and_masks)

        # Flatten now, AFTER stats have been computed
        # (compute_mask_stats() requires that the RDDs not have duplicate labels in them.)
        segments_and_masks = segments_and_masks.flatMap(lambda x: x)

        # (label, (box, mask, count))
        #   --> (label, [(box, mask, count), (box, mask, count), (box, mask, count), ...])
        masks_by_segment_id = segments_and_masks.groupByKey()
        persist_and_execute(masks_by_segment_id, "Grouping segment masks by segment label ID", logger)
        segments_and_masks.unpersist()
        del segments_and_masks

        ##
        ## Filter out small segments and/or small bodies
        ##
        keep_col = mask_stats_df['keep_segment'] & mask_stats_df['keep_body']
        if not keep_col.all():
            # Note: This array will be broadcasted to the workers.
            #       It will be potentially quite large if we're keeping most (but not all) segments.
            segments_to_keep = mask_stats_df['segment'][keep_col].values
            filtered_masks_by_segment_id = masks_by_segment_id.filter( lambda key_and_value: key_and_value[0] in segments_to_keep )
            persist_and_execute(filtered_masks_by_segment_id, "Filtering masks by segment and size", logger)
            del masks_by_segment_id
            masks_by_segment_id = filtered_masks_by_segment_id

        # Aggregate
        # --> (segment_label, (box, mask, downsample_factor))
        segment_box_mask_factor = masks_by_segment_id.mapValues( partial(combine_masks, config) )

        def _generate_mesh(box_mask_factor):
            box, mask, factor = box_mask_factor
            return generate_mesh(config, box, mask, factor)

        # --> (segment_label, (mesh_bytes, vertex_count))
        segments_meshes_counts = segment_box_mask_factor.mapValues( _generate_mesh )
        persist_and_execute(segments_meshes_counts, "Computing meshes", logger)

        with Timer("Computing mesh statistics", logger):
            mask_stats_df = self.append_mesh_stats( mask_stats_df, segments_meshes_counts )

        # Update the 'keep_body' column: Skip meshes that are too big.
        huge_bodies = (mask_stats_df['body_mesh_bytes'] > 1.9e9)
        if huge_bodies.any():
            logger.error("SOME BODY MESH GROUPS ARE TOO BIG TO PROCESS.  See dumped DataFrame for details.")
            mask_stats_df['keep_body'] &= ~huge_bodies

            # Drop them from the processing list
            segments_in_huge_bodies = mask_stats_df['segment'][huge_bodies].values
            segments_meshes_counts = segments_meshes_counts.filter(lambda seg_and_values: not (seg_and_values[0] in segments_in_huge_bodies))

        # --> (segment_label, mesh_bytes)
        def drop_count(item):
            segment_label, (mesh_bytes, _vertex_count) = item
            return (segment_label, mesh_bytes)
        segments_and_meshes = segments_meshes_counts.map(drop_count)

        # Group by body ID
        # --> ( body_id ( segment_label, mesh_bytes ) )
        grouped_body_ids_segments_meshes = self.group_by_body(segments_and_meshes)
        unpersist(segments_and_meshes)
        del segments_and_meshes

        with Timer("Writing meshes to DVID", logger) as timer:
            grouped_body_ids_segments_meshes.foreachPartition( partial(post_meshes_to_dvid, config) )


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
            joined = left.merge(right, 'outer', 'segment', suffixes=('_left', '_right'), copy=False)
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
        full_stats_df['box_size'] = full_stats_df.eval('(z1 - z0)*(y1 - y0)*(x1 - x0)')
        full_stats_df['keep_segment'] = (full_stats_df['segment_voxel_count'] >= config['options']['minimum-segment-size'])
        full_stats_df['keep_segment'] &= (full_stats_df['segment_voxel_count'] <= config['options']['maximum-segment-size'])

        max_analysis_voxels = config['options']['max-analysis-volume']
        full_stats_df['downsample_factor'] = 1 + np.power(full_stats_df['box_size'].values / max_analysis_voxels, (1./3)).astype(int)

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

            # Missing segments in the labelmap are assumed to be the identity-mapped
            full_stats_df['body'].fillna( full_stats_df['segment'], inplace=True )

            # Calculate body voxel sizes
            body_sizes = full_stats_df[['body', 'segment_voxel_count']].groupby('body').sum()
            body_sizes_df = pd.DataFrame({ 'body': body_sizes.index,
                                           'body_voxel_count': body_sizes['segment_voxel_count'] })

            full_stats_df = full_stats_df.merge(body_sizes_df, 'left', on='body', copy=False)
        else:
            # Not grouping -- Just duplicate segment stats into body columns
            full_stats_df['body'] = full_stats_df['body']
            full_stats_df['body_voxel_count'] = full_stats_df['segment_voxel_count']
        
        #logger.info(f"{full_stats_df}")
        full_stats_df['keep_body'] = (full_stats_df['body_voxel_count'] >= config['options']['minimum-agglomerated-size'])

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


    def append_mesh_stats(self, mask_stats_df, segments_meshes_counts):
        # Add mesh sizes to stats columns
        def mesh_stat_row(item):
            segment_label, (mesh_bytes, vertex_count) = item
            return (segment_label, len(mesh_bytes), vertex_count)

        import pandas as pd
        mesh_stats_df = pd.DataFrame(segments_meshes_counts.map( mesh_stat_row ).collect(),
                                      columns=['segment', 'mesh_bytes', 'vertexes'])
                    
        mask_stats_df = mask_stats_df.merge(mesh_stats_df, 'left', on='segment', copy=False)

        # Calculate body mesh size (in total bytes, not including tar overhead)
        body_mesh_bytes = mask_stats_df[['body', 'mesh_bytes']].groupby('body').sum()
        body_mesh_bytes_df = pd.DataFrame({ 'body': body_mesh_bytes.index,
                                            'body_mesh_bytes': body_mesh_bytes['mesh_bytes'] })
        
        # Add body mesh bytes column
        mask_stats_df = mask_stats_df.merge(body_mesh_bytes_df, 'left', on='body', copy=False)

        stats_gb = mask_stats_df.memory_usage().sum() / 1e9

        # Write the Stats DataFrame to a file for offline analysis.
        output_path = self.config_dir + '/mesh-stats-dataframe.pkl.xz'
        logger.info(f"Saving mesh statistics ({stats_gb:.3f} GB) to {output_path}")
        mask_stats_df.to_pickle(output_path)

        return mask_stats_df

    def load_labelmap(self):
        if self._labelmap is None:
            config = self.config_data
            grouping_scheme = config["mesh-config"]["storage"]["grouping-scheme"]
            assert grouping_scheme == 'labelmap'
            self._labelmap = load_labelmap( config["mesh-config"]["storage"]["labelmap"], self.config_dir )
        return self._labelmap


    def group_by_body(self, body_ids_and_meshes):
        config = self.config_data

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
            grouped_body_ids_and_meshes = body_ids_and_meshes.mapPartitions( prepend_mapped_group_id ) \
                                                             .filter(lambda item: item[0] not in skip_groups) \
                                                             .groupByKey(numPartitions=n_partitions)
        elif grouping_scheme in ("singletons", "no-groups"):
            # Create 'groups' of one item each, re-using the body ID as the group id.
            # (The difference between 'singletons', and 'no-groups' is in how the mesh is stored, below.)
            grouped_body_ids_and_meshes = body_ids_and_meshes.map( lambda id_mesh: (id_mesh[0], [(id_mesh[0], id_mesh[1])]) )

        persist_and_execute(grouped_body_ids_and_meshes, f"Grouping meshes with scheme: '{grouping_scheme}'", logger)
        return grouped_body_ids_and_meshes

        
        
        return grouped_body_ids_and_meshes
        

    def _init_meshes_instance(self):
        dvid_info = self.config_data["dvid-info"]
        options = self.config_data["options"]
        if is_node_locked(dvid_info["dvid"]["server"], dvid_info["dvid"]["uuid"]):
            raise RuntimeError(f"Can't write meshes: The node you specified ({dvid_info['dvid']['server']} / {dvid_info['dvid']['uuid']}) is locked.")

        node_service = retrieve_node_service( dvid_info["dvid"]["server"],
                                              dvid_info["dvid"]["uuid"],
                                              options["resource-server"],
                                              options["resource-port"] )

        node_service.create_keyvalue(dvid_info["dvid"]["meshes-destination"])


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


def combine_masks(config, boxes_and_compressed_masks ):
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
                        config["options"]["max-analysis-volume"],
                        suppress_zero=True,
                        pad=1 ) # mesh generation requires 1-px halo of zeros

    return (combined_box, combined_mask_downsampled, chosen_downsample_factor)


def generate_mesh(config, combined_box, combined_mask, downsample_factor):
    # This config factor is an option to artificially scale the meshes up before
    # writing them, on top of whatever amount the data was downsampled.
    rescale_factor = config["options"]["rescale-before-write"]
    downsample_factor *= rescale_factor
    combined_box = combined_box * rescale_factor

    mesh_bytes, vertex_count = mesh_from_array( combined_mask,
                                                combined_box[0],
                                                downsample_factor,
                                                config["mesh-config"]["simplify-ratio"],
                                                config["mesh-config"]["step-size"],
                                                config["mesh-config"]["storage"]["format"],
                                                return_vertex_count=True)
    return mesh_bytes, vertex_count


def post_meshes_to_dvid(config, partition_items):
    """
    Send the given meshes (either .obj or .drc) as key/value pairs to DVID.
    
    Args:
        config: The CreateMeshes workflow config data
            
        items: tuple (segment_id, mesh_data, error_text)
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
        for group_id, segment_ids_and_meshes in partition_items:
            for (segment_id, mesh_data) in segment_ids_and_meshes:

                @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
                def write_mesh():
                    with resource_client.access_context(dvid_server, False, 2, len(mesh_data)):
                        session.post(f'{dvid_server}/api/node/{uuid}/{instance}/key/{segment_id}', mesh_data)
                        session.post(f'{dvid_server}/api/node/{uuid}/{instance}/key/{segment_id}_info', json={ 'format': 'drc' })
                
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
                    session.post(f'{dvid_server}/api/node/{uuid}/{instance}/key/{tar_name}', tar_bytes)
            
            write_tar()

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

def fillna_inplace(df, value=0, columns=None):
    """
    Apparently there is no convenient and safe way to call fillna in-place
    with multiple dataframe columns, so here's a utility function for that.
    """
    if columns is None:
        columns = df.columns
    for col in columns:
        df[col].fillna(value, inplace=True)

