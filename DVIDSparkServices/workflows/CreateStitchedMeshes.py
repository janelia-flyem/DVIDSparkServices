import os
import copy
import tarfile
import logging
from functools import partial
from io import BytesIO
from contextlib import closing

import numpy as np

from vol2mesh.mesh import Mesh, concatenate_meshes

from dvid_resource_manager.client import ResourceManagerClient

from DVIDSparkServices.auto_retry import auto_retry
from DVIDSparkServices.util import Timer, persist_and_execute, unpersist, num_worker_nodes, cpus_per_worker, default_dvid_session
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.sparkdvid.sparkdvid import sparkdvid, retrieve_node_service 
from DVIDSparkServices.reconutils.morpho import object_masks_for_labels
from DVIDSparkServices.dvid.metadata import is_node_locked

from DVIDSparkServices.io_util.volume_service import DvidSegmentationVolumeSchema, LabelMapSchema
from DVIDSparkServices.io_util.labelmap_utils import load_labelmap

from DVIDSparkServices.io_util.brick import Grid, SparseBlockMask
from DVIDSparkServices.io_util.brickwall import BrickWall
from DVIDSparkServices.io_util.volume_service.volume_service import VolumeService
from DVIDSparkServices.io_util.volume_service.dvid_volume_service import DvidVolumeService

from DVIDSparkServices.segstats import aggregate_segment_stats_from_bricks, merge_stats_dfs, write_stats

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
                               "When using decimation, a halo of 2 or more is better to avoid artifacts.",
                "type": "integer",
                "default": 2
            },
            
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

        assert len(mesh_config["simplify-ratios"]) == 1, \
            "FIXME: The current version of the workflow only supports a single output decimation"
    

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
        
        # Bricks have a halo of 1 to ensure that there will be no gaps between meshes from different blocks
        halo = config["mesh-config"]["task-block-halo"]
        brick_wall = BrickWall.from_volume_service(volume_service, 0, None, self.sc, target_partition_size_voxels, sparse_block_mask, halo=halo)
        brick_wall.persist_and_execute("Downloading segmentation", logger)

        mesh_task_shape = np.array(config["mesh-config"]["task-block-shape"])
        if (mesh_task_shape < 1).any():
            assert (mesh_task_shape < 1).all()
            mesh_task_shape = volume_service.preferred_message_shape
        
        mesh_task_grid = Grid( mesh_task_shape )
        if not brick_wall.grid.equivalent_to( mesh_task_grid ):
            assert halo == 0, "FIXME: If you require a halo, you're not allowed to change the task grid.  I will fix this later."
            aligned_wall = brick_wall.realign_to_new_grid(mesh_task_grid)
            aligned_wall.persist_and_execute("Aligning bricks to mesh task grid...")
            brick_wall.unpersist()
            brick_wall = aligned_wall
            
        with Timer(f"Computing segmentation statistics", logger):
            seg_stats = aggregate_segment_stats_from_bricks( brick_wall.bricks, ['segment', 'voxel_count'] )
        output_path = self.config_dir + f'/segment-stats-dataframe.pkl.xz'
        write_stats(seg_stats, output_path, logger)

        # TODO HERE: Filter out segments we don't care about.
        kept_segment_ids = None
        
        def generate_meshes_for_brick( brick ):
            if kept_segment_ids is None:
                filtered_volume = brick.volume
            else:
                # Mask out segments we don't want to process
                filtered_volume = brick.volume.copy('C')
                filtered_flat = filtered_volume.reshape(-1)
                s = pd.Series(filtered_flat)
                filter_mask = ~s.isin(kept_segment_ids).values
                filtered_flat[filter_mask] = 0

            ids_and_mesh_datas = []
            for (segment_id, (box, mask, _count)) in object_masks_for_labels(filtered_volume, brick.physical_box):
                mesh_data = Mesh.from_binary_vol(mask, box)
                #assert isinstance(mesh_data, Mesh)
                ids_and_mesh_datas.append( (segment_id, mesh_data) )

            return ids_and_mesh_datas

        # Compute meshes per-block
        # --> (segment_id, mesh_for_one_block)
        segment_ids_and_mesh_blocks = brick_wall.bricks.flatMap( generate_meshes_for_brick )
        
        # Group by segment ID
        # --> (segment_id, [mesh_for_block, mesh_for_block, ...])
        mesh_blocks_grouped_by_segment = segment_ids_and_mesh_blocks.groupByKey()
        
        # Concatenate into a single mesh per segment
        # --> (segment_id, mesh)
        segment_id_and_mesh = mesh_blocks_grouped_by_segment.mapValues(concatenate_meshes)

        # Decimate
        # --> (segment_id, mesh)
        decimation_fraction = config["mesh-config"]["simplify-ratios"][0]
        def decimate(mesh):
            mesh.simplify(decimation_fraction)
            return mesh
        segment_id_and_mesh = segment_id_and_mesh.mapValues(decimate)

        # Serialize
        # --> (segment_id, mesh_bytes)
        fmt = config["mesh-config"]["storage"]["format"]
        segment_id_and_mesh_bytes = segment_id_and_mesh.mapValues( lambda mesh: mesh.serialize(fmt) )

        # Group by body ID
        # --> ( body_id, [( segment_id, mesh_bytes ), ( segment_id, mesh_bytes ), ...] )
        segment_id_and_mesh_bytes_grouped_by_body = self.group_by_body(segment_id_and_mesh_bytes)

        instance_name = config["dvid-info"]["dvid"]["meshes-destination"]
        with Timer("Writing meshes to DVID", logger):
            segment_id_and_mesh_bytes_grouped_by_body.foreachPartition( partial(post_meshes_to_dvid, config, instance_name) )

            
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

