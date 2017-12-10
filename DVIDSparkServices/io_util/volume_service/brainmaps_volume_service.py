import numpy as np
from jsonschema import validate

from dvid_resource_manager.client import ResourceManagerClient

from DVIDSparkServices.util import replace_default_entries
from DVIDSparkServices.auto_retry import auto_retry
from DVIDSparkServices.io_util.brainmaps import BrainMapsVolume

from . import VolumeServiceReader, GeometrySchema

BrainMapsSegmentationServiceSchema = \
{
    "description": "Parameters to use Google BrainMaps as a source of voxel data",
    "type": "object",
    "required": ["project", "dataset", "volume-id", "change-stack-id"],
    "default": {},
    "properties": {
        "project": {
            "description": "Project ID",
            "type": "string",
        },
        "dataset": {
            "description": "Dataset identifier",
            "type": "string"
        },
        "volume-id": {
            "description": "Volume ID",
            "type": "string"
        },
        "change-stack-id": {
            "description": "Change Stack ID. Specifies a set of changes to apple on top of the volume\n"
                           "(e.g. a set of agglomeration steps).",
            "type": "string",
            "default": ""
        }
    }
}

BrainMapsVolumeSchema = \
{
    "description": "Describes a segmentation volume from BrainMaps.",
    "type": "object",
    "default": {},
    "properties": {
        "slice-files": BrainMapsSegmentationServiceSchema,
        "geometry": GeometrySchema
    }
}

class BrainMapsVolumeServiceReader(VolumeServiceReader):
    """
    A wrapper around the BrainMaps client class that
    matches the VolumeServiceReader API.
    """

    def __init__(self, volume_config, resource_manager_client=None):
        validate(volume_config, BrainMapsVolumeSchema)
        
        if resource_manager_client is None:
            # Dummy client
            resource_manager_client = ResourceManagerClient("", 0)

        self._brainmaps_client = BrainMapsVolume( volume_config["brainmaps"]["project"],
                                                  volume_config["brainmaps"]["dataset"],
                                                  volume_config["brainmaps"]["volume-id"],
                                                  volume_config["brainmaps"]["change-stack-id"],
                                                  dtype=np.uint64 )

        block_width = volume_config["geometry"]["block-width"]
        if block_width == -1:
            # FIXME: I don't actually know what BrainMap's internal block size is...
            block_width = 64
        
        preferred_message_shape_zyx = np.array( volume_config["geometry"]["message-block-shape"][::-1] )
        replace_default_entries(preferred_message_shape_zyx, [64, 64, 6400])

        bounding_box_zyx = np.array(volume_config["geometry"]["bounding-box"])[:,::-1]
        replace_default_entries(bounding_box_zyx, self._brainmaps_client.bounding_box)

        assert  (bounding_box_zyx[0] >= self._brainmaps_client.bounding_box[0]).all() \
            and (bounding_box_zyx[1] <= self._brainmaps_client.bounding_box[1]).all(), \
            f"Specified bounding box ({bounding_box_zyx.tolist()}) extends outside the "\
            f"BrainMaps volume geometry ({self._brainmaps_client.bounding_box.tolist()})"        

        # Store members        
        self._bounding_box_zyx = bounding_box_zyx
        self._resource_manager_client = resource_manager_client
        self._preferred_message_shape_zyx = preferred_message_shape_zyx
        self._block_width = block_width

        # Overwrite config entries that we might have modified
        volume_config["geometry"]["block-width"] = self._block_width
        volume_config["geometry"]["bounding-box"] = self._bounding_box_zyx[:,::-1].tolist()
        volume_config["geometry"]["message-block-shape"] = self._preferred_message_shape_zyx[::-1].tolist()

    @property
    def dtype(self):
        return self._brainmaps_client.dtype

    @property
    def preferred_message_shape(self):
        return self._preferred_message_shape_zyx

    @property
    def block_width(self):
        return self._block_width

    @property
    def bounding_box_zyx(self):
        return self._bounding_box_zyx

    # Two-levels of auto-retry:
    # 1. Auto-retry up to three time for any reason.
    # 2. If that fails due to 504 or 503 (probably cloud VMs warming up), wait 5 minutes and try again.
    @auto_retry(2, pause_between_tries=5*60.0, logging_name=__name__,
                predicate=lambda ex: '503' in ex.args[0] or '504' in ex.args[0])
    @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
    def get_subvolume(self, box, scale=0):
        req_bytes = 8 * np.prod(box[1] - box[0])
        with self._resource_manager_client.access_context('brainmaps', True, 1, req_bytes):
            return self._brainmaps_client.get_subvolume(box, scale)
