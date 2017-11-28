import numpy as np

from dvid_resource_manager.client import ResourceManagerClient

from DVIDSparkServices.util import replace_default_entries
from DVIDSparkServices.auto_retry import auto_retry
from DVIDSparkServices.io_util.brainmaps import BrainMapsVolume

from .volume_service import VolumeServiceReader

class BrainMapsVolumeServiceReader(VolumeServiceReader):
    """
    A wrapper around the BrainMaps client class that
    matches the VolumeServiceReader API.
    """

    def __init__(self, volume_config, resource_manager_client=None):
        if resource_manager_client is None:
            # Dummy client
            resource_manager_client = ResourceManagerClient("", 0)
        
        self._preferred_message_shape_zyx = np.array( volume_config["geometry"]["message-block-shape"][::-1] )
        replace_default_entries(self._preferred_message_shape_zyx, [64, 64, 6400])

        bounding_box_zyx = np.array(volume_config["geometry"]["bounding-box"])[:,::-1]
        
        self._bounding_box_zyx = bounding_box_zyx
        self._resource_manager_client = resource_manager_client

        # Instantiate this outside of get_brainmaps_subvolume,
        # so it can be shared across an entire partition.
        self._brainmaps_client = BrainMapsVolume( volume_config["brainmaps"]["project"],
                                                  volume_config["brainmaps"]["dataset"],
                                                  volume_config["brainmaps"]["volume-id"],
                                                  volume_config["brainmaps"]["change-stack-id"],
                                                  dtype=np.uint64 )

        assert -1 not in bounding_box_zyx.flat[:], "automatic bounds not supported"
        assert  (bounding_box_zyx[0] >= self._brainmaps_client.bounding_box[0]).all() \
            and (bounding_box_zyx[1] <= self._brainmaps_client.bounding_box[1]).all(), \
            f"Specified bounding box ({bounding_box_zyx.tolist()}) extends outside the "\
            f"BrainMaps volume geometry ({self._brainmaps_client.bounding_box.tolist()})"
        
    @property
    def dtype(self):
        return self._brainmaps_client.dtype

    @property
    def preferred_message_shape(self):
        return self.brick_shape_zyx

    @property
    def block_width(self):
        # FIXME: I don't actually know what BrainMap's internal block size is...
        return 64

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
