import numpy as np

from jsonschema import validate

from dvid_resource_manager.client import ResourceManagerClient

from libdvid import DVIDException

from DVIDSparkServices.util import replace_default_entries
from DVIDSparkServices.auto_retry import auto_retry
from DVIDSparkServices.sparkdvid.sparkdvid import sparkdvid
from DVIDSparkServices.dvid.metadata import DataInstance, get_blocksize

from . import GeometrySchema, VolumeServiceReader, VolumeServiceWriter

DvidServiceSchema = \
{
    "description": "Parameters specify a DVID node",
    "type": "object",
    "required": ["server", "uuid"],

    "default": {},
    "properties": {
        "server": {
            "description": "location of DVID server to READ.",
            "type": "string",
        },
        "uuid": {
            "description": "version node for READING segmentation",
            "type": "string"
        }
    }
}

DvidGrayscaleServiceSchema = \
{
    "description": "Parameters specify a source of grayscale data from DVID",
    "type": "object",

    "allOf": [DvidServiceSchema],

    "required": DvidServiceSchema["required"] + ["grayscale-name"],
    "default": {},
    "properties": {
        "grayscale-name": {
            "description": "The grayscale instance to read/write from/to.\n"
                           "Instance must be grayscale (uint8blk).",
            "type": "string",
            "minLength": 1
        }
    }
}

DvidSegmentationServiceSchema = \
{
    "description": "Parameters specify a source of segmentation data from DVID",
    "type": "object",

    "allOf": [DvidServiceSchema],

    "required": DvidServiceSchema["required"] + ["segmentation-name"],
    "default": {},
    "properties": {
        "segmentation-name": {
            "description": "The labels instance to read/write from. \n"
                           "Instance may be either googlevoxels, labelblk, or labelarray.",
            "type": "string",
            "minLength": 1
        }
    }
}

DvidGenericVolumeSchema = \
{
    "description": "Schema for a generic dvid volume",
    "type": "object",
    "default": {},
    "properties": {
        "dvid": { "oneOf": [DvidGrayscaleServiceSchema, DvidSegmentationServiceSchema] },
        "geometry": GeometrySchema
    }
}

DvidSegmentationVolumeSchema = \
{
    "description": "Schema for a segmentation dvid volume", # (for when a generic SegmentationVolumeSchema won't suffice)
    "type": "object",
    "default": {},
    "properties": {
        "dvid": DvidSegmentationServiceSchema,
        "geometry": GeometrySchema
    }
}


class DvidVolumeService(VolumeServiceReader, VolumeServiceWriter):

    def __init__(self, volume_config, resource_manager_client=None):
        validate(volume_config, DvidGenericVolumeSchema)
        
        assert 'apply-labelmap' not in volume_config["dvid"].keys(), \
            "The apply-labelmap section should be parallel to 'dvid' and 'geometry', not nested within the 'dvid' section!"

        ##
        ## server, uuid
        ##
        if not volume_config["dvid"]["server"].startswith('http://'):
            volume_config["dvid"]["server"] = 'http://' + volume_config["dvid"]["server"]
        
        self._server = volume_config["dvid"]["server"]
        self._uuid = volume_config["dvid"]["uuid"]

        ##
        ## instance, dtype, etc.
        ##

        if "segmentation-name" in volume_config["dvid"]:
            self._instance_name = volume_config["dvid"]["segmentation-name"]
            self._dtype = np.uint64
        elif "grayscale-name" in volume_config["dvid"]:
            self._instance_name = volume_config["dvid"]["grayscale-name"]
            self._dtype = np.uint8
            
        self._dtype_nbytes = np.dtype(self._dtype).type().nbytes

        try:
            data_instance = DataInstance(self._server, self._uuid, self._instance_name)
            self._instance_type = data_instance.datatype
            self._is_labels = data_instance.is_labels()
        except ValueError:
            # Instance doesn't exist yet -- we are going to create it.
            if "segmentation-name" in volume_config["dvid"]:
                self._instance_type = 'labelarray'
                self._is_labels = True
            else:
                self._instance_type = 'uint8blk'
                self._is_labels = False

        ##
        ## Block width
        ##
        config_block_width = volume_config["geometry"]["block-width"]

        try:
            block_shape = get_blocksize(self._server, self._uuid, self._instance_name)
            assert block_shape[0] == block_shape[1] == block_shape[2], \
                "Expected blocks to be cubes."
            block_width = block_shape[0]
        except DVIDException:
            block_width = config_block_width

        if block_width == -1:
            # No block-width specified; choose default
            block_width = 64

        assert config_block_width in (-1, block_width), \
            f"DVID volume block-width ({config_block_width}) from config does not match server metadata ({block_width})"

        ##
        ## bounding-box
        ##
        bounding_box_zyx = np.array(volume_config["geometry"]["bounding-box"])[:,::-1]
        assert -1 not in bounding_box_zyx.flat[:], \
            "volume_config must specify explicit values for bounding-box"

        ##
        ## message-block-shape
        ##
        preferred_message_shape_zyx = np.array( volume_config["geometry"]["message-block-shape"][::-1] )
        replace_default_entries(preferred_message_shape_zyx, [block_width, block_width, 100*block_width])

        ##
        ## resource_manager_client
        ##
        if resource_manager_client is None:
            # Dummy client
            resource_manager_client = ResourceManagerClient("", 0)
        
        ##
        ## Store members
        ##
        self._resource_manager_client = resource_manager_client
        self._block_width = block_width
        self._bounding_box_zyx = bounding_box_zyx
        self._preferred_message_shape_zyx = preferred_message_shape_zyx

        ##
        ## Overwrite config entries that we might have modified
        ##
        volume_config["geometry"]["block-width"] = self._block_width
        volume_config["geometry"]["bounding-box"] = self._bounding_box_zyx[:,::-1].tolist()
        volume_config["geometry"]["message-block-shape"] = self._preferred_message_shape_zyx[::-1].tolist()

    @property
    def dtype(self):
        return self._dtype

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
    def get_subvolume(self, box_zyx, scale=0):
        shape = np.asarray(box_zyx[1]) - box_zyx[0]
        req_bytes = self._dtype_nbytes * np.prod(box_zyx[1] - box_zyx[0])
        throttle = (self._resource_manager_client.server_ip == "")
        with self._resource_manager_client.access_context(self._server, True, 1, req_bytes):
            return sparkdvid.get_voxels( self._server, self._uuid, self._instance_name,
                                         scale, self._instance_type, self._is_labels,
                                         shape, box_zyx[0],
                                         throttle=throttle )

    # Two-levels of auto-retry:
    # 1. Auto-retry up to three time for any reason.
    # 2. If that fails due to 504 or 503 (probably cloud VMs warming up), wait 5 minutes and try again.
    @auto_retry(2, pause_between_tries=5*60.0, logging_name=__name__,
                predicate=lambda ex: '503' in ex.args[0] or '504' in ex.args[0])
    @auto_retry(3, pause_between_tries=60.0, logging_name=__name__)
    def write_subvolume(self, subvolume, offset_zyx, scale):
        req_bytes = self._dtype_nbytes * np.prod(subvolume.shape)
        throttle = (self._resource_manager_client.server_ip == "")
        with self._resource_manager_client.access_context(self._server, True, 1, req_bytes):
            return sparkdvid.post_voxels( self._server, self._uuid, self._instance_name,
                                          scale, self._instance_type, self._is_labels,
                                          subvolume, offset_zyx,
                                          throttle=throttle )
