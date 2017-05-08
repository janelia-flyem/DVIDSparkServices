import copy
import json
import logging

import requests
import numpy as np

from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 
from DVIDSparkServices.dvid.metadata import create_labelarray 

class CopySegmentation(Workflow):
    
    DvidInfoSchema = \
    {
        "type": "object",
        "required": ["dvid-server",
                     "roi",
                     "input-uuid",
                     "input-segmentation-name",
                     "output-uuid",
                     "output-segmentation-name"],

        "additionalProperties": False,
        "properties": {
            #
            # SERVER
            #
            "dvid-server": {
                "description": "location of DVID server to READ AND WRITE",
                "type": "string",
                "property": "dvid-server"
            },

            #
            # ROI
            #
            "roi": {
                "description": "region of interest to copy",
                "type": "string",
                "minLength": 1
            },
            "partition-method": {
                "description": "Strategy to dvide the ROI into substacks for processing.",
                "type": "string",
                "minLength": 1,
                "enum": ["ask-dvid", "grid-aligned"],
                "default": "ask-dvid"
            },
            "partition-filter": {
                "description": "Optionally remove substacks from the compute set based on some criteria",
                "type": "string",
                "minLength": 1,
                "enum": ["all", "interior-only"],
                "default": "all"
            },

            #
            # INPUT
            #
            "input-uuid": {
                "description": "version node for READING segmentation",
                "type": "string"
            },
            "input-segmentation-name": {
                "description": "The labels instance to READ from. Instance may be either googlevoxels or labelblk.",
                "type": "string",
                "minLength": 1
            },
    
            #
            # OUTPUT
            #
            "output-uuid": {
                "description": "version node for WRITING segmentation",
                "type": "string"
            },
            "output-segmentation-name": {
                "description": "The labels instance to WRITE to.  If necessary, will be created (as labelblk).",
                "type": "string",
                "minLength": 1
            },
            "output-block-size": {
                "description": "The DVID blocksize for new segmentation instances. Ignored if the output segmentation instance already exists.",
                "type": "integer",
                "default": 64
            },
        }
    }

    OptionsSchema = copy.copy(Workflow.OptionsSchema)
    OptionsSchema["properties"].update(
    {
        "chunk-size": {
            "description": "Size of block to download in each thread",
            "type": "integer",
            "default": 512
        },
        "offset": {
            "description": "Offset (x,y,z) for loading data, relative to the input source coordinates",
            "type": "array",
            "items": {
                "type": "integer",
            },
           "minItems": 3,
           "maxItems": 3,
           "default": [0,0,0]
        },
        "pyramid-depth": {
            "description": "Number of pyramid levels to generate (0 means choose automatically)",
            "type": "integer",
            "default": 0
        },
        "quit-after-instance-creation": {
            # This is useful when we are using our trick of local dvids that share a google bucket back-end.
            # The metadata isn't automatically updated, so if the new label instance needs to be created,
            # then we need to restart anyway.
            "description": "If the output segmentation instance doesn't exist yet, " +
                           "create it (via the driver) and exit immediately.",
            "type": "boolean",
            "default": False # TODO: Not implemented yet.  (Do not use.)
        },
    })

    Schema = \
    {
      "$schema": "http://json-schema.org/schema#",
      "title": "Service to load raw and label data into DVID",
      "type": "object",
      "properties": {
        "dvid-info": DvidInfoSchema,
        "options" : OptionsSchema
      }
    }

    @staticmethod
    def dumpschema():
        return json.dumps(CopySegmentation.Schema)

    # name of application for DVID queries
    APP_NAME = "copysegmentation"

    def __init__(self, config_filename):
        super(CopySegmentation, self).__init__( config_filename,
                                                CopySegmentation.dumpschema(),
                                                "Copy Segmentation" )

    def execute(self):
        config_data = self.config_data
        dvid_info = config_data["dvid-info"]
        options = config_data["options"]

        # Prepend 'http://' if necessary.
        if not dvid_info['dvid-server'].startswith('http'):
            dvid_info['dvid-server'] = 'http://' + dvid_info['dvid-server']

        #input_type = get_input_instance_type(dvid_info)
        
        create_labelarray( dvid_info["dvid-server"],
                           dvid_info["uuid"],
                           dvid_info["output-segmentation-name"],
                           3*(dvid_info["output-block-size"],) )

        ## TODO:
        # Actually implement this feature...
        assert not options["quit-after-instance-creation"], "FIXME: not implemented yet." 

        

        # grab ROI subvolumes and find neighbors
        distsubvolumes = self.sparkdvid_context.parallelize_roi( dvid_info["roi"],
                                                                 options["chunk-size"],
                                                                 0,
                                                                 False, # Change to TRUE if stitching needed.
                                                                 dvid_info["partition-method"],
                                                                 dvid_info["partition-filter"] )

        # do not recompute ROI for each iteration
        distsubvolumes.persist()

        def download_and_upload_chunk(subvolume):
            seg_array = download_segmentation_chunk(dvid_info, options, subvolume)
            upload_segmentation_chunk(dvid_info, options, subvolume, seg_array)
        distsubvolumes.values().map(download_and_upload_chunk)

def download_segmentation_chunk( dvid_info, options, subvolume ):
    node_service = retrieve_node_service( dvid_info["dvid-server"], 
                                          dvid_info["input-uuid"],
                                          options["resource-server"],
                                          options["resource-port"],
                                          CopySegmentation.APP_NAME )

    start_zyx = subvolume.box[:3]
    stop_zyx = subvolume.box[3:]
    shape_zyx = np.array(stop_zyx) - start_zyx
    
    # get_labels3D() happens to work for both labelblk AND googlevoxels.
    # (DVID/libdvid can't handle googlevoxels grayscale, but segmentation works.)
    return node_service.get_labels3D( dvid_info["input-segmentation-name"], shape_zyx, start_zyx )

def upload_segmentation_chunk( dvid_info, options, subvolume, seg_array ):
    node_service = retrieve_node_service( dvid_info["dvid-server"], 
                                          dvid_info["output-uuid"],
                                          options["resource-server"],
                                          options["resource-port"],
                                          CopySegmentation.APP_NAME )

    start_zyx = subvolume.box[:3]
    stop_zyx = subvolume.box[3:]
    shape_zyx = np.array(stop_zyx) - start_zyx
    assert shape_zyx == seg_array.shape
    
    return node_service.put_labels3D( dvid_info["output-segmentation-name"], seg_array, start_zyx )


def get_input_instance_type(dvid_info):
    r = requests.get('{dvid-server}/api/node/{input-uuid}/{input-segmentation-name}/info'
                     .format(**dvid_info))
    r.raise_for_status()       

    info = r.json()
    typename = info["Base"]["TypeName"]
    assert typename in ("googlevoxels", "labelblk")
    return typename




























