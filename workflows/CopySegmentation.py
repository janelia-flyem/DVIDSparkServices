import copy
import json
import logging

import requests
import numpy as np

from DVIDSparkServices.sparkdvid import sparkdvid
from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.sparkdvid.sparkdvid import retrieve_node_service 
from DVIDSparkServices.dvid.metadata import create_labelarray
from libdvid.util.roi_utils import copy_roi, RoiInfo
#from DVIDSparkServices.dvid.local_server import ensure_dicedstore_is_running

class CopySegmentation(Workflow):
    
    DataInfoSchema = \
    {
        "type": "object",
        "default": {},
        "required": ["input", "output", "roi"],
        "additionalProperties": False,
        "properties": {
            #
            # INPUT
            #
            "input": {
                "type": "object",
                "default": {},
                "additionalProperties": False,
                "required": ["server", "uuid", "segmentation-name"],
                "properties": {
                    "server": {
                        # Note: "node-local" is not supported yet.  Start the DVID server yourself and use 127.0.0.1 
                        "description": "location of DVID server to READ.  Either IP:PORT or the special word 'node-local'.",
                        "type": "string",
                    },
                    "database-location": {
                        "description": "If 'server' is 'node-local', then this is the location of the database they'll use.",
                        "type": "string"
                    },
                    "uuid": {
                        "description": "version node for READING segmentation",
                        "type": "string"
                    },
                    "segmentation-name": {
                        "description": "The labels instance to READ from. Instance may be either googlevoxels or labelblk.",
                        "type": "string",
                        "minLength": 1
                    }
                }
            },
    
            #
            # OUTPUT
            #
            "output": {
                "type": "object",
                "default": {},
                "additionalProperties": False,
                "required": ["server", "uuid", "segmentation-name"],
                "properties": {
                    "server": {
                        "description": "location of DVID server to WRITE",
                        "type": "string",
                    },
                    "uuid": {
                        "description": "version node for WRITING segmentation",
                        "type": "string"
                    },
                    "segmentation-name": {
                        "description": "The labels instance to WRITE to.  If necessary, will be created (as labelblk).",
                        "type": "string",
                        "minLength": 1
                    },
                    "block-size": {
                        "description": "The DVID blocksize for new segmentation instances. Ignored if the output segmentation instance already exists.",
                        "type": "integer",
                        "default": 64
                    }
                }
            },

            #
            # ROI
            #
            "roi": {
                "type": "object",
                "default": {},
                "additionalProperties": False,
                "required": ["name"],
                "properties": {
                    "name": {
                        "description": "region of interest to copy",
                        "type": "string",
                        "minLength": 1
                    },
                    "partition-method": {
                        "description": "Strategy to divide the ROI into substacks for processing.",
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
                    }
                }
            }
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
            "default": 0 # automatic by default
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
        "data-info": DataInfoSchema,
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

        input_config = self.config_data["data-info"]["input"]
        output_config = self.config_data["data-info"]["output"]
        roi_config = self.config_data["data-info"]["roi"]

        for cfg in (input_config, output_config):
            # Prepend 'http://' if necessary.
            if not cfg['server'].startswith('http'):
                cfg['server'] = 'http://' + cfg['server']

        # Convert from unicode for easier C++ calls
            cfg["server"] = str(cfg["server"])
            cfg["uuid"] = str(cfg["uuid"])
            cfg["segmentation-name"] = str(cfg["segmentation-name"])

        roi_config["name"] = str(roi_config["name"])

        # create spark dvid contexts
        self.sparkdvid_input_context = sparkdvid.sparkdvid(self.sc, input_config["server"], input_config["uuid"], self)
        self.sparkdvid_output_context = sparkdvid.sparkdvid(self.sc, output_config["server"], output_config["uuid"], self)


    def execute(self):
        input_config = self.config_data["data-info"]["input"]
        output_config = self.config_data["data-info"]["output"]
        roi_config = self.config_data["data-info"]["roi"]
        options_config = self.config_data["options"]

        #input_type = get_input_instance_type(input_config)
        create_labelarray( output_config["server"],
                           output_config["uuid"],
                           output_config["segmentation-name"],
                           3*(output_config["block-size"],) )

        # Copy the ROI from source to destination
        src_info = RoiInfo(input_config["server"], input_config["uuid"], roi_config["name"])
        dest_info = RoiInfo(output_config["server"], output_config["uuid"], roi_config["name"])
        copy_roi(src_info, dest_info)

        ## TODO:
        # Actually implement this feature...
        assert not options_config["quit-after-instance-creation"], "FIXME: not implemented yet." 

        

        # (sv_id, sv)
        distsubvolumes = self.sparkdvid_input_context.parallelize_roi( roi_config["name"],
                                                                       options_config["chunk-size"],
                                                                       0,
                                                                       False, # Change to TRUE if stitching needed.
                                                                       roi_config["partition-method"],
                                                                       roi_config["partition-filter"] )

        # do not recompute ROI for each iteration
        distsubvolumes.persist()
        
        # (sv_id, data)
        seg_chunks = self.sparkdvid_input_context.map_labels64( distsubvolumes,
                                                                input_config['segmentation-name'],
                                                                0,
                                                                roi_config["name"] )

        def combine_values( item ):
            (sv_id1, sv), (sv_id2, data) = item
            assert sv_id1 == sv_id2
            return (sv_id1, (sv, data))
        
        # (sv_id, (sv, data))
        seg_chunks = distsubvolumes.zip( seg_chunks ).map( combine_values )

        self.sparkdvid_output_context.foreach_write_labels3d( output_config['segmentation-name'],
                                                              seg_chunks,
                                                              roi_config["name"],
                                                              mutateseg="yes" )

##
## FUNCTIONS BELOW THIS LINE ARE NOT USED (YET?)
##

#         def download_and_upload_chunk(subvolume):
#             seg_array = download_segmentation_chunk(input_config, options_config, subvolume)
#             upload_segmentation_chunk(output_config, options_config, subvolume, seg_array)
# 
#         distsubvolumes.values().map(download_and_upload_chunk).collect()

def download_segmentation_chunk( input_config, options_config, subvolume ):
    node_service = retrieve_node_service( input_config["server"], 
                                          input_config["uuid"],
                                          options_config["resource-server"],
                                          options_config["resource-port"],
                                          CopySegmentation.APP_NAME )

    start_zyx = subvolume.box[:3]
    stop_zyx = subvolume.box[3:]
    shape_zyx = np.array(stop_zyx) - start_zyx
    
    # get_labels3D() happens to work for both labelblk AND googlevoxels.
    # (DVID/libdvid can't handle googlevoxels grayscale, but segmentation works.)
    return node_service.get_labels3D( input_config["segmentation-name"], shape_zyx, start_zyx )

def upload_segmentation_chunk( output_config, options_config, subvolume, seg_array ):
    node_service = retrieve_node_service( output_config["server"], 
                                          output_config["uuid"],
                                          options_config["resource-server"],
                                          options_config["resource-port"],
                                          CopySegmentation.APP_NAME )

    start_zyx = subvolume.box[:3]
    stop_zyx = subvolume.box[3:]
    shape_zyx = np.array(stop_zyx) - start_zyx
    assert shape_zyx == seg_array.shape
    
    return node_service.put_labels3D( output_config["segmentation-name"], seg_array, start_zyx )


def get_input_instance_type(input_config):
    r = requests.get('{dvid-server}/api/node/{uuid}/{segmentation-name}/info'
                     .format(**input_config))
    r.raise_for_status()       

    info = r.json()
    typename = info["Base"]["TypeName"]
    assert typename in ("googlevoxels", "labelblk")
    return typename


if __name__ == "__main__":
    from DVIDSparkServices.json_util import validate_and_inject_defaults
    config = {
        "data-info": {
            "input": {
                "server": "127.0.0.1:8000",
                "uuid": "UUID1",
                "segmentation-name": "labels",
            },
            "output": {
                "server": "bergs-ws1:9000",
                "uuid": "UUID2",
                "segmentation-name": "labels",
            },
            "roi": {
                "name": "section-26"
            },
        },
        "options": {
            "corespertask": 1,
            "chunk-size": 512,
    
            "offset": [0,0,0],
            "pyramid-depth": 0,
            "quit-after-instance-creation": False,
    
            "resource-port": 0,
            "resource-server": "",
    
            "log-collector-directory": "",
            "log-collector-port": 0,
    
            "debug": False,
        }
    }

    validate_and_inject_defaults(config, CopySegmentation.Schema)
    print json.dumps(config, indent=4, separators=(',', ': '))

#     config_text = str(open('/magnetic/workspace/DVIDSparkServices/integration_tests/test_copyseg/temp_data/config.json').read())
#     config_data = json.loads(config_text)
#     print type(config_data['data-info']['input']['server'])
