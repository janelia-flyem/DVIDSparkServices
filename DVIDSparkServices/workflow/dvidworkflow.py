"""Contains information for DVID-related workflows"""

import sys
import json
from jsonschema import validate
from jsonschema import ValidationError

from DVIDSparkServices.workflow.workflow import Workflow, WorkflowError
from DVIDSparkServices.sparkdvid import sparkdvid

if sys.version_info.major > 2:
    unicode = str

# defines workflows that work over DVID
class DVIDWorkflow(Workflow):
    """ A type of workflow for DVID-specific workflows.

    Provides functionality that constrains/conforms the
    json schema interface and creates a sparkdvid instance.

    """

    # must specify server and uuid
    DVIDSchema = \
{ "$schema": "http://json-schema.org/schema#",
  "title": "Basic DVID Workflow Interface",
  "type": "object",
  "properties": {
    "dvid-info": {
      "type": "object",
      "properties": {
        "dvid-server": {
          "description": "location of DVID server",
          "type": "string",
          "minLength": 1,
          "property": "dvid-server"
        },
        "uuid": {
          "description": "version node to store segmentation",
          "type": "string",
          "minLength": 1
        }
      },
      "required" : ["dvid-server", "uuid"]
    }
  }
}
   
    # calls base initializer and verifies own schema
    def __init__(self, jsonfile, schema, appname):
        super(DVIDWorkflow, self).__init__(jsonfile, schema, appname)

        # separate schema to enforce "server" and "uuid" for all calls
        try:
            validate(self.config_data, self.DVIDSchema)
        except ValidationError as e:
            raise WorkflowError("DVID validation error: ", e.what())

        dvid_info = self.config_data['dvid-info']

        # Prepend 'http://' if necessary.
        if not dvid_info['dvid-server'].startswith('http'):
            dvid_info['dvid-server'] = 'http://' + dvid_info['dvid-server']

        if sys.version_info.major == 2:
            # Convert dvid parameters from unicode to str for easier C++ calls
            for k,v in list(dvid_info.items()):
                if isinstance(v, unicode):
                    dvid_info[k] = str(v)

        # create spark dvid context
        self.sparkdvid_context = sparkdvid.sparkdvid(self.sc,
                self.config_data["dvid-info"]["dvid-server"],
                self.config_data["dvid-info"]["uuid"], self)


    # just dumps specific DVID schema
    @classmethod
    def schema(cls):
        return DVIDWorkflow.DVIDSchema 


