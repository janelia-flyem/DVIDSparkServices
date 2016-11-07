"""Contains information for DVID-related workflows"""

import json
from jsonschema import validate
from jsonschema import ValidationError

from DVIDSparkServices.workflow.workflow import Workflow
from DVIDSparkServices.workflow import workflow
from DVIDSparkServices.sparkdvid import sparkdvid



# defines workflows that work over DVID
class DVIDWorkflow(Workflow):
    """ A type of workflow for DVID-specific workflows.

    Provides functionality that constrains/conforms the
    json schema interface and creates a sparkdvid instance.

    """

    # must specify server and uuid
    DVIDSchema = """
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
    """
   
    # calls base initializer and verifies own schema
    def __init__(self, jsonfile, schema, appname, corespertask=1):
        super(DVIDWorkflow, self).__init__(jsonfile, schema, appname, corespertask)

        # separate schema to enforce "server" and "uuid" for all calls
        try:
            validate(self.config_data, json.loads(self.DVIDSchema))
        except ValidationError, e:
            raise WorkflowError("DVID validation error: ", e.what())

        # create spark dvid context
        self.sparkdvid_context = sparkdvid.sparkdvid(self.sc,
                self.config_data["dvid-info"]["dvid-server"],
                self.config_data["dvid-info"]["uuid"], self)


    # just dumps specific DVID schema
    @staticmethod
    def dumpschema():
        return DVIDWorkflow.DVIDSchema 


