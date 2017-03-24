"""Defines the a workflow within the context of DVIDSparkServices.

This module only contains the Workflow class and a sepcial
exception type for workflow errors.

"""
import os
from jsonschema import ValidationError
import json
from DVIDSparkServices.json_util import validate_and_inject_defaults
from DVIDSparkServices.workflow.logger import WorkflowLogger



#  workflow exception
class WorkflowError(Exception):
    pass


# defines workflows that work over DVID
class Workflow(object):
    """Base class for all DVIDSparkServices workflow.

    The class handles general workflow functionality such
    as handling the json schema interface required by
    all workflows and starting a spark context.

    """

    OptionsSchema = \
    {
      "type": "object",
      "properties": {
        "corespertask": {
          "type": "integer",
          "default": 1
        },
        "resource-server": {
          "type": "string",
          "default": ""
        },
        "resource-port": {
          "type": "integer",
          "default": 0
        },
        "debug": {
          "description": "Enable certain debugging functionality.  Mandatory for integration tests.",
          "type": "boolean",
          "default": False
        }
      },
      "additionalProperties": True
    }
    
    
    def __init__(self, jsonfile, schema, appname, corespertask=1):
        """Initialization of workflow object.

        Args:
            jsonfile (dict): json config data for workflow
            schema (dict): json schema for workflow
            appname (str): name of the spark application

        """

        self.config_data = None
        schema_data = json.loads(schema)

        if jsonfile.startswith('http'):
            try:
                import requests
                self.config_data = requests.get(jsonfile).json()
            except Exception, e:
                raise WorkflowError("Coud not load file: ", str(e))
        else:
            try:
                self.config_data = json.load(open(jsonfile))
            except Exception, e:
                raise WorkflowError("Coud not load file: ", str(e))

        # validate JSON
        try:
            validate_and_inject_defaults(self.config_data, schema_data)
        except ValidationError, e:
            raise WorkflowError("Validation error: ", str(e))

        self.logger = WorkflowLogger(appname)

        # create spark context
        self.corespertask=corespertask
        self.sc = self._init_spark(appname)



    def _init_spark(self, appname):
        """Internal function to setup spark context
        
        Note: only include spark modules here so that
        the interface can be queried outside of pyspark.

        """
        from pyspark import SparkContext, SparkConf

        # set spark config
        sconfig = SparkConf()
        sconfig.setAppName(appname)

        # check config file for generic corespertask option
        corespertask = self.corespertask
        if "corespertask" in self.config_data["options"]:
            corespertask = self.config_data["options"]["corespertask"]

        # always store job info for later retrieval on master
        # set 1 cpu per task for now but potentially allow
        # each workflow to overwrite this for certain high
        # memory situations.  Maxfailures could probably be 1 if rollback
        # mechanisms exist
        sconfig.setAll([("spark.task.cpus", str(corespertask)),
                        ("spark.task.maxFailures", "2")
                       ]
                      )
        #("spark.eventLog.enabled", "true"),
        #("spark.eventLog.dir", "/tmp"), # is this a good idea -- really is temp

        # check if a server is specified that can manage load to DVID resources
        self.resource_server = ""
        self.resource_port = 0
        if "resource-server" in self.config_data["options"] and "resource-port" in self.config_data["options"]:
            self.resource_server = str(self.config_data["options"]["resource-server"])
            self.resource_port = int(self.config_data["options"]["resource-port"])

        # currently using LZ4 compression: should not degrade runtime much
        # but will help with some operations like shuffling, especially when
        # dealing with things object like highly compressible label volumes
        # NOTE: objects > INT_MAX will cause problems for LZ4
        worker_env = {}
        if "DVIDSPARK_WORKFLOW_TMPDIR" in os.environ and os.environ["DVIDSPARK_WORKFLOW_TMPDIR"]:
            worker_env["DVIDSPARK_WORKFLOW_TMPDIR"] = os.environ["DVIDSPARK_WORKFLOW_TMPDIR"]
        
        # Auto-batching heuristic doesn't work well with our auto-compressed numpy array pickling scheme.
        # Therefore, disable batching with batchSize=1
        return SparkContext(conf=sconfig, batchSize=1, environment=worker_env)

    # make this an explicit abstract method ??
    def execute(self):
        """Children must provide their own execution code"""
        
        raise WorkflowError("No execution function provided")

    # make this an explicit abstract method ??
    @staticmethod
    def dumpschema():
        """Children must provide their own json specification"""

        raise WorkflowError("Derived class must provide a schema")
         

