"""Defines the a workflow within the context of DVIDSparkServices.

This module only contains the Workflow class and a sepcial
exception type for workflow errors.

"""

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

    def __init__(self, jsonfile, schema, appname):
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
        self.sc = self._init_spark(appname)

    def _init_spark(self, appname):
        """Internal function to setup spark context
        
        Note: only include spark modules here so that
        the interface can be queried outside of pyspark.

        """
        from pyspark import SparkContext, SparkConf
        from DVIDSparkServices.sparkdvid.CompressedSerializerLZ4 import CompressedSerializerLZ4

        # set spark config
        sconfig = SparkConf()
        sconfig.setAppName(appname)
        
        # always store job info for later retrieval on master
        # set 1 cpu per task for now but potentially allow
        # each workflow to overwrite this for certain high
        # memory situations
        sconfig.setAll([("spark.task.cpus", "1"),
                        ("spark.eventLog.enabled", "true"),
                        ("spark.eventLog.dir", "/tmp") # is this a good idea -- really is temp
                       ]
                      )

        # currently using LZ4 compression: should not degrade runtime much
        # but will help with some operations like shuffling, especially when
        # dealing with things object like highly compressible label volumes
        # NOTE: objects > INT_MAX will cause problems for LZ4
        return SparkContext(conf=sconfig, serializer=CompressedSerializerLZ4())

    # make this an explicit abstract method ??
    def execute(self):
        """Children must provide their own execution code"""
        
        raise WorkflowError("No execution function provided")

    # make this an explicit abstract method ??
    @staticmethod
    def dumpschema():
        """Children must provide their own json specification"""

        raise WorkflowError("Derived class must provide a schema")
         

