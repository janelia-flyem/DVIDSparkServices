from jsonschema import validate
from jsonschema import ValidationError
import json
from DVIDSparkServices.workflow.logger import WorkflowLogger

#  workflow exception
class WorkflowError(Exception):
    pass


# defines workflows that work over DVID
class Workflow(object):
    def __init__(self, jsonfile, schema, appname):
        
        self.config_data = None
        schema_data = json.loads(schema)

        try:
            self.config_data = json.load(open(jsonfile))
        except Exception, e:
            raise WorkflowError("Coud not load file: ", str(e))

        # validate JSON
        try:
            validate(self.config_data, schema_data)
        except ValidationError, e:
            raise WorkflowError("Validation error: ", str(e))

        self.logger = WorkflowLogger(appname)

        # create spark context
        self.sc = self._init_spark(appname)


    # initialize spark context
    def _init_spark(self, appname):
        # only load spark when creating a workflow
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

        return SparkContext(conf=sconfig, serializer=CompressedSerializerLZ4())


    # make this an explicit abstract method ??
    def execute(self):
        raise WorkflowError("No execution function provided")

    # make this an explicit abstract method ??
    @staticmethod
    def dumpschema():
        raise WorkflowError("Derived class must provide a schema")
         

