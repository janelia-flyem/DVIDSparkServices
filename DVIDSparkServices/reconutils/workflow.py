from jsonschema import validate
from jsonschema import ValidationError

import json

#  workflow exception
class WorkflowError(Exception):
    pass


# defines workflows that work over DVID
class Workflow(object):
    def __init__(self, jsonfile, schema, appname):
        # only load spark when creating a workflow
        from pyspark import SparkContext
        
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

        # create spark context
        self.sc = SparkContext(None, appname)

    # make this an explicit abstract method ??
    def execute(self):
        raise WorkflowError("No execution function provided")

    # make this an explicit abstract method ??
    @staticmethod
    def dumpschema():
        raise WorkflowError("Derived class must provide a schema")
         

