"""Defines the a workflow within the context of DVIDSparkServices.

This module only contains the Workflow class and a sepcial
exception type for workflow errors.

"""
import sys
import os
import functools
import subprocess
from jsonschema import ValidationError
import json
import uuid
import socket
from DVIDSparkServices.util import mkdir_p, unicode_to_str
from DVIDSparkServices.json_util import validate_and_inject_defaults
from DVIDSparkServices.workflow.logger import WorkflowLogger

from logcollector.client_utils import make_log_collecting_decorator, noop_decorator

try:
    #driver_ip_addr = '127.0.0.1'
    driver_ip_addr = socket.gethostbyname(socket.gethostname())
except socket.gaierror:
    # For some reason, that line above fails sometimes
    # (depending on which network you're on)
    # The method below is a little hacky because it requires
    # making a connection to some arbitrary external site,
    # but it seems to be more reliable. 
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("google.com",80))
    driver_ip_addr = s.getsockname()[0]
    s.close()

#
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
        "log-collector-port": {
          "description": "If provided, a server process will be launched on the driver node to collect certain log messages from worker nodes.",
          "type": "integer",
          "default": 0
        },
        "log-collector-directory": {
          "description": "",
          "type": "string",
          "default": "" # If not provided, a temp directory will be overwritten here.
        },
        "debug": {
          "description": "Enable certain debugging functionality.  Mandatory for integration tests.",
          "type": "boolean",
          "default": False
        }
      },
      "additionalProperties": True,
      "default": {}
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

        # Convert unicode values to str (easier to pass to C++ code)
        self.config_data = unicode_to_str(self.config_data)

        self.logger = WorkflowLogger(appname)

        # create spark context
        self.corespertask=corespertask
        self.sc = self._init_spark(appname)

        # Not all workflow schemas have been ported to inherit Workflow.OptionsSchema,
        # so we have to manually provide default values
        if "log-collector-directory" not in self.config_data["options"]:
            self.config_data["options"]["log-collector-directory"] = ""
        if "log-collector-port" not in self.config_data["options"]:
            self.config_data["options"]["log-collector-port"] = 0

        # Init logcollector directory
        log_dir = self.config_data["options"]["log-collector-directory"]
        if not log_dir:
            log_dir = '/tmp/' + str(uuid.uuid1())
            self.config_data["options"]["log-collector-directory"] = log_dir

        if self.config_data["options"]["log-collector-port"]:
            mkdir_p(log_dir)

    def collect_log(self, task_key_factory=lambda *args, **kwargs: args[0]):
        """
        Use this as a decorator for functions that are executed in spark workers.
        
        task_key_factory:
            A little function that converts the arguments to your function into a key that identifies
            the log file this function should be directed to.
        
        For example, if you want to group your log messages into files according subvolumes:
        
        class MyWorkflow(Workflow):
            def execute():
                dist_subvolumes = self.sparkdvid_context.parallelize_roi(...)
                
                @self.collect_log(lambda sv: sv.box)
                def process_subvolume(subvolume):
                    logger = logging.getLogger(__name__)
                    logger.info("Processing subvolume: {}".format(subvolume.box))

                    ...
                    
                    return result
                
                dist_subvolumes.mapValues(process_subvolume)
        
        """
        port = self.config_data["options"]["log-collector-port"]
        if port == 0:
            return noop_decorator
        else:
            return make_log_collecting_decorator(driver_ip_addr, port)(task_key_factory)

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

    def run(self):
        port = self.config_data["options"]["log-collector-port"]
        self.log_dir = self.config_data["options"]["log-collector-directory"]
        
        if not self.config_data["options"]["log-collector-port"]:
            self.execute()
        else:
            # Start the log server in a separate process
            logserver = subprocess.Popen([sys.executable, '-m', 'logcollector.logserver',
                                          #'--debug=True',
                                          '--log-dir={}'.format(self.config_data["options"]["log-collector-directory"]),
                                          '--port={}'.format(self.config_data["options"]["log-collector-port"])])
            try:
                self.execute()
            finally:
                # NOTE: Apparently the flask server doesn't respond
                #       to SIGTERM if the server is used in debug mode.
                #       If you're using the logserver in debug mode,
                #       you may need to kill it yourself.
                #       See https://github.com/pallets/werkzeug/issues/58
                print("Terminating logserver with PID {}".format(logserver.pid))
                logserver.terminate()

    # make this an explicit abstract method ??
    def execute(self):
        """Children must provide their own execution code"""
        
        raise WorkflowError("No execution function provided")

    # make this an explicit abstract method ??
    @staticmethod
    def dumpschema():
        """Children must provide their own json specification"""

        raise WorkflowError("Derived class must provide a schema")
         

