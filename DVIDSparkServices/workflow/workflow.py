"""Defines the a workflow within the context of DVIDSparkServices.

This module only contains the Workflow class and a special
exception type for workflow errors.

"""
from __future__ import print_function, absolute_import
import sys
import os
import time
import requests
import subprocess
from jsonschema import ValidationError
import json
import uuid
import socket
import getpass
from io import StringIO

# ruamel.yaml supports YAML 1.2, which has
# slightly better compatibility with json.
from ruamel.yaml import YAML
yaml = YAML(typ='rt')
yaml.default_flow_style = False

from quilted.filelock import FileLock

from dvid_resource_manager.server import DEFAULT_CONFIG as DEFAULT_RESOURCE_MANAGER_CONFIG

from DVIDSparkServices import cleanup_faulthandler
from DVIDSparkServices.util import mkdir_p, unicode_to_str, kill_if_running, num_worker_nodes, get_localhost_ip_address
from DVIDSparkServices.json_util import validate_and_inject_defaults, inject_defaults
from DVIDSparkServices.workflow.logger import WorkflowLogger

import logging
from logcollector.client_utils import HTTPHandlerWithExtraData, make_log_collecting_decorator, noop_decorator

logger = logging.getLogger(__name__)

# driver_ip_addr = '127.0.0.1'
driver_ip_addr = get_localhost_ip_address()

#  workflow exception
class WorkflowError(Exception):
    pass


DRIVER_LOGNAME = '@_DRIVER_@' # <-- Funky name so it shows up at the top of the list.

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
        "description": "Options",
        "default": {},
        "additionalProperties": True,

        "properties": {
            ## RESOURCE SERVER
            "resource-server": {
                "description": "If provided, workflows MAY use this resource server to coordinate competing requests from worker nodes. \n"
                               "Set to the IP address of the (already-running) resource server, or use the special word 'driver' \n"
                               "to automatically start a new resource server on the driver node.",
                "type": "string",
                "default": ""
            },
            "resource-port": {
                "description": "Which port the resource server is running on.  (See description above.)",
                "type": "integer",
                "default": 0
            },
            "resource-server-config": {
                "type": "object",
                "default": DEFAULT_RESOURCE_MANAGER_CONFIG,
                "additionalProperties": True
            },

            ## WORKER INITIALIZATION SCRIPT
            "worker-initialization": {
                "type": "object",
                "default": {},
                "additionalProperties": False,
                "description": "The given script will be called once per worker node, before the workflow executes.",
                "properties": {
                    "script-path": {
                        "type": "string",
                        "default": ""
                    },
                    "script-args": {
                        "type": "array",
                        "items": { "type": "string" },
                        "default": []
                    },
                    "launch-delay": {
                        "description": "By default, wait for the script to complete before continuing.\n"
                                       "Otherwise, launch the script asynchronously and then pause for N seconds before continuing.",
                        "type": "integer",
                        "default": -1 # default: blocking execution
                    },
                    "log-dir": {
                        "type": "string",
                        "default": "/tmp"
                    },
                    "also-run-on-driver": {
                        "description": "Also run this initialization script on the driver machine.\n",
                        "type": "boolean",
                        "default": False
                    }
                }
            },

            ## LOG SERVER
            "log-collector-port": {
                "description": "If provided, a server process will be launched on the \n"
                               "driver node to collect certain log messages from worker nodes.",
                "type": "integer",
                "default": 0
            },
            "log-collector-directory": {
                "description": "",
                "type": "string",
                "default": "" # If not provided, a temp directory will be overwritten here.
            },

            "corespertask": {
                # DEPRECATED.
                # Configure spark-config:spark.task.cpus directly.
                "type": "integer",
                "default": 0 # Default is to use spark.task.cpus
            },
            
            "spark-config": {
                "description": "Values to override in the spark config.",
                "type": "object",
                "additionalProperties": True,
                
                "default": {
                    #"spark.eventLog.enabled": True,
                    #"spark.eventLog.dir": "/tmp"
                    "spark.task.cpus": 1,
                    "spark.task.maxFailures": 1
                }
            },

            "debug": {
                "description": "Enable certain debugging functionality.\n"
                               "Mandatory for integration tests.",
                "type": "boolean",
                "default": False
            }
        }
    }
    
    def __init__(self, jsonfile, schema, appname):
        """Initialization of workflow object.

        Args:
            jsonfile (dict): json config data for workflow
            schema (dict): json schema for workflow (already loaded as dict)
            appname (str): name of the spark application

        """

        if not jsonfile.startswith('http'):
            jsonfile = os.path.abspath(jsonfile)
        self.config_path = jsonfile
        self.config_data = None

        try:
            ext = os.path.splitext(jsonfile)[1]
            if jsonfile.startswith('http'):
                self.config_data = requests.get(jsonfile).json()
            elif ext == '.json':
                self.config_data = json.load(open(jsonfile))
            elif ext in ('.yml', '.yaml'):
                self.config_data = yaml.load(open(jsonfile))
            else:
                raise RuntimeError(f"Unknown config file extension: {ext}")
        except Exception as e:
            raise WorkflowError("Could not load config file: ", str(e))

        # validate JSON
        try:
            validate_and_inject_defaults(self.config_data, schema)
        except ValidationError as e:
            raise WorkflowError("Validation error: ", str(e))

        # Convert unicode values to str (easier to pass to C++ code)
        self.config_data = unicode_to_str(self.config_data)

        self.workflow_entry_exit_printer = WorkflowLogger(appname)

        # create spark context
        self.sc = self._init_spark(appname)
        
        self._init_logcollector_config()

        self._execution_uuid = str(uuid.uuid1())
        self._worker_task_id = 0


    def _init_spark(self, appname):
        """Internal function to setup spark context
        
        Note: only include spark modules here so that
        the interface can be queried outside of pyspark.

        """
        # currently using LZ4 compression: should not degrade runtime much
        # but will help with some operations like shuffling, especially when
        # dealing with things object like highly compressible label volumes
        # NOTE: objects > INT_MAX will cause problems for LZ4
        worker_env = {}
        if "DVIDSPARK_WORKFLOW_TMPDIR" in os.environ and os.environ["DVIDSPARK_WORKFLOW_TMPDIR"]:
            worker_env["DVIDSPARK_WORKFLOW_TMPDIR"] = os.environ["DVIDSPARK_WORKFLOW_TMPDIR"]
        
        spark_config = self.config_data["options"]["spark-config"]
        for k in list(spark_config.keys()):
            spark_config[k] = str(spark_config[k])
            if spark_config[k] in ('True', 'False'):
                spark_config[k] = spark_config[k].lower()
            
        # Backwards compatibility:
        # if 'corespertask' option exists, override it in the spark config
        if "corespertask" in self.config_data["options"] and self.config_data["options"]["corespertask"] != 0:
            if spark_config["spark.task.cpus"] != '1':
                raise RuntimeError("Bad config: You can't set both 'corespertask' and 'spark.task.cpus'.  Use 'spark.task.cpus'.")
            spark_config["spark.task.cpus"] = str(self.config_data["options"]["corespertask"])

        # set spark config
        from pyspark import SparkContext, SparkConf
        conf = SparkConf()
        conf.setAppName(appname)
        conf.setAll(list(spark_config.items()))

        # Auto-batching heuristic doesn't work well with our auto-compressed numpy array pickling scheme.
        # Therefore, disable batching with batchSize=1
        return SparkContext(conf=conf, batchSize=1, environment=worker_env)

    def relpath_to_abspath(self, relpath):
        """
        Given a path relative to the CONFIG FILE DIRECTORY (not CWD),
        return an absolute path.
        """
        if relpath.startswith('/'):
            return relpath
        
        assert not self.config_path.startswith("http"), \
            "Can't convert relpath ({}) to abspath, since config comers from an http endpoint ({})".format(relpath, self.config_path)

        abspath = os.path.normpath( os.path.join(self.config_dir, relpath) )
        return abspath

    @property
    def config_dir(self):
        """
        Return the directory that contains our config file.
        """
        return os.path.dirname( os.path.normpath(self.config_path) )

    def _init_logcollector_config(self):
        """
        If necessary, provide default values for the logcollector settings.
        Also, convert log-collector-directory to an abspath.
        """
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

        log_dir = self.relpath_to_abspath(log_dir)
        self.config_data["options"]["log-collector-directory"] = log_dir
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


    def _start_logserver(self):
        """
        If the user's config specifies a non-zero logserver port to use,
        start the logserver as a separate process and return the subprocess.Popen object.
        
        If the user's config doesn't specify a logserver port, return None.
        """
        log_port = self.config_data["options"]["log-collector-port"]
        self.log_dir = self.config_data["options"]["log-collector-directory"]
        
        if log_port == 0:
            return None, None

        # Start the log server in a separate process
        logserver = subprocess.Popen([sys.executable, '-m', 'logcollector.logserver',
                                      '--log-dir={}'.format(self.log_dir),
                                      '--port={}'.format(log_port)],
                                      #'--debug=True', # See note below about terminate() in debug mode...
                                      stderr=subprocess.STDOUT)
        
        # Wait for the server to actually start up before proceeding...
        try:
            time.sleep(2.0)
            r = requests.get('http://0.0.0.0:{}'.format(log_port), timeout=60.0 )
        except:
            # Retry once if necessary.
            time.sleep(5.0)
            r = requests.get('http://0.0.0.0:{}'.format(log_port), timeout=60.0 )

        r.raise_for_status()

        # Send all driver log messages to the server, too.
        formatter = logging.Formatter('%(levelname)s [%(asctime)s] %(module)s %(message)s')
        handler = HTTPHandlerWithExtraData( { 'task_key': DRIVER_LOGNAME },
                                              "0.0.0.0:{}".format(log_port),
                                              '/logsink', 'POST' )
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)

        logger.info(f"Started logserver on {driver_ip_addr}:{log_port}")
        return handler, logserver

    def _kill_logserver(self, handler, log_server_proc):
        if log_server_proc:
            log_port = self.config_data["options"]["log-collector-port"]
            requests.post(f"http://127.0.0.1:{log_port}/logs/shutdown")
            logger.info(f"Terminating logserver (PID {log_server_proc.pid})")
            logging.getLogger().removeHandler(handler)
            log_server_proc.terminate()
            kill_if_running(log_server_proc.pid, 10.0)

    def _start_resource_server(self):
        """
        Initialize the resource server config members and, if necessary,
        start the resource server process on the driver node.
        
        If the resource server is started locally, the "resource-server"
        setting is OVERWRITTEN in the config data with the driver IP.
        
        Returns:
            The resource server Popen object (if started), or None
        """
        # Not all workflow schemas have been ported to inherit Workflow.OptionsSchema,
        # so we have to manually provide default values
        if "resource-server" not in self.config_data["options"]:
            self.config_data["options"]["resource-server"] = ""
        if "resource-port" not in self.config_data["options"]:
            self.config_data["options"]["resource-port"] = 0

        self.resource_server = self.config_data["options"]["resource-server"]
        self.resource_port = self.config_data["options"]["resource-port"]

        if self.resource_server == "":
            return None
        
        if self.resource_port == 0:
            raise RuntimeError("You specified a resource server ({}), but no port"
                               .format(self.resource_server))
        
        if self.resource_server != "driver":
            if self.config_data["options"]["resource-server-config"]:
                raise RuntimeError("The 'resource-server-config' should only be specified when 'resource-server' is 'driver'.")
            return None

        if self.config_data["options"]["resource-server-config"]:
            tmpdir = f"/tmp/{getpass.getuser()}"
            os.makedirs(tmpdir, exist_ok=True)

            server_config_path = f'{tmpdir}/driver-resource-server-config.json'
            with open(server_config_path, 'w') as f:
                json.dump(self.config_data["options"]["resource-server-config"], f)
            config_arg = '--config-file={}'.format(server_config_path)
        else:
            config_arg = ''
        
        # Overwrite workflow config data so workers see our IP address.
        self.config_data["options"]["resource-server"] = driver_ip_addr
        self.resource_server = driver_ip_addr

        logger.info("Starting resource manager on the driver ({})".format(driver_ip_addr))
        resource_server_script = sys.prefix + '/bin/dvid_resource_manager'
        resource_server_process = subprocess.Popen("{python} {server_script} {port} {config_arg}"\
                                                   .format( python=sys.executable,
                                                            server_script=resource_server_script,
                                                            port=self.resource_port,
                                                            config_arg=config_arg),
                                                   stderr=subprocess.STDOUT,
                                                   shell=True)
        logger.info("Started resource manager")

        return resource_server_process

    def _kill_resource_server(self, resource_server_proc):
        if resource_server_proc:
            logger.info("Terminating resource manager (PID {})".format(resource_server_proc.pid))
            resource_server_proc.terminate()
            kill_if_running(resource_server_proc.pid, 10.0)

    def run(self):
        """
        Run the workflow by calling the subclass's execute() function
        (with some startup/shutdown steps before/after).
        """
        handler, log_server_proc = self._start_logserver()
        resource_server_proc = self._start_resource_server()
        worker_init_pids, driver_init_pid = self._run_worker_initializations()
        
        try:
            with self.workflow_entry_exit_printer:
                self.execute()
        finally:
            sys.stderr.flush()
            
            self._kill_initialization_procs(worker_init_pids, driver_init_pid)
            self._kill_resource_server(resource_server_proc)
            self._kill_logserver(handler, log_server_proc)

            # Only the workflow calls cleanup_faulthandler, once all spark workers have exited
            # (All spark workers share the same output file for faulthandler.)
            cleanup_faulthandler()

    def run_on_each_worker(self, func):
        """
        Run the given function once per worker node.
        """
        status_filepath = '/tmp/' + self._execution_uuid + '-' + str(self._worker_task_id)
        self._worker_task_id += 1
        
        @self.collect_log(lambda i: socket.gethostname() + '[' + func.__name__ + ']')
        def task_f(i):
            with FileLock(status_filepath):
                if os.path.exists(status_filepath):
                    return None
                
                # create empty file to indicate the task was executed
                open(status_filepath, 'w')

            result = func()
            return (socket.gethostname(), result)

        num_workers = num_worker_nodes()
        
        # It would be nice if we only had to schedule N tasks for N workers,
        # but we couldn't ensure that tasks are hashed 1-to-1 onto workers.
        # Instead, we'll schedule **LOTS** of extra tasks, but the logic in
        # task_f() will skip the unnecessary work.
        num_tasks = num_workers * 1000

        # Execute the tasks.  Returns [(hostname, result), None, None, (hostname, result), ...],
        # with 'None' interspersed for hosts that were hit multiple times.
        # (Each host only returns a single non-None result)
        host_results = self.sc.parallelize(list(range(num_tasks)), num_tasks)\
                            .repartition(num_tasks).map(task_f).collect()
        host_results = [_f for _f in host_results if _f] # Drop Nones
        
        host_results = dict(host_results)

        assert len(host_results) == num_workers, \
            "Task '{}' was not executed all workers ({}), or some tasks failed! Nodes processed: \n{}"\
            .format(func.__name__, num_workers, host_results)
        logger.info("Ran {} on {} nodes: {}".format(func.__name__, len(host_results), host_results))
        return host_results

    def _run_worker_initializations(self):
        """
        Run an initialization script (e.g. a bash script) on each worker node.
        Returns:
            (worker_init_pids, driver_init_pid), where worker_init_pids is a
            dict of { hostname : PID } containing the PIDs of the init process
            IDs running on the workers.
        """
        from subprocess import STDOUT
        from os.path import basename, splitext

        init_options = self.config_data["options"]["worker-initialization"]
        if not init_options["script-path"]:
            return ({}, None)

        init_options["script-path"] = self.relpath_to_abspath(init_options["script-path"])
        init_options["log-dir"] = self.relpath_to_abspath(init_options["log-dir"])
        mkdir_p(init_options["log-dir"])
        
        def launch_init_script():
            script_name = splitext(basename(init_options["script-path"]))[0]
            log_file = open('{}/{}-{}.log'.format(init_options["log-dir"], script_name, socket.gethostname()), 'w')

            try:
                p = subprocess.Popen( list(map(str, [init_options["script-path"]] + init_options["script-args"])),
                                      stdout=log_file,
                                      stderr=STDOUT )
            except OSError as ex:
                if ex.errno == 8: # Exec format error
                    raise RuntimeError("OSError: [Errno 8] Exec format error\n"
                                       "Make sure your script begins with a shebang line, e.g. !#/bin/bash")
                raise

            if init_options["launch-delay"] == -1:
                p.wait()
                if p.returncode == 126:
                    raise RuntimeError("Permission Error: Worker initialization script is not executable: {}"
                                       .format(init_options["script-path"]))
                assert p.returncode == 0, \
                    "Worker initialization script ({}) failed with exit code: {}"\
                    .format(init_options["script-path"], p.returncode)
                return None

            time.sleep(init_options["launch-delay"])
            return p.pid
        
        worker_init_pids = self.run_on_each_worker(launch_init_script)

        driver_init_pid = None
        if init_options["also-run-on-driver"]:
            driver_init_pid = launch_init_script()
        
        return (worker_init_pids, driver_init_pid)

    def _kill_initialization_procs(self, worker_init_pids, driver_init_pid):
        """
        Kill any initialization processes (as launched from _run_worker_initializations)
        that might still running on the workers and/or the driver.
        
        If they don't respond to SIGTERM, they'll be force-killed with SIGKILL after 10 seconds.
        """
        def kill_init_proc():
            try:
                pid_to_kill = worker_init_pids[socket.gethostname()]
            except KeyError:
                return
            else:
                kill_if_running(pid_to_kill, 10.0)
        
        if any(worker_init_pids.values()):
            self.run_on_each_worker(kill_init_proc)
        else:
            logger.info("No worker init processes to kill")
            

        if driver_init_pid:
            kill_if_running(driver_init_pid, 10.0)
        else:
            logger.info("No driver init process to kill")

    # make this an explicit abstract method ??
    def execute(self):
        """Children must provide their own execution code"""
        
        raise WorkflowError("No execution function provided")


    @classmethod
    def schema(cls):
        raise NotImplementedError

    @classmethod
    def default_config(cls, syntax="json"):
        assert syntax in ("json", "yaml", "yaml-with-comments")
        schema = cls.schema()
        output_stream = StringIO()
        if syntax == "json":
            default_instance = inject_defaults( {}, schema )
            json.dump( default_instance, output_stream, indent=4 )
        else:
            default_instance = inject_defaults( {}, schema, (syntax == "yaml-with-comments"), 2 )
            yaml.dump(default_instance, output_stream )
        return output_stream.getvalue()