import argparse
import sys
import imp
import traceback
import glob
import json
import os
import logging
logger = logging.getLogger(__name__)

from DVIDSparkServices.workflow.workflow import WorkflowError

def main(argv):
    try:
        # get workflow path
        workflowpath = sys.argv[0]
        workflowpath = workflowpath.rstrip('launchworkflow.py')
        
        parser = argparse.ArgumentParser(description="Entry point to call spark workflow plugins")

        # must specify the workflow name
        parser.add_argument("workflow", nargs='?', type=str, help="Name of workflow plugin to run")

        # specifying the config file will lead to execution
        parser.add_argument('--config-file', '-c', default="",
                help="json config file")

        # will dump the json schema
        parser.add_argument('--dump-schema', '-d', action="store_true",
                default=False, help="dump json schema")
        
        # will dump the json schema
        parser.add_argument('--list-workflows', '-w', action="store_true",
                default=False, help="list all the workflow plugins")

        args = parser.parse_args()

        # list all services if requested and exit
        if args.list_workflows:
            plugins = [os.path.split(plugin)[1].rstrip('.py') for plugin in glob.glob(workflowpath + '*.py') if plugin != workflowpath + "launchworkflow.py" and plugin != workflowpath + "__init__.py"]
            print json.dumps(plugins, indent=4)
            return

        # import plugin and grab class
        # assume plugin name and class name are the same
        workflow_source = workflowpath + args.workflow + '.py'
        workflow_name = "workflows." + args.workflow
        logger.info("Loading workflow: " + workflow_source + "as: " + workflow_name)
        workflow_mod = imp.load_source(workflow_name, workflow_source)
        workflow_cls = getattr(workflow_mod, args.workflow)

        # print the json schema for the given workflow
        if args.dump_schema:
            print workflow_cls.dumpschema()
            return

        # execute the workflow
        if args.config_file != "":
            workflow_inst = workflow_cls(args.config_file)
            workflow_inst.execute()

    # handle exceptions
    except WorkflowError, e:
        print "Workflow exception: ", str(e)
        traceback.print_exc(file=sys.stdout)
    except Exception, e:
        print "General exception: ", str(e)
        traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    DEBUG_LOGGING = True
    if DEBUG_LOGGING:
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    DEBUG_WITH_PYDEV = False
    if DEBUG_WITH_PYDEV:
        sys.path.append("/Applications/eclipse/plugins/org.python.pydev_4.1.0.201505270003/pysrc")
        import pydevd
        print "Waiting for PyDev debugger..."
        pydevd.settrace(stdoutToServer=True, stderrToServer=True)

    main(sys.argv)
