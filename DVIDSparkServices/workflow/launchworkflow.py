from __future__ import print_function
import argparse
import sys
import importlib
import glob
import json
import os
import logging
logger = logging.getLogger(__name__)

import DVIDSparkServices.workflows

def main():
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
        workflows_dir = os.path.dirname(DVIDSparkServices.workflows.__file__)
        workflow_files = map(os.path.basename, glob.glob(workflows_dir + '/*.py'))
        workflow_names = [os.path.splitext(file)[0] for file in workflow_files]
        print(json.dumps(workflow_names, indent=4))
        return

    # import plugin and grab class
    # assume plugin name and class name are the same
    module_name = "DVIDSparkServices.workflows." + args.workflow
    workflow_mod = importlib.import_module(module_name)
    workflow_cls = getattr(workflow_mod, args.workflow)

    # print the json schema for the given workflow
    if args.dump_schema:
        print(workflow_cls.dumpschema())
        return

    # execute the workflow
    if args.config_file != "":
        workflow_inst = workflow_cls(args.config_file)
        workflow_inst.run()

if __name__ == "__main__":
    sys.exit( main() )
