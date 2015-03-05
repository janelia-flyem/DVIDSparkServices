import argparse
import sys
import importlib
import traceback

from recospark.reconutils.workflow import WorkflowError

def main(argv):
    try:
        parser = argparse.ArgumentParser(description="Entry point to call spark workflow plugins")

        # must specify the workflow name
        parser.add_argument("workflow", type=str, help="Name of workflow plugin to run")

        # specifying the config file will lead to execution
        parser.add_argument('--config-file', '-c', default="",
                help="json config file")

        # will dump the json schema
        parser.add_argument('--dump-schema', '-d', action="store_true",
                default=False, help="dump json schema")

        args = parser.parse_args()

        # import plugin and grab class
        # assume plugin name and class name are the same
        workflow_mod = importlib.import_module("workflows." + args.workflow)
        workflow_cls = getattr(workflow_mod, args.workflow)

        # print the json schema for the given workflow
        if args.dump_schema:
            print workflow_cls.dumpschema()

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
    main(sys.argv)
