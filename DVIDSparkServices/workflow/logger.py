"""
Simple logging routine.  This should eventually use python logging
or be extended to write to a DVID call back.
"""
from __future__ import print_function, absolute_import
import time
import datetime

class WorkflowLogger:

    def __init__(self, appname):
        self.appname = appname

    def __enter__(self):
        self.write_data("Started")
        return self

    def __exit__(self, *args):
        self.write_data("Finished")

    # maybe support writing back to http callback
    def write_data(self, message):
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        print("%s: %s [%s]" % (self.appname, message, timestamp)) 
