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
        self.write_data("Started")

    def __del__(self):
        self.write_data("Finished")

    # maybe support writing back to http callback
    def write_data(self, message):
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        print("%s: %s [%s]" % (self.appname, message, timestamp)) 
