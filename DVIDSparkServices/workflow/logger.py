"""
Simple logging routine.  This should eventually use python logging
or be extended to write to a DVID call back.
"""
import time
import datetime

class WorkflowLogger:
    def __init__(self, appname):
        self.appname = appname
        self.log("Started")

    def __del__(self):
        self.log("Finished")

    # maybe support writing back to http callback
    def write_data(self, message):
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        print "%s: %s [%s]" % (self.appname, message, timestamp) 
