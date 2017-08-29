#!/usr/bin/env python

"""
Usage: prog [--kill-master-on-exit] <workflow name> <config file or callback address>

If the callback address does not start with an 'http' it is assumed to be a configuration path and no callback will be used

Actions:
    1.  Provides spark callback address to launching server (via server callback)
    2.  Initializes spark cluster
    3.  Runs job on server (blocking)
    4.  Examines outputs for errors and informs callback

Assumptions:
    - The environment should be properly set by the driver.
    - The master node must already be running and specified in the MASTER environment variable.

"""
from __future__ import print_function
import os
from os.path import dirname, abspath
import re
import socket
import sys
import subprocess
import traceback
import argparse
import time
import requests
import tempfile
import json
import getpass
import smtplib
from datetime import timedelta
from collections import namedtuple
from email.mime.text import MIMEText
from StringIO import StringIO

# Note: You must run this script with the same python interpreter that will run the workflow
import DVIDSparkServices.workflow

DRIVER_HOSTNAME = socket.gethostname()
SPARK_HOME = os.environ["SPARK_HOME"]
LSB_JOBNAME = os.environ['LSB_JOBNAME']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kill-master-on-exit', action='store_true')
    parser.add_argument('--email-on-exit', action='store_true')
    parser.add_argument('workflow_name')
    parser.add_argument('config_or_callback_address')
    args = parser.parse_args()
    
    # Master node must already be running and specified in the environment.
    if 'MASTER' not in os.environ:
        sys.stderr.write('Error: MASTER environment variable not set!\n')
        sys.exit(1)
    
    if not re.match("spark://.+:[0-9]+", os.environ['MASTER']):
        sys.stderr.write('MASTER environment variable is invalid:\n'
                         'MASTER="{}"\n'.format(os.environ['MASTER']))
        sys.exit(1)
    
    MASTER = os.environ['MASTER']

    if args.kill_master_on_exit:
        if 'MASTER_BJOB_ID' not in os.environ:
            sys.stderr.write('Error: MASTER_BJOB_ID environment variable not set!\n')
            sys.exit(1)
        MASTER_BJOB_ID = os.environ['MASTER_BJOB_ID']        
    
    driver_output = StringIO()
    successful = False
    
    json_header = {'content-type': 'app/json'}
    start = time.time()

    workflow_proc = None
    
    try:
        # ******** Start Job ********
        configfile = args.config_or_callback_address
    
        hascallback = args.config_or_callback_address.startswith("http")        
        if hascallback:
            # write-back callback address
            status = {}
            status["sparkAddr"] = DRIVER_HOSTNAME
            status["job_status"] = "Running"
            status_str = json.dumps(status)
    
            requests.post(args.config_or_callback_address, data=status_str, headers=json_header)
    
            configfile = configfile + "/config"
    
        # call workflow and wait
        launch_workflow_script = abspath(dirname(DVIDSparkServices.workflow.__file__)) + '/launchworkflow.py'
        workflow_proc = subprocess.Popen( [ SPARK_HOME + '/bin/spark-submit',
                                            launch_workflow_script,
                                            args.workflow_name,
                                            '-c',
                                            configfile ],
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT )
        
        # Tee each line of output to both stdout and our driver_output file
        # (so we can send it to the network callback) 
        for line in iter(workflow_proc.stdout.readline, ''):
            sys.stdout.write(line)
            sys.stdout.flush()
            driver_output.write(line)

        # Wait for final process exit.
        # (I don't think there can be anything left on stdout,
        # but just in case, we'll capture it and write it.)
        last_lines, _ = workflow_proc.communicate()
        sys.stdout.write(last_lines)
        sys.stdout.flush()
        driver_output.write(last_lines)

    finally:
        # record time
        duration = timedelta(seconds=int(time.time() - start))
        msg = "\n" + "Total Time: " + str(duration) 
        driver_output.write(msg + "\n")
        print(msg)
        
        if workflow_proc is None:
            # If something failed before we even ran the workflow,
            # Still run the cleanup below.
            workflow_proc = namedtuple('W', 'returncode')(999)

        # write status and message
        status = {}
        status["sparkAddr"] = ""
        if workflow_proc.returncode != 0:
            status["job_status"] = "Error"
        else:
            status["job_status"] = "Finished"
        status["job_message"] = driver_output.getvalue()
        status_str = json.dumps(status)
        
        if hascallback:
            requests.post(args.config_or_callback_address, data=status_str, headers=json_header)
        
        print( "Launch script done: {}".format({True: "successful", False: "UNSUCCESSFUL"}[(workflow_proc.returncode == 0)] ) )
        if workflow_proc.returncode != 0:
            print("Spark process exited with code: {}".format(workflow_proc.returncode))
        

        if args.kill_master_on_exit:
            # Kill the spark cluster
            print('Job complete; Killing Spark Master (JOB_ID={})'.format(MASTER_BJOB_ID))
            subprocess.check_call('bkill {}'.format(MASTER_BJOB_ID), shell=True)

        if args.email_on_exit:
            send_exit_email(args.workflow_name, workflow_proc.returncode, duration, args.config_or_callback_address)

        print("EXITING Driver job")
        
def send_exit_email(workflow_name, returncode, duration, config_file):
    body = "Workflow {} exited with code: {}\n"\
            "Duration: {}\n"\
            "Job name: {}\n"\
           "Config file: {}\n"\
           .format(workflow_name, returncode, duration, LSB_JOBNAME, config_file)

    msg = MIMEText(body)
    msg['Subject'] = 'Spark job exited: {}'.format(returncode)
    msg['From'] = 'sparklaunch_janelia_lsf <{}@{}>'.format(getpass.getuser(), socket.gethostname())
    msg['To'] = '{}@janelia.hhmi.org'.format(getpass.getuser())

    s = smtplib.SMTP('mail.hhmi.org')
    s.sendmail(msg['From'], [msg['To']], msg.as_string())
    s.quit()

if __name__ == "__main__":
    main()

