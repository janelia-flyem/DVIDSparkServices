#!/usr/bin/env python
"""
Usage: prog <num spark nodes> <workflow name> <job name> <callback address> <spark services workflow script>

Actions:
    1. Launches a spark cluster and master node
    2. Launches a DVIDSparkServices workflow (as a driver process)

Assumptions:
    1. The environment should be properly set by the driver.
    2. spark_launch_janelia_lsf should be in the executable path
"""

from __future__ import print_function
import sys
import os
from os.path import abspath, dirname, basename, splitext 
import re
import time
from datetime import datetime
import argparse
import subprocess
from collections import namedtuple

# Note: You must run this script with the same python interpreter that will run the workflow
import DVIDSparkServices

## NOTE: LSF jobs will inherit all of these environment variables by default. 

# Location of spark distribution
SPARK_HOME = "/usr/local/spark-current"

# spark configuration path (disable default python)
# FIXME: This won't work if DVIDSparkServices is 'installed' to the python interpreter.
#        The 'conf' dir is in the top-level of the repo and won't get installed!
CONF_DIR = abspath(dirname(DVIDSparkServices.__file__) + '/../conf')

# DVIDSparkServices will check this variable and use to override Python's tempfile.tempdir
DVIDSPARK_WORKFLOW_TMPDIR = "/scratch/" + os.environ['USER']

##################################################################


def parse_bsub_output(bsub_output):
    """
    Parse the given output from the 'bsub' command and return the job ID and the queue name.

    Example:
        
        >>> bsub_output = "Job <774133> is submitted to queue <spark>.\n"
        >>> job_id, queue_name = parse_bsub_output(bsub_output)
        >>> assert job_id == '774133'
        >>> assert queue_name == 'spark'
    """
    nonbracket_text = '[^<>]*'
    field_pattern = "{nonbracket_text}<({nonbracket_text})>{nonbracket_text}".format(**locals())

    NUM_FIELDS = 2
    field_matches = re.match(NUM_FIELDS*field_pattern, bsub_output)

    if not field_matches:
        raise RuntimeError("Could not parse bsub output: {}".format(bsub_output))

    job_id = field_matches.groups()[0]
    queue_name = field_matches.groups()[1]
    return job_id, queue_name

def get_job_hostname(job_id):
    """
    For the given job, return the name of the host it's running on.
    If it is running on more than one host, the first hostname listed by bjobs is returned.
    (For 'sparkbatch' jobs, the first host is the master.)
    """
    bjobs_output = subprocess.check_output('bjobs -X -noheader -o EXEC_HOST {}'.format(job_id), shell=True)
    hostname = bjobs_output.split(':')[0].split('*')[-1].strip()
    return hostname

def launch_spark_cluster(job_name, num_spark_workers, max_hours, job_log_dir):
    num_nodes = num_spark_workers + 1 # Add one for master
    num_slots = num_nodes * 16
    max_runtime_minutes = max_hours * 60
    
    # Add directories to PATH
    PATH_DIRS = SPARK_HOME + "/bin:" + SPARK_HOME + "/sbin"

    # set path
    curr_path = os.environ["PATH"]
    os.environ["PATH"] = PATH_DIRS + ":" + curr_path

    # set spark path
    os.environ["SPARK_HOME"] = SPARK_HOME

    # set configuration directory
    os.environ["SPARK_CONF_DIR"] = CONF_DIR

    # set exact python to be used
    os.environ["PYSPARK_PYTHON"] = sys.executable

    # DVIDSparkServices will check this variable and use to override Python's tempfile.tempdir
    os.environ["DVIDSPARK_WORKFLOW_TMPDIR"] = DVIDSPARK_WORKFLOW_TMPDIR

    # Some DVIDSparkServices functions need this information,
    # and it isn't readily available via any PySpark API.    
    os.environ["NUM_SPARK_WORKERS"] = str(num_spark_workers)
    
    cluster_launch_bsub_cmd = \
        ( "bsub"
          " -J {job_name}-cluster"                   # job name in LSF
          " -a 'sparkbatch(current)'"                # Spark environment, equivalent to old SGE '-pe spark' mode
          " -n {num_slots}"                          # CPUs for master+workers
          " -W {max_runtime_minutes}"                # Terminate after max minutes
          " -o {job_log_dir}/{job_name}-cluster.log" # stdout log
          " dummy-string"
        ).format(**locals())
     
    print("Launching spark cluster:")
    print(cluster_launch_bsub_cmd + "\n")
    bsub_output = subprocess.check_output(cluster_launch_bsub_cmd, shell=True)
    print(bsub_output)
    
    master_job_id, queue_name = parse_bsub_output(bsub_output)
    assert queue_name == 'spark', "Unexpected queue name for master job: {}".format(queue_name)

    print("Waiting for master to start...")
    wait_times = [1.0, 5.0, 10.0]
    master_hostname = get_job_hostname(master_job_id)
    while master_hostname == '-':
        time.sleep(wait_times[0])
        if len(wait_times) > 1:
            wait_times = wait_times[1:]
        master_hostname = get_job_hostname(master_job_id)

    print('...master is running on http://{}:8080\n'.format(master_hostname))
    
    return master_job_id, master_hostname

def launch_driver_job( master_job_id, master_hostname, num_driver_slots, job_log_dir, max_hours, job_name, workflow_name, config_file):
    max_runtime_minutes = max_hours * 60
    # Set MASTER now so that it will be inherited by the driver process
    os.environ["MASTER"] = "spark://{}:7077".format(master_hostname)
    
    # Set MASTER_BJOB_ID so the driver can kill the master when the workflow finishes.
    os.environ["MASTER_BJOB_ID"] = master_job_id
    
    job_cmd = "sparklaunch_janelia_lsf_int --kill-master-on-exit --email-on-exit {workflow_name} {config_file}"\
              .format(**locals())

    driver_submit_cmd = \
        ( "bsub"
          " -J {job_name}-driver"                   # job name in LSF
          " -n {num_driver_slots}"                  # CPUs for driver
          " -W {max_runtime_minutes}"               # Terminate after max minutes
          " -o {job_log_dir}/{job_name}-driver.log" # stdout log
          " '{job_cmd}'"
        ).format( **locals() )
    
    print("Launching spark driver:")
    print(driver_submit_cmd + "\n")
    bsub_output = subprocess.check_output(driver_submit_cmd, shell=True)
    print(bsub_output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver-slots', type=int, default=16)
    parser.add_argument('--job-log-dir', type=str, default='.')
    parser.add_argument('--max-hours', type=int, default=8)
    parser.add_argument('--job-name')
    parser.add_argument('num_spark_workers', type=int)
    parser.add_argument('workflow_name')
    parser.add_argument('config_file')
    args = parser.parse_args()

    if not args.job_name:
        config_name = splitext(basename(args.config_file))[0]
        args.job_name = config_name + '-{:%Y%m%d.%H%M%S}'.format(datetime.now())
    
    master_job_id, master_hostname = launch_spark_cluster(args.job_name, args.num_spark_workers, args.max_hours, args.job_log_dir)

    launch_driver_job( master_job_id,
                       master_hostname,
                       args.driver_slots,
                       args.job_log_dir,
                       args.max_hours,
                       args.job_name,
                       args.workflow_name,
                       args.config_file )

if __name__ == "__main__":
    main()
