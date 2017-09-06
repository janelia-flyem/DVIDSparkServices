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

# Note: You must run this script with the same python interpreter that will run the workflow
import DVIDSparkServices

## NOTE: LSF jobs will inherit all of these environment variables by default. 

# Location of spark distribution
SPARK_HOME = "/misc/local/spark-test" # /misc/local/spark-versions/spark-2.2.0-bin-without-hadoop

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
    bjobs_output = subprocess.check_output(f'bjobs -X -noheader -o EXEC_HOST {job_id}', shell=True).decode()
    hostname = bjobs_output.split(':')[0].split('*')[-1].strip()
    return hostname

def get_job_submit_time(job_id):
    """
    Return the job's submit_time as a datetime object.
    """
    bjobs_output = subprocess.check_output(f'bjobs -X -noheader -o SUBMIT_TIME {job_id}', shell=True).strip().decode()
    # Example:
    # Sep  6 13:10:09 2017
    submit_time = datetime.strptime(f"{bjobs_output} {time.localtime().tm_zone}", "%b %d %H:%M:%S %Y %Z")
    return submit_time

def setup_environment(num_spark_workers, config_file, job_log_dir):
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

    # DVIDSparkServices will drop faulthandler traceback logs here.
    os.environ["DVIDSPARKSERVICES_FAULTHANDLER_OUTPUT_DIR"] = abspath(job_log_dir)


def execute_bsub_command(bsub_cmd, nickname, wait_for_start=True):
    print(bsub_cmd + "\n")
    bsub_output = subprocess.check_output(bsub_cmd, shell=True).decode()
    print(bsub_output)
    
    job_id, queue_name = parse_bsub_output(bsub_output)

    submit_time = get_job_submit_time(job_id)
    submit_timestamp = int(submit_time.timestamp())
    rtm_url = ( f"http://lsf-rtm/cacti/plugins/grid/grid_bjobs.php"
                f"?action=viewjob"
                f"&tab=hostgraph"
                f"&clusterid=1"
                f"&indexid=0"
                f"&jobid={job_id}"
                f"&submit_time={submit_timestamp}" )

    print ("Host graphs:")
    print(rtm_url + "\n")

    if not wait_for_start:
        return job_id, ''

    print(f"Waiting for {nickname} to start...")
    wait_times = [1.0, 5.0, 10.0]
    hostname = get_job_hostname(job_id)
    while hostname == '-':
        time.sleep(wait_times[0])
        if len(wait_times) > 1:
            wait_times = wait_times[1:]
        hostname = get_job_hostname(job_id)

    return job_id, hostname, queue_name


def launch_spark_cluster(job_name, num_spark_workers, max_hours, job_log_dir):
    num_nodes = num_spark_workers + 1 # Add one for master
    num_slots = num_nodes * 16
    max_runtime_minutes = int(max_hours * 60)
    
    cluster_launch_bsub_cmd = \
        ( "bsub"
          " -J {job_name}-cluster"                   # job name in LSF
          " -a 'sparkbatch(test)'"                   # Spark environment, equivalent to old SGE '-pe spark' mode
          " -n {num_slots}"                          # CPUs for master+workers
          " -W {max_runtime_minutes}"                # Terminate after max minutes
          " -o {job_log_dir}/{job_name}-cluster.log" # stdout log
          " dummy-string"
        ).format(**locals())
     
    print("Launching spark cluster:")
    master_job_id, master_hostname, queue_name = execute_bsub_command(cluster_launch_bsub_cmd, 'master')
    assert queue_name == 'spark', f"Unexpected queue name for master job: {queue_name}"
    print(f'...master ({master_job_id}) is running on http://{master_hostname}:8080\n')

    return master_job_id, master_hostname

def launch_driver_job( master_job_id, master_hostname, num_driver_slots, job_log_dir, max_hours, job_name, workflow_name, config_file):
    max_runtime_minutes = int(max_hours * 60)
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
    job_id, hostname, queue_name = execute_bsub_command(driver_submit_cmd, 'driver')
    print(f'...driver ({job_id}) is running in queue "{queue_name}" on http://{hostname}:4040\n')

    return job_id, hostname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver-slots', type=int, default=16)
    parser.add_argument('--job-log-dir', type=str, default='.')
    parser.add_argument('--max-hours', type=float, default=8)
    parser.add_argument('--job-name')
    parser.add_argument('num_spark_workers', type=int)
    parser.add_argument('workflow_name')
    parser.add_argument('config_file')
    args = parser.parse_args()

    if not args.job_name:
        config_name = splitext(basename(args.config_file))[0]
        args.job_name = config_name + '-{:%Y%m%d.%H%M%S}'.format(datetime.now())

    setup_environment(args.num_spark_workers, args.config_file, args.job_log_dir)
    
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
