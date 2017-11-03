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

import sys
import os
from os.path import abspath, dirname, basename, splitext, exists
from datetime import datetime
import argparse

# Note: You must run this script with the same python interpreter that will run the workflow
import DVIDSparkServices
from .lsf_utils import Bjob, kill_job

## NOTE: LSF jobs will inherit all of these environment variables by default. 

# Location of spark distribution
SPARK_HOME = "/misc/local/spark-test" # /misc/local/spark-versions/spark-2.2.0-bin-without-hadoop

# spark configuration path
SPARK_CONF_DIR = abspath(dirname(DVIDSparkServices.__file__) + '/SPARK_CONF_DIR')
if ( not exists(f'{SPARK_CONF_DIR}/spark-defaults.conf') or not exists(f'{SPARK_CONF_DIR}/spark-env.sh') ):
    raise RuntimeError(f"SPARK_CONF_DIR does not contain necessary configuration files: {SPARK_CONF_DIR}")

# DVIDSparkServices will check this variable and use to override Python's tempfile.tempdir
DVIDSPARK_WORKFLOW_TMPDIR = "/scratch/" + os.environ['USER']

##################################################################
##################################################################

def setup_environment(num_spark_workers, config_file, job_log_dir):
    # Add directories to PATH
    PATH_DIRS = SPARK_HOME + "/bin:" + SPARK_HOME + "/sbin"

    # set path
    curr_path = os.environ["PATH"]
    os.environ["PATH"] = PATH_DIRS + ":" + curr_path

    # set spark path
    os.environ["SPARK_HOME"] = SPARK_HOME

    # set configuration directory
    os.environ["SPARK_CONF_DIR"] = SPARK_CONF_DIR

    # set exact python to be used
    os.environ["PYSPARK_PYTHON"] = sys.executable

    # DVIDSparkServices will check this variable and use to override Python's tempfile.tempdir
    os.environ["DVIDSPARK_WORKFLOW_TMPDIR"] = DVIDSPARK_WORKFLOW_TMPDIR

    # Some DVIDSparkServices functions need this information,
    # and it isn't readily available via any PySpark API.    
    os.environ["NUM_SPARK_WORKERS"] = str(num_spark_workers)

    # DVIDSparkServices will drop faulthandler traceback logs here.
    os.environ["DVIDSPARKSERVICES_FAULTHANDLER_OUTPUT_DIR"] = abspath(job_log_dir)


def launch_spark_cluster(job_name, num_spark_workers, max_hours, job_log_dir):
    num_nodes = num_spark_workers + 1 # Add one for master
    num_slots = num_nodes * 16
    
    job = Bjob( 'dummy-string',
                name=f'{job_name}-cluster',
                app_env='sparkbatch(test)',
                num_slots=num_slots,
                max_runtime_minutes=int(max_hours * 60),
                stdout_file=f'{job_log_dir}/{job_name}-cluster.log' )

    try:
        print("Launching spark cluster:")
        master_job_id, queue_name, master_hostname = job.submit()
        assert queue_name == 'spark', f"Unexpected queue name for master job: {queue_name}"

        print(f'...master ({master_job_id}) is running on http://{master_hostname}:8080\n')
        return master_job_id, master_hostname

    except KeyboardInterrupt:
        if job.job_id:
            print(f"Interrupted. Killing job {job.job_id}")
            kill_job(job.job_id)
        raise


def launch_driver_job( master_job_id, master_hostname, num_driver_slots, job_log_dir, max_hours, job_name, workflow_name, config_file):
    # Set MASTER now so that it will be inherited by the driver process
    os.environ["MASTER"] = "spark://{}:7077".format(master_hostname)
    
    # Set MASTER_BJOB_ID so the driver can kill the master when the workflow finishes.
    os.environ["MASTER_BJOB_ID"] = master_job_id
    
    job_cmd = f"sparklaunch_janelia_lsf_int --kill-master-on-exit --email-on-exit {workflow_name} {config_file}"

    job = Bjob( job_cmd,
                name=f"{job_name}-driver",
                num_slots=num_driver_slots,
                max_runtime_minutes=int(max_hours * 60),
                stdout_file=f"{job_log_dir}/{job_name}-driver.log" )


    try:
        print("Launching spark driver:")
        job_id, queue_name, hostname = job.submit()
        print(f'...driver ({job_id}) is running in queue "{queue_name}" on http://{hostname}:4040\n')
        return job_id, hostname

    except KeyboardInterrupt:
        if job.job_id:
            print(f"Interrupted. Killing job {job.job_id}")
            kill_job(job.job_id)
        raise


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
    
    master_job_id = driver_job_id = None
    
    try:
        master_job_id, master_hostname = launch_spark_cluster( args.job_name,
                                                               args.num_spark_workers,
                                                               args.max_hours + 10/60, # 10 extra minutes for the spark cluster;  
                                                               args.job_log_dir)       # it's easier to make sense of the logs when the driver dies first.
    
        driver_job_id, _driver_hostname = launch_driver_job( master_job_id,
                                                              master_hostname,
                                                              args.driver_slots,
                                                              args.job_log_dir,
                                                              args.max_hours,
                                                              args.job_name,
                                                              args.workflow_name,
                                                              args.config_file )
    except BaseException as ex:
        if isinstance(ex, KeyboardInterrupt):
            print("User Interrupted!")
        if master_job_id:
            print(f"Killing master (job {master_job_id})")
            kill_job(master_job_id)
        if driver_job_id:
            print(f"Killing driver (job {driver_job_id})")
            kill_job(driver_job_id)
        if isinstance(ex, KeyboardInterrupt):
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit( main() )
