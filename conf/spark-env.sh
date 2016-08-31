#!/usr/bin/env bash

# This file is sourced when running various Spark programs.
# Copy it as spark-env.sh and edit that to configure Spark for your site.

# Options read when launching programs locally with 
# ./bin/run-example or ./bin/spark-submit
# - HADOOP_CONF_DIR, to point Spark towards Hadoop configuration files
# - SPARK_LOCAL_IP, to set the IP address Spark binds to on this node
# - SPARK_PUBLIC_DNS, to set the public dns name of the driver program
# - SPARK_CLASSPATH, default classpath entries to append

# Options read by executors and drivers running inside the cluster
# - SPARK_LOCAL_IP, to set the IP address Spark binds to on this node
# - SPARK_PUBLIC_DNS, to set the public DNS name of the driver program
# - SPARK_CLASSPATH, default classpath entries to append
# - SPARK_LOCAL_DIRS, storage directories to use on this node for shuffle and RDD data
# - MESOS_NATIVE_LIBRARY, to point to your libmesos.so if you use Mesos

# Options read in YARN client mode
# - HADOOP_CONF_DIR, to point Spark towards Hadoop configuration files
# - SPARK_EXECUTOR_INSTANCES, Number of workers to start (Default: 2)
# - SPARK_EXECUTOR_CORES, Number of cores for the workers (Default: 1).
# - SPARK_EXECUTOR_MEMORY, Memory per Worker (e.g. 1000M, 2G) (Default: 1G)
# - SPARK_DRIVER_MEMORY, Memory for Master (e.g. 1000M, 2G) (Default: 512 Mb)
# - SPARK_YARN_APP_NAME, The name of your application (Default: Spark)
# - SPARK_YARN_QUEUE, The hadoop queue to use for allocation requests (Default: ‘default’)
# - SPARK_YARN_DIST_FILES, Comma separated list of files to be distributed with the job.
# - SPARK_YARN_DIST_ARCHIVES, Comma separated list of archives to be distributed with the job.

# Options for the daemons used in the standalone deploy mode:
# - SPARK_MASTER_IP, to bind the master to a different IP address or hostname
# - SPARK_MASTER_PORT / SPARK_MASTER_WEBUI_PORT, to use non-default ports for the master
# - SPARK_MASTER_OPTS, to set config properties only for the master (e.g. "-Dx=y")
# - SPARK_WORKER_CORES, to set the number of cores to use on this machine
# - SPARK_WORKER_MEMORY, to set how much total memory workers have to give executors (e.g. 1000m, 2g)
# - SPARK_WORKER_PORT / SPARK_WORKER_WEBUI_PORT, to use non-default ports for the worker
# - SPARK_WORKER_INSTANCES, to set the number of worker processes per node
# - SPARK_WORKER_DIR, to set the working directory of worker processes
# - SPARK_WORKER_OPTS, to set config properties only for the worker (e.g. "-Dx=y")
# - SPARK_HISTORY_OPTS, to set config properties only for the history server (e.g. "-Dx=y")
# - SPARK_DAEMON_JAVA_OPTS, to set config properties for all daemons (e.g. "-Dx=y")
# - SPARK_PUBLIC_DNS, to set the public dns name of the master or workers

ulimit -n 65535
export SCALA_HOME=/usr/local/scala-2.10.3

export SPARK_WORKER_DIR=/scratch/spark/work
export JAVA_HOME=/usr/local/jdk1.7.0_67
export SPARK_LOG_DIR=~/.spark/logs/$JOB_ID/
export SPARK_EXECUTOR_MEMORY=80g
export SPARK_DRIVER_MEMORY=50g
export SPARK_WORKER_MEMORY=80g
#export SPARK_DAEMON_JAVA_OPTS=-Dspark.worker.timeout=300 -Dspark.akka.timeout=300 -Dspark.storage.blockManagerHeartBeatMs=30000 -Dspark.akka.retry.wait=30 -Dspark.akka.frameSize=10000 -Djobid=$JOB_ID 

#Local hdd
#export SPARK_LOCAL_DIRS=/scratch/spark/tmp

#tier2 localdirs
#export SPARK_LOCAL_DIRS=/tier2/sparktest

#nrs localdirs
#export SPARK_LOCAL_DIRS=/nrs/sparklocaldir

#1 local ssd
#export SPARK_LOCAL_DIRS=/scratch-ssd1/sparklocaldir

#2 local ssds
#export SPARK_LOCAL_DIRS=/scratch-ssd1/sparklocaldir,/scratch-ssd2/sparklocaldir

#3 local ssds
export SPARK_LOCAL_DIRS=/data1/sparklocaldir,/data2/sparklocaldir,/data3/sparklocaldir

#dm11 sas
#export SPARK_LOCAL_DIRS=/misc/sparksas

#dm11 ssd
#export SPARK_LOCAL_DIRS=/misc/sparkssd

#################################


#export PYSPARK_PYTHON=/usr/local/python-2.7.6/bin/python
#export PYSPARK_PYTHON=/groups/scheffer/home/plazas/development/buildem_sparkcluster/bin/python
export SPARK_SLAVES=/scratch/spark/tmp/slaves
export SPARK_SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=30"
export SPARK_PUBLIC_DNS=$HOSTNAME
