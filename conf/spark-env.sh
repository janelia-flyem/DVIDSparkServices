#!/usr/bin/env bash

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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
# - MESOS_NATIVE_JAVA_LIBRARY, to point to your libmesos.so if you use Mesos

# Options read in YARN client mode
# - HADOOP_CONF_DIR, to point Spark towards Hadoop configuration files
# - SPARK_EXECUTOR_INSTANCES, Number of executors to start (Default: 2)
# - SPARK_EXECUTOR_CORES, Number of cores for the executors (Default: 1).
# - SPARK_EXECUTOR_MEMORY, Memory per Executor (e.g. 1000M, 2G) (Default: 1G)
# - SPARK_DRIVER_MEMORY, Memory for Driver (e.g. 1000M, 2G) (Default: 1G)
# - SPARK_YARN_APP_NAME, The name of your application (Default: Spark)
# - SPARK_YARN_QUEUE, The hadoop queue to use for allocation requests (Default: ‘default’)
# - SPARK_YARN_DIST_FILES, Comma separated list of files to be distributed with the job.
# - SPARK_YARN_DIST_ARCHIVES, Comma separated list of archives to be distributed with the job.

# Options for the daemons used in the standalone deploy mode
# - SPARK_MASTER_IP, to bind the master to a different IP address or hostname
# - SPARK_MASTER_PORT / SPARK_MASTER_WEBUI_PORT, to use non-default ports for the master
# - SPARK_MASTER_OPTS, to set config properties only for the master (e.g. "-Dx=y")
# - SPARK_WORKER_CORES, to set the number of cores to use on this machine
# - SPARK_WORKER_MEMORY, to set how much total memory workers have to give executors (e.g. 1000m, 2g)
# - SPARK_WORKER_PORT / SPARK_WORKER_WEBUI_PORT, to use non-default ports for the worker
# - SPARK_WORKER_INSTANCES, to set the number of worker processes per node
# - SPARK_WORKER_DIR, to set the working directory of worker processes
# - SPARK_WORKER_OPTS, to set config properties only for the worker (e.g. "-Dx=y")
# - SPARK_DAEMON_MEMORY, to allocate to the master, worker and history server themselves (default: 1g).
# - SPARK_HISTORY_OPTS, to set config properties only for the history server (e.g. "-Dx=y")
# - SPARK_SHUFFLE_OPTS, to set config properties only for the external shuffle service (e.g. "-Dx=y")
# - SPARK_DAEMON_JAVA_OPTS, to set config properties for all daemons (e.g. "-Dx=y")
# - SPARK_PUBLIC_DNS, to set the public dns name of the master or workers

# Generic options for the daemons used in the standalone deploy mode
# - SPARK_CONF_DIR      Alternate conf dir. (Default: ${SPARK_HOME}/conf)
# - SPARK_LOG_DIR       Where log files are stored.  (Default: ${SPARK_HOME}/logs)
# - SPARK_PID_DIR       Where the pid file is stored. (Default: /tmp)
# - SPARK_IDENT_STRING  A string representing this instance of spark. (Default: $USER)
# - SPARK_NICENESS      The scheduling priority for daemons. (Default: 0)


ulimit -n 65535
export SCALA_HOME=/misc/local/scala-2.11.8

export SPARK_WORKER_DIR=/scratch/$USER/work
export JAVA_HOME=/misc/local/jdk1.8.0_102
export SPARK_LOG_DIR=~/.spark/logs/$(date +%H-%F)/
export SPARK_EXECUTOR_MEMORY=90g
export SPARK_DRIVER_MEMORY=60g
export SPARK_WORKER_MEMORY=90g
export SPARK_WORKER_OPTS=-Dspark.worker.cleanup.enabled=true

###################################
#set disk for shuffle and spilling

#wide GPFS
#HDLIST=""
#for i in {0..15}
#    do
#        HDLIST=/gpfs1/spark_local/`hostname -s`/$i,$HDLIST 
#    done
#export SPARK_LOCAL_DIRS=$HDLIST

#Narrow GPFS
#export SPARK_LOCAL_DIRS=/gpfs1/spark_local/`hostname -s`

#dm11 - wittenbachj's homedir for testing
#export SPARK_LOCAL_DIRS=/groups/freeman/home/wittenbachj/sparkscratch

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

#export PYSPARK_PYTHON=/groups/scheffer/home/plazas/miniconda2/envs/cluster/bin/python
#export PYSPARK_PYTHON=/usr/local/python-2.7.11/bin/python
export PYSPARK_PYTHON=/groups/flyem/proj/cluster/miniconda/envs/flyem/bin/python
export SPARK_SLAVES=/scratch/spark/tmp/slaves
export SPARK_SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=30"
export SPARK_PUBLIC_DNS=$HOSTNAME

## pull in users environment variables on the workers to that PYTHONPATH will transfer (change by: wittenbachj)
if [[ -z "${DO_NOT_SOURCE_PROFILE}" ]]; then
    source $HOME/.bash_profile
fi

#explict pathing to hadoop install for SL7
export SPARK_DIST_CLASSPATH=$(/misc/local/hadoop-2.6.4/bin/hadoop classpath)

# Change requested by Eric Trautman to enable newer version of classpath for FlyTEM jobs
if [[ -n "${SPARK_DIST_PRE_CLASSPATH}" ]]; then
        export SPARK_DIST_CLASSPATH="${SPARK_DIST_PRE_CLASSPATH}:${SPARK_DIST_CLASSPATH}"
fi

#updated to remove memory flags by Ken Carlile 5/11/16
#updated for prod/FlyTEM by Ken Carlile 9/7/16
