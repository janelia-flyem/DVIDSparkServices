#!/bin/bash

##
## Script to launch a local DVID instance, intended for spark worker nodes.
##
## Usage: launch-worker-dvid.sh <dvid-config.toml>
##
## (The dvid toml must reside in this directory. Don't use an absolute path.)
##
## Notes:
##
## 1. The given .toml can be treated as a template, which will be
##    copied and modified to give the worker a unique log file path.
##    This way, worker DVID logs can be saved to the NFS and easily
##    inspected without logging into the individual worker nodes to
##    find the log file.
##
## 2. Edit the script below to customize the DVID binary location.
##

THIS_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


FLYEM_ENV=/groups/flyem/proj/cluster/miniforge/envs/flyem
BUILDEM_DIR=${FLYEM_ENV}/buildem-dir
DVID_CONFIG_DIR=${THIS_SCRIPT_DIR}
#DVID_CONFIG_DIR=/groups/flyem/proj/cluster/miniconda/envs/flyem/dvid-configs
#GOPATH=${FLYEM_ENV}/gopath

if [ $# != 1 ]; then
    1>&2 echo "Usage: $0 <dvid-config.toml>"
    1>&2 echo "(The DVID config TOML must reside in ${DVID_CONFIG_DIR}.)"
    exit 1
fi

DVID_CONFIG_NAME=$1
DVID_TOML=${DVID_CONFIG_DIR}/${DVID_CONFIG_NAME}

if [ ! -e ${DVID_TOML} ]; then
    1>&2 echo "DVID TOML not found: ${DVID_TOML}"
    exit 2
fi

# Kill any accidentally running DVIDs from a previous run.
OLD_DVID_TOP_INFO=$(top -b -n1 | grep $(whoami) | grep dvid)
if [ "${OLD_DVID_TOP_INFO}" != "" ] ; then
    OLD_DVID_PID=$(echo "${OLD_DVID_TOP_INFO}" | python -c 'import sys; print(sys.stdin.read().split()[0])')
    kill -9 ${OLD_DVID_PID}
fi

##
## CUSTOMIZE HERE:
## Choose whether or not to store logs on NFS vs. worker /tmp/
##
WRITE_WORKER_LOGS_TO_NFS=true

if $WRITE_WORKER_LOGS_TO_NFS; then
    # Instead of using the same TOML for every worker (and logging to /tmp)
    # Copy the toml for each worker and overwrite the logfile location with something unique
    WORKER_TOML=${DVID_CONFIG_DIR}/worker-dvid-tomls/$(uname -n)_${DVID_CONFIG_NAME}
    WORKER_LOG=${DVID_CONFIG_DIR}/worker-dvid-logs/$(uname -n)_${DVID_CONFIG_NAME}.log

    mkdir -p $(dirname ${WORKER_TOML})
    mkdir -p $(dirname ${WORKER_LOG})

    sed 's|logfile =.*|logfile = "'${WORKER_LOG}'"|g' < ${DVID_TOML} > ${WORKER_TOML}
    DVID_TOML=${WORKER_TOML}
    
    echo "[$(date)] Launching Worker DVID..." >> ${WORKER_LOG}
fi

##
## CUSTOMIZE HERE:
## Choose which dvid build to use -- from conda or custom-built in Buildem
##
DVID_PREFIX=${FLYEM_ENV}   # from conda
#DVID_PREFIX=${BUILDEM_DIR} # from buildem

export PATH=${DVID_PREFIX}/bin:${PATH}
export LD_LIBRARY_PATH=${DVID_PREFIX}/lib:${LD_LIBRARY_PATH}

echo "Launching DVID:"
echo "${DVID_PREFIX}/bin/dvid -verbose serve \"${DVID_TOML}\""
${DVID_PREFIX}/bin/dvid -verbose serve "${DVID_TOML}" &

DVID_PID=$!

# If we're terminated with SIGTERM, send SIGTERM to the subprocess.
trap 'kill -TERM $DVID_PID' EXIT

wait $DVID_PID
