#!/bin/bash
# author by Ethan
# usage:
#   1. set this file on your server
#       e.g. /nfs/project/ethan/env/tensorboard_daemon.sh
#   2. add this command into your remote server ~/.bashrc:
#       alias tensorboard-luban="bash /nfs/project/ethan/env/tensorboard_daemon.sh"
#   3. add this into your pc ~/.bashrc
#       alias tensorboard-luban="sshpass -p ${LUBAN_PW} ssh -L 16006:127.0.0.1:8000 luban@xx -p 8022"
#
#
#
CURDIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_PATH=$(echo $CURDIR | xargs dirname | xargs dirname)
#REPOSITORY_DIR=$(echo $PROJECT_PATH | xargs dirname)
echo "source --> ${PROJECT_PATH}/deeplearning/bin/util.sh"
source ${PROJECT_PATH}/deeplearning/bin/util.sh

if [ $# -ne 1 ]; then
  FATAL "1 param expected !"
  FATAL "sh bin/tensorboard-daemon.sh MODEL_DIR "
  EXIT
fi

# step 1 run tensorflow
# step 2 映射鲁班的8000~8100端口到本机的16006
# ssh -L 16006:127.0.0.1:8000 luban@ip -p 8022

# run tensorboard

MODEL_DIR=${1}
log_dir=${PROJECT_PATH}/log
ENSURE_DIR $log_dir
LOG_PATH=$log_dir/tensorboad-daemon.log

INFO "============== TensorBloard Daemon Begin >>>>>>>>>>>>>>>>>>>"
INFO ">>> MODEL_DIR : ${MODEL_DIR}"
INFO ">>> LOG_PATH : ${LOG_PATH}"

KILL_APP tensorboard &&
# ssh -L 16006:127.0.0.1:6006 luban@ip -p 8022
/nfs/project/ethan/anaconda3/bin/tensorboard --logdir="${MODEL_DIR}" --port=6006 --host 127.0.0.1 > "${LOG_PATH}" 2>&1 &
# tensorboard --logdir="${MODEL_DIR}" --port=8000 --host 127.0.0.1 > "${LOG_PATH}" 2>&1 &

INFO "new on your local pc execute: tensorboard-luban"
