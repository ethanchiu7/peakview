#!/bin/bash
# 2021-06-18
# Author by Ethan
# --------------------
CURDIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_DIR=$(echo $CURDIR | xargs dirname | xargs dirname)
#REPOSITORY_DIR=$(echo $PROJECT_PATH | xargs dirname)
echo "source --> ${PROJECT_DIR}/deeplearning/bin/util.sh"
source ${PROJECT_DIR}/deeplearning/bin/util.sh

#if [ $# -ne 2 ]; then
#        FATAL "Usage : $0 RUN_MODE MODELING"
#        exit 1
#fi
PYTHON="/yourselfpath/xx/bin/python"


main() {
  JOB_SCRIPT="${PROJECT_DIR}/deeplearning/consume_files.py"
  LOG_DIR=${PROJECT_DIR}/deeplearning/logs
  ENSURE_DIR ${LOG_DIR}
  LOG_PATH="${LOG_DIR}/consume_files.log"

  INFO "JOB BEGIN ..."
  INFO "${PYTHON} ${JOB_SCRIPT} > ${LOG_PATH} 2>&1"

  ${PYTHON} ${JOB_SCRIPT} \
    --spawning=proto_to_tfrecord    \
    --process_num=16                        \
    --is_file_patterns=True                 \
    --input_files=/xx/xx/20210321/2021*/*.proto  \
    --output_base=${PROJECT_DIR}/tfrecord/xx   \
    --inherit_dir_num=1                     \
    > ${LOG_PATH} 2>&1

INFO "JOB END ..."

}

main

