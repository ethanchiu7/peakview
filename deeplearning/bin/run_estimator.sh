#!/bin/bash
# 2019-08-18
# Author by Ethan
CURDIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_PATH=$(echo $CURDIR | xargs dirname | xargs dirname)
#REPOSITORY_DIR=$(echo $PROJECT_PATH | xargs dirname)
echo "source --> ${PROJECT_PATH}/deeplearning/bin/util.sh"
source ${PROJECT_PATH}/deeplearning/bin/util.sh

if [ $# -ne 2 ]; then
        FATAL "Usage : $0 RUN_MODE MODELING"
        exit 1
fi
PYTHON="/home/yourname/anaconda3/bin/python"
RUN_MODE=${1}
MODELING=${2}
TRAIN_FILE="/xx/xx/*.tfrecord,/xx/xx/*.tfrecord,"
EVAL_FILE="/xx/xx/*.tfrecord,/xx/xx/*.tfrecord,"
PREDICT_FILE="/xx/xx/*.tfrecord,/xx/xx/*.tfrecord,"


main() {
  JOB_SCRIPT="${PROJECT_PATH}/deeplearning/estimator_app.py"
  LOG_DIR=${PROJECT_PATH}/deeplearning/logs
  ENSURE_DIR ${LOG_DIR}
  LOG_PATH="${LOG_DIR}/estimator_app-${RUN_MODE}-${MODELING}.log"

  INIT_DIR="${PROJECT_PATH}/deeplearning/model_dir/${MODELING}"
  MODEL_DIR="${PROJECT_PATH}/deeplearning/model_dir/${MODELING}"
  INFO "JOB BEGIN ..."
  INFO "${PYTHON} ${JOB_SCRIPT} > ${LOG_PATH} 2>&1"
  INFO "MODEL_DIR : ${MODEL_DIR}"

  ${PYTHON} ${JOB_SCRIPT} \
    --run_mode="${RUN_MODE}"  \
    --modeling=${MODELING}    \
    --use_gpu=True      \
    --init_checkpoint="${INIT_DIR}"     \
    --model_dir="${MODEL_DIR}"      \
    --clear_model_dir=False         \
    --is_file_patterns=True         \
    --train_file=${TRAIN_FILE}    \
    --train_batch_size=64           \
    --train_epoch=4                 \
    --eval_file=${EVAL_FILE}    \
    --eval_batch_size=32            \
    --predict_file=${PREDICT_FILE}    \
    --predict_batch_size=8          \
    --num_actual_predict_examples=0   \
    > ${LOG_PATH} 2>&1

INFO "JOB END ..."

}

main

