#!/bin/bash
#
# author by Ethan
CURDIR=$(cd "$(dirname "$0")"; pwd)
PROJECT_PATH=$(echo $CURDIR | xargs dirname | xargs dirname)
REPOSITORY_DIR=$(echo $PROJECT_PATH | xargs dirname)
JARVIS_DIR=${REPOSITORY_DIR}/jarvis
echo "source --> ${JARVIS_DIR}/bash_lib/util.sh"
source ${JARVIS_DIR}/bash_lib/util.sh

if [ $# -ne 2 ]; then
        FATAL "Usage : $0 RUN_MODE MODELING"
        exit 1
fi
#MODELING=medallion_02
RUN_MODE=${1}
MODELING=${2}
TRAIN_FILE="/nfs/project/ethan/deepquant/tfrecord/stock/history/daily/train/*.tfrecord,"
EVAL_FILE="/nfs/project/ethan/deepquant/tfrecord/stock/history/daily/eval/*.tfrecord,"
PREDICT_FILE="/nfs/project/ethan/deepquant/tfrecord/stock/history/daily/train/2018-12-28.tfrecord,"


main() {
  PYTHON="/home/luban/anaconda3/bin/python"
  #PYTHON="/nfs/project/ethan/anaconda3/bin/python"
  echo "PROJECT_PATH : ${PROJECT_PATH}"
  JOB_SCRIPT="${PROJECT_PATH}/deeplearning/estimator_app.py"
  LOG_DIR=${PROJECT_PATH}/deeplearning/logs
  ENSURE_DIR ${LOG_DIR}
  LOG_PATH="${LOG_DIR}/estimator_app-${RUN_MODE}-${MODELING}.log"

  INIT_DIR="${PROJECT_PATH}/deeplearning/model_dir/${MODELING}"
  MODEL_DIR="${PROJECT_PATH}/deeplearning/model_dir/${MODELING}"
  INFO "JOB BEGIN ..."
  INFO "${PYTHON} ${JOB_SCRIPT} > ${LOG_PATH} 2>&1"
  INFO "MODEL_DIR : ${MODEL_DIR}"

  ${PYTHON} ${JOB_SCRIPT}     \
    --run_mode="${RUN_MODE}"  \
    --modeling=${MODELING}    \
    --use_gpu=True            \
    --init_checkpoint="${INIT_DIR}" \
    --model_dir="${MODEL_DIR}"      \
    --clean_model_dir=True         \
    --is_file_patterns=True         \
    --train_file=${TRAIN_FILE}      \
    --train_batch_size=64           \
    --train_epoch=5                 \
    --shuffle_train_files=False     \
    --eval_file=${EVAL_FILE}        \
    --eval_batch_size=32            \
    --predict_file=${PREDICT_FILE}  \
    --predict_batch_size=8          \
    --num_actual_predict_examples=0 \
    > ${LOG_PATH} 2>&1

INFO "JOB END ..."

}

main
