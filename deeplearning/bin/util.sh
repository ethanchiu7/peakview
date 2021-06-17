#!/bin/bash
# author by Ethan

#CURDIR=$(cd "$(dirname "$0")"; pwd)
#PROJECT_PATH=$(echo $CURDIR | xargs dirname | xargs dirname)
#REPOSITORY_DIR=$(echo $PROJECT_PATH | xargs dirname)
#JARVIS_DIR=${REPOSITORY_DIR}/jarvis
#echo "source --> ${JARVIS_DIR}/bash_lib/util.sh"
#source ${JARVIS_DIR}/bash_lib/util.sh
#
#if [ $# -ne 2 ]; then
#        FATAL "Usage : $0 RUN_MODE MODELING"
#        exit 1
#fi

function STATEMENT() {
  USAGE='
=============== Welcome for using this util file ! =================
                                               <  Author by Ethan  >
  If your main shell want the project path :                       |

How to source this file ?

#!/bin/bash
# author by Ethan
CURDIR=$(cd "$(dirname "$0")"; pwd)
project_dir=$(echo $CURDIR | xargs dirname)
repository_dir=$(echo $project_dir | xargs dirname)
jarvis_dir=${repository_dir}/jarvis
echo "source --> ${jarvis_dir}/bash_lib/util.sh"
source ${jarvis_dir}/bash_lib/util.sh

>>>>>>>>>>>>>>>>>>>>>>>>>> Gooood Luck !!! >>>>>>>>>>>>>>>>>>>>>>>>>
'
printf "%s\\n" "$USAGE"
}

function ENSURE_ARGV() {
    if [ $# -eq 0 ];then
       echo "Usage: $0 params"
       exit 1
    fi
}

function TIME() {
    # time=$(TIME)
    time=$(date +"%Y%m%d_%H%M%S")
    echo $time
}

function TIME_STEMP() {
    # time=$(TIME)
    time=$(date +"%s")
    echo $time
}

function INFO()
{
    time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[INFO] $time FUNCNAME:[${FUNCNAME[*]:1:7}] $1"
}

function NOTICE()
{
    time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[NOTICE] $time FUNCNAME:[${FUNCNAME[*]:1:7}] $1"
}

function WARN()
{
    time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[WARN] $time FUNCNAME:[${FUNCNAME[*]:1:7}] $1"
}

function FATAL()
{
    time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[FATAL] $time FUNCNAME:[${FUNCNAME[*]:1:7}] $1"
}

function SUCCESS()
{
    time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[SUCCESS] $time FUNCNAME:[${FUNCNAME[*]:1:7}] $1"
}

function FAILED()
{
    time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[FAILED] $time FUNCNAME:[${FUNCNAME[*]:1:7}] $1"
}

function EXIT()
{
    time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[EXIT] $time FUNCNAME:[${FUNCNAME[*]:1:7}] $1"
    exit 1
}


function STRING_EMPTY() {
    # empty return 1 else return 0
    if [ $# -eq 0 ];then
       echo "Usage: $0 string"
       exit 1
    fi
    msg=$2
    if [ "ETHAN" = "ETHAN${1}" ]; then
        NOTICE "${msg} that is EMPTY"
        return 1
    fi
    return 0
}

function ENSURE_DIR() {
    dir=$1
    if [ ! -d "${dir}" ]; then
        mkdir -p "${dir}"
    fi
}

function ENSURE_DIRS() {
    if [ $# -eq 0 ];then
       echo "Usage: $0 params"
       exit 1
    fi
    for arg in "$@"
    do
        ENSURE_DIR "${arg}"
    done
}

function REMOVE() {
    if [ $# -ne 1 ]; then
        FATAL "Usage : $0 file_path"
        exit 1
    fi
    if [ ! -f "$1" ]; then
        return 0
    fi
    rm -rf "$1"
    CHECK_STATUS $? "REMOVE ${1}"
}

function ERASE_FILE() {
    if [ $# -ne 1 ]; then
        FATAL "Usage : $0 file_path"
        exit 1
    fi
    REMOVE "$1"
    touch "$1"
}

function FILE_EXIST() {
    if [ $# -ne 1 ]; then
        FATAL "Usage : $0 file_path"
        exit 1
    fi
    dir=$1
    if [[ ! -f ${dir} ]]; then
        FATAL "File doesn't exist : ${dir}"
        exit 1
    fi
}

function ERASE_DIR() {
    dir=$1
    if [ ! -d ${dir} ]; then
        mkdir -p ${dir}
    else
        rm -rf ${dir}
        mkdir -p ${dir}
    fi
}

function BACKUP() {
    FILE_PATH=$1
    NEW_FILE_DIR=$2
    RESULT_TS=$(date "+%s")
    file_name=$(echo ${FILE_PATH} | awk -F '/' '{print $NF}')
    NOTICE "cp ${FILE_PATH} ${NEW_FILE_DIR}/${file_name}.${RESULT_TS}"
    cp ${FILE_PATH} ${NEW_FILE_DIR}/${file_name}.${RESULT_TS}
}

function CLEANDIR() {
    DIR=$1
    KEYWORD=$2
    if [ $# -eq 2 ]; then
        EXPIRE_DAYS=1
    elif [ $# -eq 3 ]; then
        EXPIRE_DAYS=$3
    fi
    # clean data
    # find ${DIR} -name "*${KEYWORD}*" -mtime ${EXPIRE_DAYS} -delete -or -name "*txt" -mtime ${EXPIRE_DAYS} -delete
    # clean log
    # find ${PROJECT_PATH}/log -name "*log" -mtime ${EXPIRE_DAYS} -delete -or -name "*.txt" -mtime ${EXPIRE_DAYS} -delete
    find ${DIR} -name "*${KEYWORD}*" -mtime ${EXPIRE_DAYS} -delete
}

function CHECK_STATUS()
{
    # verify exit status of the last executed command
    if [ $# -lt 1 ]; then
        FATAL "The exit status of the last executed command should be given !"
        FATAL "Exit current process ..."
        exit 1
    fi
    if [ $# -eq 1 ]; then
        status=$1
        if [ "${status}" -ne 0 ]; then
            FATAL "Failed last executed command , EXIT not 0 !"
            EXIT "Exit current process ..."
        fi
        SUCCESS "last executed commend success !"
        return 0
    elif [ $# -eq 2 ]; then
        status=$1
        job_name=$2
        if [ "${status}" -ne 0 ]; then
            FATAL "Failed : ${job_name}  , EXIT not 0 !"
            EXIT "Exit current process ..."
        fi
        SUCCESS "${job_name} success !"
        return 0
    fi
    EXIT "CHECK_STATUS : param ERROR !"
}

function CHECK_APP() {
  if [ $# -eq 1 ]; then
    app_name=${1}
    pid=$(pgrep "${app_name}")
  elif [ $# -eq 2 ]; then
    app_name=${1}
    app_param=${2}
    pid=$(ps -aux | grep ${app_name} | grep ${app_param} | awk '{print $2}')
  fi
  STRING_EMPTY "${pid}"
  if [ $? -eq 0 ]; then
     INFO "${app_name} is running"
     return 0
  fi
  INFO "${app_name} is not running"
  return 1

}

function KILL_APP() {
  app_name=${1}
  pid=$(pgrep "${app_name}")
  if [ "ETHAN" = "ETHAN${pid}" ]; then
      NOTICE "${app_name} is not running !"
      return 0
  fi
  kill -9 "${pid}" && INFO "killed ${app_name}"
  return 0
}

# ================ HDFS util ===============

function HCAT_TOP_N() {
    # Usage : hdfs_file_path=$(GET_TOP_N abc/abc 2019 1)
    if [ ! $# -eq 4 ]; then
        FATAL "Usage: HCAT_TOP_N options"
        exit 1
    fi
    parent_directory=$1
    key_word=$2
    top_n=$3
    local_path=$4

    result_paths=$(hdfs dfs -ls ${parent_directory} | awk '{print $8}' | grep "${key_word}" | head -n ${top_n})
    STRING_EMPTY ${result_paths} "${parent_directory}/*${key_word}*"
    CHECK_STATUS $?

    ERASE_FILE ${local_path}
    for i in ${result_paths}
    do
        NOTICE "save ${i}/part*  --> ${local_path}"
        hdfs dfs -cat ${i}/part* >> ${local_path}
    done
}

function HPUT() {
    if [ ! $# -eq 2 ]; then
        FATAL "Usage: HPUT local_file hdfs_path"
        exit 1
    fi
    if [ ! -f "$1" ]; then
        FATAL "local_file [$1] not exist"
        exit 1
    fi
    hdfs dfs -test -d "$2"
    if [ ! $? -eq 0 ]; then
        hdfs dfs -mkdir -p "$2"
    fi
    hdfs dfs -put -f "$1" "$2"
}

function HGET() {
    if [ ! $# -eq 2 ]; then
        FATAL "Usage: HGET hdfs_file local_file"
        exit 1
    fi

    hdfs dfs -test -f $1
    if [ ! $? -eq 0 ]; then
        FATAL "hdfs_file [$1] not exist"
        exit 1
    fi

    if [ -f $2 ]; then
        rm -f $2
    fi

    local_path=$(dirname $2)
    if [ ! -d ${local_path} ]; then
        mkdir -p ${local_path}
    fi

    hdfs dfs -get $1 $2
}

function HRM() {
    if [ ! $# -eq 1 ]; then
        FATAL "Usage: HRM hdfs_file_path"
        exit 1
    fi
    hdfs dfs -test -e $1
    if [ ! $? -eq 0 ]; then
        NOTICE "HDFS_PATH [$1] not exist"
        return 0
    fi
    hdfs dfs -rm -r -f $1
}

function ARCHIVE_FILES_TO_HDFS() {
    if [ $# -le 2 ]; then
        FATAL "Usage: ARCHIVE_FILES_TO_HDFS files_arr package_name hdfs_path"
        exit 1
    fi

    name=$1[@]
    job_name=$1
    hdfs_path=$2

    files_arr=("$@")
    ((last1_id=${#files_arr[@]} - 1))
    ((last2_id=${#files_arr[@]} - 2))
    job_name=${files_arr[last2_id]}
    hdfs_path=${files_arr[last1_id]}
    unset files_arr[last1_id]
    unset files_arr[last2_id]

    ERASE_DIR ${job_name}
    cd ${job_name}

    for i in "${files_arr[@]}"
    do
       echo "cp ${i} -> $(pwd;)"
       cp ${i} ./
    done
    tar -zcvf ./${job_name}.tar ./*

    NOTICE "$(pwd;)/${job_name}.tar --> ${hdfs_path}"
    HPUT ./${job_name}.tar ${hdfs_path}

    NOTICE "cd .. && rm -rf ./${job_name}"
    cd .. && rm -rf ./${job_name}

}

#STATEMENT%