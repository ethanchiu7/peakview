#!/bin/bash
source /etc/profile
source /etc/bashrc

:<<!
    使用方法：
        -d job_dir :
                定义python执行文件的路径，其中 job_dir/main.py 为主程序
                在这个目录可以定义其他python文件模块 在main.py中直接import
                必须 创建 main.py 作为spark主程序
                (Option) job_dir 目录下可以 存在其他 *.py 模块 便于main.py中import
                本脚本 会自动打包 job_dir 目录下全部 *.py 并分发给 Spark集群进行分布式计算
        -p script_params:
                定义传递给 job_dir/main.py 的参数 可以是多个
        例如:
                bash run_pyspark.sh -d job_dir_example -p "1 2 3 20211118"
!
while getopts ":d:p:h" optname
do
    case "${optname}" in
       "d")
        job_dir="${OPTARG}"
        ;;
       "p")
        script_params="${OPTARG}"
        ;;
       ":")
        echo "No argument value for option $OPTARG"
        exit
        ;;
       "?")
        echo "Unknown option $OPTARG"
        exit
        ;;
       *)
        echo "Unknown error while processing options"
        exit
        ;;
    esac
    #echo "option index is $OPTIND"
done
PROJECT_PATH=$(readlink -f $0 | xargs dirname | xargs dirname | xargs dirname)
job_file="${PROJECT_PATH}/bigdata/${job_dir}/main.py"
echo "job_dir: ${job_dir}"
echo "script_params: ${script_params}"
echo "job_file: ${job_file}"

test -f ${job_file}
if [ ! $? -eq 0 ]; then
    echo "job_file not exist: ${job_file}"
    exit 1
fi

SCIPT_DIR=$(readlink -f $0 | xargs dirname | xargs dirname)
cd ${SCIPT_DIR}

QUEUE=your.queue.name

cd ${PROJECT_PATH}/bigdata/${job_dir}
rm -f ${PROJECT_PATH}/bigdata/${job_dir}/${job_dir}.zip
zip -r ${job_dir}.zip ./*.py

PYSPARK_DRIVER_PYTHON="/your/local/python3.6.9/bin/python3"
remote_python="/your/hdfs/python/python3.6.9.tgz#python3.6.9"
PYSPARK_PYTHON="./python3.6.9/bin/python3"

export PYSPARK_DRIVER_PYTHON=${PYSPARK_DRIVER_PYTHON} \
    && export PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}/bigdata" \
    && export PYSPARK_PYTHON=${PYSPARK_PYTHON}   \
    && source /etc/profile  \
    && spark-submit \
        --queue ${QUEUE} --conf spark.speculation=true \
        --conf spark.yarn.dist.files=hdfs://DClusterNmg/admin/ranger/livy/hive-site.xml  \
        --conf spark.yarn.dist.archives=${remote_python}  \
        --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON="${PYSPARK_PYTHON}" \
        --conf spark.pyspark.python=${PYSPARK_PYTHON} \
        --conf spark.pyspark.driver.python=${PYSPARK_DRIVER_PYTHON} \
        --conf spark.driver.memory=10g  \
        --conf spark.executor.memory=13g  \
        --py-files=${PROJECT_PATH}/datapipline/${job_dir}/${job_dir}.zip \
        --jars ${PROJECT_PATH}/bigdata/jars/spark-tensorflow-connector_2.11-1.15.0.jar \
        ${job_file} ${script_params}

date
echo "spark job end"
