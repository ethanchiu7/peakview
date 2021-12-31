#!/bin/bash
source /etc/profile
source /etc/bashrc
source /home/xiaoju/.bashrc

PROJECT_PATH=$(readlink -f $0 | xargs dirname | xargs dirname | xargs dirname)
SCIPT_DIR=$(readlink -f $0 | xargs dirname | xargs dirname)
cd ${SCIPT_DIR}
QUEUE=your.queue.name

if [ $# -eq 0 ];then
    echo "Usage: $0 run_date"
    exit 1
fi
script_params=$@
job_dir="job_dir_example"
job_file="${PROJECT_PATH}/bigdata/${job_dir}/pyspark_example.py"

cd ${PROJECT_PATH}/bigdata/${job_dir}
rm -f ${PROJECT_PATH}/bigdata/${job_dir}/${job_dir}.zip
zip -r ${job_dir}.zip ./*.py

PYSPARK_DRIVER_PYTHON="/your/local/python3.6.9/bin/python3"
remote_python="/your/hdfs/python/python3.6.9.tgz#python3.6.9"
PYSPARK_PYTHON="./python3.6.9/bin/python3"

export PYSPARK_DRIVER_PYTHON=${PYSPARK_DRIVER_PYTHON} \
    && export PYTHONPATH="/your/local/python3.6.9/lib/python3.6/site-packages:${PROJECT_PATH}/bigdata" \
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
