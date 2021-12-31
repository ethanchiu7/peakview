# -*- coding: utf-8 -*-
"""
    Author  ：   Ethan
    Time    ：   2021/01/17 4:45 下午
    Site    :   
    Suggestion  ：
    Description : pyspark 处理数据并直接向HDFS写入TFRecord
"""
# import base64
import json
import time
from datetime import datetime
import numpy as np
import os
import sys
import copy
# import jieba
from pyspark import SparkContext, SparkConf
from pyspark.sql import HiveContext
from pyspark.sql.types import StructType, StructField, ArrayType, IntegerType, LongType, FloatType, DoubleType, StringType
# from nlp_util import rm_special_characters, replace_sentence_by_words, ch_int_to_str
from common import util
logger = util.Logger(__file__)


def parse_input_data(line: str):
    line = line.strip()
    l = line.split("\t")
    names = ["a", "b", "c", "d"]
    assert len(l) == len(names)

    d = dict()
    for k, v in zip(names, l):
        d[k] = v
        
    if not d["a"]:
        return None

    return (d["a"], d)


def read_kv_file_as_rdd(sc, hdfs_file_path):
    rdd = sc.textFile(hdfs_file_path)
    return rdd


def parse_rdd_to_broadcast(rdd):
    def parse(raw):
        raw = raw.strip()
        if not raw:
            return None
        l = raw.split(" ")
        if len(l) != 2:
            return None
        return (l[0], l[1])
    rdd = rdd.map(parse).filter(lambda i: i is not None)
    kv_dict = dict(rdd.collect())
    bc = sc.broadcast(kv_dict)
    return bc


def read_kv_file_as_broadcast(sc, hdfs_file_path):
    rdd = sc.textFile(hdfs_file_path)
    return parse_rdd_to_broadcast(rdd)


def save_rdd_to_tfrecord(hc, rdd, token_idx_bc, tfrecord_output_file):
    def parse_example(r, token_idx_bc):
        PAD = 0
        UNKNOWN = 1
        START = 3
        END = 4
        min_seq_len = 16
        max_seq_len = 1024
        per_file_limit = 50000
        token_list = [int(i) for i in r["tokens"].strip().split(",")]

        token_idx_l = [int(token_idx_bc.value[str(i)]) + 10 if str(i) in token_idx_bc.value else UNKNOWN for i in token_list]
        
        inputs = [START] + token_idx_l
        targets = token_idx_l + [END]

        result = []
        result.append(len(inputs))
        result.append(int(r["order_id"]))
        result.append(int(r["ts"]))
        result.append(inputs)
        result.append(targets)

        return result

    # 定义数据结构
    fields = [
        StructField("token_len", LongType()),
        StructField("order_id", LongType()),
        StructField("ts", LongType()),
        StructField("inputs", ArrayType(LongType(), True)),
        StructField("targets", ArrayType(LongType(), True)),
    ]
    schema = StructType(fields)

    # 过滤异常数据
    rdd = rdd.filter(lambda r: len([int(i) for i in r["tokens"].strip().split(",")]) > 3)

    sample_count = int(rdd.count())
    partitions = sample_count // 50000
    print("sample_count: {}, partitions: {}".format(sample_count, partitions))

    result_rdd = rdd.map(lambda i: parse_example(i[1], token_idx_bc))
    df = hc.createDataFrame(result_rdd, schema)

    df = df.orderBy(df["token_len"], -df["order_id"])

    df.repartition(partitions).write\
        .format("tfrecords")\
        .option("recordType", "Example")\
        .save(tfrecord_output_file, mode="overwrite")
    logger.info("\nTFRecord已保存至: {}".format(tfrecord_output_file))
    return None


if __name__ == '__main__':
    # param
    date_str = "20201101"
    overwrite = False
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
        print(">>> using passed in date_str: {}".format(date_str))
    else:
        print(">>> using default date_str: {}".format(date_str))
    year = date_str[:4]
    month = date_str[4: 6]
    day = date_str[-2:]
    logger.info(
        "len(sys.argv): {}, date_str: {}, year: {}, month: {}, day: {}".format(len(sys.argv), date_str, year, month,
                                                                               day))

    output_dir = "/your/hdfs/tfrecord_output/{}".format(date_str)

    util.HdfsUtil.check_success(output_dir, overwrite=overwrite)

    APP_NAME = '{}_{}'.format(__file__, str(time.time()))
    logger.info("\nJOB BEGIN !".format(APP_NAME))
    logger.info("\nAPP_NAME: {}".format(APP_NAME))
    # SparkSession available as 'spark'.
    conf = SparkConf().set("spark.hadoop.validateOutputSpecs", "False").setAppName(APP_NAME)
    # sc = spark.sparkContext
    sc = SparkContext(conf=conf)
    hc = HiveContext(sc)

    input_files = "/your/hdfs/input_data/{}/{}/{}/part-*.txt".format(year, month, day)
    input_rdd = sc.textFile(input_files)
    print("input_files 原始数据总量: ", input_rdd.count())

    input_rdd = input_rdd.map(lambda i: parse_input_data(i)).filter(lambda i: i is not None)
    print("input_files 解析后数据总量: ", input_rdd.count())

    # broadcast
    token_idx_dict = "/your/hdfs/token_idx_dict/part-00000"
    token_idx_bc = read_kv_file_as_broadcast(sc, token_idx_dict)
    print("token_idx_bc len: {}".format(len(token_idx_bc.value)))

    save_rdd_to_tfrecord(hc, input_files, token_idx_bc, output_dir)





