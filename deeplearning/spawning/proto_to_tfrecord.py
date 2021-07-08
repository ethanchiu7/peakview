# -*- coding: utf-8 -*-
"""
    Author  ：   Ethan
    Time    ：   2021/6/18 下午4:26
    Site    :   
    Suggestion  ：
    Description :
    File    :   proto_to_tfrecord.py
    Software    :   PyCharm
"""
import os
import math
from collections import Iterable
import random
from absl import flags
import tensorflow as tf
from common import utils
from spawning import nn_order_sample_pb2
from common.tfrecode_builder import TFFeatures

logger = utils.Logger(__file__)
PROJECT_DIR = utils.DirUtils.get_parent_dir(__file__, 2)


def load_proto(file_name):
    with open(file_name, "rb") as f:
        nn_set = nn_order_sample_pb2.NNOrderSampleSet()
        nn_str = nn_set.FromString(f.read())
        return nn_str.samples


def process_time(ts):
    ts = ts / 3600.0  # hour
    floor_raw = int(ts)
    floor_w = ts - floor_raw
    ceil_w = 1 - floor_w
    floor_idx1 = floor_raw % 24  # 24hours
    ceil_idx1 = (floor_idx1 + 1) % 24
    floor_idx2 = floor_raw % (24 * 7)  # hour of weekday
    ceil_idx2 = (floor_idx2 + 1) % (24 * 7)
    return [floor_idx1, ceil_idx1, floor_idx2, ceil_idx2], [ceil_w, floor_w]


def has_nan_or_inf(nums):
    if isinstance(nums, Iterable):
        return any([math.isnan(x) or math.isinf(x) for x in list(nums)])
    else:
        return math.isnan(nums) or math.isinf(nums)


def sample_to_TFFeatures(sample):
    qid, pid, did, od_grid, od_dist, dt, cityid, others_type, pos_list, neg_list = \
        sample.qid, sample.pid, sample.did, sample.od_grid, sample.od_dist, sample.timestamp, sample.cityid, sample.others_type,\
        sample.pos_feats, sample.neg_feats
    features = []
    for pos_instance in pos_list:
        if sample.qid is None or sample.pid is None or sample.did is None or \
                len(sample.od_grid) < 1 or len(sample.od_dist) < 1:
            continue

        if len(pos_instance.link_list) > 1000 or has_nan_or_inf(pos_instance.numeric_val) or \
            len(pos_instance.numeric_val) < 1 or \
            len(pos_instance.link_list) < 1 or \
            len(pos_instance.xgb_preds) < 1 or \
                False:
            continue
        if len(neg_list) > 4:
            neg_list = [neg_list[i] for i in random.sample(range(len(neg_list)), 4)]
        for neg_instance in neg_list[:4]:
            if False or \
                    len(neg_instance.link_list) > 1000 or has_nan_or_inf(neg_instance.numeric_val) or \
                    len(neg_instance.numeric_val) < 1 or \
                    len(neg_instance.link_list) < 1 or \
                    len(neg_instance.xgb_preds) < 1:
                continue

            timeIdxFeat, timeWFeat = process_time(sample.timestamp)

            feature = TFFeatures(qidFeat=str(sample.qid),
                                 label=int(1),
                                 pidFeat=int(sample.pid),
                                 didFeat=int(sample.did),
                                 od_gridFeat=[int(i) for i in sample.od_grid],
                                 od_distFeat=[float(i) for i in sample.od_dist],
                                 # cityidFeat=int(cityid),
                                 timeIdxFeat=[int(i) for i in timeIdxFeat],
                                 timeWFeat=[float(i) for i in timeWFeat],

                                 statValFeat_pos=[float(i) for i in pos_instance.numeric_val],
                                 linkidFeat_pos=[int(i) for i in pos_instance.link_list],
                                 xgbFeat_pos=[int(i) for i in pos_instance.xgb_preds],

                                 statValFeat_neg=[float(i) for i in neg_instance.numeric_val],
                                 linkidFeat_neg=[int(i) for i in neg_instance.link_list],
                                 xgbFeat_neg=[int(i) for i in neg_instance.xgb_preds],
                                 )

            features.append(feature)

    return features


def spawn(lock, index, input_file, output_dir):
    """

    :param lock: 进程锁
    :param index: 文件序号
    :param input_file: 输入
    :param output_file: 输出
    :return: None

        进程锁 控制对资源的 占用和释放
        lock.acquire()
        lock.release()
    """
    input_base_name = ''.join(os.path.basename(input_file).split('.')[:-1])
    output_file = os.path.join(output_dir, "{}.tfrecord".format(input_base_name))
    os.system("rm -f {}".format(output_file))

    samples = load_proto(input_file)

    print(""""... 开始处理第 {} 个文件
    samples len: {}, input file: {}
    output file: {}
    
    """.format(index, len(samples), input_file, output_file))

    if len(samples) < 1:
        logger.warning("samples < 1, file_path: {}".format(input_file))
        return

    # writer = tf.data.experimental.TFRecordWriter(output_file)
    write_count = 0
    with tf.io.TFRecordWriter(output_file) as writer:
        for sample in samples:
            tf_features = sample_to_TFFeatures(sample)
            if len(tf_features) < 1:
                continue
            # print("features len: {}".format(len(tf_features)))
            for tf_feature in tf_features:
                tf_example = tf_feature.to_TFExample()
                writer.write(tf_example.SerializeToString())
                write_count += 1
    print("write_count {}, output_file: {}".format(write_count, output_file))

    return


