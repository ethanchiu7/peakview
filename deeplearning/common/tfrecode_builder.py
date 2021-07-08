# -*- coding: utf-8 -*-
"""
    Author  ：   Ethan Chiu
    Time    ：   2021/6/21 下午2:34
    Site    :   
    Suggestion  ：
    Description :
    File    :   tfrecode_builder.py
    Software    :   PyCharm
"""
import collections
import tensorflow as tf


def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f


def create_float_feature(values):
    f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return f


def create_bytes_feature(values):
    values = [v.encode() if isinstance(v, str) else v for v in values]
    f = tf.train.Feature(bytes_list=tf.train.BytesList(value=list(values)))
    return f


class TFFeatures(object):
    """
    tf_feature = TFFeatures(qidFeat=str(order_id),
                                 label=int(1),
                                 seq_a=[int(i) for i in seq_a],
                                 seq_b=[float(i) for i in seq_b],
                                 )
    with tf.io.TFRecordWriter(output_file) as writer:
        tf_example = tf_feature.to_TFExample()
        writer.write(tf_example.SerializeToString())

    """

    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            self.__dict__[key] = value

    def to_TFExample(self):
        features = collections.OrderedDict()
        for key, value in self.__dict__.items():
            if not isinstance(value, (list, tuple)):
                value = [value]
            if isinstance(value[0], str):
                features[key] = create_bytes_feature(value)
            elif isinstance(value[0], int):
                features[key] = create_int_feature(value)
            elif isinstance(value[0], float):
                features[key] = create_float_feature(value)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        return tf_example


