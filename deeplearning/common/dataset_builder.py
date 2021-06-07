# -*- coding: utf-8 -*-
"""
    Author  ：   Ethan Chiu
    Time    ：   2021/3/30 下午12:15
    Site    :   
    Suggestion  ：
    Description :
    File    :   model_builder.py
    Software    :   PyCharm
"""
import re
import collections
from collections.abc import Sequence, Mapping, Iterator
import tensorflow as tf
import abc
import numpy as np
import six
import json
import copy
from common import utils
from common import tf_utils


class DataSource(object):
    TFRECORD = "tfrecord"
    TEXT = "text"
    ARRAY = "array"


class DataSetBuilder(object):
    def __init__(self):
        pass

    @staticmethod
    def pattern_to_files(input_files, is_file_patterns):
        """

        :param patterns:
        :param is_file_patterns:
        :return:
        """
        input_file_paths = []
        if is_file_patterns:
            if isinstance(input_files, str):
                input_files = input_files.split(",")
            assert isinstance(input_files, (list, tuple))
            for input_pattern in input_files:
                input_file_paths.extend(tf.gfile.Glob(input_pattern))
        else:
            if isinstance(input_files, str):
                input_files = input_files.split(",")
            assert isinstance(input_files, (list, tuple))
            input_file_paths.extend(input_files)
        return input_file_paths

    @classmethod
    def get_feature_dict_input_fn(cls,
                                  input_files,
                                  batch_size=64,
                                  epoch=1,
                                  is_training=True,
                                  num_cpu_threads=4):
        pass

    @classmethod
    def get_text_input_fn(cls,
                          input_files,
                          batch_size=64,
                          epoch=1,
                          is_training=True,
                          num_cpu_threads=4):
        pass

    @classmethod
    def get_tfrecord_input_fn(cls, input_files, name_to_features, batch_size, epoch, is_training, num_cpu_threads):
        """

        :param input_files: list of file path, or list of file pattern, or string which split by ","
        :param name_to_features: how to parse a tfrecode record, tensor shape and tf.type

            BERT  :
            name_to_features = {
                "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
                "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
                "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
                "label_ids": tf.FixedLenFeature([], tf.int64),
                "is_real_example": tf.FixedLenFeature([], tf.int64),
            }
            GPT   :
            name_to_features = {
              "inputs": tf.io.VarLenFeature(tf.int64),
              "targets": tf.io.VarLenFeature(tf.int64),
            }

        :param batch_size:
        :param epoch:
        :param is_training:
        :param num_cpu_threads:
        :return: input_fn
        """

        def _decode_record(record, name_to_features, sparse_to_dense=True):
            """Decodes a record to a TensorFlow example."""
            example = tf.io.parse_single_example(record, name_to_features)

            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.cast(t, tf.int32)
                if sparse_to_dense and isinstance(t, tf.SparseTensor):
                    t = tf.sparse.to_dense(t)
                example[name] = t
            return example

        def input_fn(params):
            """The actual input function."""
            tf.logging.info("[input_fn] batch_size : {}, epoch : {}".format(batch_size, epoch))
            # data_fields = {
            #     "inputs": tf.io.VarLenFeature(tf.int64),
            #     "targets": tf.io.VarLenFeature(tf.int64)
            # }

            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            if is_training:
                d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
                d = d.repeat(count=epoch)
                d = d.shuffle(buffer_size=len(input_files))

                # `cycle_length` is the number of parallel files that get read.
                cycle_length = min(num_cpu_threads, len(input_files))

                # `sloppy` mode means that the interleaving is not exact. This adds
                # even more randomness to the training pipeline.
                d = d.apply(
                    tf.data.experimental.parallel_interleave(
                        # tf.contrib.data.parallel_interleave(
                        tf.data.TFRecordDataset,
                        sloppy=is_training,
                        cycle_length=cycle_length))
                d = d.shuffle(buffer_size=100)
            else:
                d = tf.data.TFRecordDataset(input_files)
                # Since we evaluate for a fixed number of steps we don't want to encounter
                # out-of-range exceptions.
                d = d.repeat(count=epoch)

            # We must `drop_remainder` on training because the TPU requires fixed
            # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
            # and we *don't* want to drop the remainder, otherwise we wont cover
            # every sample.
            d = d.apply(
                tf.data.experimental.map_and_batch(
                    # tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size=batch_size,
                    num_parallel_batches=num_cpu_threads,
                    drop_remainder=True))
            # d = d.map(lambda batch_data: {i: tf.sparse.to_dense(batch_data[i]) for i in batch_data})
            return d

        return input_fn