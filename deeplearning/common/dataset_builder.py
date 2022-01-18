# -*- coding: utf-8 -*-
"""
    Author  ：   Ethan Chiu
    Time    ：   2021/3/30 下午12:15
    Site    :   
    Suggestion  ：
    Description :
    File    :   modeling_base.py
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
from common import tf_embed


class DataSource(object):
    TFRECORD = "data"
    TEXT = "text"
    ARRAY = "array"


class DataSetBuilder(object):
    def __init__(self):
        pass

    @staticmethod
    def pattern_to_files(input_files, is_file_patterns, is_hdfs_file=False):
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
                if is_hdfs_file and "ClusterNMG" not in input_pattern:
                    input_pattern = "viewfs://ClusterNMG" + input_pattern
                input_file_paths.extend(tf.io.gfile.glob(input_pattern))
        else:
            if isinstance(input_files, str):
                input_files = input_files.split(",")
            assert isinstance(input_files, (list, tuple))
            for f in input_files:
                if is_hdfs_file and "ClusterNMG" not in f:
                    f = "viewfs://ClusterNMG" + f
                if tf.io.gfile.exists(f):
                    input_file_paths.append(f)
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
    def get_tfrecord_input_fn(cls, input_files, name_to_features, batch_size, epoch, shuffle_input_files, num_cpu_threads):
        if not num_cpu_threads:
            num_cpu_threads = 4
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
        :param shuffle_input_files:
        :param num_cpu_threads:
        :return: input_fn
        """

        def _decode_record(record, name_to_features):
            """Decodes a record to a TensorFlow example."""
            example = tf.io.parse_single_example(record, name_to_features)

            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                # if t.dtype == tf.int64:
                #     t = tf.cast(t, tf.int32)
                # if isinstance(t, tf.SparseTensor):
                #     t = tf.sparse.to_dense(t)
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

            if shuffle_input_files:
                d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
                d = d.repeat(count=epoch)
                d = d.shuffle(buffer_size=len(input_files))

                # `cycle_length` is the number of parallel files that get read.
                cycle_length = min(max(num_cpu_threads, 1), len(input_files))

                # `sloppy` mode means that the interleaving is not exact. This adds
                # even more randomness to the training pipeline.
                d = d.apply(
                    tf.data.experimental.parallel_interleave(
                        # tf.contrib.data.parallel_interleave(
                        tf.data.TFRecordDataset,
                        sloppy=shuffle_input_files,
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
            d = d.map(lambda batch_data:
                        {
                            i: tf.sparse.to_dense(batch_data[i]) if isinstance(batch_data[i], tf.SparseTensor) else batch_data[i] for i in batch_data
                        }
                      , num_parallel_calls=num_cpu_threads
                     )
            # d = d.map(lambda batch_data: {i: tf.sparse.to_dense(batch_data[i]) for i in batch_data})

            return d

        return input_fn


def test_tfrecord_input_fn_builder():
    # input_files = "/Users/didi/PycharmProjects/dd-peakview/data/rp_rank_pairwise/1_beijing/3.4.2/20211021/29_numerical_fea.data"
    input_files = "/Users/didi/PycharmProjects/dd-peakview/data/often_route_tfrecord/part-r-00118"

    name_to_features = {
        "labels": tf.io.FixedLenFeature([], tf.float32),
        "order_id": tf.io.FixedLenFeature([], tf.int64),
        "order_max_link_len": tf.io.FixedLenFeature([], tf.int64),

        "req_time": tf.io.FixedLenFeature([], tf.int64),
        "passenger_id": tf.io.FixedLenFeature([], tf.int64),
        "driver_id": tf.io.FixedLenFeature([], tf.int64),
        "spos": tf.io.FixedLenFeature([2], tf.float32),
        "dpos": tf.io.FixedLenFeature([2], tf.float32),

        "links": tf.io.VarLenFeature(tf.int64),
        "num_fea": tf.io.FixedLenFeature([87], tf.float32),
        "num_fea_mat_norm_minmax": tf.io.FixedLenFeature([87], tf.float32),
        "num_fea_mat_norm_z_score": tf.io.FixedLenFeature([87], tf.float32),

        "links_neg": tf.io.VarLenFeature(tf.int64),
        "num_fea_neg": tf.io.FixedLenFeature([87], tf.float32),
        "num_fea_mat_norm_minmax_neg": tf.io.FixedLenFeature([87], tf.float32),
        "num_fea_mat_norm_z_score_neg": tf.io.FixedLenFeature([87], tf.float32),
    }

    # if True:
    #     name_to_features["label"] = tf.io.FixedLenFeature([], tf.int64)

    input_fn = DataSetBuilder.get_tfrecord_input_fn(input_files=[input_files],
                                                    name_to_features=name_to_features, batch_size=2, epoch=1, shuffle_input_files=False, num_cpu_threads=None)

    ds = input_fn(params=None)
    # ds = tf.data.TFRecordDataset(input_files)

    link_id2index = utils.DataLoader(file_path="/Users/didi/PycharmProjects/dd-peakview/model_dir/often_route_pairwise_v1/link_dict.txt", sep=" ", key_idx=0, value_idx=1).result
    first_index_token = "0"
    vocabulary_token_list = [first_index_token] + list(link_id2index.keys())
    vocabulary_token_list = [int(i) for i in vocabulary_token_list]
    vocabulary_index_list = [first_index_token] + list(link_id2index.values())
    vocabulary_index_list = [int(i) for i in vocabulary_index_list]

    ori_link_dict_size = len(vocabulary_index_list)
    print("ori_link_dict_size: ", ori_link_dict_size)
    link_idx_table_layer = tf_embed.StaticVocabularyTableLayer(vocabulary_token_list,
                                                               vocabulary_index_list=vocabulary_index_list,
                                                               default_index_start=0, num_oov_buckets=4,
                                                               lookup_key_dtype=tf.int64)

    for i, batch_data in enumerate(ds):
        if i >= 1:
            break
        # print("----------------------- {} ----------------------".format(i))
        # tf.print(tf_utils.convert_ndarrays_to_strings(tf_utils.convert_tensors_to_ndarrays(batch_data)))
        # tf.print(batch_data)
        # tf.print(batch_data["order_id"])
        # tf.print(batch_data["order_max_link_len"])
        print(batch_data["links"])
        print(">>>>>>>>>>>>")
        print(link_idx_table_layer(batch_data["links"], tf.int32))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.enable_eager_execution()

    test_tfrecord_input_fn_builder()
