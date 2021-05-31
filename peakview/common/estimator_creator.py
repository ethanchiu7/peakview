# -*- coding: utf-8 -*-
"""
    Author  ：   Ethan Chiu
    Time    ：   2021/3/30 下午12:15
    Site    :   
    Suggestion  ：
    Description :
    File    :   estimator_creator.py
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
import util
import tf_util


class DataSource(object):
    TFRECORD = "tfrecord"
    TEXT = "text"
    ARRAY = "array"


class DataSetCreator(object):
    def __init__(self, input_files, is_file_patterns=False, name_to_features=None, data_source=DataSource.TFRECORD):
        if name_to_features is None and data_source == DataSource.TFRECORD:
            name_to_features = ModelCreatorABC.get_name_to_features()
        self.input_file_paths = util.FileUtils.pattern_to_files(input_files, is_file_patterns)
        tf.logging.info(" [DataSetCreator] input_file_paths count {} :".format(len(self.input_file_paths)))
        for file_path in self.input_file_paths:
            tf.logging.info("  %s" % file_path)

        self.name_to_features = name_to_features
        self.data_source = data_source

    def input_fn_builder(self, batch_size=64, epoch=1, is_training=True, num_cpu_threads=4):
        if self.data_source == DataSource.TFRECORD:
            tf.logging.info(" [DataSetCreator] -> tfrecord_input_fn_builder ")
            return self.tfrecord_input_fn_builder(self.input_file_paths, self.name_to_features, batch_size, epoch, is_training, num_cpu_threads)
        if self.data_source == DataSource.TEXT:
            return
        if self.data_source == DataSource.ARRAY:
            return

    @classmethod
    def feature_dict_input_fn_builder(cls,
                                      input_files,
                                      batch_size=64,
                                      epoch=1,
                                      is_training=True,
                                      num_cpu_threads=4):
        pass

    @classmethod
    def text_input_fn_builder(cls,
                              input_files,
                              batch_size=64,
                              epoch=1,
                              is_training=True,
                              num_cpu_threads=4):
        pass

    @classmethod
    def tfrecord_input_fn_builder(cls, input_files, name_to_features, batch_size, epoch, is_training, num_cpu_threads):
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


class ModelCreatorABC(abc.ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.probabilities = None

    @abc.abstractmethod
    def get_model_conf(self):
        pass

    @classmethod
    def get_name_to_features(cls):
        seq_length = 128
        max_predictions_per_seq = 15
        # name_to_features = {
        #     "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        #     "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        #     "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        #     "label_ids": tf.io.FixedLenFeature([], tf.int64)
        # }
        # name_to_features = {
        #     "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        #     "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        #     "sentence_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        #     "tts_dist": tf.io.FixedLenFeature([seq_length], tf.float32),
        #     "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        #     "label": tf.io.FixedLenFeature([], tf.int64),
        #     "sample_id": tf.io.FixedLenFeature([], tf.int64)
        # }
        name_to_features = {
            "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),

            "masked_lm_positions": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights": tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
            "next_sentence_labels": tf.io.FixedLenFeature([], tf.int64),

            "sentence_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
            "tts_dist": tf.io.FixedLenFeature([seq_length], tf.float32),
            "sample_id": tf.io.FixedLenFeature([], tf.int64),
            "order_id": tf.io.FixedLenFeature([], tf.string),
            "code_links": tf.io.FixedLenFeature([], tf.string)
        }
        return name_to_features

    @classmethod
    def get_assignment_map_from_checkpoint(cls, tvars, init_checkpoint, verbose=False):
        """Compute the union of the current variables and checkpoint variables."""
        assignment_map = collections.OrderedDict()
        trainable_variable_in_ckpt = {}
        trainable_variable_not_in_ckpt = {}
        ckpt_variables_in_trainable = {}
        ckpt_variables_not_in_trainable = {}

        name_to_variable = collections.OrderedDict()
        for var in tvars:
            name = var.name
            m = re.match("^(.*):\\d+$", name)
            if m is not None:
                name = m.group(1)
            name_to_variable[name] = var
        try:
            init_vars = tf.train.list_variables(init_checkpoint)
        except:
            init_vars = []
        for (name, shape) in init_vars:
            if name not in name_to_variable:
                ckpt_variables_not_in_trainable[name] = 1
                ckpt_variables_not_in_trainable[name + ":0"] = 1
                continue
            assignment_map[name] = name
            ckpt_variables_in_trainable[name] = shape
            ckpt_variables_in_trainable[name + ":0"] = shape

        trainable_variables_log_s = "[get_assignment_map_from_checkpoint] trainable_variables\n"
        for var in tvars:
            name = var.name
            m = re.match("^(.*):\\d+$", name)
            if m is not None:
                name = m.group(1)
            if name in ckpt_variables_in_trainable:
                trainable_variable_in_ckpt[name] = var.shape
                trainable_variables_log_s += "*** INIT_FROM_CKPT_TRAINABLE_VAR *** name = {}, shape = {}\n".format(name, str(var.shape))
            else:
                trainable_variable_not_in_ckpt[name] = var.shape
                trainable_variables_log_s += " name = {}, shape = {}\n".format(name, str(var.shape))

        assert len(trainable_variable_in_ckpt) == len(ckpt_variables_in_trainable) / 2
        tf.logging.info("""[get_assignment_map_from_checkpoint] trainable_variables: {}, init_checkpoint: {}
        trainable_variable_in_ckpt: {}, trainable_variable_not_in_ckpt: {}
        ckpt_variables_in_trainable: {}, ckpt_variables_not_in_trainable: {}
        """.format(len(tvars), init_checkpoint,
                   len(trainable_variable_in_ckpt), len(trainable_variable_not_in_ckpt),
                   len(ckpt_variables_in_trainable) / 2, len(ckpt_variables_not_in_trainable) / 2))

        if verbose:
            tf.logging.info(trainable_variables_log_s)
            ckpt_variables_not_in_trainable_log_s = "[get_assignment_map_from_checkpoint] ckpt_variables_not_in_trainable:\n"
            for name in ckpt_variables_not_in_trainable:
                if ":0" in name:
                    continue
                ckpt_variables_not_in_trainable_log_s += "name = {}, shape = {} \n".format(name, str(ckpt_variables_not_in_trainable[name]))
            tf.logging.info(ckpt_variables_not_in_trainable_log_s)

        return (assignment_map, trainable_variable_in_ckpt, trainable_variable_not_in_ckpt, ckpt_variables_in_trainable, ckpt_variables_not_in_trainable)

    @abc.abstractmethod
    def create_model(self, features, labels, is_training):
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        labels = features["label"]
        num_labels = 2

        output_layer = tf.cast(input_ids, tf.float32)

        hidden_size = output_layer.shape[-1].value
        output_weights = tf.get_variable("output_weights", [num_labels, hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            if is_training:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)
            self.probabilities = probabilities
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            batch_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            batch_mean_loss = tf.reduce_mean(batch_loss)

            return (logits, probabilities, batch_mean_loss, batch_loss)

    @staticmethod
    def _decay_warmup_lr(global_step, init_lr, num_decay_steps, end_learning_rate, decay_pow, num_warmup_steps):
        learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

        # Implements linear decay of the learning rate.
        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            global_step,
            num_decay_steps,
            end_learning_rate=end_learning_rate,
            power=decay_pow,
            cycle=False)

        # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
        # learning rate will be `global_step/num_warmup_steps * init_lr`.
        if num_warmup_steps:
            global_steps_int = tf.cast(global_step, tf.int32)
            warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

            global_steps_float = tf.cast(global_steps_int, tf.float32)
            warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            warmup_percent_done = global_steps_float / warmup_steps_float
            warmup_learning_rate = init_lr * warmup_percent_done

            is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
            learning_rate = ((1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
        return learning_rate

    @classmethod
    def create_train_op(cls, loss, init_lr, num_decay_steps, end_learning_rate, decay_pow, num_warmup_steps, use_tpu=False):
        """Creates an optimizer training op."""
        global_step = tf.train.get_or_create_global_step()
        learning_rate = cls._decay_warmup_lr(global_step, init_lr, num_decay_steps, end_learning_rate, decay_pow, num_warmup_steps)

        # It is recommended that you use this optimizer for fine tuning, since this
        # is how the model was trained (note that the Adam m/v variables are NOT
        # loaded from init_checkpoint.)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-6)

        if use_tpu:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)

        # This is how the model was pre-trained.
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

        train_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step=global_step)

        # Normally the global step update is done inside of `apply_gradients`.
        # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
        # a different optimizer, you should probably take this line out.
        # new_global_step = global_step + 1
        # train_op = tf.group(train_op, [global_step.assign(new_global_step)])
        return train_op

    @classmethod
    def create_metric_ops(cls, batch_loss, labels, logits, is_real_example=1):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=batch_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

    def create_predict_ops(self):
        predictions = {"probabilities": self.probabilities}
        return predictions


class ModelConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               num_labels,
               seq_length,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    """Constructs BertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    self.seq_length = seq_length
    self.num_labels = num_labels
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = ModelConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def test_tfrecord_input_fn_builder():
    input_files = "/Users/didi/PycharmProjects/nightingale/deeplearning/tfrecord/finetune/part-00009.tfrecord"
    show_line_num = 1
    dataset_creator = DataSetCreator(input_files=input_files,
                                     is_file_patterns=True, name_to_features=ModelCreatorABC.get_name_to_features(),
                                     data_source=DataSource.TFRECORD)
    input_fn = dataset_creator.input_fn_builder(batch_size=4, epoch=1, is_training=False, num_cpu_threads=1)
    ds = input_fn(params=None)
    # ds = tf.data.TFRecordDataset(input_files)

    for i, batch_data in enumerate(ds):
        if i >= 1:
            break
        tf.print("----------------------- {} ----------------------".format(i))
        tf.print(tf_util.convert_ndarrays_to_strings(tf_util.convert_tensors_to_ndarrays(batch_data)))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.enable_eager_execution()

    test_tfrecord_input_fn_builder()
