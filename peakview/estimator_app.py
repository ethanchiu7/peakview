# -*- coding: utf-8 -*-
"""
    Author  ：   Ethan Chiu
    Time    ：   2021/3/30 下午12:15
    Site    :
    Suggestion  ：
    Description :
    File    :   model_builder.py
    Based on Tensorflow 1.14
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import enum
import json
import numpy as np
import tensorflow as tf
from common import util
from common import tf_util

PROJECT_DIR = util.DirUtils.get_parent_dir(__file__, 2)
from common import dataset_builder

# ========= If want to use other models, just need change here ========
from models import bert_finetune as modeling
# ====================================================================

# import importlib
# modellib = importlib.import_module("models.{}".format("bert_finetune"))
# ====================================================================
running_config = modeling.RunningConfig()
network_config = modeling.ModelConfig()
model_builder = modeling.ModelBuilder()


class RunMode(enum.Enum):
    TRAIN = 0
    EVAL = 1
    TRAIN_WITH_EVAL = 2
    PREDICT = 3


class LogVerbosity(enum.Enum):
    DEBUG = tf.logging.DEBUG
    INFO = tf.logging.INFO
    WARN = tf.logging.WARN
    ERROR = tf.logging.ERROR
    FATAL = tf.logging.FATAL


def define_flags():
    ## ------  Required parameters
    flags.DEFINE_enum("run_mode", RunMode.PREDICT.name, [e.name for e in RunMode], "Run this py mode, TRAIN/EVAL/TRAIN_WITH_EVAL/PREDICT")
    flags.DEFINE_enum("log_verbosity", LogVerbosity.INFO.value, [e.name for e in LogVerbosity],
                      "tf logging set_verbosity, DEBUG/INFO/WARN/ERROR/FATAL")
    flags.DEFINE_boolean("use_gpu", False, "If use GPU.")

    flags.DEFINE_string("model_builder", "models.bert_finetune", "define how to build current models")

    flags.DEFINE_string("model_dir", "{}/model_dir/{}".format(PROJECT_DIR, running_config.model_name),
                        "The output directory where the models checkpoints will be written.")
    flags.DEFINE_string("init_checkpoint", "{}/model_dir/{}".format(PROJECT_DIR, running_config.model_name),
                        "Initial checkpoint (usually from a pre-trained models).")

    flags.DEFINE_boolean("clear_model_dir", running_config.clear_model_dir, "If remove model_dir.")
    flags.DEFINE_integer("save_checkpoints_steps", running_config.save_checkpoints_steps, "How often to save the models checkpoint.")

    flags.DEFINE_boolean("is_file_patterns", True, "If train_file / eval_file / predict_file is file patterns.")
    # /nfs/project/ethan/nightingale/deeplearning/tfrecord/*.tfrecord
    flags.DEFINE_string("train_file", running_config.train_file, "Input TF example files (can be a glob or comma separated).")
    flags.DEFINE_integer("train_batch_size", 4, "Total batch size for training.")
    flags.DEFINE_integer("train_epoch", 2, "Total number of training epochs to perform.")

    flags.DEFINE_string("eval_file", running_config.eval_file, "Input TF example files (can be a glob or comma separated).")
    flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

    flags.DEFINE_string("predict_file", running_config.predict_file,"Input TF example files (can be a glob or comma separated).")
    flags.DEFINE_integer("predict_batch_size", 10, "Total batch size for predict.")
    flags.DEFINE_integer("num_actual_predict_examples", 10, "The num of examples during predict mode.")

    # learning rate polynomial_decay
    flags.DEFINE_float("learning_rate", running_config.learning_rate, "The initial learning rate for Adam.")
    flags.DEFINE_integer("decay_steps", 2000, "polynomial_decay args : decay_steps")
    flags.DEFINE_float("end_learning_rate", 0.0001, "polynomial_decay args : end_learning_rate")
    flags.DEFINE_float("decay_pow", 1.0, "polynomial_decay args : power")
    flags.DEFINE_integer("warmup_steps", 1000, "polynomial_decay args : decay_steps")


def model_fn_builder(init_checkpoint, learning_rate, decay_steps, end_learning_rate, decay_pow, warmup_steps, use_tpu=False):
  """Returns `model_fn` closure for GPUEstimator."""
  is_real_example = 1

  def model_fn(features, labels, mode, params, config):
    log_prefix = "[model_fn] [mode: {}]".format(mode)
    """
          * `features`: This is the first item returned from the `input_fn`
                 passed to `train`, `evaluate`, and `predict`. This should be a
                 single `tf.Tensor` or `dict` of same.
          * `labels`: This is the second item returned from the `input_fn`
                 passed to `train`, `evaluate`, and `predict`. This should be a
                 single `tf.Tensor` or `dict` of same (for multi-head models).
                 If mode is `tf.estimator.ModeKeys.PREDICT`, `labels=None` will
                 be passed. If the `model_fn`'s signature does not accept
                 `mode`, the `model_fn` must still be able to handle
                 `labels=None`.
          * `mode`: Optional. Specifies if this training, evaluation or
                 prediction. See `tf.estimator.ModeKeys`.
          * `params`: Optional `dict` of hyperparameters.  Will receive what
                 is passed to Estimator in `params` parameter. This allows
                 to configure Estimators from hyper parameter tuning.
          * `config`: Optional `estimator.RunConfig` object. Will receive what
                 is passed to Estimator as its `config` parameter, or a default
                 value. Allows setting up things in your `model_fn` based on
                 configuration such as `num_ps_replicas`, or `model_dir`."""

    tf.logging.info("****** {} ****** Features".format(log_prefix))
    for name in sorted(features.keys()):
      tf.logging.info("****** %s ******  name = %s, shape = %s" % (log_prefix, name, features[name].shape))

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    model_builder.create_model(features, labels, is_training)

    tvars = tf.trainable_variables()
    tf.logging.info("****** {} ****** global_variables len: {}, local_variables len: {}"
                    .format(log_prefix, len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)), len(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES))))
    tf.logging.info("****** {} ****** trainable_variables len: {}, trainable_variables parameter size: {}"
                    .format(log_prefix, len(tvars), tf_util.count_variable_parameter_size(tvars)))
    if init_checkpoint and tf.compat.v1.train.checkpoint_exists(init_checkpoint):
      (assignment_map, trainable_variable_in_ckpt, trainable_variable_not_in_ckpt, ckpt_variables_in_trainable, ckpt_variables_not_in_trainable)\
          = model_builder.get_assignment_map_from_checkpoint(tvars, init_checkpoint, verbose=True)
      if len(assignment_map) > 0:
        tf.logging.info("****** {} ****** init some variables from other checkpoint: {}".format(log_prefix, str(init_checkpoint)))
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
      else:
        tf.logging.info("****** {} ****** len(assignment_map) <= 0, do not init_from_checkpoint".format(log_prefix))
    else:
        tf.logging.info("****** {} ****** init_checkpoint does not exist: {}".format(log_prefix, str(init_checkpoint)))

    output_spec = None

    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = model_builder.get_train_op()
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=model_builder.batch_mean_loss,
          train_op=train_op,
          training_hooks=model_builder.get_training_hooks())

    elif mode == tf.estimator.ModeKeys.EVAL:
      output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=model_builder.batch_mean_loss,
                                               eval_metric_ops=model_builder.get_metric_ops(),
                                               evaluation_hooks=model_builder.get_evaluation_hooks())
    else:
      predict_ops = model_builder.get_predict_ops()
      output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predict_ops, prediction_hooks=model_builder.get_prediction_hooks())
    return output_spec

  return model_fn


def main(_):
    tf.logging.info("[BEGIN] {}".format(__file__))
    tf.logging.info("PARENT_DIR: {}, Run Mode: {}".format(PROJECT_DIR, FLAGS.run_mode))
    if FLAGS.clear_model_dir:
        tf.logging.info("DeleteRecursively: {}".format(FLAGS.model_dir))
        tf.gfile.DeleteRecursively(FLAGS.model_dir)
    tf.gfile.MakeDirs(FLAGS.model_dir)

    name_to_features = model_builder.get_name_to_features()

    def get_input_fn_train():
        tf.logging.info("*** Input Files For Train ***")
        train_dataset_creator = dataset_builder.DataSetCreator(FLAGS.train_file,
                                                             is_file_patterns=FLAGS.is_file_patterns,
                                                             name_to_features=name_to_features,
                                                             data_source=dataset_builder.DataSource.TFRECORD)

        train_input_fn = train_dataset_creator.input_fn_builder(batch_size=FLAGS.train_batch_size,
                                                                epoch=FLAGS.train_epoch,
                                                                is_training=True,
                                                                num_cpu_threads=4)
        return train_input_fn

    def get_input_fn_eval():
        tf.logging.info("*** Input Files For Eval ***")
        eval_dataset_creator = dataset_builder.DataSetCreator(FLAGS.eval_file,
                                                            is_file_patterns=FLAGS.is_file_patterns,
                                                            name_to_features=name_to_features,
                                                            data_source=dataset_builder.DataSource.TFRECORD)
        eval_input_fn = eval_dataset_creator.input_fn_builder(batch_size=FLAGS.train_batch_size,
                                                              epoch=1,
                                                              is_training=False,
                                                              num_cpu_threads=1)
        return eval_input_fn

    def get_input_fn_predict():
        tf.logging.info("*** Input Files For Predict ***")
        predict_dataset_creator = dataset_builder.DataSetCreator(FLAGS.predict_file,
                                                               is_file_patterns=FLAGS.is_file_patterns,
                                                               name_to_features=name_to_features,
                                                               data_source=dataset_builder.DataSource.TFRECORD)
        predict_input_fn = predict_dataset_creator.input_fn_builder(batch_size=FLAGS.predict_batch_size,
                                                                    epoch=1,
                                                                    is_training=False,
                                                                    num_cpu_threads=1)
        return predict_input_fn

    # model_fn
    model_fn = model_fn_builder(
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        decay_steps=FLAGS.decay_steps,
        end_learning_rate=FLAGS.end_learning_rate,
        decay_pow=FLAGS.decay_pow,
        warmup_steps=FLAGS.warmup_steps)

    # model_params
    model_params = None

    # run_config
    dist_strategy = None
    if FLAGS.use_gpu:
        # dist_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
        # dist_strategy = tf.distribute.MirroredStrategy(devices=None, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        dist_strategy = tf.distribute.MirroredStrategy(devices=None, cross_device_ops=tf.distribute.NcclAllReduce())
    ''' IF ERROR COULD TRY
    dist_strategy = tf.contrib.distribute.MirroredStrategy(
        devices=["device:GPU:%d" % i for i in range(FLAGS.n_gpus)],
        cross_tower_ops=tf.distribute.HierarchicalCopyAllReduce())
    '''
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        save_summary_steps=100,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=5,
        keep_checkpoint_every_n_hours=10,
        log_step_count_steps=100,
        train_distribute=dist_strategy,
        eval_distribute=dist_strategy)

    # estimator
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=model_params,
        warm_start_from=None)

    # do_train
    if FLAGS.run_mode == RunMode.TRAIN.name:
        input_fn_train = get_input_fn_train()
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        estimator.train(input_fn=input_fn_train, steps=None)

    # do_eval
    output_eval_file = os.path.join(FLAGS.model_dir, "eval_results.txt")
    if FLAGS.run_mode == RunMode.EVAL.name:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        result = estimator.evaluate(input_fn=get_input_fn_eval())
        tf_util.write_eval_result(result, output_eval_file)

    # do_train_with_eval
    if FLAGS.run_mode == RunMode.TRAIN_WITH_EVAL.name:
        tf.logging.info("***** Running training with evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        train_spec = tf.estimator.TrainSpec(input_fn=get_input_fn_train())
        eval_spec = tf.estimator.EvalSpec(input_fn=get_input_fn_eval())
        result = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        tf_util.write_eval_result(result, output_eval_file)

    # do_predict
    output_predict_file = os.path.join(FLAGS.model_dir, "predict_results.txt")
    if FLAGS.run_mode == RunMode.PREDICT.name:
        tf.logging.info("***** Running prediction *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        predict_result_it = estimator.predict(input_fn=get_input_fn_predict(), yield_single_examples=True)
        tf_util.parse_and_record_predict_result(predict_result_it, output_predict_file, FLAGS.num_actual_predict_examples)

    tf.logging.info("[FINISH] {}".format(__file__))
    exit(0)


if __name__ == "__main__":
    define_flags()
    flags = tf.flags
    FLAGS = flags.FLAGS
    tf.logging.set_verbosity(LogVerbosity[FLAGS.log_verbosity].value)

    tf.app.run()

