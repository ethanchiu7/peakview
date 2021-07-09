# -*- coding: utf-8 -*-
"""
    Author  ：   Ethan
    Time    ：   2021/7/9 5:38 下午
    Site    :   
    Suggestion  ：
    Description :
    File    :   layers.py
    Software    :   PyCharm
"""
import tensorflow as tf


class ClassifierLossLayer(tf.keras.layers.Layer):
  """Final classifier layer with loss."""

  def __init__(self,
               hidden_size,
               num_labels,
               dropout_prob=0.0,
               initializer=None,
               use_bias=True,
               name="classifier"):
    super(ClassifierLossLayer, self).__init__(name=name)
    self.hidden_size = hidden_size
    self.num_labels = num_labels
    self.initializer = initializer
    self.dropout = tf.keras.layers.Dropout(dropout_prob)
    self.use_bias = use_bias

    with tf.compat.v1.variable_scope(name):
      self.w = tf.compat.v1.get_variable(
          name="kernel",
          shape=[self.hidden_size, self.num_labels],
          initializer=self.initializer)
      if self.use_bias:
        self.b = tf.compat.v1.get_variable(
            name="bias",
            shape=[self.num_labels],
            initializer=tf.zeros_initializer)
      else:
        self.b = None

  def call(self, input_tensor, labels=None, training=None):
    input_tensor = self.dropout(input_tensor, training)

    logits = tf.matmul(input_tensor, self.w)
    if self.use_bias:
      logits = tf.nn.bias_add(logits, self.b)

    log_probs = tf.nn.log_softmax(logits, axis=-1)
    if labels is not None:
      one_hot_labels = tf.one_hot(labels, depth=self.num_labels,
                                  dtype=tf.float32)
      per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
      batch_mean_loss = tf.reduce_mean(per_example_loss)
    else:
      per_example_loss = None
      batch_mean_loss = None

    return logits, log_probs, per_example_loss, batch_mean_loss
