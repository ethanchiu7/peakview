# -*- coding: utf-8 -*-
"""
    Author  ：   Ethan
    Time    ：   2021/11/9 5:24 下午
    Site    :   
    Suggestion  ：
    Description :
    File    :   norm.py
    Software    :   PyCharm
"""
import tensorflow as tf


class NormLayer(tf.keras.layers.Layer):
  """Replacement for contrib_layers.layer_norm."""

  def __init__(self, hdim, dtype=tf.float32, name="LayerNorm"):
    super(NormLayer, self).__init__(name=name)
    self._dtype = dtype

    with tf.variable_scope(name):
      self.beta = tf.get_variable(
          "beta", [hdim], dtype=dtype, initializer=tf.zeros_initializer())
      self.gamma = tf.get_variable(
          "gamma", [hdim], dtype=dtype, initializer=tf.ones_initializer())

  def call(self, inputs):
    inputs_shape = inputs.shape

    # Compute norm along last axis
    mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
    # Compute layer normalization using the batch_normalization function.
    # Note that epsilon must be increased for float16 due to the limited
    # representable range.
    variance_epsilon = 1e-12 if self._dtype != tf.float16 else 1e-3
    outputs = tf.nn.batch_normalization(
        inputs,
        mean,
        variance,
        offset=self.beta,
        scale=self.gamma,
        variance_epsilon=variance_epsilon)
    outputs.set_shape(inputs_shape)
    return outputs


# def layer_norm(inputs, hdim, dtype=tf.float32, name="layer_norm"):
#     with tf.variable_scope(name) as scope:
#         beta = tf.get_variable("beta", [hdim], dtype=dtype, initializer=tf.zeros_initializer())
#         gamma = tf.get_variable("gamma", [hdim], dtype=dtype, initializer=tf.ones_initializer())
#     inputs_shape = inputs.shape
#     mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
#     variance_epsilon = 1e-12 if dtype != tf.float16 else 1e-3
#     outputs = tf.nn.batch_normalization(
#         inputs,
#         mean,
#         variance,
#         offset=beta,
#         scale=gamma,
#         variance_epsilon=variance_epsilon)
#     outputs.set_shape(inputs_shape)
#     return outputs


def layer_norm(inputs, name="layer_norm"):
    with tf.variable_scope(name) as scope:
        return tf.layers.batch_normalization(inputs, axis=[0, 1, 2])
