# -*- coding: utf-8 -*-
"""
    Author  ：   Ethan
    Time    ：   2021/11/9 4:27 下午
    Site    :   
    Suggestion  ：
    Description :
    File    :   cnn.py
    Software    :   PyCharm
"""
import tensorflow as tf


def cnn_2d_block(inputs, filters=256, kernel_size=3, padding="same", activation="relu", dropout=0, name="cnn_block"):
    with tf.variable_scope(name) as scope:
        output = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, padding=padding,
                                  activation=activation,
                                  name="cnn")
        if dropout > 0:
            output = tf.nn.dropout(output, rate=dropout)
        output = tf.layers.batch_normalization(output)
    return output


# def inception_block(inputs):
#     cnn_1 = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size, padding=padding,
#                               activation=activation,
#                               name="cnn")
#     cnn_1 = tf.



