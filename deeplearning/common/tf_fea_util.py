# -*- coding: utf-8 -*-
"""
    Author  ：   Ethan Chiu
    Time    ：   2021/12/20 下午4:37
    Site    :   
    Suggestion  ：
    Description :
    File    :   tf_fea_util.py
    Software    :   PyCharm
"""
import tensorflow as tf


def discretize(continuous_tensor, step_value=60.*60, grid_num=24):
    """
    连续性变量 特征离散化
    默认为 将10位时间戳 转换为 0 ~ 23 小时 id
    :param continuous_tensor:  tf.int32 or tf.float32
    :param step_value: 多少值作为间隔 (精度)
    :param grid_num:  划分的 id 数量
    :return:
    """
    continuous_tensor = tf.cast(continuous_tensor, tf.float32)
    step_value = tf.cast(step_value, tf.float32)
    grid_num = tf.cast(grid_num, tf.int32)
    id = tf.cast(continuous_tensor // step_value, tf.int32)
    return id % grid_num


def get_geo_point_bound(point_tensor, grid_num=400):
    """

    :param point_tensor: [B, 2] [B, lnt lat]
    :param grid_num: 网格横纵数量
    :return: tf.int32 0 ~ grid_num ** 2 - 1
    """
    point_bound = discretize(point_tensor, step_value=1.0/grid_num, grid_num=grid_num)
    key = point_bound[:, 0] * grid_num + point_bound[:, 1]
    return key


