# -*- coding: utf-8 -*-
"""
    Author  ：   Ethan Chiu
    Time    ：   2021/3/30 下午12:15
    Site    :
    Suggestion  ：
    Description : The main BERT modeling and related functions.
    File    :   modeling_base.py
    Based on Tensorflow 1.14
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from common import config_base


class RunningConfig(config_base.RunningConfig):
    def __init__(self):
        super(RunningConfig, self).__init__(learning_rate=5e-5)
        self.model_name = os.path.basename(__file__.split(".")[0])
        self.learning_rate = 5e-5
        self.train_file = "/Users/didi/PycharmProjects/deepquant/tfrecord/stock/history/daily/train/*.tfrecord"
        self.eval_file = "/Users/didi/PycharmProjects/deepquant/tfrecord/stock/history/daily/eval/*.tfrecord"
        self.predict_file = "/Users/didi/PycharmProjects/deepquant/tfrecord/stock/history/daily/train/*.tfrecord"
        self.is_file_patterns = True
