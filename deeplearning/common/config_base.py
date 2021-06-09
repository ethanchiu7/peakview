# -*- coding: utf-8 -*-
"""
    Author  ：   Ethan Chiu
    Time    ：   2021/3/30 下午12:15
    Site    :
    Suggestion  ：
    Description :
    Based on Tensorflow 1.14
"""
import os
import copy
import time
import json
import six
import tensorflow as tf


class Config(object):
    """Configuration for `BertModel`."""

    def __init__(self, *args, **kwargs):
        self.vocab_size = 512

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = Config(vocab_size=None)
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


class RunningConfig(Config):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # learning_rate = self._decay_warmup_lr(global_step, kwargs["init_lr"], kwargs["num_decay_steps"],
        #                                 kwargs["end_learning_rate"], kwargs["decay_pow"], kwargs["num_warmup_steps"])
        self.learning_rate = 0.005
        self.clear_model_dir = False
        self.save_checkpoints_steps = 10000
        self.train_file = ""
        self.eval_file = ""
        self.predict_file = ""
        self.is_file_patterns = True


class ModelConfig(Config):

    def __init__(self, model_name=os.path.basename(__file__.split(".")[0])):
        super(ModelConfig, self).__init__()
        self.model_name = model_name
        self.learning_rate = 0.005

        self.vocab_size = 512



