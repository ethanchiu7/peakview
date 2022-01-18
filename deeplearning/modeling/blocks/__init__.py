# -*- coding: utf-8 -*-
"""
    Author  ：   Ethan
    Time    ：   2021/11/9 4:27 下午
    Site    :   
    Suggestion  ：
    Description :
    File    :   __init__.py.py
    Software    :   PyCharm
"""
from absl import flags
from absl import app as absl_app


def define_flags():
    flags.DEFINE_enum(
        name="flag_name", short_name="", default="",
        enum_values=['', '', ''],
        help="")
    flags.DEFINE_boolean(
        name="", default=True, help="")


def main(_):


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    define_flags()
    absl_app.run(main)
