# -*- coding: utf-8 -*-
"""
    Author  ：   Ethan
    Time    ：   2021/5/31 下午3:39
    Site    :   
    Suggestion  ：
    Description :
    File    :   decorators.py
    Software    :   PyCharm
"""
from absl import flags
from absl import app as absl_app
import warnings
import functools


def define_flags():
    # flags.DEFINE_enum(
    #     name="flag_name", default="",
    #     enum_values=['', '', ''],
    #     help="")
    # flags.DEFINE_boolean(
    #     name="", default=True, help="")
    pass


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


def test_deprecated():
    @deprecated
    def some_old_function(x, y):
        return x + y

    print(some_old_function(1, 2))

    @deprecated
    class Xx(object):
        def xx(self, x):
            print("haha", x)
    x = Xx()
    x.xx('xixi')


# This is for timing
def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('\n{0} cost time {1} s\n'.format(func.__name__, time_spend))
        return result
    return func_wrapper


def test_timer():
    @timer
    def some_old_function(x, y):
        return x + y

    print(some_old_function(1, 2))


def main(_):
    test_deprecated()
    # test_timer()


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    define_flags()
    absl_app.run(main)
