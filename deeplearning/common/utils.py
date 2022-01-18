# -*- coding: utf-8 -*-
"""
    Author  ：   Ethan
    Time    ：   2020/8/10 2:43 下午
    Site    :
    Suggestion  ：
    Description :
    File    :   util.py
    Software    :   PyCharm
"""
import os
import sys
import collections
import logging
from abc import abstractmethod
# import tensorflow as tf
from absl import app as absl_app
from absl import flags
from datetime import date, timedelta, datetime
import time
import hashlib
import tensorflow as tf
import shutil
import time
import functools


def define_flags():
    flags.DEFINE_enum(name="mode", default="train", enum_values=["train", "predict"],
                      help="")


class Logger(logging.Logger, object):

    def __init__(self, name="default", level=logging.INFO, log_file=None, stdout=True):

        super(Logger, self).__init__(name, level)

        # Gets or creates a logger
        # self._logger = logging.getLogger(logger_name)

        # set log level
        # self._logger.setLevel((level))

        # set formatter
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')

        if log_file is not None:
            if not os.path.exists(log_file):
                os.makedirs(log_file)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.addHandler(file_handler)

        if stdout:
            # define stream handler
            stream_handler = logging.StreamHandler(sys.stdout)
            # set handler
            stream_handler.setFormatter(formatter)
            # add file handler to logger
            self.addHandler(stream_handler)


logger = Logger(__file__)


def check_status(status, job_name="job"):
    if int(status) == 0:
        print("{} success".format(job_name))
        return True
    else:
        print(">>>>>  {} failed !!!".format(job_name))
        exit(1)


class DirUtils(object):
    @staticmethod
    def get_parent_dir(file, parent_num=1):
        parent_dir = os.path.abspath(file)
        for i in range(parent_num):
            parent_dir = os.path.dirname(parent_dir)
        return parent_dir

    @staticmethod
    def remove_dir(dir):
        if os.path.exists(dir):
            cmd = "rm -rf {}".format(dir)
            print(cmd)
            os.system(cmd)

    @staticmethod
    def ensure_dir(dir):
        if not os.path.exists(dir):
            print("mkdir: {}".format(dir))
            os.makedirs(dir)

    @staticmethod
    def refresh_dir(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            cmd = "rm -rf {}/*".format(dir)
            print(cmd)
            os.system(cmd)


class FileUtils(object):
    @staticmethod
    def remove_file(file_path):
        if os.path.exists(file_path):
            print("remove file: {}".format(file_path))
            os.remove(file_path)

    @staticmethod
    def pattern_to_files(input_files, is_file_patterns):
        """

        :param patterns:
        :param is_file_patterns:
        :return:
        """
        input_file_paths = []
        if is_file_patterns:
            if isinstance(input_files, str):
                input_files = input_files.split(",")
            assert isinstance(input_files, (list, tuple))
            for input_pattern in input_files:
                input_file_paths.extend(tf.gfile.Glob(input_pattern))
        else:
            if isinstance(input_files, str):
                input_files = input_files.split(",")
            assert isinstance(input_files, (list, tuple))
            input_file_paths.extend(input_files)
        return input_file_paths


class DataWriter(object):
    @staticmethod
    def save_lines_to_file(lines, file_path):
        print("[save_lines_to_file] file_path: {}".format(file_path))
        if len(lines) < 1:
            logger.warning("[save_list_to_file], len(lines): {} < 1 ".format(len(lines)))
            return
        print("[save_list_to_file], len(lines): {} -> file path: {}".format(len(lines), file_path))
        DirUtils.ensure_dir(os.path.dirname(file_path))
        FileUtils.remove_file(file_path)
        c = 0
        with open(file_path, "w") as f:
            for line in lines:
                f.write("{}\n".format(line))
                c += 1
        print("[save_list_to_file] has been write {} lines to : {}".format(c, file_path))

    @staticmethod
    def save_dict_to_file(data_dict, file_path):
        if len(data_dict) < 1:
            logger.warning("[save_dict_to_file], len(data_dict): {} < 1 ".format(len(data_dict)))
            return
        print("[save_dict_to_file], len(data_dict): {} -> file path: {}".format(len(data_dict), file_path))
        DirUtils.ensure_dir(os.path.dirname(file_path))
        FileUtils.remove_file(file_path)
        with open(file_path, "w") as f:
            for key in data_dict:
                line = ",".join([str(i) for i in data_dict[key]])
                f.write("{} {}\n".format(key, line))


class DataLoader(object):
    """"
    value_idx 支持 None int / list of int
    当 value_idx 为 None时 所有 value 都是 1
    """
    def __init__(self, file_path, report_interval=100, sep="\t", key_idx=0, value_idx=None):
        if not os.path.isfile(file_path):
            raise FileExistsError(file_path)
        self.value_always_same = False
        self.result = collections.OrderedDict()
        if not isinstance(value_idx, (int, list)):
            # 所有value 都是 1
            self.value_always_same = True
        # result
        self.file_path = file_path
        self.report_interval = report_interval
        self.sep = sep
        self.key_idx = key_idx
        self.value_idx = value_idx
        self.read_line(self.file_path)

        print("已从文件: {}\n加载 result len: {}".format(self.file_path, len(self.result)))
        print("======== result =======")
        print(list(self.result.items())[:100])
        print("======== result =======")

    def read_line(self, file_path):
        for line in open(file_path, "r"):
            self.parse_line(line)

    def parse_line(self, line):
        line = line.strip()
        if not line:
            return None

        l = line.split(self.sep)

        value = None
        if isinstance(self.value_idx, int):
            if not (len(l) > 1):
                return
            value = l[self.value_idx]
        if isinstance(self.value_idx, list):
            if not (len(l) > max(self.value_idx)):
                return
            value = [l[i] for i in self.value_idx]
        if self.value_always_same:
            value = None

        self.result[l[self.key_idx]] = value

        if len(self.result) % self.report_interval == 1:
            print("len result_dict: {}".format(len(self.result)))

    def get_value_by_key(self, key):
        assert isinstance(key, str)
        return self.result[key]


class DataReader(object):
    """
    读取超大文件迭代器
    """
    def __init__(self, report_interval=100):
        super(DataReader, self).__init__()
        self.report_interval = report_interval

    def _parse_line(self, line):
        line = line.strip()
        return line

    def read_file(self, file_path):
        has_read_lines = 0
        progress_rate = 0
        # file_total_lines = sum([1 for line in open(file_path)])
        for line in open(file_path, "r"):
            has_read_lines += 1
            # self.progress_rate = self.has_read_lines / file_total_lines
            # if self.has_read_lines % self.report_interval == 0:
            #     print("[DataReader] has_read_lines: {}, file_total_lines: {}, progress_rate: {} ......".format(
            #         self.has_read_lines, file_total_lines, round(self.progress_rate, 5)))
            parsed_line = self._parse_line(line)
            if parsed_line:
                yield parsed_line

    def __call__(self, file_path):
        return self.read_file(file_path)


class Reducer(object):
    def __init__(self, reduce_fn=None):
        if reduce_fn:
            self.reduce_fn = reduce_fn
        self.last_key = None
        self.value_list_tmp = []

    @staticmethod
    def reduce_fn(k, v_l):
        print(k, len(v_l))

    def __call__(self, key, value):
        if self.last_key is None:
            self.last_key = key
            self.value_list_tmp.append(value)
            return
        if self.last_key != key:
            self.reduce_fn(self.last_key, self.value_list_tmp)
            self.last_key = key
            self.value_list_tmp = []
        self.value_list_tmp.append(value)

    def __del__(self):
        if len(self.value_list_tmp) > 0:
            self.reduce_fn(self.last_key, self.value_list_tmp)


class HdfsUtil(object):
    @staticmethod
    def hdfs2local(hdfs_path, local_path, reload=False, merge=False):
        if reload is False and os.path.exists(local_path):
            print("local_path already exists: {}".format(local_path))
            return 1
        cmd = "hdfs dfs -test -e {}".format(hdfs_path)
        status = os.system(cmd)
        if status != 0:
            print("HDFS not exists: ", hdfs_path)
            return 1
        FileUtils.remove_file(local_path)
        DirUtils.ensure_dir(os.path.dirname(local_path))
        print("downloading from HDFS: {} --> Local: {}".format(hdfs_path, local_path))
        cmd = "hdfs dfs -get {} {}".format(hdfs_path, local_path)
        if merge:
            cmd = "hdfs dfs -getmerge {} {}".format(hdfs_path, local_path)
        status = os.system(cmd)
        if int(status) != 0:
            logger.fatal("failed: {}".format(cmd))
            return 1
        print("download finish: {} -> {}".format(hdfs_path, local_path))
        return 0

    @staticmethod
    def hdfsdir2local(hdfs_dir, local_dir, reload=False, merge=False):
        cmd = "hdfs dfs -test -e {}".format(hdfs_dir)
        status = os.system(cmd)
        if status != 0:
            print("HDFS not exists: ", hdfs_dir)
            return 1
        if reload is False and os.path.isdir(local_dir):
            print("local_dir already exists: {}".format(local_dir))
            return 1
        DirUtils.refresh_dir(local_dir)
        print("downloading from HDFS: {} --> Local: {}".format(hdfs_dir, local_dir))
        cmd = "hdfs dfs -get {} {}".format(hdfs_dir, local_dir)
        if merge:
            cmd = "hdfs dfs -getmerge {} {}".format(hdfs_dir, local_dir)
        status = os.system(cmd)
        if int(status) != 0:
            logger.fatal("failed: {}".format(cmd))
            return 1
        print("download finish: {} -> {}".format(hdfs_dir, local_dir))
        return 0

# def timer(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         begin_time = time.perf_counter()
#         res = func(*args, **kwargs)
#         time_elapsed = time.perf_counter() - begin_time
#         print("[Timer] {} | {} sec".format(func.__name__, time_elapsed))
#         return res
#     return wrapper


class TimeUtils(object):
    @staticmethod
    def str2timestamp(date_string, format_str="%Y-%m-%d %H:%M:%S"):
        return int(time.mktime(datetime.strptime(date_string, format_str).timetuple()))

    @staticmethod
    def str2datetime(date_string, format_str="%Y-%m-%d-%H"):
        return datetime.strptime(date_string, format_str)

    @staticmethod
    def datetime2str(dt, format_str="%Y-%m-%d-%H"):
        return dt.strftime(format_str)

    @staticmethod
    def get_date(days=None, fmt=None):
        days = int(days) if days else 0

        if fmt:
            date = (datetime.now() + timedelta(days)).strftime(fmt)
        else:
            date = (datetime.now() + timedelta(days))
        return date

    @staticmethod
    def get_date_list_by_timedelta(start_dt, delta):
        days = []
        for i in range(delta.days + 1):
            day = start_dt + timedelta(days=i)
            days.append(day)
        return days

    @classmethod
    def get_datetime_list_by_start_end(cls, start_dt, end_dt, to_str=False, format_str="%Y%m%d"):
        """
        s = str2datetime("20200901", "%Y%m%d")
        e = str2datetime("20200911", "%Y%m%d")
        days = get_datetime_list_by_start_end(s, e)
        days = [datetime2str(i, "%Y%m%d") for i in days]
        print(days)
        :param to_str:
        :param start_dt:
        :param end_dt:
        :return:
        """
        if isinstance(start_dt, str):
            start_dt = cls.str2datetime(start_dt, format_str)
        if isinstance(end_dt, str):
            end_dt = cls.str2datetime(end_dt, format_str)

        reverse_tag = False
        if end_dt < start_dt:
            delta = start_dt - end_dt  # as timedelta
            date_l = cls.get_date_list_by_timedelta(end_dt, delta)
            reverse_tag = True
        else:
            delta = end_dt - start_dt  # as timedelta
            date_l = cls.get_date_list_by_timedelta(start_dt, delta)
        if reverse_tag:
            date_l = date_l[::-1]
        if to_str:
            date_l = [cls.datetime2str(i, format_str) for i in date_l]
        return date_l


def get_vocab_by_hash_str(vocab, to_hash_str):
    hash_value = int(hashlib.sha1(to_hash_str.encode("utf-8")).hexdigest(), 16)
    vocab_size = len(vocab)
    return vocab[hash_value % vocab_size]


# def test_get_vocab_by_hash_str():
#     v = [1, 3, 5]
#     s = "hkjjjnnn"
#     print(get_vocab_by_hash_str(v, s))


def run_loop(flags_obj):
    # print(TimeUtils.get_datetime_list_by_start_end("20201111", "20201101", to_str=True))
    # test_get_vocab_by_hash_str()
    print("TEST ...", TimeUtils.get_date(-1, '%Y-%m-%d'))

    return


def main(_):
    run_loop(flags.FLAGS)


if __name__ == '__main__':
    define_flags()
    absl_app.run(main)