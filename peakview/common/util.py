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
            logger.info(cmd)
            os.system(cmd)

    @staticmethod
    def ensure_dir(dir):
        if not os.path.exists(dir):
            logger.info("mkdir: {}".format(dir))
            os.makedirs(dir)

    @staticmethod
    def refresh_dir(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            cmd = "rm -rf {}/*".format(dir)
            logger.info(cmd)
            os.system(cmd)


class FileUtils(object):
    @staticmethod
    def remove_file(file_path):
        if os.path.exists(file_path):
            logger.info("remove file: {}".format(file_path))
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
        logger.info("[save_lines_to_file] file_path: {}".format(file_path))
        if len(lines) < 1:
            logger.warning("[save_list_to_file], len(lines): {} < 1 ".format(len(lines)))
            return
        logger.info("[save_list_to_file], len(lines): {} -> file path: {}".format(len(lines), file_path))
        DirUtils.ensure_dir(os.path.dirname(file_path))
        FileUtils.remove_file(file_path)
        c = 0
        with open(file_path, "w") as f:
            for line in lines:
                f.write("{}\n".format(line))
                c += 1
        logger.info("[save_list_to_file] has been write {} lines to : {}".format(c, file_path))

    @staticmethod
    def save_dict_to_file(data_dict, file_path):
        if len(data_dict) < 1:
            logger.warning("[save_dict_to_file], len(data_dict): {} < 1 ".format(len(data_dict)))
            return
        logger.info("[save_dict_to_file], len(data_dict): {} -> file path: {}".format(len(data_dict), file_path))
        DirUtils.ensure_dir(os.path.dirname(file_path))
        FileUtils.remove_file(file_path)
        with open(file_path, "w") as f:
            for key in data_dict:
                line = ",".join([str(i) for i in data_dict[key]])
                f.write("{} {}\n".format(key, line))


class DataLoader(object):
    def __init__(self, file_path, report_interval=100, sep="\t", key_idx=0, value_idx=1):
        # result
        self.result_dict = collections.OrderedDict()
        self.file_path = file_path
        self.report_interval = report_interval
        self.sep = sep
        self.key_idx = key_idx
        if not isinstance(value_idx, (int, list)):
            logger.fatal("[DataLoader] value_idx must be a int or a list of int !")
            exit(-1)
        self.value_idx = value_idx
        self.read_line(self.file_path)

    def read_line(self, file_path):
        for line in open(file_path, "r"):
            self.parse_line(line)

    def parse_line(self, line):
        l = line.strip().split(self.sep)
        value = None
        if isinstance(self.value_idx, int):
            if not (len(l) > 1):
                return
            value = l[self.value_idx]

        if isinstance(self.value_idx, list):
            if not (len(l) > max(self.value_idx)):
                return
            value = [l[i] for i in self.value_idx]
        if len(self.result_dict) % self.report_interval == 1:
            logger.info("len result_dict: {}".format(len(self.result_dict)))
        if value:
            self.result_dict[l[self.key_idx]] = value


class DataReader(object):
    def __init__(self, file_path, report_interval=100):
        super(DataReader, self).__init__()
        self.file_path = file_path
        self.report_interval = report_interval
        self.file_total_lines = sum([1 for line in open(self.file_path)])
        self.has_read_lines = 0
        self.progress_rate = 0

    @abstractmethod
    def parse_line(self, line):
        pass

    def read_line(self):
        for line in open(self.file_path, "r"):
            self.has_read_lines += 1
            self.progress_rate = self.has_read_lines / self.file_total_lines
            if self.has_read_lines % self.report_interval == 0:
                logger.info("[DataReader] has_read_lines: {}, file_total_lines: {}, progress_rate: {} ......".format(
                    self.has_read_lines, self.file_total_lines, round(self.progress_rate, 5)))
            yield self.parse_line(line)


class HdfsUtil(object):
    @staticmethod
    def hdfs2local(hdfs_path, local_path, reload=False, merge=False):
        if reload is False and os.path.exists(local_path):
            logger.info("local_path already exists: {}".format(local_path))
            return 1
        else:
            FileUtils.remove_file(local_path)
            DirUtils.ensure_dir(os.path.dirname(local_path))
            logger.info("downloading from HDFS: {} --> Local: {}".format(hdfs_path, local_path))
            cmd = "hdfs dfs -get {} {}".format(hdfs_path, local_path)
            if merge:
                cmd = "hdfs dfs -getmerge {} {}".format(hdfs_path, local_path)
            status = os.system(cmd)
            if int(status) != 0:
                logger.fatal("failed: {}".format(cmd))
                return 1
            logger.info("download finish: {} -> {}".format(hdfs_path, local_path))
            return 0


class TimeUtils(object):
    @staticmethod
    def str2timestamp(date_string, format_str="%Y-%m-%d %H:%M:%S"):
        return int(time.mktime(datetime.strptime(date_string, format_str).timetuple()))

    @staticmethod
    def str2datetime(date_string, format_str="%Y-%m-%d-%H"):
        return datetime.strptime(date_string, format_str)

    @staticmethod
    def datetime2str(dt: datetime, format_str="%Y-%m-%d-%H"):
        return dt.strftime(format_str)

    @staticmethod
    def get_date_list_by_timedelta(start_dt: date, delta: timedelta):
        days = []
        for i in range(delta.days + 1):
            day = start_dt + timedelta(days=i)
            days.append(day)
        return days

    @classmethod
    def get_datetime_list_by_start_end(cls, start_dt, end_dt, to_str=False):
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
            start_dt = cls.str2datetime(start_dt, "%Y%m%d")
        if isinstance(end_dt, str):
            end_dt = cls.str2datetime(end_dt, "%Y%m%d")

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
            date_l = [cls.datetime2str(i, "%Y%m%d") for i in date_l]
        return date_l


def get_vocab_by_hash_str(vocab, to_hash_str):
    hash_value = int(hashlib.sha1(to_hash_str.encode("utf-8")).hexdigest(), 16)
    vocab_size = len(vocab)
    return vocab[hash_value % vocab_size]


def test_get_vocab_by_hash_str():
    v = [1, 3, 5]
    s = "hkjjjnnn"
    print(get_vocab_by_hash_str(v, s))


def run_loop(flags_obj):
    print(TimeUtils.get_datetime_list_by_start_end("20201111", "20201101", to_str=True))
    test_get_vocab_by_hash_str()

    return


def main(_):
    run_loop(flags.FLAGS)


if __name__ == '__main__':
    define_flags()
    absl_app.run(main)