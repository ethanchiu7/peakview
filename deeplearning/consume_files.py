# -*- coding: utf-8 -*-
"""
    Author  ：   Ethan Chiu
    Time    ：   2021/3/8 下午3:22
    Site    :
    Suggestion  ：
    Description :   通用 多进程 文件处理工具
    File    :   consume_files.py
"""
import os
import collections
import json
import random
import tensorflow as tf
from absl import app as absl_app
from absl import flags

FLAGS = flags.FLAGS
import enum
import tqdm
from common import utils
import importlib
from typing import List, Mapping
from common import dataset_builder
from multiprocessing import Pool, Manager


def define_flags():
    flags.DEFINE_string("spawning", "rp_rank_pairwise_proto_to_tfrecord", "define how to build current modeling")
    flags.DEFINE_integer("process_num", 4, "num of process")
    flags.DEFINE_boolean("is_file_patterns", True,
                         "If train_file / eval_file / predict_file is file patterns.")
    flags.DEFINE_string(name="input_files",
                        default="/Users/name/xx/xx.proto",
                        help="input files")
    flags.DEFINE_string(name="output_base", default="{}/xx/xx".format(PROJECT_DIR),
                        help="由于输入文件的路径可能存在多个不同dir，比如 20210818、202010819，"
                             "因此需要在output_base路径基础上继承 指定数量的input_dir, 作为output_dir")
    flags.DEFINE_integer("inherit_dir_num", 1, "输出路径从输入路径继承的dir数量，"
                                               "如输入 /aa/bb/cc.txt 继承数量1 就是 /output_base/bb 为output_dir")


logger = utils.Logger(__file__)
PROJECT_DIR = utils.DirUtils.get_parent_dir(__file__, 2)


def run_loop(flags_obj):
    is_multiprocess = False
    if FLAGS.process_num > 1:
        is_multiprocess = True
    to_import_module = "spawning.{}".format(FLAGS.spawning)
    print("======== import : {}".format(to_import_module))
    spawning = importlib.import_module(to_import_module)
    spawn = spawning.spawn

    pool = None
    lock = None
    if is_multiprocess:
        pool = Pool(processes=FLAGS.process_num)
        manager = Manager()
        lock = manager.Lock()

    input_file_paths = dataset_builder.DataSetBuilder.pattern_to_files(FLAGS.input_files,
                                                                   is_file_patterns=FLAGS.is_file_patterns)
    logger.info("input_file_paths len: {} \n{}".format(len(input_file_paths), '\n'.join(input_file_paths)))

    job_count = 0
    for index, input_file in enumerate(tqdm.tqdm(input_file_paths)):
        if not os.path.exists(input_file):
            logger.warning("proto file path not exist: {}".format(input_file))
            continue

        inherit_dirs = []
        for i in range(FLAGS.inherit_dir_num):
            inherit_dirs.append(str(input_file).split("/")[-2-i])

        output_dir = os.path.join(FLAGS.output_base, *inherit_dirs)

        utils.DirUtils.ensure_dir(output_dir)

        if is_multiprocess:
            print("""">>> 添加任务到 进程池，第 {} 个文件
                    input file: {}
                    output file: {}
                    """.format(index, input_file, output_dir))
            res = pool.apply_async(spawn, args=(lock, index, input_file, output_dir))
            job_count += 1
        else:
            logger.info("""">>> 开始执行，第 {} 个文件
                    input file: {}
                    output file: {}
                    """.format(index, input_file, output_dir))
            spawn(lock, index, input_file, output_dir)
        # print(res.get(timeout=1))
    if is_multiprocess:
        pool.close()
        logger.info("Now the pool is closed and no longer available")
        logger.info("===== 进程池 任务数量一共: {}, 开始多进程执行 =====".format(job_count))
        pool.join()

    logger.info("===== [END] =====")


def main(_):
    run_loop(flags.FLAGS)


if __name__ == '__main__':
    # test()
    define_flags()
    absl_app.run(main)
