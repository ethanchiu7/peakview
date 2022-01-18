# -*- coding: utf-8 -*-
"""
    Author  ：   Ethan Chiu
    Time    ：   2021/12/22 上午11:34
    Site    :   
    Suggestion  ：
    Description :
    File    :   tf_embed.py
    Software    :   PyCharm
"""

import tensorflow as tf


class StaticHashTableLayer(object):
    def __init__(self, vocabulary_token_list=None, vocabulary_index_list=None, default_index_start=0, default_value=-1,
                 lookup_key_dtype=tf.string):
        self.vocabulary_token_list = vocabulary_token_list
        self.lookup_key_dtype = lookup_key_dtype
        self.vocabulary_index_list = vocabulary_index_list
        if not self.vocabulary_index_list and self.vocabulary_token_list:
            self.vocabulary_index_list = [i + default_index_start for i in range(len(self.vocabulary_token_list))]

        self.built = False

    def build(self):
        assert self.vocabulary_token_list and self.vocabulary_index_list
        table_initializer_key = tf.constant(self.vocabulary_token_list, dtype=self.lookup_key_dtype)
        table_initializer_value = tf.constant(self.vocabulary_index_list, dtype=tf.int64)
        table_initializer = tf.lookup.KeyValueTensorInitializer(keys=table_initializer_key, values=table_initializer_value)
        # table = tf.lookup.StaticVocabularyTable(table_initializer, self.num_oov_buckets)
        table = tf.lookup.StaticHashTable(table_initializer, default_value=-1)

        self.table = table
        self.built = True

    def __call__(self, input_tensor, out_dtype=tf.int32):
        if not self.built:
            self.build()
        out = self.table.lookup(input_tensor)
        out = tf.cast(out, out_dtype)
        return out


class StaticVocabularyTableLayer(object):
    def __init__(self, vocabulary_token_list=None, vocabulary_index_list=None, default_index_start=0, num_oov_buckets=1e5,
                 lookup_key_dtype=tf.string):
        self.vocabulary_token_list = vocabulary_token_list
        self.lookup_key_dtype = lookup_key_dtype
        self.vocabulary_index_list = vocabulary_index_list
        if not self.vocabulary_index_list and self.vocabulary_token_list:
            self.vocabulary_index_list = [i + default_index_start for i in range(len(self.vocabulary_token_list))]
        self.num_oov_buckets = num_oov_buckets

        self.built = False

    def build(self):
        if self.vocabulary_token_list and self.vocabulary_index_list:
            table_initializer_key = tf.constant(self.vocabulary_token_list, dtype=self.lookup_key_dtype)
            table_initializer_value = tf.constant(self.vocabulary_index_list, dtype=tf.int64)
            table_initializer = tf.lookup.KeyValueTensorInitializer(keys=table_initializer_key, values=table_initializer_value)
            table = tf.lookup.StaticVocabularyTable(table_initializer, self.num_oov_buckets)
            # table = tf.lookup.StaticHashTable(table_initializer, default_value=[-1, -1])
        else:
            table = tf.lookup.StaticVocabularyTable(None, num_oov_buckets=self.num_oov_buckets, lookup_key_dtype=self.lookup_key_dtype)

        self.table = table
        self.built = True

    def __call__(self, input_tensor, out_dtype=tf.int32):
        if not self.built:
            self.build()
        out = self.table.lookup(input_tensor)
        out = tf.cast(out, out_dtype)
        return out


def test_IndexConverter():
    """
        num_oov_buckets = 3
        input_tensor = tf.constant(["emerson", "lake", "palmer", "king", "crimnson"])
        table = tf.lookup.StaticVocabularyTable(
        tf.TextFileIdTableInitializer(filename), num_oov_buckets)
        out = table.lookup(input_tensor).
        table.init.run()
        print(out.eval())

    """

    vocabulary_list = ['PAD', 'a', 'b', 'c']
    inputs = ['b', 'PAD', 'x', 'xx', 'a']
    vocabulary_list = [11, 12, 13, 14]
    # vocabulary_list = None
    vocabulary_index_list = [11, 22, 33, 44]
    # vocabulary_index_list = None
    inputs = [11, 12, 14, 8]
    table_layer = StaticVocabularyTableLayer(vocabulary_list, vocabulary_index_list=vocabulary_index_list,
                                             default_index_start=0, num_oov_buckets=4, lookup_key_dtype=tf.int64)
    inputs = tf.constant(inputs, tf.int64)
    out_op = table_layer(inputs)
    print(out_op)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.enable_eager_execution()
    test_IndexConverter()




