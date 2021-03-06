# -*- coding: utf-8 -*-
"""
    Author  ：   Ethan Chiu
    Time    ：   2021/4/8 下午3:55
    Site    :   
    Suggestion  ：
    Description :
    File    :   tf_util.py
    Software    :   PyCharm
"""
import six
import json
import numpy as np
import tensorflow as tf


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, None)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def count_variable_parameter_size(variables):
    """
    计算变量的参数规模
    :param variables:
    :return:
    """
    if not variables:
        variables = tf.compat.v1.trainable_variables()
    total_parameters = 0
    for variable in variables:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
    return total_parameters


def bucket_by_boundaries(inputs, boundaries):
    """
    t = tf.constant([[1.3, 2.0], [2.99, 9.9]], tf.float32)
    b = [2, 3, 8]
    print(bucket_by_boundaries(t, b))
    ...
    tf.Tensor(
    [[0 1]
    [1 3]], shape=(2, 2), dtype=int32)
    :param inputs:
    :param boundaries:
    :return:
    """
    input_shape = tf.shape(inputs)
    batch = get_shape_list(inputs)[0]
    vocab_size = len(boundaries) + 1

    source = tf.reshape(inputs, [-1, 1])
    # source = tf.expand_dims(inputs, axis=-1)
    bucket_table = tf.constant([tf.float32.min] + list(boundaries), tf.float32, name="bucket_table")
    bucket_table = tf.expand_dims(bucket_table, axis=0)
    tile_multiples = tf.shape(source)
    bucket_table = tf.tile(bucket_table, tile_multiples)

    compared_table = tf.cast((source >= bucket_table), tf.int32)
    bucket_index = tf.reduce_sum(compared_table, axis=-1) - 1
    output = tf.reshape(bucket_index, input_shape)
    return output


def test_bucket_by_boundaries():
    t = tf.constant([[1.3, 2.0], [2.99, 9.9]], tf.float32)
    b = [2, 3, 8]
    print(bucket_by_boundaries(t, b))


def one_hot_lookup_table(input_ids, table):
    """
    This vocab will be small so we always do one-hot here, since it is always faster for a small vocabulary.
    i = tf.constant([[1, 2], [3, 4]], tf.int32)
    t = [[0.2, 0.4], [1.4, 1.5], [2.4, 2.8], [3.4, 3.8], [4.4, 4.8]]
    print(one_hot_lookup_table(i, t))
    tf.Tensor(
        [[[1.4 1.5]
        [2.4 2.8]]

        [[3.4 3.8]
        [4.4 4.8]]], shape=(2, 2, 2), dtype=float32)
    :param input_ids:
    :param table:
    :return:
    """
    input_shape = get_shape_list(input_ids)
    table_shape = get_shape_list(table)
    output_shape = tf.concat([input_shape, tf.expand_dims(table_shape[-1], 0)], axis=0)
    flat_input_ids = tf.reshape(input_ids, [-1])
    one_hot_ids = tf.one_hot(flat_input_ids, depth=table_shape[0])
    input_embedding = tf.matmul(one_hot_ids, table)
    input_embedding = tf.reshape(input_embedding, output_shape)
    return input_embedding


def normalize_with_moments(x, axes=None, epsilon=1e-8):
    """
    z-score
    :param x:
    :param axes:
    :param epsilon:
    :return:
    """
    if axes is None:
        axes = [0, 1]
    mean, variance = tf.nn.moments(x, axes=axes, keepdims=False)
    x_normed = (x - mean) / tf.sqrt(variance + epsilon) # epsilon to avoid dividing by zero
    return x_normed


def batch_accuracy_binary(prob, label, weights):
    prob = tf.reshape(prob, [-1])
    label = tf.reshape(label, [-1])
    predictions = tf.cast(prob >= 0.5, tf.float32)
    accuracy = tf.cast(tf.equal(predictions, label), tf.float32)
    accuracy = accuracy * weights
    return tf.reduce_mean(accuracy)


def convert_tensors_to_ndarrays(inputs):
    outputs = {}
    for j in inputs:
        outputs[j] = inputs[j].numpy()
    return outputs


def convert_ndarrays_to_lists(inputs):
    outputs = {}
    for j in inputs:
        if isinstance(inputs[j], np.ndarray):
            outputs[j] = list(map(lambda x: str(x.decode()) if isinstance(x, bytes) else x.item(), inputs[j].tolist()))
        else:

            outputs[j] = inputs[j].decode() if isinstance(inputs[j], bytes) else inputs[j].item()
    return outputs


def convert_ndarrays_to_strings(inputs):
    outputs = {}
    for j in inputs:
        if isinstance(inputs[j], np.ndarray):
            outputs[j] = ";".join(map(lambda x: str(x.decode()) if isinstance(x, bytes) else str(x), inputs[j].tolist()))
        else:
            outputs[j] = str(inputs[j].item())
    return outputs


def write_eval_result(result, output_eval_file):
    """
    output_eval_file = os.path.join(FLAGS.model_dir, "eval_results.txt")
    if FLAGS.run_mode == RunMode.EVAL.value:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        result = estimator.evaluate(input_fn=get_input_fn_eval())
        tf_util.write_eval_result(result, output_eval_file)
    :param result:
    :param output_eval_file:
    :return:
    """
    with tf.gfile.GFile(output_eval_file, "w") as writer:
        tf.logging.info("***** Eval results *****\n output_eval_file: {}".format(output_eval_file))
        for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def parse_and_record_predict_result(predict_result_it, output_predict_file, num_actual_predict_examples=None,
                                    to_csv=True, csv_sep=" "):
    """
    output_predict_file = os.path.join(FLAGS.model_dir, "predict_results.txt")
    if FLAGS.run_mode == RunMode.PREDICT.value:
        tf.logging.info("***** Running prediction *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        predict_result_it = estimator.predict(input_fn=get_input_fn_predict(), yield_single_examples=True)
        tf_util.parse_and_record_predict_result(predict_result_it, output_predict_file, FLAGS.num_actual_predict_examples)
    :param predict_result_it:
    :param output_predict_file:
    :param num_actual_predict_examples:
    :return:
    """

    def parse_value(v):
        if isinstance(v, bytes):
            return v.decode()
        elif isinstance(v, np.ndarray):
            return str(v.item())
        else:
            return str(v)

    with tf.gfile.GFile(output_predict_file, "w") as writer:
        num_written_lines = 0
        tf.logging.info("***** Predict results *****\n  output_predict_file: {}".format(output_predict_file))
        for (i, prediction) in enumerate(predict_result_it):
            if i % 100000 == 0:
                print("---- i : {} ----".format(i))
            if num_actual_predict_examples and num_actual_predict_examples > 0 and num_written_lines >= num_actual_predict_examples:
                break
            if to_csv:
                if i == 0:
                    writer.write(csv_sep.join([k for k in prediction]) + "\n")
                writer.write(csv_sep.join([parse_value(prediction[k]) for k in prediction]) + "\n")
            else:
                output_line = json.dumps(convert_ndarrays_to_lists(prediction), ensure_ascii=False)
                output_line += "\n"
                writer.write(output_line)
            num_written_lines += 1


def deepfm(embedded, seq_mask=None):
    """

    :param embedded: [batch, seq, embedding]
    :param seq_mask: [batch, seq]
    :return:
    """
    if seq_mask:
        embedded = tf.multiply(embedded, seq_mask)
    # [batch, embedding_size]
    left = tf.square(tf.reduce_sum(embedded, 1))
    # [batch, embedding_size]
    right = tf.reduce_sum(tf.square(embedded), 1)
    # [batch, embedding_size]
    fm = 0.5 * (left - right)
    return fm


def scale_l2_2d(x, norm_length):
    """

    :param x: 2D tensor
    :param norm_length:
    :return:
    """
    alpha = tf.reduce_max(tf.abs(x), 1, keep_dims=True) + 1e-12
    l2_norm = alpha * tf.sqrt(
        tf.reduce_sum(tf.pow(x / alpha, 2), 1, keep_dims=True) + 1e-6)
    x_unit = x / l2_norm
    return norm_length * x_unit


def scale_l2_3d(x, norm_length):
    """

    :param x: 3D tensor
    :param norm_length:
    :return:
    """
    # shape(x) = (batch, num_timesteps, d)
    # Divide x by max(abs(x)) for a numerically stable L2 norm.
    # 2norm(x) = a * 2norm(x/a)
    # Scale over the full sequence, dims (1, 2)
    alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
    l2_norm = alpha * tf.sqrt(
        tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
    x_unit = x / l2_norm
    return norm_length * x_unit


def test_one_hot_lookup_table():
    i = tf.constant([[1, 2], [3, 4]], tf.int32, name="i")
    t = [[0.2, 0.4], [1.4, 1.5], [2.4, 2.8], [3.4, 3.8], [4.4, 4.8]]
    t = tf.constant(t, name="t")
    print(one_hot_lookup_table(i, t))


if __name__ == '__main__':
    tf.enable_eager_execution()
    test_bucket_by_boundaries()
    # test_one_hot_lookup_table()