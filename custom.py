from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops

import tensorflow as tf


def _sum_rows(x):
  """Returns a vector summing up each row of the matrix x."""
  # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
  # a matrix.  The gradient of _sum_rows(x) is more efficient than
  # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
  # we use _sum_rows(x) in the nce_loss() computation since the loss
  # is mostly used for training.
  cols = array_ops.shape(x)[1]
  ones_shape = array_ops.stack([cols, 1])
  ones = array_ops.ones(ones_shape, x.dtype)
  return array_ops.reshape(math_ops.matmul(x, ones), [-1])

def _compute_true_logits(inputs, true_w, num_true):
    dim = array_ops.shape(true_w)[1:2]
    new_true_w_shape = array_ops.concat([[-1, num_true], dim], 0)
    row_wise_dots = math_ops.multiply(
        array_ops.expand_dims(inputs, 1),
        array_ops.reshape(true_w, new_true_w_shape))
    dots_as_matrix = array_ops.reshape(row_wise_dots,
                                       array_ops.concat([[-1], dim], 0))
    true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true])
    return true_logits

def _compute_sampled_logits(embedding,
                            biases,
                            labels,
                            inputs,
                            num_sampled,
                            num_classes,
                            num_true=1,
                            sampled_values=None,
                            subtract_log_q=True,
                            remove_accidental_hits=False,
                            partition_strategy="mod",
                            name=None):
  if biases is not None and not isinstance(biases, list):
    biases = [biases]
  if not isinstance(labels, list):
    labels = [labels]
  if not isinstance(num_classes, list):
    num_classes = [num_classes]
  with ops.name_scope(name, "compute_sampled_logits",
                      [embedding, biases, inputs, labels]):
    for i in range(len(labels)):
        if labels[i].dtype != dtypes.int64:
            labels[i] = math_ops.cast(labels[i], dtypes.int64)
    labels_flat = [array_ops.reshape(label, [-1]) for label in labels]

    if sampled_values is None:
      sampled_values = [candidate_sampling_ops.log_uniform_candidate_sampler(true_classes=true_classes,
                                                                             num_true=num_true,
                                                                             num_sampled=num_sampled,
                                                                             unique=True,
                                                                             range_max=range_max)
                        for true_classes, range_max in zip(labels, num_classes)]

    true_w = embedding(*labels_flat)
    true_logits = _compute_true_logits(inputs, true_w, num_true)

    sampled_w = embedding(*[sample[0] for sample in sampled_values])
    sampled_logits = math_ops.matmul(inputs, sampled_w, transpose_b=True)

    if biases is not None:
        true_b = math_ops.add_n([embedding_ops.embedding_lookup(bias, label)
                                 for bias, label in zip(biases, labels_flat)])
        true_b = array_ops.reshape(true_b, [-1, num_true])
        true_logits += true_b

        sampled_b = math_ops.add_n([embedding_ops.embedding_lookup(bias, sample[0])
                                   for bias, sample in zip(biases, sampled_values)])
        sampled_logits += sampled_b

    if subtract_log_q:
      # Subtract log of Q(l), prior probability that l appears in sampled.
      true_logits -= math_ops.add_n([math_ops.log(sample[1]) for sample in sampled_values])
      sampled_logits -= math_ops.add_n([math_ops.log(sample[2]) for sample in sampled_values])

    # Construct output logits and labels. The true labels/logits start at col 0.
    out_logits = array_ops.concat([true_logits, sampled_logits], 1)
    # true_logits is a float tensor, ones_like(true_logits) is a float tensor
    # of ones. We then divide by num_true to ensure the per-example labels sum
    # to 1.0, i.e. form a proper probability distribution.
    out_labels = array_ops.concat([
        array_ops.ones_like(true_logits) / num_true,
        array_ops.zeros_like(sampled_logits)
    ], 1)

  return out_logits, out_labels


def nce_loss(weights,
             biases,
             labels,
             inputs,
             num_sampled,
             num_classes,
             num_true=1,
             sampled_values=None,
             remove_accidental_hits=False,
             partition_strategy="mod",
             name="nce_loss"):
  logits, labels = _compute_sampled_logits(
      embedding=weights,
      biases=biases,
      labels=labels,
      inputs=inputs,
      num_sampled=num_sampled,
      num_classes=num_classes,
      num_true=num_true,
      sampled_values=sampled_values,
      subtract_log_q=True,
      remove_accidental_hits=remove_accidental_hits,
      partition_strategy=partition_strategy,
      name=name)
  sampled_losses = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=labels, logits=logits, name="sampled_losses")
  # sampled_losses is batch_size x {true_loss, sampled_losses...}
  # We sum out true and sampled losses.
  return _sum_rows(sampled_losses)
