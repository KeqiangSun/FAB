from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def l1_loss(predictions, targets):
  total_elements = (tf.shape(targets)[0] * tf.shape(targets)[1] * tf.shape(targets)[2]
      * tf.shape(targets)[3])
  total_elements = tf.to_float(total_elements)

  loss = tf.reduce_sum(tf.abs(predictions- targets))
  loss = tf.div(loss, total_elements)

  return loss


def l2_loss(predictions, targets):
  total_elements = (tf.shape(targets)[0] * tf.shape(targets)[1] * tf.shape(targets)[2]
      * tf.shape(targets)[3])
  total_elements = tf.to_float(total_elements)

  loss = tf.reduce_sum(tf.square(predictions-targets))
  loss = tf.div(loss, total_elements)

  return loss

