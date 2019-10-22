from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def bilinear_interp(im, x, y, name):
  with tf.variable_scope(name):
    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])

    num_batch = tf.shape(im)[0]
    _, height, width, channels = im.get_shape().as_list()

    x = tf.to_float(x)
    y = tf.to_float(y)

    height_f = tf.cast(height, 'float32')
    width_f = tf.cast(width, 'float32')
    zero = tf.constant(0, dtype=tf.int32)

    max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')
    max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
    x = (x + 1.0) * (width_f - 1.0) / 2.0
    y = (y + 1.0) * (height_f - 1.0) / 2.0

    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    dim2 = width
    dim1 = width * height

    base = tf.range(num_batch) * dim1
    base = tf.reshape(base, [-1, 1])
    base = tf.tile(base, [1, height * width])
    base = tf.reshape(base, [-1])

    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    im_flat = tf.reshape(im, tf.stack([-1, channels]))
    im_flat = tf.to_float(im_flat)
    pixel_a = tf.gather(im_flat, idx_a)
    pixel_b = tf.gather(im_flat, idx_b)
    pixel_c = tf.gather(im_flat, idx_c)
    pixel_d = tf.gather(im_flat, idx_d)

    x1_f = tf.to_float(x1)
    y1_f = tf.to_float(y1)

    wa = tf.expand_dims((x1_f - x) * (y1_f - y), 1)
    wb = tf.expand_dims((x1_f - x) * (1.0 - (y1_f - y)), 1)
    wc = tf.expand_dims((1.0 - (x1_f - x)) * (y1_f - y), 1)
    wd = tf.expand_dims((1.0 - (x1_f - x)) * (1.0 - (y1_f - y)), 1)

    output = tf.add_n([wa*pixel_a, wb*pixel_b, wc*pixel_c, wd*pixel_d])
    output = tf.reshape(output, shape=tf.stack([num_batch, height, width, channels]))

    return output

def meshgrid(height, width):
  with tf.variable_scope('meshgrid'):
    x_t = tf.matmul(
        tf.ones(shape=tf.stack([height,1])),
        tf.transpose(
              tf.expand_dims(
                  tf.linspace(-1.0,1.0,width),1),[1,0]))
    y_t = tf.matmul(
        tf.expand_dims(
            tf.linspace(-1.0, 1.0, height), 1),
        tf.ones(shape=tf.stack([1, width])))
    x_t_flat = tf.reshape(x_t, (1,-1))
    y_t_flat = tf.reshape(y_t, (1,-1))
    grid_x = tf.reshape(x_t_flat, [1, height, width])
    grid_y = tf.reshape(y_t_flat, [1, height, width])

    return grid_x, grid_y

