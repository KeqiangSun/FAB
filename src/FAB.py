from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import datetime
import numpy as np
import os
import time
import math
import skimage.io
import skimage.transform

from utils.loss_utils import l2_loss
from utils.geo_layer_utils import bilinear_interp
from utils.geo_layer_utils import meshgrid
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from utils.config import Config


class FAB(object):
    def __init__(self, structure_predictor_is_train=True, deblur_is_train=True,
                 resnet_is_train=True, is_training=True,
                 MOVING_AVERAGE_DECAY=0.9997, BN_EPSILON=0.001,
                 CONV_WEIGHT_DECAY=0.0005, CONV_WEIGHT_STDDEV=0.1,
                 FC_WEIGHT_DECAY=0.0005, FC_WEIGHT_STDDEV=0.01,
                 RESNET_VARIABLES='RESNET_VARIABLES',
                 UPDATE_OPS_COLLECTION='resnet_update_ops',
                 IMAGENET_MEAN_BGR=[103.062623801, 115.902882574, 123.151630838, ],
                 input_size = 224):

        self.structure_predictor_is_train = structure_predictor_is_train
        self.deblur_is_train = deblur_is_train
        self.resnet_is_train = resnet_is_train

        self.MOVING_AVERAGE_DECAY = MOVING_AVERAGE_DECAY
        self.BN_DECAY = self.MOVING_AVERAGE_DECAY
        self.BN_EPSILON = BN_EPSILON
        self.CONV_WEIGHT_DECAY = CONV_WEIGHT_DECAY
        self.CONV_WEIGHT_STDDEV = CONV_WEIGHT_STDDEV
        self.FC_WEIGHT_DECAY = FC_WEIGHT_DECAY
        self.FC_WEIGHT_STDDEV = FC_WEIGHT_STDDEV
        self.RESNET_VARIABLES = RESNET_VARIABLES
        self.UPDATE_OPS_COLLECTION = UPDATE_OPS_COLLECTION
        self.IMAGENET_MEAN_BGR = IMAGENET_MEAN_BGR
        self.input_size = input_size

### loss function ###
    def l1_loss_(self, logits, labels):
        logits = tf.cast(logits,tf.float32)
        labels = tf.cast(labels,tf.float32)
        losses = tf.reduce_sum(tf.abs(tf.subtract(logits,labels)), axis=1)
        losses_mean = tf.reduce_mean(losses)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_ = tf.add_n([losses_mean] + regularization_losses)

        return loss_

    def l2_loss_(self, logits, labels):
        logits = tf.cast(logits,tf.float32)
        labels = tf.cast(labels,tf.float32)
        losses = tf.nn.l2_loss(tf.subtract(logits,labels))
        losses_mean = tf.reduce_mean(losses)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_ = tf.add_n([losses_mean] + regularization_losses)

        return loss_

    def wing_loss(self, logits, labels, w=10.0, epsilon=2.0):
        logits = tf.cast(logits,tf.float32)
        labels = tf.cast(labels,tf.float32)
        x = tf.subtract(logits,labels)
        C = w * (1.0 - math.log(1.0 + w/epsilon))
        absolute_x = tf.abs(x)
        losses = tf.where(tf.greater(w, absolute_x),
                          w * tf.log(1.0 + absolute_x/epsilon),
                          absolute_x - C)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_ = tf.add_n([losses] + regularization_losses)

        return loss_

    def calculate_NME(self, logits, labels):
        logits = tf.cast(logits,tf.float32)
        labels = tf.cast(labels,tf.float32)

        subtract_square_distance = tf.square(tf.subtract(logits, labels))
        mean_distance = tf.reduce_mean([tf.sqrt(tf.add(subtract_square_distance[:, column],
                                                       subtract_square_distance[:, column+1])) for column in range(0, 136, 2)], axis=0)

        outer_eye_x = tf.square(tf.subtract(labels[:, 72], labels[:, 90]))
        outer_eye_y = tf.square(tf.subtract(labels[:, 73], labels[:, 91]))
        inter_ocular_distance = tf.sqrt(tf.add(outer_eye_x, outer_eye_y))

        normalized_mean_error = tf.divide(mean_distance, inter_ocular_distance,
                                          name='normalized_mean_error')
        loss_ = tf.reduce_mean(normalized_mean_error)

        return loss_

### structure predictor model ###
    def structure_predictor_inference(self,input_images_boundary,batch_size):
        with tf.variable_scope('structure_predictor_model_'):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0001)):

                batch_norm_params = {'decay': 0.9997,
                                     'epsilon': 0.0001,
                                     'is_training': self.structure_predictor_is_train}

                with slim.arg_scope([slim.batch_norm],
                                    is_training = self.structure_predictor_is_train,
                                    updates_collections=None):
                    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
                        net = slim.conv2d(input_images_boundary, 64, [5, 5], stride=1, scope='conv1')
                        net = slim.max_pool2d(net, [2, 2], scope='pool1')
                        net = slim.conv2d(net, 128, [5, 5], stride=1, scope='conv2')
                        net = slim.max_pool2d(net, [2, 2], scope='pool2')
                        net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv3')
                        net = slim.max_pool2d(net, [2, 2], scope='pool3')
                        net = tf.image.resize_bilinear(net, [64,64])
                        net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv4')
                        net = tf.image.resize_bilinear(net, [128,128])
                        net = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv5')
                        net = tf.image.resize_bilinear(net, [256,256])
                        net = slim.conv2d(net, 64, [5, 5], stride=1, scope='conv6')

            net = slim.conv2d(net, 3, [5, 5], stride=1, activation_fn=tf.tanh,
                              normalizer_fn=None, scope='conv7')
            flow = net[:, :, :, 0:2]
            mask = tf.expand_dims(net[:, :, :, 2], 3)

            grid_x, grid_y = meshgrid(256, 256)
            grid_x = tf.tile(grid_x, [batch_size, 1, 1])
            grid_y = tf.tile(grid_y, [batch_size, 1, 1])

            coor_x_1 = grid_x + flow[:, :, :, 0]*2
            coor_y_1 = grid_y + flow[:, :, :, 1]*2
            coor_x_2 = grid_x + flow[:, :, :, 0]
            coor_y_2 = grid_y + flow[:, :, :, 1]

            output_1 = bilinear_interp(input_images_boundary[:, :, :, 0:1],
                                       coor_x_1, coor_y_1, 'extrapolate')
            output_2 = bilinear_interp(input_images_boundary[:, :, :, 1:2],
                                       coor_x_2, coor_y_2, 'extrapolate')

            mask = 0.33 * (1.0 + mask)
            mask = tf.tile(mask, [1, 1, 1, 3])
            next_frame = tf.multiply(mask, output_1) + tf.multiply(1.0 - mask, output_2)

            return next_frame

### video deblur function ###
    def get_shape(self, x, i):
        return x.get_shape().as_list()[i]

    def weight_variable(self, shape, stddev=0.02, name = 'weight'):
        w = tf.get_variable(name, shape,
                            initializer=tf.random_normal_initializer(stddev=stddev),
                            trainable=self.deblur_is_train)
        return w

    def bias_variable(self, shape, name):
        b = tf.get_variable(name, initializer = tf.zeros(shape),
                            trainable= self.deblur_is_train)
        return b

    def conv2d(self, x, W, stride = 1):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

    def conv2d_transpose(self, x, w, output_shape, stride = 2):
        return tf.nn.conv2d_transpose(x, w, output_shape=output_shape,
                                      strides=[1, stride, stride, 1], padding='SAME')

    def bn(self, x):
        net = x
        out_channels = self.get_shape(net, 3)
        mean, var = tf.nn.moments(net, axes=[0,1,2])
        beta = self.bias_variable([out_channels], name="beta")
        gamma = self.weight_variable([out_channels], name="gamma")
        net = tf.nn.batch_normalization(net, mean, var, beta, gamma, 0.001)
        return net

    def conv_bn(self, x, filter_shape):
        net = x
        net = tf.nn.conv2d(net, self.weight_variable(filter_shape, name = "weight"),
                           strides=[1, 1, 1, 1], padding="SAME")
        out_channels = filter_shape[3]
        mean, var = tf.nn.moments(net, axes=[0,1,2])
        beta = self.bias_variable([out_channels], name="beta")
        gamma = self.weight_variable([out_channels], name="gamma")
        net = tf.nn.batch_normalization(net, mean, var, beta, gamma, 0.001)
        return net

    def resnet_block(self, x, out_channel, filter_size = 3):
        x_channel = x.get_shape().as_list()[3]
        with tf.variable_scope("conv_bn_relu"):
            net = self.conv_bn(x, filter_shape=[filter_size,
                                                filter_size,
                                                out_channel,
                                                out_channel])
            net = tf.nn.relu(net)
        with tf.variable_scope("conv_bn"):
            net = self.conv_bn(net, filter_shape=[filter_size,
                                                  filter_size,
                                                  out_channel,
                                                  out_channel])
        net = net + x
        tf.nn.relu(net)
        return net

    def dynamic_fusion(self, x, h, filter_size = 5):
        n_channel = self.get_shape(x, 3)
        t = tf.concat([x, h], 3)
        similarity = tf.nn.conv2d(t, self.weight_variable([filter_size,
                                                           filter_size,
                                                           n_channel*2,
                                                           n_channel],
                                                           name = "wt"),
                                                           strides=[1, 1, 1, 1],
                                                           padding='VALID')
        epsilon = self.bias_variable([1], name = 'bias_epsilon')
        alpha = 2*tf.abs(tf.sigmoid(similarity) - 0.5) + epsilon
        alpha = tf.clip_by_value(alpha, 0, 1)
        hflt_filter_size = filter_size // 2
        alpha = tf.pad(alpha-1, [[0, 0],
                                 [hflt_filter_size, hflt_filter_size],
                                 [hflt_filter_size, hflt_filter_size],
                                 [0, 0]], "CONSTANT") + 1
        y = alpha*x + (1-alpha)*h
        return y, alpha

    def video_deblur_inference(self, X, F, H, net_channel = 64):
        with tf.variable_scope('video_deblur_model_'):
            H_curr = []
            with tf.variable_scope("encoding"):
                with tf.variable_scope("conv1"):
                    filter_size = 5
                    net_X = self.conv2d(X, self.weight_variable([filter_size,
                                                                 filter_size,
                                                                 self.get_shape(X, 3),
                                                                 net_channel]))
                    net_X = tf.nn.relu(net_X)
                with tf.variable_scope("conv2"):
                    filter_size = 3
                    net_X = self.conv2d(net_X, self.weight_variable([filter_size,
                                                                     filter_size,
                                                                     self.get_shape(net_X, 3),
                                                                     net_channel//2]),
                                                                     stride = 2)
                    net_X = tf.nn.relu(net_X)
                net = tf.concat([net_X, F], 3)
                f0 = net
                filter_size = 3
                num_resnet_layers = 8
                for i in range (num_resnet_layers):
                    with tf.variable_scope('resnet_block%d' % (i+1)):
                        net = self.resnet_block(net, net_channel)
                        if i == 3:
                            (net, alpha) = self.dynamic_fusion(net, H[0])
                            h = tf.expand_dims(net, axis=0)
                            H_curr = h
            with tf.variable_scope("feat_out"):
                F = self.conv2d(net, self.weight_variable([filter_size,
                                                           filter_size,
                                                           self.get_shape(net, 3),
                                                           net_channel//2],
                                                           name = 'conv_F'))
                F = tf.nn.relu(F)
            with tf.variable_scope("img_out"):
                filter_size = 4
                shape = [self.get_shape(X, 0),
                         self.get_shape(X, 1),
                         self.get_shape(X, 2),
                         net_channel]
                Y = self.conv2d_transpose(net, self.weight_variable([filter_size,
                                                                     filter_size,
                                                                     net_channel,
                                                                     net_channel],
                                                                     name = "deconv"),
                                                                     shape,
                                                                     stride = 2)
                Y = tf.nn.relu(Y)
                filter_size = 3
                Y = self.conv2d(Y, self.weight_variable([filter_size,
                                                         filter_size,
                                                         self.get_shape(Y, 3),
                                                         3],
                                                         name = 'conv'))
            return Y, F, H_curr

### resnet inference ###
    def resnet_inference(self,
                        input_images_blur,
                        batch_size,
                        num_classes=136,
                        num_blocks=[2, 2, 2, 2],
                        use_bias=False,
                        bottleneck=True,
                        dropout_ratio=1.0):
    ####resnet_model####
        with tf.variable_scope('resnet_model_'):
            c = Config()
            c['bottleneck'] = bottleneck
            c['is_training'] = tf.convert_to_tensor(self.resnet_is_train,
                                                    dtype='bool',
                                                    name='is_training')
            c['ksize'] = 3
            c['stride'] = 1
            c['use_bias'] = use_bias
            c['fc_units_out'] = num_classes
            c['num_blocks'] = num_blocks
            c['stack_stride'] = 2

            with tf.variable_scope('scale1'):
                c['conv_filters_out'] = 16
                c['ksize'] = 7
                c['stride'] = 2
                x = self.conv(input_images_blur, c)
                x = self.resnet_bn(x, c)
                x = self.activation(x)

            with tf.variable_scope('scale1_pool'):
                x = self._max_pool(x, ksize=3, stride=2)
                x = self.resnet_bn(x, c)
                x = self.activation(x)

            with tf.variable_scope('scale2'):
                x = self._max_pool(x, ksize=3, stride=2)
                c['num_blocks'] = num_blocks[0]
                c['stack_stride'] = 1
                c['block_filters_internal'] = 8
                x = self.stack(x, c)

            with tf.variable_scope('scale3'):
                c['num_blocks'] = num_blocks[1]
                c['block_filters_internal'] = 16
                assert c['stack_stride'] == 2
                x = self.stack(x, c)

            with tf.variable_scope('scale4'):
                c['num_blocks'] = num_blocks[2]
                c['block_filters_internal'] = 32
                x = self.stack(x, c)

            with tf.variable_scope('scale5'):
                c['num_blocks'] = num_blocks[3]
                c['block_filters_internal'] = 64
                x = self.stack(x, c)

            x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

            if num_classes != None:
                with tf.variable_scope('fc1'):
                    c['fc_units_out'] = 256
                    x = self.fc(x, c)

                with tf.variable_scope('dropout1'):
                    x = tf.nn.dropout(x, dropout_ratio)

                with tf.variable_scope('fc2'):
                    c['fc_units_out'] = 256
                    x = self.fc(x, c)

                with tf.variable_scope('dropout2'):
                    x = tf.nn.dropout(x, dropout_ratio)

                with tf.variable_scope('fc3'):
                    c['fc_units_out'] = 136
                    landmark_localization = self.fc(x, c)

        return landmark_localization

    def stack(self, x, c):
        for n in range(c['num_blocks']):
            s = c['stack_stride'] if n == 0 else 1
            c['block_stride'] = s
            with tf.variable_scope('block%d' % (n + 1)):
                x = self.block(x, c, n)
        return x

    def block(self, x, c, n):
        filters_in = x.get_shape()[-1]
        m = 4 if c['bottleneck'] else 1
        filters_out = m * c['block_filters_internal']
        c['conv_filters_out'] = c['block_filters_internal']

        shortcut = x

        if c['bottleneck']:
            if n == 1:
                with tf.variable_scope('pre_activation'):
                    x = self.resnet_bn(x, c)
                    x = self.activation(x)

            with tf.variable_scope('a'):
                c['ksize'] = 1
                c['stride'] = c['block_stride']
                x = self.conv(x, c)
                x = self.resnet_bn(x, c)
                x = self.activation(x)

            with tf.variable_scope('b'):
                x = self.conv(x, c)
                x = self.resnet_bn(x, c)
                x = self.activation(x)

            with tf.variable_scope('c'):
                c['conv_filters_out'] = filters_out
                c['ksize'] = 1
                assert c['stride'] == 1
                x = self.conv(x, c)
        else:
            with tf.variable_scope('A'):
                c['stride'] = c['block_stride']
                assert c['ksize'] == 3
                x = self.conv(x, c)
                x = self.resnet_bn(x, c)
                x = self.activation(x)

            with tf.variable_scope('B'):
                c['conv_filters_out'] = filters_out
                assert c['ksize'] == 3
                assert c['stride'] == 1
                x = self.conv(x, c)
                x = self.resnet_bn(x, c)

        with tf.variable_scope('shortcut'):
            if filters_out != filters_in or c['block_stride'] != 1:
                c['ksize'] = 1
                c['stride'] = c['block_stride']
                c['conv_filters_out'] = filters_out
                shortcut = self.conv(shortcut, c)

        if n == 0:
            return x + shortcut
        elif n == 1:
            x = self.resnet_bn(x+shortcut, c)
            return self.activation(x)

    def resnet_bn(self, x, c):
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]

        if c['use_bias']:
            bias = self._get_variable('bias',
                                      params_shape,
                                      initializer=tf.zeros_initializer)
            return x + bias

        axis = list(range(len(x_shape) - 1))
        beta = self._get_variable('beta',
                            params_shape,
                            initializer=tf.zeros_initializer)
        gamma = self._get_variable('gamma',
                            params_shape,
                            initializer=tf.ones_initializer)

        moving_mean = self._get_variable('moving_mean',
                                    params_shape,
                                    initializer=tf.zeros_initializer,
                                    trainable=False)
        moving_variance = self._get_variable('moving_variance',
                                        params_shape,
                                        initializer=tf.ones_initializer,
                                        trainable=False)

        # These ops will only be preformed when training.
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, self.BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, self.BN_DECAY)

        tf.add_to_collection(self.UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(self.UPDATE_OPS_COLLECTION, update_moving_variance)

        mean, variance = control_flow_ops.cond(
            c['is_training'], lambda: (mean, variance),
            lambda: (moving_mean, moving_variance))

        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, self.BN_EPSILON)

        return x

    def activation(self, x):
        alphas = tf.get_variable('alpha', x.get_shape()[-1],
                            initializer=tf.constant_initializer(0.25),
                            dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alphas * (x - abs(x)) * 0.5

        return pos + neg

    def fc(self, x, c):
        num_units_in = x.get_shape()[1]
        num_units_out = c['fc_units_out']
        weights_initializer = tf.truncated_normal_initializer(
            stddev=self.FC_WEIGHT_STDDEV)
        weights = self._get_variable('weights',
                                shape=[num_units_in, num_units_out],
                                initializer=weights_initializer,
                                weight_decay=self.FC_WEIGHT_STDDEV)
        biases = self._get_variable('biases',
                            shape=[num_units_out],
                            initializer=tf.zeros_initializer)
        x = tf.nn.xw_plus_b(x, weights, biases)

        return x

    def stack_fc(self, x, c):
        num_units_in = x.get_shape()[1]

        weights_initializer = tf.truncated_normal_initializer(
            stddev=self.FC_WEIGHT_STDDEV)

        weights = self._get_variable('weights',
                                shape=[num_units_in, 256],
                                initializer=weights_initializer,
                                weight_decay=self.FC_WEIGHT_STDDEV)
        biases = self._get_variable('biases',
                            shape=[256],
                            initializer=tf.zeros_initializer)
        x = tf.nn.xw_plus_b(x, weights, biases)

        weights_2 = self._get_variable('weights_2',
                                shape=[256, 256],
                                initializer=weights_initializer,
                                weight_decay=self.FC_WEIGHT_STDDEV)
        biases_2 = self._get_variable('biases_2',
                            shape=[256],
                            initializer=tf.zeros_initializer)
        x = tf.nn.xw_plus_b(x, weights_2, biases_2)

        num_units_out = c['fc_units_out']

        weights_3 = self._get_variable('weights_3',
                                shape=[256, num_units_out],
                                initializer=weights_initializer,
                                weight_decay=self.FC_WEIGHT_STDDEV)
        biases_3 = self._get_variable('biases_3',
                            shape=[num_units_out],
                            initializer=tf.zeros_initializer)
        x = tf.nn.xw_plus_b(x, weights_3, biases_3)

        return x

    def _get_variable(self, name,
                    shape,
                    initializer,
                    weight_decay=0.0,
                    dtype='float',
                    trainable=True):
        if weight_decay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None
        collections = [tf.GraphKeys.VARIABLES, self.RESNET_VARIABLES]

        return tf.get_variable(name,
                            shape=shape,
                            initializer=initializer,
                            dtype=dtype,
                            regularizer=regularizer,
                            collections=collections,
                            trainable=trainable)

    def conv(self, x, c):
        ksize = c['ksize']
        stride = c['stride']
        filters_out = c['conv_filters_out']

        filters_in = x.get_shape()[-1]
        shape = [ksize, ksize, filters_in, filters_out]
        initializer = tf.truncated_normal_initializer(stddev=self.CONV_WEIGHT_STDDEV)
        weights = self._get_variable('weights',
                                shape=shape,
                                dtype='float',
                                initializer=initializer,
                                weight_decay=self.CONV_WEIGHT_DECAY)

        return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')

    def _max_pool(self, x, ksize=3, stride=2):
        return tf.nn.max_pool(x,
                            ksize=[1, ksize, ksize, 1],
                            strides=[1, stride, stride, 1],
                            padding='SAME')

### FAB model ###
    def FAB_inference(self,
                      input_images_boundary,
                      input_images_blur,
                      F,H,
                      batch_size,
                      net_channel=64,
                      num_classes=136,
                      num_blocks=[2, 2, 2, 2],
                      use_bias=False,
                      bottleneck=True,
                      dropout_ratio=1.0):

    ####structure_predictor_model####
        with tf.variable_scope('structure_predictor_model_'):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0001)):

                batch_norm_params = {
                'decay': 0.9997,
                'epsilon': 0.001,
                'is_training': self.structure_predictor_is_train,
                }
                with slim.arg_scope([slim.batch_norm],
                                    is_training=self.structure_predictor_is_train,
                                    updates_collections=None):
                    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
                        net = slim.conv2d(input_images_boundary, 64, [5, 5], stride=1, scope='conv1')
                        net = slim.max_pool2d(net, [2, 2], scope='pool1')
                        net = slim.conv2d(net, 128, [5, 5], stride=1, scope='conv2')
                        net = slim.max_pool2d(net, [2, 2], scope='pool2')
                        net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv3')
                        net = slim.max_pool2d(net, [2, 2], scope='pool3')
                        net = tf.image.resize_bilinear(net, [64,64])
                        net = slim.conv2d(net, 256, [3, 3], stride=1, scope='conv4')
                        net = tf.image.resize_bilinear(net, [128,128])
                        net = slim.conv2d(net, 128, [3, 3], stride=1, scope='conv5')
                        net = tf.image.resize_bilinear(net, [256,256])
                        net = slim.conv2d(net, 64, [5, 5], stride=1, scope='conv6')
            net = slim.conv2d(net, 3, [5, 5], stride=1,
                              activation_fn=tf.tanh, normalizer_fn=None, scope='conv7')
            flow = net[:, :, :, 0:2]
            mask = tf.expand_dims(net[:, :, :, 2], 3)

            grid_x, grid_y = meshgrid(256, 256)
            grid_x = tf.tile(grid_x, [batch_size, 1, 1])
            grid_y = tf.tile(grid_y, [batch_size, 1, 1])

            coor_x_1 = grid_x + flow[:, :, :, 0]*2
            coor_y_1 = grid_y + flow[:, :, :, 1]*2
            coor_x_2 = grid_x + flow[:, :, :, 0]
            coor_y_2 = grid_y + flow[:, :, :, 1]

            output_1 = bilinear_interp(input_images_boundary[:, :, :, 0:1],
                                       coor_x_1, coor_y_1, 'extrapolate')
            output_2 = bilinear_interp(input_images_boundary[:, :, :, 1:2],
                                       coor_x_2, coor_y_2, 'extrapolate')

            mask = 0.5 * (1.0 + mask)
            mask = tf.tile(mask, [1, 1, 1, 3])
            self.next_frame = tf.multiply(mask, output_1) + tf.multiply(1.0 - mask, output_2)
            self.structure_predictor_output = tf.concat([self.next_frame,input_images_blur],3)

    ####video_deblur_model####
        with tf.variable_scope('video_deblur_model_'):
            H_curr = []
            with tf.variable_scope("encoding"):

                with tf.variable_scope("conv1"):
                    filter_size = 5
                    net_X = self.conv2d(self.structure_predictor_output, self.weight_variable([filter_size,
                                                                                 filter_size,
                                                                                 self.get_shape(self.structure_predictor_output, 3),
                                                                                 net_channel]))
                    net_X = tf.nn.relu(net_X)

                with tf.variable_scope("conv2"):
                    filter_size = 3
                    net_X = self.conv2d(net_X, self.weight_variable([filter_size,
                                                                     filter_size,
                                                                     self.get_shape(net_X, 3),
                                                                     net_channel//2]),
                                                                     stride = 2)
                    net_X = tf.nn.relu(net_X)

                net = tf.concat([net_X, F], 3)
                f0 = net
                filter_size = 3
                num_resnet_layers = 8

                for i in range (num_resnet_layers):
                    with tf.variable_scope('resnet_block%d' % (i+1)):
                        net = self.resnet_block(net, net_channel)

                        if i == 3:
                            (net, alpha) = self.dynamic_fusion(net, H[0])
                            h = tf.expand_dims(net, axis=0)
                            H_curr = h

            with tf.variable_scope("feat_out"):
                F = self.conv2d(net, self.weight_variable([filter_size,
                                                           filter_size,
                                                           self.get_shape(net, 3),
                                                           net_channel//2],
                                                           name = 'conv_F'))
                F = tf.nn.relu(F)

            with tf.variable_scope("img_out"):
                filter_size = 4
                shape = [self.get_shape(self.structure_predictor_output, 0),
                         self.get_shape(self.structure_predictor_output, 1),
                         self.get_shape(self.structure_predictor_output, 2),
                         net_channel]
                Y = self.conv2d_transpose(net, self.weight_variable([filter_size,
                                                                     filter_size,
                                                                     net_channel,
                                                                     net_channel],
                                                                     name = "deconv"),
                                                                     shape,
                                                                     stride = 2)
                Y = tf.nn.relu(Y)
                filter_size = 3
                self.video_deblur_output = self.conv2d(Y, self.weight_variable([filter_size,
                                                                         filter_size,
                                                                         self.get_shape(Y, 3),
                                                                         3],
                                                                         name = 'conv'))

    ####resnet_model####
        with tf.variable_scope('resnet_model_'):
            c = Config()
            c['bottleneck'] = bottleneck
            c['is_training'] = tf.convert_to_tensor(self.resnet_is_train,
                                                    dtype='bool',
                                                    name='is_training')
            c['ksize'] = 3
            c['stride'] = 1
            c['use_bias'] = use_bias
            c['fc_units_out'] = num_classes
            c['num_blocks'] = num_blocks
            c['stack_stride'] = 2

            with tf.variable_scope('scale1'):
                c['conv_filters_out'] = 16
                c['ksize'] = 7
                c['stride'] = 2
                x = self.conv(self.video_deblur_output, c)
                x = self.resnet_bn(x, c)
                x = self.activation(x)

            with tf.variable_scope('scale1_pool'):
                x = self._max_pool(x, ksize=3, stride=2)
                x = self.resnet_bn(x, c)
                x = self.activation(x)

            with tf.variable_scope('scale2'):
                x = self._max_pool(x, ksize=3, stride=2)
                c['num_blocks'] = num_blocks[0]
                c['stack_stride'] = 1
                c['block_filters_internal'] = 8
                x = self.stack(x, c)

            with tf.variable_scope('scale3'):
                c['num_blocks'] = num_blocks[1]
                c['block_filters_internal'] = 16
                assert c['stack_stride'] == 2
                x = self.stack(x, c)

            with tf.variable_scope('scale4'):
                c['num_blocks'] = num_blocks[2]
                c['block_filters_internal'] = 32
                x = self.stack(x, c)

            with tf.variable_scope('scale5'):
                c['num_blocks'] = num_blocks[3]
                c['block_filters_internal'] = 64
                x = self.stack(x, c)

            x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

            if num_classes != None:
                with tf.variable_scope('fc1'):
                    c['fc_units_out'] = 256
                    x = self.fc(x, c)

                with tf.variable_scope('dropout1'):
                    x = tf.nn.dropout(x, dropout_ratio)

                with tf.variable_scope('fc2'):
                    c['fc_units_out'] = 256
                    x = self.fc(x, c)

                with tf.variable_scope('dropout2'):
                    x = tf.nn.dropout(x, dropout_ratio)

                with tf.variable_scope('fc3'):
                    c['fc_units_out'] = 136
                    self.logits = self.fc(x, c)

        return F, H_curr
