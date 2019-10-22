from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
import cv2
import time
import copy
import numpy as np
import tensorflow as tf

from utils.curve import points_to_heatmap_rectangle_68pt
from six.moves import xrange
from six.moves import urllib
from datagen import DataGenerator
from datagen import ensure_dir
from FAB import FAB

MOMENTUM = 0.9
POINTS_NUM = 68
IMAGE_SIZE = 256
PIC_CHANNEL = 3
num_input_imgs = 3
NUM_CLASSES = POINTS_NUM*2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
structure_predictor_net_channel = 64

FLAGS = tf.app.flags.FLAGS
# address
tf.app.flags.DEFINE_string('structure_predictor_train_dir', '', """Directory where to write train_checkpoints.""")
tf.app.flags.DEFINE_string('video_deblur_train_dir', '', """Directory where to write train_checkpoints.""")
tf.app.flags.DEFINE_string('resnet_train_dir', '', """Directory where to write train_checkpoints.""")
tf.app.flags.DEFINE_string('end_2_end_train_dir', '', """Directory where to write train_checkpoints.""")
tf.app.flags.DEFINE_string('end_2_end_valid_dir', '', """Directory where to write valid logs.""")
tf.app.flags.DEFINE_string('data_dir', '', """Directory where the dataset stores.""")
tf.app.flags.DEFINE_string('img_list', '', """Directory where the img_list stores.""")
tf.app.flags.DEFINE_string('data_dir_valid', '', """Directory where the valid image stores. Only used for pretraining on 300W datasets.""")
tf.app.flags.DEFINE_string('img_list_valid', '', """Directory where the valid image_list stores. Only used for pretraining on 300W datasets.""")
# parameters
tf.app.flags.DEFINE_float('learning_rate', 0.00003, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 1, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 2000000, "max steps")
tf.app.flags.DEFINE_boolean('resume_structure_predictor', True, """Resume from latest saved state.""")
tf.app.flags.DEFINE_boolean('resume_resnet', True, """Resume from latest saved state.""")
tf.app.flags.DEFINE_boolean('resume_video_deblur', True, """Resume from latest saved state.""")
tf.app.flags.DEFINE_boolean('resume_all', False, """Resume from latest saved state.""")
tf.app.flags.DEFINE_boolean('minimal_summaries', False, """Produce fewer summaries to save HD space.""")
tf.app.flags.DEFINE_string('training_period', 'pretrain', """Choose the training period: pretrain/train.""")
tf.app.flags.DEFINE_boolean('use_bn', False, """Use batch normalization. Otherwise use biases.""")

def resume(sess, do_resume, ckpt_path, key_word):
    var = tf.global_variables()
    if do_resume:
        structure_predictor_latest = tf.train.latest_checkpoint(ckpt_path)
        if not structure_predictor_latest:
            print ("\n No checkpoint to continue from in ", ckpt_path, '\n')
        structure_predictor_var_to_restore = [val  for val in var if key_word in val.name]
        saver_structure_predictor = tf.train.Saver(structure_predictor_var_to_restore)
        saver_structure_predictor.restore(sess, structure_predictor_latest)

def train(resnet_model, is_training, F, H, F_curr, H_curr,
          input_images_blur, input_images_boundary, next_boundary_gt, labels,
          data_dir, data_dir_valid, img_list, img_list_valid,
          dropout_ratio):

    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    # define the losses.
    lambda_ = 1e-5

    loss_1 = resnet_model.l2_loss_(resnet_model.logits, labels)
    loss_2 = resnet_model.l2_loss_(resnet_model.next_frame,next_boundary_gt)
    loss_3 = resnet_model.l2_loss_(input_images_blur[:,:,:,-3:],resnet_model.video_deblur_output)
    loss_ = loss_1+loss_2+loss_3+tf.reduce_sum(tf.square(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))*lambda_

    ema = tf.train.ExponentialMovingAverage(resnet_model.MOVING_AVERAGE_DECAY, global_step)
    tf.add_to_collection(resnet_model.UPDATE_OPS_COLLECTION, ema.apply([loss_]))
    tf.summary.scalar('loss_avg', ema.average(loss_))

    ema = tf.train.ExponentialMovingAverage(0.9, val_step)
    val_op = tf.group(val_step.assign_add(1), ema.apply([loss_]))
    tf.summary.scalar('loss_valid', ema.average(loss_))

    tf.summary.scalar('learning_rate', FLAGS.learning_rate)

    # define the optimizer and back propagate.
    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = opt.compute_gradients(loss_)
    for grad, var in grads:
        if grad is not None and not FLAGS.minimal_summaries:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    batchnorm_updates = tf.get_collection(resnet_model.UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    saver_all = tf.train.Saver(tf.all_variables())

    summary_op = tf.summary.merge_all()

    # initialize all variables
    init = tf.initialize_all_variables()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)

    summary_writer = tf.summary.FileWriter(FLAGS.end_2_end_train_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(FLAGS.end_2_end_valid_dir)
    val_save_root = os.path.join(FLAGS.end_2_end_valid_dir,'visualization')
    compare_save_root = os.path.join(FLAGS.end_2_end_valid_dir,'deblur_compare')

    # resume weights
    resume(sess, FLAGS.resume_structure_predictor, FLAGS.structure_predictor_train_dir, 'voxel_flow_model_')
    resume(sess, FLAGS.resume_video_deblur, FLAGS.video_deblur_train_dir, 'video_deblur_model_')
    resume(sess, FLAGS.resume_resnet, FLAGS.resnet_train_dir, 'resnet_model_')
    resume(sess, FLAGS.resume_all, FLAGS.end_2_end_train_dir, '')
    
    # create data generator
    if FLAGS.training_period == 'pretrain':
        dataset = DataGenerator(data_dir, img_list, data_dir_valid, img_list_valid)
        dataset._create_train_sets_for_300W()
        dataset._create_valid_sets_for_300W()
    elif FLAGS.training_period == 'train':
        dataset = DataGenerator(data_dir,img_list)
        dataset._create_train_table()
        dataset._create_sets_for_300VW()
    else:
        raise NameError("No such training_period!")
    train_gen = dataset._aux_generator(batch_size = FLAGS.batch_size,
                                    num_input_imgs = num_input_imgs,
                                    NUM_CLASSES = POINTS_NUM*2,
                                    sample_set='train')
    valid_gen = dataset._aux_generator(batch_size = FLAGS.batch_size,
                                    num_input_imgs = num_input_imgs,
                                    NUM_CLASSES = POINTS_NUM*2,
                                    sample_set='valid')
    
    # main training process.
    for x in xrange(FLAGS.max_steps + 1):

        start_time = time.time()
        step = sess.run(global_step)
        i = [train_op, loss_]
        write_summary = step > 1 and not (step % 100)
        if write_summary:
            i.append(summary_op)
        i.append(resnet_model.logits)
        i.append(F_curr)
        i.append(H_curr)

        train_line_num, frame_name, input_boundaries, boundary_gt_train, input_images_blur_generated, landmark_gt_train = next(train_gen)

        if (frame_name == '2.jpg'):
            input_images_boundary_init = copy.deepcopy(input_boundaries)
            F_init = np.zeros([FLAGS.batch_size,
                               IMAGE_SIZE//2,
                               IMAGE_SIZE//2,
                               structure_predictor_net_channel//2], dtype=np.float32)

            H_init = np.zeros([1,
                               FLAGS.batch_size,
                               IMAGE_SIZE//2,
                               IMAGE_SIZE//2,
                               structure_predictor_net_channel], dtype=np.float32)
            feed_dict={
                    input_images_boundary:input_images_boundary_init,
                    input_images_blur:input_images_blur_generated,
                    F:F_init,
                    H:H_init,
                    labels:landmark_gt_train,
                    next_boundary_gt:boundary_gt_train,
                    dropout_ratio:0.5
                    }
        else:
            output_points = o[-3]
            output_points = np.reshape(output_points,(POINTS_NUM,2))

            boundary_from_points = points_to_heatmap_rectangle_68pt(output_points)
            boundary_from_points = np.expand_dims(boundary_from_points,axis=0)
            boundary_from_points = np.expand_dims(boundary_from_points,axis=3)
            input_images_boundary_init = np.concatenate([input_images_boundary_init[:,:,:,1:2],
                                                         boundary_from_points], axis=3)
            feed_dict={
                    input_images_boundary:input_images_boundary_init,
                    input_images_blur:input_images_blur_generated,
                    F:o[-2],
                    H:o[-1],
                    labels:landmark_gt_train,
                    next_boundary_gt:boundary_gt_train,
                    dropout_ratio:0.5
                    }

        o = sess.run(i,feed_dict=feed_dict)
        loss_value = o[1]
        duration = time.time() - start_time
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step > 1 and step % 300 == 0:
            examples_per_sec = FLAGS.batch_size / float(duration)
            format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (step, loss_value, examples_per_sec, duration))

        if write_summary:
            summary_str = o[2]
            summary_writer.add_summary(summary_str, step)

        if step > 1 and step % 300 == 0:
            checkpoint_path = os.path.join(FLAGS.end_2_end_train_dir, 'model.ckpt')
            ensure_dir(checkpoint_path)
            saver_all.save(sess, checkpoint_path, global_step=global_step)

        # Run validation periodically
        if step > 1 and step % 300 == 0:
            valid_line_num, frame_name, input_boundaries, boundary_gt_valid, input_images_blur_generated, landmark_gt_valid = next(valid_gen)

            if (frame_name == '2.jpg')  or valid_line_num <= 3:
                input_images_boundary_init = copy.deepcopy(input_boundaries)
                F_init = np.zeros([FLAGS.batch_size,
                                   IMAGE_SIZE//2,
                                   IMAGE_SIZE//2,
                                   structure_predictor_net_channel//2], dtype=np.float32)

                H_init = np.zeros([1, FLAGS.batch_size,
                                   IMAGE_SIZE//2,
                                   IMAGE_SIZE//2,
                                   structure_predictor_net_channel], dtype=np.float32)

                feed_dict={input_images_boundary:input_images_boundary_init,
                        input_images_blur:input_images_blur_generated,
                        F:F_init,
                        H:H_init,
                        labels:landmark_gt_valid,
                        next_boundary_gt:boundary_gt_valid,
                        dropout_ratio:1.0
                        }
            else:
                output_points = o_valid[-3]
                output_points = np.reshape(output_points,(POINTS_NUM,2))
                boundary_from_points = points_to_heatmap_rectangle_68pt(output_points)
                boundary_from_points = np.expand_dims(boundary_from_points,axis=0)
                boundary_from_points = np.expand_dims(boundary_from_points,axis=3)

                input_images_boundary_init = np.concatenate([input_images_boundary_init[:,:,:,1:2],
                                                             boundary_from_points], axis=3)
                feed_dict={
                        input_images_boundary:input_images_boundary_init,
                        input_images_blur:input_images_blur_generated,
                        F:o_valid[-2],
                        H:o_valid[-1],
                        labels:landmark_gt_valid,
                        next_boundary_gt:boundary_gt_valid,
                        dropout_ratio:1.0
                        }
            i_valid = [loss_,resnet_model.logits,F_curr,H_curr]
            o_valid = sess.run(i_valid,feed_dict=feed_dict)
            print('Validation top1 error %.2f' % o_valid[0])
            if write_summary:
                val_summary_writer.add_summary(summary_str, step)
            img_video_deblur_output = sess.run(resnet_model.video_deblur_output,feed_dict=feed_dict)[0]*255
            img = input_images_blur_generated[0,:,:,0:3]*255
            compare_img = np.concatenate([img,img_video_deblur_output],axis=1)
            points = o_valid[1][0]*255

            for point_num in range(int(points.shape[0]/2)):
                cv2.circle(img,(int(round(points[point_num*2])),int(round(points[point_num*2+1]))),1,(55,225,155),2)
            val_save_path = os.path.join(val_save_root,str(step)+'.jpg')
            compare_save_path = os.path.join(compare_save_root,str(step)+'.jpg')
            ensure_dir(val_save_path)
            ensure_dir(compare_save_path)
            cv2.imwrite(val_save_path,img)
            cv2.imwrite(compare_save_path,compare_img)

def main(argv=None):
    resnet_model = FAB(structure_predictor_is_train=False,
                       deblur_is_train=True,
                       resnet_is_train=False)

    is_training = tf.placeholder('bool', [], name='is_training')

    input_images_boundary = tf.placeholder(tf.float32,shape=(FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 2))
    input_images_blur = tf.placeholder(tf.float32,shape=(FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, PIC_CHANNEL*3))
    next_boundary_gt = tf.placeholder(tf.float32,shape=(FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 1))
    labels = tf.placeholder(tf.float32,shape=(FLAGS.batch_size,NUM_CLASSES))
    dropout_ratio = tf.placeholder(tf.float32)
    F = tf.placeholder(tf.float32, [FLAGS.batch_size, IMAGE_SIZE//2, IMAGE_SIZE//2, structure_predictor_net_channel//2])
    H = tf.placeholder(tf.float32, [1, FLAGS.batch_size, IMAGE_SIZE//2, IMAGE_SIZE//2, structure_predictor_net_channel])

    F_curr, H_curr = resnet_model.FAB_inference(input_images_boundary, input_images_blur, F, H, FLAGS.batch_size,
                                    net_channel=structure_predictor_net_channel, num_classes=136,
                                    num_blocks=[2, 2, 2, 2], use_bias=(not FLAGS.use_bn),
                                    bottleneck=True,dropout_ratio=1.0)

    train(resnet_model, is_training, F, H, F_curr, H_curr,
          input_images_blur, input_images_boundary, next_boundary_gt, labels,
          FLAGS.data_dir, FLAGS.data_dir_valid, FLAGS.img_list, FLAGS.img_list_valid,
          dropout_ratio)

if __name__ == '__main__':
    tf.app.run()
