from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
import cv2
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
tf.app.flags.DEFINE_string('structure_predictor_train_dir', '', """Directory where to write train_checkpoints.""")
tf.app.flags.DEFINE_string('video_deblur_train_dir', '', """Directory where to write train_checkpoints.""")
tf.app.flags.DEFINE_string('resnet_train_dir', '', """Directory where to write train_checkpoints.""")
tf.app.flags.DEFINE_string('end_2_end_train_dir', '', """Directory where to write train_checkpoints.""")
tf.app.flags.DEFINE_string('end_2_end_test_dir', '', """Directory where to write test logs.""")
tf.app.flags.DEFINE_string('data_dir', '', """Directory where the dataset stores.""")
tf.app.flags.DEFINE_string('img_list', '', """Directory where the img_list stores.""")

tf.app.flags.DEFINE_float('learning_rate', 0.0, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 1, "batch size")
tf.app.flags.DEFINE_boolean('resume_structure_predictor', False, """Resume from latest saved state.""")
tf.app.flags.DEFINE_boolean('resume_resnet', False, """Resume from latest saved state.""")
tf.app.flags.DEFINE_boolean('resume_video_deblur', False, """Resume from latest saved state.""")
tf.app.flags.DEFINE_boolean('resume_all', False, """Resume from latest saved state.""")
tf.app.flags.DEFINE_boolean('minimal_summaries', False, """Produce fewer summaries to save HD space.""")
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

def test(resnet_model, is_training, F, H, F_curr, H_curr, input_images_blur,
        input_images_boundary, next_boundary_gt, labels, data_dir, img_list,
        dropout_ratio):

    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    init = tf.initialize_all_variables()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)
    val_save_root = os.path.join(FLAGS.end_2_end_test_dir,'visualization')

    ################################ resume part #################################

    # resume weights
    resume(sess, FLAGS.resume_structure_predictor, FLAGS.structure_predictor_train_dir, 'voxel_flow_model_')
    resume(sess, FLAGS.resume_video_deblur, FLAGS.video_deblur_train_dir, 'video_deblur_model_')
    resume(sess, FLAGS.resume_resnet, FLAGS.resnet_train_dir, 'resnet_model_')
    resume(sess, FLAGS.resume_all, FLAGS.end_2_end_train_dir, '')

    ##############################################################################

    gt_file_path = os.path.join(FLAGS.end_2_end_test_dir,'gt.txt')
    pre_file_path = os.path.join(FLAGS.end_2_end_test_dir,'pre.txt')
    ensure_dir(gt_file_path)
    ensure_dir(pre_file_path)
    gt_file = open(gt_file_path,'w')
    pre_file = open(pre_file_path,'w')

    dataset = DataGenerator(data_dir,img_list)
    dataset._create_train_table()
    dataset._create_sets_for_300VW()
    test_gen = dataset._aux_generator(batch_size = FLAGS.batch_size, num_input_imgs = num_input_imgs,
                                       NUM_CLASSES = POINTS_NUM*2, sample_set='test')

    test_break_flag = False
    for x in xrange(len(dataset.train_table)-2):

        step = sess.run(global_step)

        if not test_break_flag:
            test_line_num, frame_name, input_boundaries, boundary_gt_test, input_images_blur_generated, landmark_gt_test, names, test_break_flag = next(test_gen)

        if (frame_name == '2.jpg')  or test_line_num <= 3:
            input_images_boundary_init = copy.deepcopy(input_boundaries)
            F_init = np.zeros([FLAGS.batch_size, IMAGE_SIZE//2,
                               IMAGE_SIZE//2, structure_predictor_net_channel//2], dtype=np.float32)

            H_init = np.zeros([1, FLAGS.batch_size, IMAGE_SIZE//2,
                               IMAGE_SIZE//2, structure_predictor_net_channel], dtype=np.float32)

            feed_dict={
                    input_images_boundary:input_images_boundary_init,
                    input_images_blur:input_images_blur_generated,
                    F:F_init,
                    H:H_init,
                    labels:landmark_gt_test,
                    next_boundary_gt:boundary_gt_test,
                    dropout_ratio:1.0
                    }
        else:
            output_points = o[0]
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
                    labels:landmark_gt_test,
                    next_boundary_gt:boundary_gt_test,
                    dropout_ratio:1.0
                    }

        i = [resnet_model.logits, F_curr, H_curr]
        o = sess.run(i, feed_dict=feed_dict)
        pres = o[0]

        for batch_num,pre in enumerate(pres):
            for v in pre:
                pre_file.write(str(v*255.0)+' ')
            if len(names) > 1:
                pre_file.write(names[-1])
            else:
                pre_file.write(names[batch_num])
            pre_file.write('\n')
        for batch_num,g in enumerate(landmark_gt_test):
            for v in g:
                gt_file.write(str(v*255.0)+' ')
            if len(names) > 1:
                gt_file.write(names[-1])
            else:
                gt_file.write(names[batch_num])
            gt_file.write('\n')

        img = input_images_blur_generated[0,:,:,0:3]*255
        points = o[0][0]*255

        for point_num in range(int(points.shape[0]/2)):
            cv2.circle(img,(int(round(points[point_num*2])),int(round(points[point_num*2+1]))),1,(55,225,155),2)
        val_save_path = os.path.join(val_save_root,str(step)+'.jpg')
        ensure_dir(val_save_path)
        cv2.imwrite(val_save_path,img)

        global_step = global_step + 1
    print('Test done!')

def main(argv=None):

    resnet_model = FAB()

    is_training = tf.placeholder('bool', [], name='is_training')
    input_images_boundary = tf.placeholder(tf.float32,shape=(FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 2))
    input_images_blur = tf.placeholder(tf.float32,shape=(FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, PIC_CHANNEL*3))
    next_boundary_gt = tf.placeholder(tf.float32,shape=(FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 1))
    labels = tf.placeholder(tf.float32,shape=(FLAGS.batch_size,NUM_CLASSES))
    dropout_ratio = tf.placeholder(tf.float32)
    F = tf.placeholder(tf.float32, [FLAGS.batch_size, IMAGE_SIZE//2, IMAGE_SIZE//2, structure_predictor_net_channel//2])
    H = tf.placeholder(tf.float32, [1, FLAGS.batch_size, IMAGE_SIZE//2, IMAGE_SIZE//2, structure_predictor_net_channel])
    F_curr, H_curr= \
        resnet_model.FAB_inference(input_images_boundary, input_images_blur, F, H, FLAGS.batch_size,
                                net_channel=structure_predictor_net_channel, num_classes=136, num_blocks=[2, 2, 2, 2],
                                use_bias=(not FLAGS.use_bn), bottleneck=True, dropout_ratio=1.0)

    test(resnet_model, is_training, F, H, F_curr, H_curr, input_images_blur,
            input_images_boundary, next_boundary_gt, labels, FLAGS.data_dir, FLAGS.img_list,
            dropout_ratio)

if __name__ == '__main__':
    tf.app.run()
