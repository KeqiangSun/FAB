from __future__ import print_function
import cv2
import os
import random
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as scm
import tensorflow as tf

from PIL import Image
from utils import affine_transformation
from utils.color_jitter import ImageJitter
from skimage import transform, util
from utils.curve import points_to_heatmap_rectangle_68pt

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

class DataGenerator():

    def __init__(self, img_dir=None, train_list_file=None,
                 img_dir_valid=None, valid_list_file=None):
        self.img_dir = img_dir
        self.img_dir_valid = img_dir_valid
        self.train_list_file = train_list_file
        self.valid_list_file = valid_list_file

    def _create_train_table(self):
        self.train_table = []
        input_file = open(self.train_list_file, 'r')
        for line in input_file.readlines():
            self.train_table.append(line)
        input_file.close()

    def _randomize(self):
        random.shuffle(self.train_table)

    def _create_train_sets_for_300W(self):
        self.train_set = []
        input_file = open(self.train_list_file, 'r')
        for line in input_file.readlines():
            self.train_set.append(line)
        input_file.close()

    def _create_valid_sets_for_300W(self):
        self.valid_set = []
        input_file = open(self.valid_list_file, 'r')
        for line in input_file.readlines():
            self.valid_set.append(line)
        input_file.close()

    def _create_sets_for_300VW(self, validation_rate = 0.05):
        self.sample = len(self.train_table)
        valid_sample = int(self.sample * validation_rate)
        self.train_set = self.train_table[:self.sample - valid_sample]
        self.valid_set = self.train_table[self.sample - valid_sample:]
        self.test_set = self.train_table[:]

    def _aux_generator(self, batch_size = 1, NUM_CLASSES = 136,
                       num_input_imgs = 3, normalize = True, sample_set = 'train'):
        train_line_num = 0
        valid_line_num = 0
        test_line_num = 0
        test_break_flag = False

        while True:
            train_img = np.zeros((batch_size, 256,256,3*num_input_imgs), dtype = np.float32)
            train_gtmap = np.zeros((batch_size, NUM_CLASSES), dtype = np.float32)
            i = 0
            names = []
            max_lines = 3

            while i < batch_size:
                input_boundaries = []

                if sample_set == 'train':
                    if train_line_num+1 == len(self.train_set) or train_line_num+2 == len(self.train_set) :
                        train_line_num = 0
                elif sample_set == 'valid':
                    if valid_line_num+1 == len(self.valid_set) or valid_line_num+2 == len(self.valid_set):
                        valid_line_num = 0
                elif sample_set == 'test':
                    if test_line_num+1 == len(self.test_set):
                        print('The end of the testing set!')
                        test_break_flag = True

                for cntr in range(max_lines):
                    if sample_set == 'train':
                        line = self.train_set[train_line_num]
                        train_line_num += 1
                    elif sample_set == 'valid':
                        line = self.valid_set[valid_line_num]
                        valid_line_num += 1
                    elif sample_set == 'test':
                        line = self.test_set[test_line_num]
                        test_line_num += 1

                    eles = line.strip().split()
                    frame_path = eles[-1]
                    name = frame_path.split('/')[-1]
                    names.append(name)
                    gt = np.array(map(float,eles[:-1]))
                    gt_flatten = np.reshape(gt,(gt.shape[0]/2,2))

                    boundary_gt_train = points_to_heatmap_rectangle_68pt(gt_flatten)
                    boundary_gt_train = np.expand_dims(boundary_gt_train,axis=0)
                    boundary_gt_train = np.expand_dims(boundary_gt_train,axis=3)
                    input_boundaries.append(boundary_gt_train)

                    if sample_set == 'train':
                        if name != '0.jpg' and name != '1.jpg':
                            break
                    elif sample_set == 'valid':
                        if (name != '0.jpg' and name != '1.jpg' and valid_line_num > 2):
                            break
                    elif sample_set == 'test':
                        if (name != '0.jpg' and name != '1.jpg' and test_line_num > 2):
                            break

                input_boundaries = input_boundaries[:-1]
                if len(input_boundaries) > 0:
                    input_boundaries = np.concatenate(input_boundaries,axis=3)

                path_eles = frame_path.split('/')
                name_eles = path_eles[-1].split('.')
                frame_num = int(name_eles[0])

                frame_path_2 = os.path.join(path_eles[0],str(frame_num-2)+'.'+name_eles[-1])
                input_img_path_2 = os.path.join(self.img_dir, frame_path_2)
                img_2 = self.open_img(input_img_path_2)
                img_2 = scm.imresize(img_2, (256,256))

                frame_path_1 = os.path.join(path_eles[0],str(frame_num-1)+'.'+name_eles[-1])
                input_img_path_1 = os.path.join(self.img_dir, frame_path_1)
                img_1 = self.open_img(input_img_path_1)
                img_1 = scm.imresize(img_1, (256,256))

                frame_path_0 = os.path.join(path_eles[0],str(frame_num)+'.'+name_eles[-1])
                input_img_path_0 = os.path.join(self.img_dir, frame_path_0)
                img_0 = self.open_img(input_img_path_0)
                img_0 = scm.imresize(img_0, (256,256))

                img = np.concatenate([img_2,img_1,img_0],axis=2)

                if normalize:
                    train_img[i] = img.astype(np.float32) / 255
                    train_gtmap[i] = gt.astype(np.float32) /255
                else :
                    train_img[i] = img.astype(np.float32)
                    train_gtmap[i] = gt.astype(np.float32)

                i = i + 1

                if sample_set == 'train':
                    yield train_line_num, name, input_boundaries, boundary_gt_train, train_img, train_gtmap
                elif sample_set == 'valid':
                    yield valid_line_num, name, input_boundaries, boundary_gt_train, train_img, train_gtmap
                elif sample_set == 'test':
                    print("name = {}".format(name))
                    yield test_line_num, name, input_boundaries, boundary_gt_train, train_img, train_gtmap, names, test_break_flag

    def _voxel_flow_generator_(self, batch_size = 1, sample_set = 'train'):

        train_line_num = 0
        valid_line_num = 0

        while True:
            input_boundaries = np.zeros((batch_size, 256, 256, 2), dtype = np.float32)
            boundary_gts_train = np.zeros((batch_size, 256, 256, 1), dtype = np.float32)
            i = 0
            max_lines = 3

            while i < batch_size:
                input_boundary = []

                if sample_set == 'train':
                    if train_line_num+1 == len(self.train_set) or train_line_num+2 == len(self.train_set) :
                        train_line_num = 0
                    line_num = copy.deepcopy(train_line_num)
                elif sample_set == 'valid':
                    if valid_line_num+1 == len(self.valid_set) or valid_line_num+2 == len(self.valid_set):
                        valid_line_num = 0
                    line_num = copy.deepcopy(valid_line_num)

                for cntr in range(max_lines):
                    if sample_set == 'train':
                        line = self.train_set[line_num]
                    elif sample_set == 'valid':
                        line = self.valid_set[line_num]

                    line_num += 1
                    eles = line.strip().split()
                    frame_path = eles[-1]
                    gt = np.array(map(float,eles[:-1]))
                    gt_flatten = np.reshape(gt,(gt.shape[0]/2,2))

                    boundary_gt_train = points_to_heatmap_rectangle_68pt(gt_flatten)
                    boundary_gt_train = np.expand_dims(boundary_gt_train,axis=2)
                    boundary_gt_train = np.expand_dims(boundary_gt_train,axis=0)
                    input_boundary.append(boundary_gt_train[0])

                train_line_num += 1
                valid_line_num += 1
                input_boundary = input_boundary[:-1]
                input_boundaries[i] = np.concatenate(input_boundary,axis=2)
                boundary_gts_train[i] = boundary_gt_train[0]

                i = i + 1

            if sample_set == 'train':
                yield input_boundaries, boundary_gts_train
            elif sample_set == 'valid':
                yield input_boundaries, boundary_gts_train

    def _video_deblur_generator_(self, batch_size = 1,normalize = True,
                                 num_input_imgs = 3,sample_set='train'):

        train_line_num = 0
        valid_line_num = 0

        while True:
            train_img = np.zeros((batch_size, 256, 256, 3*num_input_imgs), dtype = np.float32)
            i = 0
            max_lines = 3

            while i < batch_size:
                input_images = []

                if sample_set == 'train':
                    if train_line_num+1 == len(self.train_set) or train_line_num+2 == len(self.train_set) :
                        train_line_num = 0
                    line_num = copy.deepcopy(train_line_num)
                elif sample_set == 'valid':
                    if valid_line_num+1 == len(self.valid_set) or valid_line_num+2 == len(self.valid_set):
                        valid_line_num = 0
                    line_num = copy.deepcopy(valid_line_num)

                for cntr in range(max_lines):
                    if sample_set == 'train':
                        line = self.train_set[line_num]
                    elif sample_set == 'valid':
                        line = self.valid_set[line_num]
                    line_num += 1

                    eles = line.strip().split()
                    frame_path = eles[-1]
                    input_img_path = os.path.join(self.img_dir, frame_path)
                    name = frame_path.split('/')[-1]

                    img = self.open_img(input_img_path)
                    img = scm.imresize(img, (256,256))

                    if normalize:
                        input_images.append(img.astype(np.float32) / 255)
                    else :
                        input_images.append(img.astype(np.float32))

                train_line_num += 1
                valid_line_num += 1
                train_img[i] = np.concatenate(input_images,axis=2)

                i = i + 1

            if sample_set == 'train':
                yield train_line_num, name, train_img
            elif sample_set == 'valid':
                yield valid_line_num, name, train_img

    def _resnet_generator(self, batch_size = 16, NUM_CLASSES = 136,
                          normalize = True, sample_set = 'train'):

        while True:
            train_img = np.zeros((batch_size, 256,256,3), dtype = np.float32)
            train_gtmap = np.zeros((batch_size, NUM_CLASSES), dtype = np.float32)
            i = 0

            while i < batch_size:
                if sample_set == 'train':
                    line = random.choice(self.train_set)
                elif sample_set == 'valid':
                    line = random.choice(self.valid_set)

                eles = line.strip().split()
                name = eles[-1]
                if sample_set == 'train':
                    input_img_path = os.path.join(self.img_dir, name)
                elif sample_set == 'valid':
                    input_img_path = os.path.join(self.img_dir_valid, name)

                img = self.open_img(input_img_path)

                if sample_set == 'train':
                    gt = np.array(list(map(float, eles[:-1])))
                    gt = gt.reshape(-1, 2)

                    transform_matrix, do_mirror = affine_transformation.get_affine_mat(
                        width=256, height=256,
                        max_trans=40, max_rotate=30, max_zoom=1.1,
                        min_trans=-40, min_rotate=-30, min_zoom=0.9)

                    img = affine_transformation.affine2d(img, transform_matrix, output_img_width=256,
                                             output_img_height=256, center=True,
                                             is_landmarks=False, do_mirror=do_mirror)
                    gt = affine_transformation.affine2d(gt, transform_matrix, output_img_width=256,
                                            output_img_height=256, center=True,
                                            is_landmarks=True, do_mirror=do_mirror)

                    transformdict = {'Brightness':0.5025, 'Contrast':0.5136,
                                     'Sharpness':0.5568, 'Color':0.5203}
                    image_jitter = ImageJitter(transformdict)
                    img = Image.fromarray(img)
                    img = image_jitter(img)
                    img = np.array(img)

                    img = util.random_noise(img, mode='gaussian')
                    img = (img*255).astype(np.uint8)
                    gt = gt.reshape(1, -1).squeeze()

                elif sample_set == 'valid':
                   gt = np.array(map(float,eles[:-1]))

                if normalize:
                    train_img[i] = img.astype(np.float32) / 255
                    train_gtmap[i] = gt.astype(np.float32) /255
                else:
                    train_img[i] = img.astype(np.float32)
                    train_gtmap[i] = gt.astype(np.float32)

                i = i + 1

            yield train_img, train_gtmap

    def open_img(self, img_path, color = 'RGB'):
        img = cv2.imread(img_path)
        if color == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif color == 'BGR':
            return img
        elif color == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')
