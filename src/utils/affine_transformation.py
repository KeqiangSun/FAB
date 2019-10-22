import random
import copy
import cv2
import numpy as np


def get_affine_mat(width, height,
                   max_trans, max_rotate, max_zoom,
                   min_trans, min_rotate, min_zoom):
    rotate = random.uniform(min_rotate, max_rotate)
    trans = random.uniform(min_trans, max_trans)
    zoom = random.uniform(min_zoom, max_zoom)

    # rotate
    transform_matrix = np.zeros((3,3))
    center = (width/2.-0.5, height/2.-0.5)
    M = cv2.getRotationMatrix2D(center, rotate, 1)
    transform_matrix[:2,:] = copy.deepcopy(M)
    transform_matrix[2,:] = np.array([0, 0, 1])

    # translate
    transform_matrix[0,2] += trans
    transform_matrix[1,2] += trans

    # zoom
    for i in range(3):
        transform_matrix[0,i] *= zoom
        transform_matrix[1,i] *= zoom
    transform_matrix[0,2] += (1.0 - zoom) * center[0]
    transform_matrix[1,2] += (1.0 - zoom) * center[1]

    # random horizontal mirror
    do_mirror = False
    mirror_rng = random.uniform(0.,1.)
    if mirror_rng>0.5:
        do_mirror = True

    return transform_matrix,do_mirror

def AffinePoint(points, affine_mat):
    """
    Affine a 2d point
    """
    assert(affine_mat.shape[0] == 2)
    assert(affine_mat.shape[1] == 3)
    assert(points.shape[1] == 2)
    results = np.zeros(points.shape)
    for i in range(points.shape[0]):
        point_x = points[i,0]
        point_y = points[i,1]
        results[i,0] = affine_mat[0,0] * point_x + \
                    affine_mat[0,1] * point_y + \
                    affine_mat[0,2]
        results[i,1] = affine_mat[1,0] * point_x + \
                    affine_mat[1,1] * point_y + \
                    affine_mat[1,2]

    return results

def affine2d(x, matrix, output_img_width, output_img_height,
             center=True, is_landmarks=False, do_mirror=False):
    assert(len(matrix.shape) == 2)
    if is_landmarks:
        transform_matrix = matrix[:2,:]
        src = x.squeeze()
        dst = np.empty((src.shape[0],2), dtype=np.float32)
        for i in range(src.shape[0]):
            dst[i,:] = AffinePoint(np.expand_dims(src[i,:], axis=0), transform_matrix)
        if do_mirror:
            results = exchange_landmarks(dst,np.array([0,16,1,15,2,14,3,13,4,12,5,11,6,10,7,9,17,26,18,25,19,24,20,23,21,22,36,45,37,44,38,
                                                       43,39,42,41,46,40,47,31,35,32,34,48,54,49,53,50,52,60,64,61,63,67,65,59,55,58,56]).reshape(-1, 2))
    else:
        if do_mirror:
            matrix[0,0] = -matrix[0,0]
            matrix[0,1] = -matrix[0,1]
            matrix[0,2] = float(output_img_width)-matrix[0,2]
        transform_matrix = matrix[:2,:]
        src = x.astype(np.uint8)
        dst = cv2.warpAffine(src, transform_matrix,
                             (output_img_width, output_img_height),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(127,127,127))

        if len(dst.shape) == 2:
            dst = np.expand_dims(np.asarray(dst), axis=2)

    return dst

def exchange_landmarks(input_tf, corr_list):
    for i in range(corr_list.shape[0]):
        temp = copy.deepcopy(input_tf[corr_list[i][0], :])
        input_tf[corr_list[i][0], :] = input_tf[corr_list[i][1], :]
        input_tf[corr_list[i][1], :] = temp

    return input_tf
