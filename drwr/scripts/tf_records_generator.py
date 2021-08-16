import startup

import sys
import os
import glob
import re
import random
import math

import numpy as np
from scipy.io import loadmat
from imageio import imread

from skimage.transform import resize as im_resize

from util.fs import mkdir_if_missing
from util.data import tf_record_options
from run.fuzz_pc_631 import get_z_candidate3
from run.possion import PoissonDiskSampler
import tensorflow as tf

from tensorflow import app

flags = tf.app.flags

flags.DEFINE_string('split_dir',
                    '',
                    'Directory path containing the input rendered images.')

flags.DEFINE_string('inp_dir_renders',
                    '',
                    'Directory path containing the input rendered images.')

flags.DEFINE_string('inp_dir_voxels',
                    '',
                    'Directory path containing the input voxels.')
flags.DEFINE_string('out_dir',
                    '',
                    'Directory path to write the output.')

flags.DEFINE_string('synth_set', '02691156',
                    '')

flags.DEFINE_boolean('store_camera', False, '')
flags.DEFINE_boolean('store_voxels', False, '')
flags.DEFINE_boolean('store_depth', False, '')
flags.DEFINE_string('split_path', '', '')

flags.DEFINE_integer('num_views', 10, 'Num of viewpoints in the input data.')
flags.DEFINE_integer('image_size', 64,
                     'Input images dimension (pixels) - width & height.')
flags.DEFINE_integer('vox_size', 32, 'Voxel prediction dimension.')
flags.DEFINE_boolean('tfrecords_gzip_compressed', False, 'Voxel prediction dimension.')

THRESHOLD = 0.6
FLAGS = flags.FLAGS


def read_camera(filename):
    cam = loadmat(filename)
    extr = cam["extrinsic"]
    pos = cam["pos"]
    return extr, pos


def loadDepth(dFile, minVal=0, maxVal=10):
    dMap = imread(dFile)
    dMap = dMap.astype(np.float32)
    dMap = dMap*(maxVal-minVal)/(pow(2,16)-1) + minVal
    return dMap


def _dtype_feature(ndarray):
    ndarray = ndarray.flatten()
    """match appropriate tf.train.Feature class with dtype of ndarray. """
    assert isinstance(ndarray, np.ndarray)
    dtype_ = ndarray.dtype
    if dtype_ == np.float64 or dtype_ == np.float32:
        return tf.train.Feature(float_list=tf.train.FloatList(value=ndarray))
    elif dtype_ == np.int64:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=ndarray))
    else:
        raise ValueError("The input should be numpy ndarray. \
                           Instaed got {}".format(ndarray.dtype))


def _string_feature(s):
    s = s.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[s]))
    

def binaryzation(mask):
    mask = np.squeeze(mask)
    l,h = mask.shape
    for i in range(l):
        for j in range(h):      
            if mask[i][j] >= 255 * THRESHOLD:
                mask[i][j] = 255
            else:
                mask[i][j] = 0
    return mask, 0,0

   
def binaryzation2(image):
    image = np.squeeze(image)
    w,h = image.shape
    k =0.95
    wd = np.where(image >= k)
    bd = np.where(image < k)
    wd = np.array(wd)
    wd = np.transpose(wd)
    bd = np.array(bd)
    bd = np.transpose(bd)
    return image, wd, bd


def sdf(image, white_point, w_nSample):
    scale = 255 / (64 * math.sqrt(2))
    image = 255 * image
    w,h = image.shape
    for i in range(w):
        for j in range(h):
            if image[i][j] == 0:
                diffMat = np.tile([i,j],(w_nSample,1)) - white_point
                sqDiffMat = diffMat**2
                sqDistances = sqDiffMat.sum(axis = 1)
                distances = sqDistances ** 0.5
                min_dis = min(distances)
                image[i][j] = 220 - scale * min_dis
    image = image / 255.0
    return image


def mask_sdf(image):
    images = np.squeeze(image)
    image_erzhi, white_point, black_point = binaryzation2(images)
    w_nSample = white_point.shape[0]
    b_nSample = black_point.shape[0]
    if w_nSample == 0 or b_nSample == 0:
        return image
    image_res = sdf(images, white_point, w_nSample)
    image_res = np.expand_dims(image_res, 2)
    return image_res


def create_record(synth_set, split_name, models):
    im_size = FLAGS.image_size
    num_views = FLAGS.num_views
    num_models = len(models)

    mkdir_if_missing(FLAGS.out_dir)

    # address to save the TFRecords file
    train_filename = "{}/{}_{}.tfrecords".format(FLAGS.out_dir, synth_set, split_name)
    # open the TFRecords file
    options = tf_record_options(FLAGS)
    print(train_filename)
    writer = tf.python_io.TFRecordWriter(train_filename, options=options)
   
    render_dir = os.path.join(FLAGS.inp_dir_renders, synth_set)
    voxel_dir = os.path.join(FLAGS.inp_dir_voxels, synth_set)
    for j, model in enumerate(models):
      
        print("{}/{}: {}".format(j, num_models, model))

        if FLAGS.store_voxels:
            voxels_file = os.path.join(voxel_dir, "{}.mat".format(model))
            voxels = loadmat(voxels_file)["Volume"].astype(np.float32)
            voxels = np.transpose(voxels, (1, 0, 2))
            voxels = np.flip(voxels, axis=1)

        im_dir = os.path.join(render_dir, model)
        images = sorted(glob.glob("{}/render_*.png".format(im_dir)))

        rgbs = np.zeros((num_views, im_size, im_size, 3), dtype=np.float32)
        masks = np.zeros((num_views, im_size, im_size, 1), dtype=np.float32)
        mask_sdfs = np.zeros((num_views, im_size, im_size, 1), dtype=np.float32)
        cameras = np.zeros((num_views, 4, 4), dtype=np.float32)
        cam_pos = np.zeros((num_views, 3), dtype=np.float32)
        depths = np.zeros((num_views, im_size, im_size, 1), dtype=np.float32)

        assert(len(images) >= num_views)
        
        error_flag = 0
        for k in range(num_views):
            im_file = images[k]
            img = imread(im_file)
            rgb = img[:, :, 0:3]
            mask = img[:, :, [3]]
            a,b,_ = mask.shape
            erzhi_mask, _ , _ = binaryzation(mask)
            mask = mask / 255.0
            masks_sdf = mask_sdf(mask)
            if True:  
                mask_fg = np.repeat(mask, 3, 2)
                mask_bg = 1.0 - mask_fg
                rgb = rgb * mask_fg + np.ones(rgb.shape)*255.0*mask_bg
            rgb = rgb / 255.0
            actual_size = rgb.shape[0]
            if im_size != actual_size:
                rgb = im_resize(rgb, (im_size, im_size), order=3)
                mask = im_resize(mask, (im_size, im_size), order=3)
            rgbs[k, :, :, :] = rgb
            masks[k, :, :, :] = mask
            mask_sdfs[k,:,:] = masks_sdf

            fn = os.path.basename(im_file)
            img_idx = int(re.search(r'\d+', fn).group())

            if FLAGS.store_camera:
                cam_file = "{}/camera_{}.mat".format(im_dir, img_idx)
                cam_extr, pos = read_camera(cam_file)
                cameras[k, :, :] = cam_extr
                cam_pos[k, :] = pos

            if FLAGS.store_depth:
                depth_file = "{}/depth_{}.png".format(im_dir, img_idx)
                depth = loadDepth(depth_file)
                d_max = 10.0
                d_min = 0.0
                depth = (depth - d_min) / d_max
                depth_r = im_resize(depth, (im_size, im_size), order=0)
                depth_r = depth_r * d_max + d_min
                depths[k, :, :] = np.expand_dims(depth_r, -1)
      
        if error_flag == 1:
            continue
        # Create a feature
        feature = {"image": _dtype_feature(rgbs),
                   "mask": _dtype_feature(masks),
                   "mask_sdfs": _dtype_feature(mask_sdfs),
                   "name": _string_feature(model)}
        if FLAGS.store_voxels:
            feature["vox"] = _dtype_feature(voxels)

        if FLAGS.store_camera:
            feature["extrinsic"] = _dtype_feature(cameras)
            feature["cam_pos"] = _dtype_feature(cam_pos)

        if FLAGS.store_depth:
            feature["depth"] = _dtype_feature(depths)

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


SPLIT_DEF = [("val", 0.05), ("train", 0.95)]


def generate_splits(input_dir):
    files = [f for f in os.listdir(input_dir) if os.path.isdir(f)]
    models = sorted(files)
    random.shuffle(models)
    num_models = len(models)
    models = np.array(models)
    out = {}
    first_idx = 0
    for k, splt in enumerate(SPLIT_DEF):
        fraction = splt[1]
        num_in_split = int(np.floor(fraction * num_models))
        end_idx = first_idx + num_in_split
        if k == len(SPLIT_DEF)-1:
            end_idx = num_models
        models_split = models[first_idx:end_idx]
        out[splt[0]] = models_split
        first_idx = end_idx
    return out


def load_drc_split(base_dir, synth_set):
    filename = os.path.join(base_dir, "{}.file".format(synth_set))
    lines = [line.rstrip('\n') for line in open(filename)]

    k = 3
    split = {}
    while k < len(lines):
        _,_,name,_,_,num = lines[k:k+6]
        k += 6
        num = int(num)
        split_curr = []
        for i in range(num):
            _, _, _, _, model_name = lines[k:k+5]
            k += 5
            split_curr.append(model_name)
        split[name] = split_curr

    return split


def generate_records(synth_set):
    base_dir = FLAGS.split_dir
    split = load_drc_split(base_dir, synth_set)
    for key, value in split.items():
            if key == 'val':
                continue
            create_record(synth_set, key, value)
            break

def read_split(filename):
    f = open(filename, "r")
    lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def main(_):
    generate_records(FLAGS.synth_set)


if __name__ == '__main__':


    app.run()
