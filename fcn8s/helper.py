# MIT License
# 
# Copyright (c) 2017-2018 Udacity, Inc
# Copyright (c) Modifications 2018, 2019 Pablo R. Palafox (pablo.rodriguez-palafox [at] tum.de)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
# of the Software, and to permit persons to whom the Software is furnished to do 
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
import cv2
from glob import glob
from urllib.request import urlretrieve
from distutils.version import LooseVersion
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime


class DLProgress(tqdm):
    last_block = 0
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def check_compatibility():
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
    print('TensorFlow Version: {}'.format(tf.__version__))
    
    if not tf.test.gpu_device_name():
        print('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def img_size(img):
    return (img.shape[0], img.shape[1])


def random_crop(img, gt):
    h,w = img_size(img)
    nw = random.randint(768, w-2) # Random crop size
    nh = int(nw / 2) # Keep original aspect ration
    x1 = random.randint(0, w - nw) # Random position of crop
    y1 = random.randint(0, h - nh)
    return img[y1:(y1+nh), x1:(x1+nw), :], gt[y1:(y1+nh), x1:(x1+nw)]


def bc_img(img, s = 1.0, m = 0.0):
    img = img.astype(np.int)
    img = img * s + m
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img  


def get_files_paths(gt_dir, imgs_dir):
    """
    Get training data filenames
    """    
    cities = os.listdir(imgs_dir)
    gt = []
    imgs = []
    for city in cities:
        new_gt_path = os.path.join(gt_dir, city)
        new_imgs_path = os.path.join(imgs_dir, city)
        gt += glob(os.path.join(new_gt_path, "*_gtFine_labelIds.png"))
        imgs += glob(os.path.join(new_imgs_path, "*.png"))
    gt.sort()
    imgs.sort()
    return gt, imgs


def get_num_imgs_in_folder(imgs_dir):
    """
    Sum the number of images contained in each city
    """
    cities = os.listdir(imgs_dir)
    num_imgs = 0
    for city in cities:
        city_path = os.path.join(imgs_dir, city)
        num_imgs += len(os.listdir(city_path)) 

    return num_imgs


def prepare_ground_truth(dataset, img, num_classes, mode='train'):
    """
    Prepare ground truth for cityscape data
    """
    new_image = np.zeros((img.shape[0], img.shape[1], num_classes))
    
    # road
    road_mask = img == 7

    # Depending on the dataset, the ``fence_mask`` will be generated differently
    if dataset[0:4] == 'city':
        if mode == 'train':
            # construction[building, wall, fence, guard_rail, bridge, tunnel]
            fence_mask = np.logical_or.reduce((img == 11, img == 12, img == 13,
                                              img == 14, img == 15, img == 16))
        elif mode == 'test':  
            fence_mask = img == 13
    
    elif dataset[0:4] == 'robo':
        fence_mask = img == 13

    # everything else
    else_mask = np.logical_not(np.logical_or.reduce((road_mask, fence_mask)))

    new_image[:,:,0] = road_mask
    new_image[:,:,1] = fence_mask
    new_image[:,:,2] = else_mask

    return new_image.astype(np.float32)


def gen_batch_function(train_gt_dir, train_imgs_dir, 
                       val_gt_dir, val_imgs_dir, 
                       test_gt_dir, test_imgs_dir, 
                       image_shape, dataset):
    """
    Generate function to create batches of training data
    """
    def get_batches_fn(batch_size=1, mode='train', num_classes=3, print_flag=False):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        if mode == 'train':

            # Get only the path of the imgs. Ground truth images' paths will be obtained later
            _, imgs_paths = get_files_paths(train_gt_dir, train_imgs_dir)

            #background_color = np.array([255, 0, 0])
            #road_color = np.array([128, 64, 128, 255])
            #car_color = np.array([0, 0, 142, 255])

            random.shuffle(imgs_paths)

            for batch_i in range(0, len(imgs_paths), batch_size):

                images    = []
                gt_images = []

                for image_file in imgs_paths[batch_i:batch_i+batch_size]:

                    # Get gt_image_file by first finding the city name and then renaming the basename of image_file
                    city = os.path.basename(image_file).partition("_")[0]
                    gt_type = 'gtFine_labelIds.png'
                    gt_image_file = os.path.join(train_gt_dir, city, os.path.basename(image_file)[:-15]+gt_type)
                    
                    # Read images and groundtruth images
                    image    = scipy.misc.imread(image_file)
                    gt_image = scipy.misc.imread(gt_image_file)

                    # Show images and gt_images as they are
                    if print_flag:
                        plt.figure(figsize=(16, 8))
                        plt.subplot(2,2,1)
                        plt.imshow(image)
                        plt.subplot(2,2,2)
                        plt.imshow(gt_image)

                    #####################################################
                    # AUGMENTATION #
                    #Random crop augmentation
                    image, gt_image = random_crop(image, gt_image) 
                    image = scipy.misc.imresize(image, image_shape)
                    gt_image = scipy.misc.imresize(gt_image, image_shape)

                    # Contrast augmentation
                    contr = random.uniform(0.85, 1.15) 
                    # Brightness augmentation
                    bright = random.randint(-40, 30) 
                    image = bc_img(image, contr, bright)
                    #####################################################

                    #####################################################
                    # PREPARE GROUND TRUTH
                    gt_image = prepare_ground_truth(dataset, gt_image, num_classes)
                    #####################################################

                    images.append(image)
                    gt_images.append(gt_image)

                    if print_flag:
                        plt.subplot(2,2,3)
                        plt.imshow(image)
                        plt.subplot(2,2,4)
                        gt_image = scipy.misc.imresize(gt_image, image_shape)
                        plt.imshow(gt_image)
                        plt.show()


                yield np.array(images), np.array(gt_images)


        elif mode == 'val':

            _, imgs_paths = get_files_paths(val_gt_dir, val_imgs_dir)

            #background_color = np.array([255, 0, 0])
            #road_color = np.array([128, 64, 128, 255])
            #car_color = np.array([0, 0, 142, 255])

            random.shuffle(imgs_paths)

            for batch_i in range(0, len(imgs_paths), batch_size):

                images    = []
                gt_images = []

                for image_file in imgs_paths[batch_i:batch_i+batch_size]:

                    # Get gt_image_file by first finding the city name and then renaming the basename of image_file
                    city = os.path.basename(image_file).partition("_")[0]
                    gt_type = 'gtFine_labelIds.png'
                    gt_image_file = os.path.join(val_gt_dir, city, os.path.basename(image_file)[:-15]+gt_type)
                    
                    # Read images and groundtruth images
                    image    = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                    gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                    # Show images and gt_images as they are
                    if print_flag:
                        plt.figure(figsize=(16, 8))
                        plt.subplot(2,2,1)
                        plt.imshow(image)
                        plt.subplot(2,2,2)
                        plt.imshow(gt_image)

                    #####################################################
                    # PREPARE GROUND TRUTH
                    gt_image = prepare_ground_truth(dataset, gt_image, num_classes)
                    #####################################################

                    images.append(image)
                    gt_images.append(gt_image)

                    if print_flag:
                        plt.subplot(2,2,3)
                        plt.imshow(image)
                        plt.subplot(2,2,4)
                        gt_image = scipy.misc.imresize(gt_image, image_shape)
                        plt.imshow(gt_image)
                        plt.show()

                yield np.array(images), np.array(gt_images)

    return get_batches_fn