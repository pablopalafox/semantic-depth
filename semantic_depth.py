# This file is licensed under a GPLv3 License.
#
# GPLv3 License
# Copyright (C) 2018-2019 Pablo R. Palafox (pablo.palafox@tum.de)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Roborace Vision Pipeline

1. Read frame
2. Segment frame and generate:
    -> FENCE mask
    -> ROAD  mask
3. Produce disparity map by using monodepth network
4. Generate 3D Point Cloud from disparity map
5. Apply masks to 3D Point Cloud and obtain:
    -> road3D Point Cloud
    -> fence3D Point Cloud
6. Compute:
    a) 1. width of road at every depth 
    b) 1. Fit plane to road
       2. Fit planes to fences (there can be 1, 2 or 3 fence objects visible)
       3. intersections -> obtain lane borders
       4. Compute distance between lane borders
'''

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

#import imageio
#imageio.plugins.ffmpeg.download()

import os
import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
#from moviepy.editor import *
import cv2

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import cv2

#----
from monodepth_lib.monodepth_model import *
from monodepth_lib.monodepth_dataloader import *
from monodepth_lib.average_gradients import *
#----

from semantic_depth_lib.point_cloud_2_ply import PointCloud2Ply
import semantic_depth_lib.pcl as pcl

from open3d import *

#########################################################################
#########################################################################

'''
Class for processing frames
'''
class FrameProcessor():    
    def __init__(self, is_city, input_frame, output_directory, 
                 output_name, frame_segmenter, frame_depther, 
                 input_shape, approach, depth, save_data, verbose):

        self.is_city           = is_city
        self.input_frame       = input_frame
        self.output_directory  = output_directory
        self.output_name       = output_name
        self.frame_segmenter   = frame_segmenter
        self.frame_depther     = frame_depther
        self.input_shape       = input_shape
        self.approach          = approach
        self.depth             = depth
        self.save_data         = save_data
        self.verbose           = verbose

    def process_frame(self):

        tic_global = time.time()

        tic_read = time.time()
        # Read frame from its path and store its shape
        print(self.input_frame)
        self.original_frame = cv2.imread(self.input_frame)
        original_shape = self.original_frame.shape
        original_height = original_shape[0]
        original_width = original_shape[1]
        disparity_mult = original_width #original_width
        # Resize the frame to the shape the monodepth network requires
        frame = cv2.resize(self.original_frame, (self.input_shape[1], self.input_shape[0]), 
                           interpolation = cv2.INTER_CUBIC)
        toc_read = time.time()
        time_read_resize = toc_read - tic_read

        ##########################################################################
        ## 1. SEGMENTATION and MASKS ##
        if self.verbose:
            print("\nSegmenting frame...")
        tic_semantic = time.time()
        road_mask, fence_mask, segmented_frame = self.frame_segmenter.segment_frame(frame)
        toc_semantic = time.time()
        time_semantic = toc_semantic - tic_semantic
        if self.verbose:
            print("Semantic Segmentation time: ", time_semantic)
        # Squeeze unnecessary dimensions of the masks
        road_mask  = road_mask.squeeze()  # Remove 3rd-dimension
        fence_mask = fence_mask.squeeze() # Remove 3rd-dimension
        # 10. Save image
        if self.save_data:
            self.segmented_frame = cv2.resize(segmented_frame, (original_width, original_height), 
                                              interpolation = cv2.INTER_CUBIC)
            cv2.imwrite('{}.png'.format(self.output_name), self.segmented_frame)
        
        exit()
        ##########################################################################
        ## 2. DISPARITY MAP ##
        if self.verbose:
            print("\nComputing frame's disparity map...")
        tic_disparity = time.time()
        # Disparities in monodepth are normalized, so we need to scale them by 
        # the full resolution width (2048 for Cityscapes)
        #                           (4032 in an iPhone 8)
        disparity = self.frame_depther.compute_disparity(frame)
        disparity = disparity * disparity_mult
        toc_disparity = time.time()
        time_disparity = toc_disparity - tic_disparity
        if self.verbose:
            print("\nMonodepth time: ", time_disparity)
        if self.save_data:
            self.frame_depther.disp_to_image(disparity, self.output_name, original_height, original_width)
            

        #############################################################################
        ## 3. 3D POINTS: Get 3D points from disparity map and create corresponding ##
        #                color's array
        tic_to3D = time.time()
        if self.verbose:
            print("\nConverting disparity map to 3D Point Cloud...")
        points3D = self.frame_depther.compute_3D_points(disparity)
        colors   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.save_data:
            # For EVERYTHING
            point_cloud = PointCloud2Ply(points3D, colors, '{}_raw'.format(self.output_name))
            point_cloud.prepare_and_save_point_cloud()

        ##########################################################################
        # -- IMPORTANT: NO UTILITY --
        # 4. MASKED IMAGES: Convert RGB image to GRAY and apply masks to obtain 
        #    gray scale images with either a road or a fence on them 
        gray_frame  = cv2.cvtColor(colors, cv2.COLOR_RGB2GRAY)
        road_image  = np.multiply(gray_frame, road_mask)
        fence_image = np.multiply(gray_frame, fence_mask)
        if self.save_data:
            cv2.imwrite('{}_road_mask.png'.format(self.output_name), road_image)
            cv2.imwrite('{}_fence_mask.png'.format(self.output_name), fence_image)

        ##########################################################################
        # 5. Apply masks to the whole 3D points matrix (to colors as well)
        #    to only get road or fence 3D points
        #: ROAD
        road3D       = points3D[road_mask]
        road_colors  = colors[road_mask]
        #: FENCE
        fence3D      = points3D[fence_mask]
        fence_colors = colors[fence_mask]
        
        toc_to3D = time.time()
        time_to3D = toc_to3D - tic_to3D
        if self.verbose:
            print("\nTo 3D time: ", time_to3D)

        if self.save_data:
            np.savez('{}_pointCloud.npz'.format(self.output_name), 
                     road3D=road3D, road_colors=road_colors, 
                     fence3D=fence3D, fence_colors=fence_colors)


        ##########################################################################
        # 6. Remove noise and fit planes        
        #    Remove noise from road 3D point cloud:
        tic_road = time.time()
        
        #    Compute Median Absolute Deviation along 'z' axis in the ROAD Point Cloud
        road3D, road_colors = pcl.remove_from_to(road3D, road_colors, 2, 0.0, 7.0)

        #    Compute Median Absolute Deviation along 'y' axis in the ROAD Point Cloud
        road3D, road_colors = pcl.remove_noise_by_mad(road3D, road_colors, 1, 15.0)

        #    Compute Median Absolute Deviation along 'x' axis in the ROAD Point Cloud
        road3D, road_colors = pcl.remove_noise_by_mad(road3D, road_colors, 0, 2.0)

        #    Find best fitting plane and remove all points too far away from this plane 
        (road3D, road_colors, road_plane3D, road_colors_plane, 
        road_plane_coeff) = pcl.remove_noise_by_fitting_plane(road3D, road_colors,
                                                              axis=1, 
                                                              threshold=5.0, 
                                                              plane_color=[200, 200, 200])
        toc_road = time.time()
        time_road = toc_road - tic_road


        #################################################################################

        # read into open3d 
        road3D_pcd = PointCloud()
        road3D_pcd.points = Vector3dVector(road3D)
        road3D_pcd.colors = Vector3dVector(road_colors)
        # write_point_cloud("test_road.ply", road3D_pcd)

        # remove some more outliers
        print("Statistical oulier removal")
        cl,ind = statistical_outlier_removal(road3D_pcd,
            nb_neighbors=10, std_ratio=0.5)
        inlier_cloud = select_down_sample(road3D_pcd, ind)

        cl,ind = radius_outlier_removal(inlier_cloud,
            nb_points=80, radius=0.5)

        inlier_cloud = select_down_sample(inlier_cloud, ind)

        # go back to numpy array
        road3D = np.asarray(inlier_cloud.points)
        road_colors = np.asarray(inlier_cloud.colors)

        #################################################################################
        #################################################################################

        #################################################################################
        ####################### rw APPROACH ##########################################
        #   Get 3D points that define a horizontal line at a certain depth
        tic_rw = time.time()
        left_pt_rw, right_pt_rw = pcl.get_end_points_of_road(road3D, 
                                                                   self.depth-0.02)
        # np.savez('{}_nai.npz'.format(self.output_name),
        #          left_pt_rw=left_pt_rw, right_pt_rw=right_pt_rw)
        #dist_rw = pcl.compute_distance_in_3D(left_pt_rw, right_pt_rw)
        dist_rw = abs(left_pt_rw[0][0] - right_pt_rw[0][0])
        if self.verbose:
            print("Road width", dist_rw)
        line_rw, colors_line_rw = pcl.create_3Dline_from_3Dpoints(left_pt_rw, 
                                                                        right_pt_rw,
                                                                        [250,0,0])
        line_rw[:,2] += 0.2 # for better visualization, shift it a bit

        toc_rw = time.time()
        time_rw = toc_rw - tic_rw
        if self.verbose:
            print("\nrw time: ", time_rw)
        #################################################################################

        if self.approach == 'both':
            tic_fences = time.time()
            ##########################################################################
            # 6.B Remove noise from fence 3D point cloud:
            # 0. Separate into LEFT and RIGHT fence
            #   0.1 But before, remove outliers that go to infinity upwards
            fence3D, fence_colors = pcl.remove_noise_by_mad(fence3D,  fence_colors, 
                                                            1, 5.0)
            #   0.2 Then, remove all points whose 'z' (2) value is greater than 
            #       a certain value (we set it to 30.0)
            fence3D, fence_colors = pcl.threshold_complete(fence3D, fence_colors,
                                                           2, 35.0)
            #   0.3 Separate into LEFT and RIGHT fences
            (fence3D_left, fence_left_colors, 
                fence3D_right, fence_right_colors) = pcl.extract_pcls(fence3D, fence_colors)

            #### -- LEFT FENCE
            # 1. Compute Median Absolute Deviation along 'x' axis in the LEFT FENCE Point Cloud
            fence3D_left, fence_left_colors = pcl.remove_noise_by_mad(fence3D_left, fence_left_colors, 0, 5.0)
            # 2. Find best fitting plane and remove all points too far away from this plane 
            
            (fence3D_left, fence_left_colors, fence_left_plane3D, fence_left_colors_plane, 
                fence_left_plane_coeff) = pcl.remove_noise_by_fitting_plane(fence3D_left, fence_left_colors,
                                                                            axis=0, 
                                                                            threshold=1.0, 
                                                                            plane_color=[40, 70, 40])
            
            #### -- RIGHT FENCE
            # 1. Compute Median Absolute Deviation along 'x' axis in the RIGHT FENCE Point Cloud
            fence3D_right, fence_right_colors = pcl.remove_noise_by_mad(fence3D_right, fence_right_colors, 0, 1.0)
            # 2. Find best fitting plane and remove all points too far away from this plane
            
            (fence3D_right, fence_right_colors, fence_right_plane3D, fence_right_colors_plane,
                fence_right_plane_coeff) = pcl.remove_noise_by_fitting_plane(fence3D_right, fence_right_colors,
                                                                             axis=0, 
                                                                             threshold=1.0, 
                                                                             plane_color=[40, 70, 40])                                                    
            toc_fences = time.time()
            time_fences = toc_fences - tic_fences

            ####################################################################################
            ############################ f2f APPROACH ##########################################
            ######## ROAD-LEFT_FENCE intersection at a certain depth ###########################
            tic_f2f = time.time()
            left_pt_f2f  = pcl.planes_intersection_at_certain_depth(road_plane_coeff, 
                                                                         fence_left_plane_coeff, 
                                                                         z=self.depth)

            right_pt_f2f = pcl.planes_intersection_at_certain_depth(road_plane_coeff, 
                                                                         fence_right_plane_coeff, 
                                                                         z=self.depth)
            dist_f2f = pcl.compute_distance_in_3D(left_pt_f2f, right_pt_f2f)
            if self.verbose:
                print("Distance from fence to fence:", dist_f2f)
            line_f2f, colors_line_f2f = pcl.create_3Dline_from_3Dpoints(left_pt_f2f, 
                                                                                  right_pt_f2f,
                                                                                  [0,255,0])
        
            toc_f2f = time.time()
            time_f2f = toc_f2f - tic_f2f
            if self.verbose:
                print("\nf2f time: ", time_f2f)
        

        ##########################################################################
        # 9. Draw letters in the image
        if self.save_data:

            self.segmented_frame = cv2.resize(segmented_frame, (original_width, original_height), 
                                              interpolation = cv2.INTER_CUBIC)

            # Save image featuring only the segmentation
            cv2.imwrite('{}_only_segmentation.png'.format(self.output_name), self.segmented_frame)

            h = original_height
            w = original_width

            if self.is_city:
                thickness = 2
                fontScale = 2
                left = 0.01
                right = 0.68
                middle = 0.33
                h_zero = 0.05 * h
                h_first = 0.12 * h
                h_second = 0.18 * h
            else:
                thickness = 5
                fontScale = 4
                left = 0.01
                right = 0.67
                middle = 0.33
                h_zero = 0.05 * h
                h_first = 0.12 * h
                h_second = 0.18 * h


            cv2.rectangle(self.segmented_frame,(0,0),(w, int(0.2*h)),(156, 157, 159), -1)
            cv2.putText(self.segmented_frame, 'At {:.2f}m depth:'.format(self.depth),
                        (int(middle*w), int(h_zero)),
                        fontFace = 16, fontScale = fontScale, color=(255,255,255), thickness = thickness)
        

            if self.approach == 'both':
                cv2.putText(self.segmented_frame, '{:.2f}m to l fence'.format(-left_pt_f2f[0][0]), 
                            (int(left*w), int(h_first)),
                            fontFace = 16, fontScale = fontScale, color=(255,255,255), thickness = thickness)
                cv2.putText(self.segmented_frame, '{:.2f}m to r fence'.format(right_pt_f2f[0][0]), 
                            (int(right*w), int(h_first)),
                            fontFace = 16, fontScale = fontScale, color=(255,255,255), thickness = thickness)
                cv2.putText(self.segmented_frame, 'Fence2Fence: {:.2f}m'.format(dist_f2f), 
                        (int(middle*w), int(h_first)),
                        fontFace = 16, fontScale = fontScale, color=(255,255,255), thickness = thickness)
            
            cv2.putText(self.segmented_frame, '{:.2f}m to road\'s l'.format(-left_pt_rw[0][0]), 
                        (int(left*w), int(h_second)),
                        fontFace = 16, fontScale = fontScale, color=(255,255,255), thickness = thickness)
            cv2.putText(self.segmented_frame, '{:.2f}m to road\'s r'.format(right_pt_rw[0][0]), 
                        (int(right*w), int(h_second)),
                        fontFace = 16, fontScale = fontScale, color=(255,255,255), thickness = thickness)
            cv2.putText(self.segmented_frame, 'Road\'s width: {:.2f}m'.format(dist_rw), 
                        (int(middle*w), int(h_second)),
                        fontFace = 16, fontScale = fontScale, color=(255,255,255), thickness = thickness)

            ##########################################################################
            # 10. Save image
            cv2.imwrite('{}.png'.format(self.output_name), self.segmented_frame)

            ######################################################
            # 98. Save Point Cloud to ply file to check results

            # semantic 3D Point Cloud
            point_cloud_road = PointCloud2Ply(road3D, road_colors, '{}_ROAD'.format(self.output_name))
            point_cloud_road.prepare_and_save_point_cloud()
            
            if self.approach == 'both':
                point_cloud_fence = PointCloud2Ply(fence3D_left, fence_left_colors, '{}_FENCE'.format(self.output_name))
                point_cloud_fence.add_extra_point_cloud(fence3D_right, fence_right_colors)
                point_cloud_fence.prepare_and_save_point_cloud()

            # For ROAD
            #point_cloud = PointCloud2Ply(road3D, road_colors, self.output_name)
            #point_cloud.add_extra_point_cloud(road_plane3D, road_colors_plane)

            # For FENCEs and ROAD
            point_cloud = PointCloud2Ply(road3D, road_colors, self.output_name)
            point_cloud.add_extra_point_cloud(road_plane3D, road_colors_plane)
            point_cloud.add_extra_point_cloud(line_rw, colors_line_rw)

            if self.approach == 'both':

                point_cloud.add_extra_point_cloud(fence3D_left, fence_left_colors)
                point_cloud.add_extra_point_cloud(fence3D_right, fence_right_colors)
                
                point_cloud.add_extra_point_cloud(fence_left_plane3D, fence_left_colors_plane)
                point_cloud.add_extra_point_cloud(fence_right_plane3D, fence_right_colors_plane)
                point_cloud.add_extra_point_cloud(line_f2f, colors_line_f2f)
            
            point_cloud.prepare_and_save_point_cloud()
            
            # For EVERYTHING
            point_cloud = PointCloud2Ply(points3D, colors, '{}_ALL'.format(self.output_name))
            point_cloud.add_extra_point_cloud(line_rw, colors_line_rw)
            if self.approach == 'both':
                point_cloud.add_extra_point_cloud(line_f2f, colors_line_f2f)
            point_cloud.prepare_and_save_point_cloud()

        toc_global = time.time()
        time_global = toc_global - tic_global

        #################################################################################
        # Save data to files
        with open('{}_times.txt'.format(self.output_name), 'w') as f:
            f.write("Time read:       {}\n".format(time_read_resize))
            f.write("Time semantic:   {}\n".format(time_semantic))
            f.write("Time disparity:  {}\n".format(time_disparity))
            f.write("Time to3D:       {}\n".format(time_to3D))
            f.write("Time road:       {}\n".format(time_road))
            f.write("Time rw:      {}\n".format(time_rw))
            f.write("Time fences:     {}\n".format(time_fences))
            f.write("Time f2f:   {}\n".format(time_f2f))
            f.write("Time global:     {}\n".format(time_global))

        with open('{}_distances.txt'.format(self.output_name), 'w') as f:
            f.write("rw distance:    {}\n".format(dist_rw))
            f.write("f2f distance: {}\n".format(dist_f2f))

        return dist_rw, dist_f2f


class SegmentFrame():
    def __init__(self, input_shape, model_var_dir, use_frozen, use_xla, CUDA_DEVICE_NUMBER):
        self.input_shape = input_shape
        self.model_var_dir = model_var_dir
        self.CUDA_DEVICE_NUMBER = CUDA_DEVICE_NUMBER

        self.restore_model(use_frozen, use_xla)


    def load_graph(self, graph_file, use_xla):
    
        jit_level = 0
        config = tf.ConfigProto()
        
        if use_xla:
            
            jit_level = tf.OptimizerOptions.ON_1
            config.graph_options.optimizer_options.global_jit_level = jit_level

        with tf.Session(graph=tf.Graph(), config=config) as sess:
            
            gd = tf.GraphDef()
            
            with tf.gfile.Open(graph_file, 'rb') as f:
                data = f.read()
                gd.ParseFromString(data)
            
            tf.import_graph_def(gd, name='')
            
            ops = sess.graph.get_operations()
            n_ops = len(ops)
            
            return sess, ops


    def restore_model(self, use_frozen=True, use_xla=False):

        if use_frozen:
            print("\n\nRestoring (frozen) segmentation model...")
            
            graph_file = '{}/optimized_graph.pb'.format(self.model_var_dir)
            sess, _ = self.load_graph(graph_file, use_xla)

            self.sess = sess
            graph = self.sess.graph

            self.keep_prob   = graph.get_tensor_by_name('keep_prob:0')
            self.input_image = graph.get_tensor_by_name('image_input:0')
            self.logits      = graph.get_tensor_by_name('logits:0')
            
            print("Segmentation model successfully restored!")

        else:
            print("\n\nRestoring segmentation model...")

            os.environ["CUDA_VISIBLE_DEVICES"]=self.CUDA_DEVICE_NUMBER
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.visible_device_list = "0"

            #config = tf.ConfigProto(allow_soft_placement=True)

            self.sess = tf.Session(config=config)
            
            model_meta_file = "{}/variables/saved_model.meta".format(self.model_var_dir)
            # print(model_meta_file)

            new_saver = tf.train.import_meta_graph(model_meta_file)
            new_saver.restore(self.sess, tf.train.latest_checkpoint(self.model_var_dir+"/variables"))
            
            graph = tf.get_default_graph()
            
            self.keep_prob   = graph.get_tensor_by_name('keep_prob:0')
            self.input_image = graph.get_tensor_by_name('image_input:0')
            self.logits      = graph.get_tensor_by_name('logits:0')
            
            self.sess.run(tf.local_variables_initializer())
            
            print("Segmentation model successfully restored!")


    def segment_frame(self, frame):

        # Note that the frame has already been resized by this time 
        # to the ``input_shape`` dimensions
        street_im = scipy.misc.toimage(frame)

        im_softmax = self.sess.run(
                [tf.nn.softmax(self.logits)],
                {self.keep_prob: 1.0, self.input_image: [frame]})

        # Road
        im_softmax_road = im_softmax[0][:, 0].reshape(self.input_shape[0], self.input_shape[1])
        segmentation_road = (im_softmax_road > 0.5).reshape(self.input_shape[0], self.input_shape[1], 1)
        road_mask = np.dot(segmentation_road, np.array([[128, 64, 128, 64]]))
        road_mask = scipy.misc.toimage(road_mask, mode="RGBA")
        #scipy.misc.imsave('road.png', road_mask)
        street_im.paste(road_mask, box=None, mask=road_mask)

        # Fence
        im_softmax_fence = im_softmax[0][:, 1].reshape(self.input_shape[0], self.input_shape[1])
        segmentation_fence = (im_softmax_fence > 0.5).reshape(self.input_shape[0], self.input_shape[1], 1)
        fence_mask = np.dot(segmentation_fence, np.array([[160, 10, 10, 64]]))
        fence_mask = scipy.misc.toimage(fence_mask, mode="RGBA")
        #scipy.misc.imsave('fence.png', fence_mask)
        street_im.paste(fence_mask, box=None, mask=fence_mask)
        #scipy.misc.imsave('fence_overlaid.png', street_im)

        return segmentation_road, segmentation_fence, np.array(street_im)


class DepthFrame():
    def __init__(self, is_city=False, encoder='vgg', input_height=256, input_width=512, 
             checkpoint_path='models/monodepth/model_cityscapes/model_cityscapes',
             f=None):

        self.is_city         = is_city
        self.encoder         = encoder
        self.input_height    = input_height
        self.input_width     = input_width
        self.checkpoint_path = checkpoint_path  

        # Computed after calibrating chessboard images taken by iPhone 8 
        #fx = 480.08864363
        #fy = 322.31613675
        self.f = float(f) if f is not None else None
        # trial and error! No other way, since the monodepth network is obscure when 
        # working with images that are not from the dataset on which you trained
        
        if self.is_city:
            print("Setting params from Cityscapes")
            # Cityscapes
            self.cx = 1048.64 / 4
            self.cy = 519.277 / 4
            self.b = 0.6
            if self.f is None:
                self.f = 500 # found empirically
        else:
            print("Setting params from iPhone 6 rear camera")
            # Munich test set (our own dummy dataset)
            self.cx = 314.05519001 
            self.cy = 124.09658151
            self.b  = 1
            if self.f is None:
                self.f = 380 # works well for most images in the Munich test set

        self.params = monodepth_parameters(
            encoder=self.encoder,
            height=self.input_height,
            width=self.input_width,
            batch_size=2,
            num_threads=1,
            num_epochs=1,
            do_stereo=False,
            wrap_mode="border",
            use_deconv=False,
            alpha_image_loss=0,
            disp_gradient_loss_weight=0,
            lr_loss_weight=0,
            full_summary=False)

        self.restore_model()
        

    def restore_model(self):
        print("\n\nRestoring monodepth model...")

        self.graph_depth = tf.Graph()
        with self.graph_depth.as_default():

            self.left  = tf.placeholder(tf.float32, [2, self.input_height, self.input_width, 3])
            self.model = MonodepthModel(self.params, "test", self.left, None)

            # SESSION
            config = tf.ConfigProto(allow_soft_placement=True)
            self.sess = tf.Session(config=config)

            # SAVER
            train_saver = tf.train.Saver()

            # INIT
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coordinator)

            # RESTORE
            restore_path = self.checkpoint_path
            train_saver.restore(self.sess, restore_path)
        
        print("Monodepth model successfully restored!")


    def post_processing(self, disp):
        _, h, w = disp.shape
        l_disp = disp[0,:,:]
        r_disp = np.fliplr(disp[1,:,:])
        m_disp = 0.5 * (l_disp + r_disp)
        l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
        r_mask = np.fliplr(l_mask)
        return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


    def compute_disparity(self, frame):
        
        # Note that the frame has already been resized by this time 
        # to the ``input_shape`` dimensions
        frame = frame.astype(np.float32) / 255
        input_frames = np.stack((frame, np.fliplr(frame)), 0)

        with self.graph_depth.as_default():
            disp = self.sess.run(self.model.disp_left_est[0], feed_dict={self.left: input_frames})
        disp_pp = self.post_processing(disp.squeeze()).astype(np.float32)

        return disp_pp


    def disp_to_image(self, disp_pp, output_name, original_height, original_width):
        disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width])
        plt.imsave("{}_disp.png".format(output_name), disp_to_img, cmap='gray') # cmap='plasma'


    def compute_3D_points(self, disp):

        print("focal length", self.f)
        print("baseline", self.b)

        Q = np.float32([[1, 0, 0, - self.cx  ],
                        [0,-1, 0,   self.cy  ],  # turn points 180 deg around x-axis,
                        [0, 0, 0, - self.f   ],  #     so that y-axis looks up, and z-axis looks to you
                        [0, 0, 1/self.b, 0 ]]) #     Therefore, points will have negative z values

        points3D = cv2.reprojectImageTo3D(disp, Q)
        return points3D


def main():

    parser = argparse.ArgumentParser(description="Read frame and "
                            "compute the distance from the center "
                            "of the car to the fences.")

    parser.add_argument("--input_folder", help="Path to all frames we want to work with.", 
                        default="data/test_images_munich")

    parser.add_argument("--input_frame", help="COMPLETE path to frame we want to work with. "\
                                              "Only set to test the system on one single image ", 
                        default="data/test_images_munich/test_3.png")
                        #default="media/images/bielefeld_018644.png")

    parser.add_argument("--semantic_model", help="Path to semantic segmentation model.", 
                       default="models/sem_seg/100-Epochs-roborace750") # 100-Epochs-roborace750
    

    parser.add_argument("--monodepth_checkpoint", help="Path to monodepthcheckpoint.", 
                        default="models/monodepth/model_cityscapes/model_cityscapes")

    parser.add_argument('--monodepth_encoder', type=str,   
                        help='type of encoder, vgg or resnet50', default='vgg')

    parser.add_argument('--input_height', type=int, 
                        help='input height', 
                        default=256)

    parser.add_argument('--input_width', type=int, 
                        help='input width', 
                        default=512)

    parser.add_argument('--approach', type=str,
                        help='approach for measuring road width',
                        default='both')

    parser.add_argument('--depth', type=float,
                        help='depth at which to compute road\'s width',
                        default=10)

    parser.add_argument('--f', type=float,
                        help='focal length',
                        default=350)

    parser.add_argument('--save_data',
                        help='If set, images and ply files are saved to disk',
                        action='store_true')

    parser.add_argument('--use_frozen',
                        help='If set, uses frozen model',
                        action='store_true')

    parser.add_argument('--use_xla',
                        help='If set, uses xla',
                        action='store_true')

    parser.add_argument('--CUDA_DEVICE_NUMBER',
                        help='Number of GPU device to use (e.g., 0, 1, 2, ...)',
                        default="0")

    parser.add_argument('--verbose',
                        help='If set, prints info',
                        action='store_true')

    parser.add_argument('--is_city',
                        help='Set if using images from Cityscapes dataset',
                        action='store_true')


    args = parser.parse_args()

    # Input size
    input_shape = (args.input_height, args.input_width)

    ################################################################################
    # Create a DepthFrame object
    ################################################################################
    frame_depther = DepthFrame(args.is_city,
                               args.monodepth_encoder, 
                               args.input_height, 
                               args.input_width,
                               args.monodepth_checkpoint,
                               args.f)

    ################################################################################
    # Create a SegmentFrame object
    ################################################################################
    frame_segmenter = SegmentFrame(input_shape, args.semantic_model,
                                   args.use_frozen, args.use_xla, 
                                   args.CUDA_DEVICE_NUMBER)
    
    ################################################################################
    # Apply pipeline to frame/frames
    ################################################################################

    # Only 1 frame
    if args.input_frame: 

        print()
        print()
        print("##########################################################")
        print("##### {} - focal length: {}  #####".format(args.input_frame, args.f))
        print("##########################################################")

        input_frame = args.input_frame

        # Create output frame path
        output_directory = "results"
      
        output_name = os.path.basename(input_frame)
        output_name = os.path.splitext(output_name)[0]
        
        output_directory = os.path.join(output_directory, output_name)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        output_name  = os.path.join(output_directory, "{}_output".format(output_name))

        # Create a FrameProcessor object
        frame_processor = FrameProcessor(args.is_city,
                                         input_frame,
                                         output_directory,
                                         output_name, 
                                         frame_segmenter,
                                         frame_depther,
                                         input_shape,
                                         args.approach,
                                         args.depth,
                                         args.save_data,
                                         args.verbose)

        # Process input frame
        frame_processor.process_frame()

    # A series of frames
    else:

        input_frames = {"test_1.png": 5.3, "test_2.png": 4.4, "test_3.png": 5.4, "test_4.png": 3.1, "test_5.png": 4.6}
        print()
        print()
        print("Series of frames: ")
        print(input_frames)

        if args.f is None:

            best_mae_rw = -1
            best_f_rw = None

            best_mae_f2f = -1
            best_f_f2f = None

            best_mae_overall = -1
            best_f_overall = None

            focal_lengths = [380, 580]

            for f in focal_lengths:

                # Update focal in lenght in frame_depther object
                frame_depther.f = f

                # Create the corresponding folder for this new focal length trial
                f_directory = os.path.join('results', str(f))
                if not os.path.exists(f_directory):
                        os.makedirs(f_directory)

                all_data = []
                for input_frame, real_distance in sorted(input_frames.items()):

                    print()
                    print()
                    print("####################################################")
                    print("#####    focal length: {} - images: {}".format(f, input_frame))
                    print("#####    real distance at 10 m:   {}".format(real_distance))
                    print("####################################################")
                    input_frame = os.path.join(args.input_folder, input_frame)

                    # Create output frame path              
                    output_name = os.path.basename(input_frame)
                    output_name = os.path.splitext(output_name)[0]
                    
                    output_directory = os.path.join(f_directory, output_name)
                    if not os.path.exists(output_directory):
                        os.makedirs(output_directory)
                    
                    output_name  = os.path.join(output_directory, "{}_output".format(output_name))

                    # Create a FrameProcessor object
                    frame_processor = FrameProcessor(args.is_city,
                                                     input_frame,
                                                     output_directory,
                                                     output_name, 
                                                     frame_segmenter,
                                                     frame_depther,
                                                     input_shape,
                                                     args.approach,
                                                     args.depth,
                                                     args.save_data,
                                                     args.verbose)

                    # Process input frame
                    dist_rw, dist_f2f = frame_processor.process_frame()

                    test_data = []
                    test_data.extend( ( real_distance, dist_rw, dist_f2f, abs(real_distance - dist_rw), abs(real_distance - dist_f2f) ) )
                    all_data.append(test_data)

                all_data_array = np.asarray(all_data)

                # Compute MAE for both the rw and f2f approaches
                abs_errors_rw    = all_data_array[:,3]
                abs_errors_f2f = all_data_array[:,4]

                mae_rw    = np.sum(abs_errors_rw)    / len(input_frames)
                mae_f2f = np.sum(abs_errors_f2f) / len(input_frames)
                mae_overall  = mae_rw + mae_f2f

                mae_for_file      = np.zeros((1,5))
                mae_for_file[:,3] = mae_rw
                mae_for_file[:,4] = mae_f2f

                # Register the best focal length
                if best_mae_rw == -1 or mae_rw < best_mae_rw:
                    best_mae_rw = mae_rw
                    best_f_rw = f

                if best_mae_f2f == -1 or mae_f2f < best_mae_f2f:
                    best_mae_f2f = mae_f2f
                    best_f_f2f = f

                if best_mae_overall == -1 or mae_overall < best_mae_overall:
                    best_mae_overall = mae_overall
                    best_f_overall = f

                all_data_array = np.concatenate((all_data_array, mae_for_file))

                np.savetxt("{}/data.txt".format(f_directory), all_data_array, fmt='%1.4f')
                print("Data saved for focal length: {}".format(f))

            results_directory = "results"
            with open('{}/best_focal_lengths.txt'.format(results_directory), 'w') as f:
                f.write("Best f road's width: {}\n".format(best_f_rw))
                f.write("Best f fence2fence:  {}\n".format(best_f_f2f))
                f.write("Best f overall:      {}\n".format(best_f_overall))
            print("Best focal lenghts file generated!")

        else: # if argument f has been set

            # Update focal in lenght in frame_depther object
            frame_depther.f = f

            # Create the corresponding folder for this new focal length trial
            f_directory = os.path.join('results', str(f))
            if not os.path.exists(f_directory):
                    os.makedirs(f_directory)

            all_data = []
            for input_frame, real_distance in sorted(input_frames.items()):

                print()
                print()
                print("##########################################################")
                print("#####    {} - f: {}             #####".format(input_images, f))
                print("#####    real distance at 10 m: {}            #####".format(real_distance))
                print("##########################################################")

                input_frame = os.path.join(args.input_folder, input_frame)

                # Create output frame path              
                output_name = os.path.basename(input_frame)
                output_name = os.path.splitext(output_name)[0]
                
                output_directory = os.path.join(f_directory, output_name)
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory)
                
                output_name  = os.path.join(output_directory, "{}_output".format(output_name))

                # Create a FrameProcessor object
                frame_processor = FrameProcessor(args.is_city,
                                                 input_frame,
                                                 output_directory,
                                                 output_name, 
                                                 frame_segmenter,
                                                 frame_depther,
                                                 input_shape,
                                                 args.approach,
                                                 args.depth,
                                                 args.save_data)

                # Process input frame
                dist_rw, dist_f2f = frame_processor.process_frame()

                test_data = []
                test_data.extend( ( real_distance, dist_rw, dist_f2f, abs(real_distance - dist_rw), abs(real_distance - dist_f2f) ) )
                all_data.append(test_data)

            all_data_array = np.asarray(all_data)

            # Compute MAE for both the rw and f2f approaches
            abs_errors_rw    = all_data_array[:,3]
            abs_errors_f2f = all_data_array[:,4]

            mae_rw    = np.sum(abs_errors_rw)    / len(input_frames)
            mae_f2f = np.sum(abs_errors_f2f) / len(input_frames)
            mae_overall  = mae_rw + mae_f2f

            mae_for_file      = np.zeros((1,5))
            mae_for_file[:,3] = mae_rw
            mae_for_file[:,4] = mae_f2f

            all_data_array = np.concatenate((all_data_array, mae_for_file))

            np.savetxt("{}/data.txt".format(f_directory), all_data_array, fmt='%1.4f')
            print("Data saved for focal lenght: {}".format(f))


if __name__ == "__main__":
    main()