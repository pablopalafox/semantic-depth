# This file is licensed under a GPLv3 License.
#
# GPLv3 License
# Copyright (C) 2018-2019 Pablo R. Palafox (pablo.rodriguez-palafox@tum.de)
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

1. Read frame from cam
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

import imageio

imageio.plugins.ffmpeg.download()

import os
import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt
from moviepy.editor import *
import cv2
import glob

#----
from thirdparty.monodepth_lib.monodepth_model import *
from thirdparty.monodepth_lib.monodepth_dataloader import *
from thirdparty.monodepth_lib.average_gradients import *
#----

from semantic_depth_lib.point_cloud_2_ply   import PointCloud2Ply
import semantic_depth_lib.pcl as pcl

'''
Class for processing frames
'''
class FrameProcessor():

    disp_multiplier = 3800 # PARTICULAR FOR CITYSAPES (width of original images in dataset)

    def __init__(self, frame_segmenter, frame_depther, input_shape, approach, depth,
                 verbose):

        self.frame_segmenter   = frame_segmenter
        self.frame_depther     = frame_depther
        self.input_shape       = input_shape
        self.approach          = approach
        self.depth             = depth
        self.verbose           = verbose

    def process_frame(self, input_frame, output_name,
                      result_images_dir, result_ply_dir):

        print("\n\nPROCESSING NEW FRAME! \n")

        # Read frame from its path and store its shape
        original_frame = cv2.imread(input_frame)
        original_shape = original_frame.shape
        h = original_shape[0]
        w = original_shape[1]
        # Resize the frame to the shape the monodepth network requireds
        #frame = scipy.misc.imresize(original_frame, self.input_shape, interp='lanczos')
        frame = cv2.resize(original_frame, (self.input_shape[1], self.input_shape[0]), 
                           interpolation = cv2.INTER_CUBIC)
        ##########################################################################
        # 1. SEGMENTATION and MASKS
        print("\nSegmenting frame...")
        road_mask, fence_mask, segmented_frame = self.frame_segmenter.segment_frame(frame)
        road_mask  = road_mask.squeeze()  # Remove 3rd-dimension
        fence_mask = fence_mask.squeeze() # Remove 3rd-dimension
        #segmented_frame = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2RGB)

        ##########################################################################
        # 2. DISPARITY MAP
        print("\nComputing frame's disparity map...")
        disparity = self.frame_depther.compute_disparity(frame)
        # Disparities in monodepth are normalized, so we need to scale them by 
        # the full resolution width (2048 for Cityscapes)
        #disparity = disparity * original_shape[1]
        disparity = disparity * self.disp_multiplier

        ##########################################################################
        # 3. 3D POINTS: Get 3D points from disparity map and create corresponding 
        #    color's array
        print("\nConverting disparity map to 3D Point Cloud...")
        points3D = self.frame_depther.compute_3D_points(disparity)
        colors   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ##########################################################################
        # 4. MASKED IMAGES: Convert RGB image to GRAY and apply masks to obtain 
        #    gray scale images with either a road or a fence on them
        # gray_frame  = cv2.cvtColor(colors, cv2.COLOR_RGB2GRAY)
        # road_image  = np.multiply(gray_frame, road_mask)
        # fence_image = np.multiply(gray_frame, fence_mask)

        ##########################################################################
        # 5. Apply masks to the whole 3D points matrix (to colors as well)
        #     to only get road or fence 3D points
        #: ROAD
        road3D       = points3D[road_mask]
        road_colors  = colors[road_mask]
        #: FENCE
        fence3D      = points3D[fence_mask]
        fence_colors = colors[fence_mask]

        ##########################################################################
        # 6. Remove noise and fit planes        
        #    Remove noise from road 3D point cloud:
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
        
        #################################################################################
        ####################### NAIVE APPROACH ##########################################
        #   Get 3D points that define a horizontal line at a certain depth
        left_pt_naive, right_pt_naive = pcl.get_end_points_of_road(road3D, 
                                                                   self.depth-0.02)
        # np.savez('{}_nai.npz'.format(self.output_name),
        #          left_pt_naive=left_pt_naive, right_pt_naive=right_pt_naive)
        #dist_naive = pcl.compute_distance_in_3D(left_pt_naive, right_pt_naive)
        dist_naive = abs(left_pt_naive[0][0] - right_pt_naive[0][0])
        if self.verbose:
            print("Road width", dist_naive)
        line_naive, colors_line_naive = pcl.create_3Dline_from_3Dpoints(left_pt_naive, 
                                                                        right_pt_naive,
                                                                        [250,0,0])

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

            ####################################################################################
            ####################### ADVANCED APPROACH ##########################################
            ######## ROAD-LEFT_FENCE intersection at a certain depth ###########################
            left_pt_advanced  = pcl.planes_intersection_at_certain_depth(road_plane_coeff, 
                                                                         fence_left_plane_coeff, 
                                                                         z=self.depth)

            right_pt_advanced = pcl.planes_intersection_at_certain_depth(road_plane_coeff, 
                                                                         fence_right_plane_coeff, 
                                                                         z=self.depth)
            dist_advanced = pcl.compute_distance_in_3D(left_pt_advanced, right_pt_advanced)
            if self.verbose:
                print("Distance from fence to fence:", dist_advanced)
            line_advanced, colors_line_advanced = pcl.create_3Dline_from_3Dpoints(left_pt_advanced, 
                                                                                  right_pt_advanced,
                                                                                  [0,255,0])
        
            if self.verbose:
                print("\nAdvanced time: ", time_advanced)

        ##########################################################################
        # 9. Draw letters in the image
        self.segmented_frame = cv2.resize(segmented_frame, (w, h), interpolation = cv2.INTER_CUBIC)

        thickness = int(0.0035 * h) 
        fontScale = int(0.001  * h)

        cv2.rectangle(self.segmented_frame,(0,0),(w, int(0.2*h)),(156, 157, 159), -1)
        cv2.putText(self.segmented_frame, 'At {:.2f} m depth:'.format(self.depth),
                    (int(0.36*w), int(0.035*h)),
                    fontFace = 16, fontScale = fontScale+0.2, color=(255,255,255), thickness = thickness)
        if self.approach == 'advanced' or self.approach == 'both':
            cv2.putText(self.segmented_frame, '{:.2f} m to left fence'.format(-left_pt_advanced[0][0]), 
                        (int(0.05*w), int(0.08*h)),
                        fontFace = 16, fontScale = fontScale, color=(255,255,255), thickness = thickness)
            cv2.putText(self.segmented_frame, '{:.2f} m to right fence'.format(right_pt_advanced[0][0]), 
                        (int(0.6*w), int(0.08*h)),
                        fontFace = 16, fontScale = fontScale, color=(255,255,255), thickness = thickness)
        cv2.putText(self.segmented_frame, '{:.2f} m to road\'s left end'.format(-left_pt_naive[0][0]), 
                    (int(0.05*w), int(0.13*h)),
                    fontFace = 16, fontScale = fontScale, color=(255,255,255), thickness = thickness)
        cv2.putText(self.segmented_frame, '{:.2f} m to road\'s right end'.format(right_pt_naive[0][0]), 
                    (int(0.6*w), int(0.13*h)),
                    fontFace = 16, fontScale = fontScale, color=(255,255,255), thickness = thickness)
        cv2.putText(self.segmented_frame, 'Road\'s width: {:.2f} m'.format(dist_naive), 
                    (int(0.35*w), int(0.18*h)),
                    fontFace = 16, fontScale = fontScale, color=(255,255,255), thickness = thickness)

        ##########################################################################
        # 8.A Project the 3D points that define the line to the image plane
        #self.print_line_on_image(left_pt_naive, right_pt_naive, (0,0,255))
        #self.print_line_on_image(left_pt_advanced, right_pt_advanced, (0,255,0))
        ##########################################################################
        # 10. Save image
        cv2.imwrite('{}/{}.png'.format(result_images_dir, output_name), self.segmented_frame)

        ######################################################
        # 98. Save Point Cloud to ply file to check results
        # For ROAD
        #point_cloud = PointCloud2Ply(road3D, road_colors, self.output_name)
        #point_cloud.add_extra_point_cloud(road_plane3D, road_colors_plane)

        """
        # For FENCEs and ROAD (Advanced approach + Naive approach)
        point_cloud = PointCloud2Ply(fence3D_left, fence_left_colors, '{}/{}_naive'.format(result_ply_dir, output_name))
        point_cloud.add_extra_point_cloud(fence_left_plane3D, fence_left_colors_plane)
        point_cloud.add_extra_point_cloud(fence3D_right, fence_right_colors)
        point_cloud.add_extra_point_cloud(fence_right_plane3D, fence_right_colors_plane)
        point_cloud.add_extra_point_cloud(road3D, road_colors)
        point_cloud.add_extra_point_cloud(road_plane3D, road_colors_plane)
        point_cloud.add_extra_point_cloud(line_advanced, colors_line_advanced)
        point_cloud.add_extra_point_cloud(line_naive, colors_line_naive)
        point_cloud.prepare_and_save_point_cloud()
        """

        # For FENCEs and ROAD (Naive approach)
        point_cloud = PointCloud2Ply(road3D, road_colors, '{}/{}_naive'.format(result_ply_dir, output_name))
        point_cloud.add_extra_point_cloud(line_naive, colors_line_naive)
        point_cloud.prepare_and_save_point_cloud()

        """
        # For ALL with Naive Approach
        point_cloud = PointCloud2Ply(points3D, colors, '{}/{}_naive'.format(result_ply_dir, output_name))
        point_cloud.add_extra_point_cloud(line_naive, colors_line_naive)
        point_cloud.prepare_and_save_point_cloud()
        """

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
            print(model_meta_file)

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
        fence_mask = np.dot(segmentation_fence, np.array([[190, 153, 153, 64]]))
        fence_mask = scipy.misc.toimage(fence_mask, mode="RGBA")
        #scipy.misc.imsave('fence.png', fence_mask)
        street_im.paste(fence_mask, box=None, mask=fence_mask)

        return segmentation_road, segmentation_fence, np.array(street_im)


class DepthFrame():
    def __init__(self, encoder='vgg', input_height=256, input_width=512, 
             checkpoint_path='models/monodepth/model_cityscapes/model_cityscapes',
             fov=322):


        self.encoder         = encoder
        self.input_height    = input_height
        self.input_width     = input_width
        self.checkpoint_path = checkpoint_path  

        # CITYSCAPES INTRINSIC PARAMS #
        f = 2268
        baseline = 0.222126

        self.Q = np.float32([[1, 0, 0, -0.5*input_width ],
                             [0,-1, 0,  0.5*input_height],    # turn points 180 deg around x-axis,
                             [0, 0, 0,    -f*baseline   ],    # so that y-axis looks up
                             [0, 0, 1,      0           ]])

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
        points3D = cv2.reprojectImageTo3D(disp, self.Q)
        return points3D


def main():

    parser = argparse.ArgumentParser(description="Read frame and "
                            "compute the distance from the center "
                            "of the car to the fences.")

    parser.add_argument("--input_folder", help="Path to folder where the input images are.", 
                        default="data/stuttgart_video_test/*.png")

    parser.add_argument("--semantic_model", help="Path to semantic segmentation model.", 
                        default="models/sem_seg/30-Epochs-cityscapes")

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
                        default='naive')

    parser.add_argument('--use_frozen',
                        help='If set, uses frozen model',
                        action='store_true')

    parser.add_argument('--use_xla',
                        help='If set, uses xla',
                        action='store_true')

    parser.add_argument('--CUDA_DEVICE_NUMBER',
                        help='Number of GPU device to use (e.g., 0, 1, 2, ...)',
                        default="0")

    parser.add_argument('--depth', type=float,
                        help='depth at which to compute road\'s width',
                        default=10)

    parser.add_argument('--verbose',
                        help='If set, prints info',
                        action='store_true')

    args = parser.parse_args()


    # Input size
    input_shape = (args.input_height, args.input_width)

    # Create a DepthFrame object
    frame_depther = DepthFrame(args.monodepth_encoder, 
                               args.input_height, 
                               args.input_width,
                               args.monodepth_checkpoint)

    # Create a SegmentFrame object
    frame_segmenter = SegmentFrame(input_shape, args.semantic_model, 
                                   args.use_frozen, args.use_xla, 
                                   args.CUDA_DEVICE_NUMBER)

    # Create a FrameProcessor object
    frame_processor = FrameProcessor(frame_segmenter,
                                     frame_depther,
                                     input_shape,
                                     args.approach,
                                     args.depth,
                                     args.verbose)



    # Create output frame path
    output_directory  = "results/stuttgart_video"
    result_images_dir = os.path.join(output_directory, 
                                     'result_sequence_imgs')
    result_ply_dir    = os.path.join(output_directory, 'result_sequence_ply')

    if not os.path.exists(result_images_dir):
        print("Creating directory for storing result frame")
        os.makedirs(result_images_dir)

    if not os.path.exists(result_ply_dir):
        print("Creating directory for storing result ply")
        os.makedirs(result_ply_dir)

    
    # Process input frames
    for input_frame in sorted(glob.glob(args.input_folder)):

        print("Processing", input_frame)

        output_name = os.path.basename(input_frame)
        output_name = os.path.splitext(output_name)[0]
        frame_processor.process_frame(input_frame, output_name, 
                                      result_images_dir, result_ply_dir)


if __name__ == "__main__":
    main()