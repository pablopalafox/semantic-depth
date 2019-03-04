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

import numpy as np
import scipy.misc
import tensorflow as tf
from tqdm import tqdm 
from moviepy.editor import *
import os.path
from glob import glob
import sys
import time

from load_graph import load_graph

'''
city   = input("Enter the name of the CITY (lowercase string) whose video you want to segment (e.g. montreal): ")
epochs = int(input("Enter the number of EPOCHS (integer) on which the model you want to use was trained: "))
dataset = input("Enter the DATASET on which the model you want to use was trained: (e.g roborace350) ")
seconds = int(input("Enter the number of SECONDS (integer) you want to segment from the video: "))
'''


city   = "montreal"
epochs = "100"
dataset = "roborace350"
seconds = 10


class SegmentVideo(object):

    '''
    Constructor with param setting
    '''
    def __init__(self, params):
        for p in params:
            setattr(self, p, params[p])


    '''
    Segments the image
    '''
    def segment_frame(self, frame):

        start_time = time.time()

        frame = scipy.misc.imresize(frame, self.image_shape)
        street_im = scipy.misc.toimage(frame)
        


        #config = tf.ConfigProto()
        #jit_level = tf.OptimizerOptions.ON_1
        #config.graph_options.optimizer_options.global_jit_level = jit_level
        with tf.Session(graph=self.graph) as sess:

            feed_dict = {
                self.keep_prob:   1.0,
                self.input_image: [frame]
            }

            im_softmax = sess.run(
                                    [tf.nn.softmax(self.logits)],
                                    feed_dict=feed_dict)
        

        '''
        feed_dict = {
            self.keep_prob:   1.0,
            self.input_image: [frame]
        }
        im_softmax = self.sess.run(
            [tf.nn.softmax(self.logits)],
            feed_dict=feed_dict)
        '''

        
        # Road
        im_softmax_r = im_softmax[0][:, 0].reshape(self.image_shape[0], self.image_shape[1])
        segmentation_r = (im_softmax_r > 0.5).reshape(self.image_shape[0], self.image_shape[1], 1)
        mask = np.dot(segmentation_r, np.array([[50, 200, 50, 64]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im.paste(mask, box=None, mask=mask)

        # Fence
        im_softmax_r = im_softmax[0][:, 1].reshape(self.image_shape[0], self.image_shape[1])
        segmentation_r = (im_softmax_r > 0.5).reshape(self.image_shape[0], self.image_shape[1], 1)
        mask = np.dot(segmentation_r, np.array([[255, 0, 0, 64]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im.paste(mask, box=None, mask=mask)

        print(time.time() - start_time)

        return np.array(street_im)


    '''
    Main processing loop
    '''
    def process_video(self):
        print("Applying inference to input video")

        # new_frames = []
        # video      = VideoFileClip(self.input_video)
        # for frame in video.iter_frames():
        #     new_frame = self.segment_image(frame)
        #     new_frames.append(new_frame)
        #     print(len(new_frames))
        # new_video = ImageSequenceClip(new_frames, fps=video.fps)
        # new_video.write_videofile(self.output_video, audio=False)

        if not os.path.exists(self.output_path):
            print("Creating directory for storing video")
            os.makedirs(self.output_path)
        self.output_video = os.path.join(self.output_path, self.output_video)

        clip = VideoFileClip(self.input_video).subclip(0,seconds)
        new_clip = clip.fl_image(self.segment_frame)

        new_clip.write_videofile(self.output_video, audio=False)



    '''
    Restore model and retrieve pertinent tensors
    '''
    def restore_model(self):
        print("Restoring saved model...")
    
        '''
        # 1
        self.sess = tf.Session() 

        model_meta_file = self.model_var_dir + '/saved_model.meta'

        new_saver = tf.train.import_meta_graph(model_meta_file)
        new_saver.restore(self.sess, tf.train.latest_checkpoint(self.model_var_dir))
        
        all_vars = tf.get_collection('vars')
        for v in all_vars:
            v_ = sess.run(v)
            print(v_)
        
        graph = tf.get_default_graph()
        self.keep_prob   = graph.get_tensor_by_name('keep_prob:0')
        self.input_image = graph.get_tensor_by_name('image_input:0')
        self.logits      = graph.get_tensor_by_name('logits:0')
        '''

        
        # 2
        graph_filename = "models/100-Epochs-roborace350/optimized_graph.pb"
        graph, ops = load_graph(graph_filename, True)
        self.keep_prob   = graph.get_tensor_by_name('keep_prob:0')
        self.input_image = graph.get_tensor_by_name('image_input:0')
        self.logits      = graph.get_tensor_by_name('logits:0')
        self.graph = graph



        print("Model successfully restored")


    '''
    Run the segmentation
    '''
    def run(self):
        self.restore_model()
        self.process_video()



'''
Entry point
'''
if __name__=='__main__':

    params = {
        'input_video':   'videos/complete_1_{}.mp4'.format(city),
        'output_path':   'videos/results/{}-Epochs-{}'.format(epochs, dataset),
        'output_video':  'segmented_{}seconds_{}.mp4'.format(seconds, city),
        'model_var_dir': 'models/{}-Epochs-{}/variables'.format(epochs, dataset),
        'image_shape':   (256, 512)
    }

    sv = SegmentVideo(params)
    sv.run()

