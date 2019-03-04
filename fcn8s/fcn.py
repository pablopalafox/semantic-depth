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


'''
FCN-8s that trains on the Cityscapes set (or datset with similar structure)
'''

import matplotlib
matplotlib.use('Agg')
import math
import time
import os.path
import tensorflow as tf
import helper
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from glob import glob
import re
import numpy as np
import scipy.misc
from datetime import datetime
import csv
import argparse


class FCN(object):

    '''
    Constructor for setting params
    '''
    def __init__(self, params):
        for p in params:
            setattr(self, p, params[p])

        # Check for compatibility, data set and conditionally download VGG16 model
        helper.check_compatibility() 
        helper.maybe_download_pretrained_vgg(self.data_dir)

        # Define static project constants
        self.vgg_path        = os.path.join(self.data_dir, 'vgg')

        self.train_gt_dir    = os.path.join(self.data_dir, self.dataset, self.train_gt_subdir)
        self.train_imgs_dir  = os.path.join(self.data_dir, self.dataset, self.train_imgs_subdir)

        self.val_gt_dir      = os.path.join(self.data_dir, self.dataset, self.val_gt_subdir)
        self.val_imgs_dir    = os.path.join(self.data_dir, self.dataset, self.val_imgs_subdir)

        self.test_gt_dir     = os.path.join(self.data_dir, self.dataset, self.test_gt_subdir)
        self.test_imgs_dir   = os.path.join(self.data_dir, self.dataset, self.test_imgs_subdir)

        # Define the batching function
        self.get_batches_fn  = helper.gen_batch_function(self.train_gt_dir, self.train_imgs_dir,
                                                   self.val_gt_dir, self.val_imgs_dir, 
                                                   self.test_gt_dir, self.test_imgs_dir,
                                                   self.image_shape, self.dataset)

    '''
    Load the VGG16 model
    '''
    def load_vgg(self, sess):

        # Load the saved model
        tf.saved_model.loader.load(sess, ['vgg16'], self.vgg_path)

        # Get the relevant layers for constructing the skip-layers out of the graph
        graph       = tf.get_default_graph()
        image_input = graph.get_tensor_by_name('image_input:0')
        keep_prob   = graph.get_tensor_by_name('keep_prob:0')
        l3          = graph.get_tensor_by_name('layer3_out:0')
        l4          = graph.get_tensor_by_name('layer4_out:0')
        l7          = graph.get_tensor_by_name('layer7_out:0')
        
        return image_input, keep_prob, l3, l4, l7

    '''
    Restore model and retrieve pertinent tensors
    '''
    def restore_model(self, sess):

        print("Restoring saved model...")
        model_var_dir = '{}/{}/variables'.format(self.model_dir, self.model)
                                                        
        model_meta_file = model_var_dir + '/saved_model.meta'
        new_saver = tf.train.import_meta_graph(model_meta_file)
        new_saver.restore(sess, tf.train.latest_checkpoint(model_var_dir))
        
        all_vars = tf.get_collection('vars')
        for v in all_vars:
            v_ = sess.run(v)
            print(v_)

        graph = tf.get_default_graph()
        keep_prob   = graph.get_tensor_by_name('keep_prob:0')
        image_input = graph.get_tensor_by_name('image_input:0')
        logits      = graph.get_tensor_by_name('logits:0')

        # For computing IoU metric
        correct_label      = tf.placeholder(dtype = tf.float32, shape = (None, None, None, self.num_classes))
        predictions_argmax = graph.get_tensor_by_name('predictions_argmax:0')
    
        # Define iou metric operation
        labels_argmax = tf.argmax(correct_label, axis=-1, output_type=tf.int64)
        iou, iou_op = tf.metrics.mean_iou(labels_argmax, 
                                          predictions_argmax, 
                                          self.num_classes)
        sess.run(tf.local_variables_initializer())
        print("Model successfully restored")

        return iou, iou_op, image_input, correct_label, keep_prob, logits

    '''
    Save the model
    '''
    def save_model(self, sess):

        # Create model dir if it doesn't exist
        model_var_dir = os.path.join(self.model_dir, 
                                     self.model, 
                                     'variables')
        if os.path.exists(model_var_dir):
            shutil.rmtree(model_var_dir)
        os.makedirs(model_var_dir)

        # Create a Saver object
        saver = tf.train.Saver()
        
        print("Saving model to: {}".format(model_var_dir))
        saver.save(sess, model_var_dir + '/saved_model')
        tf.train.write_graph(sess.graph_def, 
                             os.path.join(self.model_dir, 
                                          self.model), 
                                          'saved_model.pb', False)

    ''' 
    Define the layers
    '''
    def layers(self, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):

        kernel_initializer = tf.truncated_normal_initializer(stddev = 0.01)
        weights_regularized_l2 = 1e-3

        # We generate the 1x1 convolutions of layers 3, 4 and 7 of the VGG model
        conv_1x1_of_7 = tf.layers.conv2d(inputs=vgg_layer7_out, 
                                         filters=num_classes, 
                                         kernel_size=(1,1), strides=(1,1), 
                                         kernel_initializer=kernel_initializer,
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        
        conv_1x1_of_4 = tf.layers.conv2d(inputs=vgg_layer4_out, 
                                         filters=num_classes, 
                                         kernel_size=(1,1), strides=(1,1), 
                                         kernel_initializer=kernel_initializer,
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

        conv_1x1_of_3 = tf.layers.conv2d(inputs=vgg_layer3_out, 
                                         filters=num_classes, 
                                         kernel_size=(1,1), 
                                         strides=(1,1), 
                                         kernel_initializer=kernel_initializer,
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

        # Decoder, with upsampling and skipped connections
        # Upsampling conv_1x1_of_7
        deconv1 = tf.layers.conv2d_transpose(inputs=conv_1x1_of_7, 
                                             filters=num_classes, 
                                             kernel_size=(4,4), 
                                             strides=(2,2), padding='same', 
                                             kernel_initializer=kernel_initializer,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        # Skip connections from VGG16 layer 4
        first_skip = tf.add(deconv1, conv_1x1_of_4)
        
        # Upsampling first_skip
        deconv2 = tf.layers.conv2d_transpose(inputs=first_skip, 
                                             filters=num_classes, 
                                             kernel_size=(4,4), 
                                             strides=(2,2), 
                                             padding='same', 
                                             kernel_initializer=kernel_initializer,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        # Skip connections from VGG16 layer 3
        second_skip = tf.add(deconv2, conv_1x1_of_3)
        
        # Upsampling l7_decoder
        deconv3 = tf.layers.conv2d_transpose(inputs=second_skip, 
                                             filters=num_classes, 
                                             kernel_size=(16,16), 
                                             strides=(8,8), 
                                             padding='same', 
                                             kernel_initializer=kernel_initializer,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

        return deconv3


    def build_predictor(self, nn_last_layer):
        softmax_output = tf.nn.softmax(nn_last_layer)
        predictions_argmax = tf.argmax(softmax_output, 
                                       axis=-1, 
                                       output_type=tf.int64, 
                                       name='predictions_argmax')
        return predictions_argmax


    def build_metrics(self, correct_label, predictions_argmax, num_classes):
          labels_argmax = tf.argmax(correct_label, 
                                    axis=-1, 
                                    output_type=tf.int64, 
                                    name='labels_argmax')
          iou, iou_op = tf.metrics.mean_iou(labels_argmax, predictions_argmax, num_classes)
          return iou, iou_op

    '''
    Optimizer based on cross entropy
    '''
    def optimize_cross_entropy(self, nn_last_layer, correct_label, learning_rate, num_classes):

        # Reshape logits and label for computing cross entropy
        logits        = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
        correct_label = tf.reshape(correct_label, (-1, num_classes), name='correct_label')

        # For computing accuracy on test set
        #acc, acc_op = tf.metrics.accuracy(labels=correct_label, predictions=logits)

        # Compute cross entropy and loss
        cross_entropy_logits = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
        cross_entropy_loss   = tf.reduce_mean(cross_entropy_logits)

        # Define a training operation using the Adam optimizer (allows to have variable learning rate)
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)
        
        return logits, train_op, cross_entropy_loss


    ''' 
    Define training op
    '''
    def train_nn(self, sess, train_op, cross_entropy_loss, image_input, correct_label, keep_prob, learning_rate, iou, iou_op):

        print("\nStarting training with a learning_rate of: {}".format(self.learning_rate))
        
        train_mean_losses_list = []
        train_mean_ious_list  = []
        val_mean_losses_list   = []
        val_mean_ious_list    = []
        
        # Iterate over epochs
        for epoch in range(1, self.epochs+1):

            print("\nEpoch: {}/{}".format(epoch, self.epochs))

            ##########################################################################
            print("\n## TRAINING of epoch {} ##".format(epoch))

            ## TRAINING DATA ##
            # Iterate over batches of training data using the batch generation function
            train_losses   = []
            train_ious     = []
            train_batch    = self.get_batches_fn(self.batch_size, mode='train')
            total_num_imgs = helper.get_num_imgs_in_folder(self.train_imgs_dir)
            train_size     = math.ceil(total_num_imgs / self.batch_size)

            for i, d in tqdm(enumerate(train_batch), desc="Epoch {}: Train Batch".format(epoch), total=train_size):

                image, label = d

                # Create the feed dictionary
                feed_dict_train = { 
                    image_input   : image,
                    correct_label : label,
                    keep_prob     : self.dropout,
                    learning_rate : self.learning_rate
                }

                # Create the feed dictionary
                feed_dict_train_iou = { 
                    image_input   : image,
                    correct_label : label,
                    keep_prob     : 1.0,
                }

                # Train and compute the loss of the current train BATCH
                #_, train_loss, _ = sess.run([train_op, cross_entropy_loss, iou_op], feed_dict=feed_dict_train)
                _, train_loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict_train)
                _ = sess.run([iou_op], feed_dict=feed_dict_train_iou)
                train_iou = sess.run(iou)

                print('   loss: {}'.format(train_loss))
                print('   iou:  {}'.format(train_iou))
                train_losses.append(train_loss)
                train_ious.append(train_iou)

            ### LOSS ###
            # Compute the mean loss of the current EPOCH based on the losses from each batch
            train_mean_loss = sum(train_losses) / len(train_losses)
            print("TRAIN: mean loss of current epoch: {}".format(train_mean_loss))
            # Append the mean loss of the current epoch to a list of mean losses of the whole training
            train_mean_losses_list.append(train_mean_loss)

            ### IOU ###
            # Compute the mean IoU of the current EPOCH based on the IoUs from each batch
            train_mean_iou = sum(train_ious) / len(train_ious)
            print("TRAIN: mean iou of current epoch: {}".format(train_mean_iou))
            # Append the mean loss of the current epoch to a list of mean losses of the whole training
            train_mean_ious_list.append(train_mean_iou)

            ##########################################################################
            print("\n## VALIDATION of epoch {} ##".format(epoch))
            ## VALIDATION DATA ##
            # Iterate over batches of training data using the batch generation function
            val_losses     = []
            val_ious       = []
            val_batch      = self.get_batches_fn(self.batch_size, mode='val')
            total_num_imgs = helper.get_num_imgs_in_folder(self.val_imgs_dir)
            val_size       = math.ceil(total_num_imgs / self.batch_size)

            for i, d in tqdm(enumerate(val_batch), desc="Epoch {}: Val Batch".format(epoch), total=val_size):

                image, label = d

                # Create the feed dictionary
                feed_dict_val = { 
                    image_input   : image,
                    correct_label : label,
                    keep_prob     : 1.0,
                }

                # Compute the loss of the current val BATCH
                val_loss, _ = sess.run([cross_entropy_loss, iou_op], feed_dict=feed_dict_val)
                val_iou = sess.run(iou)

                print('    loss: {}'.format(val_loss))
                print('    iou:  {}'.format(val_iou))
                val_losses.append(val_loss)
                val_ious.append(val_iou)

            ### LOSS ###
            # Compute the mean loss of the current EPOCH based on the losses from each batch
            val_mean_loss = sum(val_losses) / len(val_losses)
            print("VAL: mean loss of current epoch: {}".format(val_mean_loss))
            # Append the mean loss of the current epoch to a list of mean losses of the whole training
            val_mean_losses_list.append(val_mean_loss)

            ### IOU ###
            # Compute the mean IoU of the current EPOCH based on the IoUs from each batch
            val_mean_iou = sum(val_ious) / len(val_ious)
            print("VAL: mean iou of current epoch: {}".format(val_mean_iou))
            # Append the mean loss of the current epoch to a list of mean losses of the whole training
            val_mean_ious_list.append(val_mean_iou)

        print("\nTraining completed")

        # Save logging info into images
        epochs_list = list(range(1, self.epochs+1))
        self.logging('loss', train_mean_losses_list, val_mean_losses_list, epochs_list)
        self.logging('iou',  train_mean_ious_list,   val_mean_ious_list, epochs_list)     


    '''
    Apply inference over test set and obtain model accuracy
    '''    
    def inference(self, sess, iou, iou_op, image_input, correct_label, keep_prob, logits):

        # Make folder for current run if it doesn't exist
        time_str = datetime.now()
        time_str = "{}_{}_{} {}-{}".format(time_str.year, time_str.month, time_str.day, time_str.hour, time_str.minute)
        output_dir = os.path.join(self.runs_dir, 
                                  self.model, 
                                  time_str)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Get ground truth and image file names
        gt_paths, imgs_paths = helper.get_files_paths(self.test_gt_dir, self.test_imgs_dir)
        both = zip(gt_paths, imgs_paths)
        test_size = len(gt_paths)

        times = []
        test_ious = []
        for gt_file, image_file in tqdm(both, desc="Test Batch", total=test_size):

            t_init = time.time()

            # Read Ground Truth and prepare it as a depth 3 image
            gt = scipy.misc.imresize(scipy.misc.imread(gt_file), self.image_shape)
            gt_image = helper.prepare_ground_truth(self.dataset, gt, self.num_classes, 'test')

            # Read the input image
            image = scipy.misc.imresize(scipy.misc.imread(image_file), self.image_shape)
            street_im = scipy.misc.toimage(image)

            #########################################
            ## 1. Compute IoU of test set

            # Convert ground truth and image to (1, 256, 512, 3) format size
            gt_image = np.expand_dims(gt_image, axis=0)
            image = np.expand_dims(image, axis=0)

            # Create the feed dictionary
            feed_dict_test = { 
                image_input   : image,
                correct_label : gt_image,
                keep_prob     : 1.0,
            }

            sess.run([iou_op], feed_dict=feed_dict_test)
            test_iou = sess.run(iou)
            test_ious.append(test_iou)

            #######################################
            ## 2. Apply inference on test set
            # im_softmax is a matrix of heigh*width rows and 'num_classes' columns
            # which codifies the probability that a pixel belongs to a class
            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, image_input: image})

            t1 = time.time() - t_init

            ### Road
            # 1. For the first class, we only select the values corresponding to the first column
            # We also convert the vector of pixels into a matrix
            im_softmax_r = im_softmax[0][:, 0].reshape(self.image_shape[0], self.image_shape[1])
            # 2. We create a matrix of height*width*depth with boolean values (True or False) depending
            # on wheter the probability of belonging to class 'road' is higher than 0.5
            segmentation_r = (im_softmax_r > 0.5).reshape(self.image_shape[0], self.image_shape[1], 1)
            mask = np.dot(segmentation_r, np.array([[128, 64, 128, 64]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im.paste(mask, box=None, mask=mask)

            ### Fence
            im_softmax_r = im_softmax[0][:, 1].reshape(self.image_shape[0], self.image_shape[1])
            segmentation_r = (im_softmax_r > 0.5).reshape(self.image_shape[0], self.image_shape[1], 1)
            mask = np.dot(segmentation_r, np.array([[190, 153, 153, 64]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im.paste(mask, box=None, mask=mask)

            t2 = time.time() - t_init

            t_both = "{} {}\n".format(t1, t2)
            times.append(t_both)

            # Save output image
            image_file = os.path.basename(image_file)
            output_path = os.path.join(output_dir, image_file)
            image = np.array(street_im)
            scipy.misc.imsave(output_path, image)

        ### Inference time log ###
        with open("times.txt", "w") as file:
            for pair in times:
                file.write(pair)

        ### IoU ###
        # Compute the mean IoU of the whole test set
        test_mean_iou = sum(test_ious) / len(test_ious)
        print("TEST: mean iou of test set: {}".format(test_mean_iou))

        # Create txt file in which to store IoU of testing set
        metric_type = 'iou'
        metric_path = os.path.join(self.logging_dir, self.model, metric_type)
        if not os.path.exists(metric_path):
            print("Creating '{}' directory for storing {} info of Testing set".format(metric_path, metric_type))
            os.makedirs(metric_path)
        metric_file_path = os.path.join(metric_path, "test_set_iou_{}.txt".format(time_str))
        with open(metric_file_path, "w") as text_file:
            for iou in test_ious:
                text_file.write("{}\n".format(iou))
            text_file.write("IoU metric of Testing set: {}".format(test_mean_iou))
 
    '''
    Plot loss vs epochs, iou vs epochs
    '''
    def logging(self, metric_type, train_mean_metric_list, val_mean_metric_list, epochs_list):

        print("Plotting '{} vs epochs' and saving as image".format(metric_type))

        # Create logging dir for metric if it doesn't exist already
        metric_path = os.path.join(self.logging_dir, self.model, metric_type)
        if not os.path.exists(metric_path):
            print("Creating '{}' directory for storing {} info of Training and Validation sets".format(metric_path, metric_type))
            os.makedirs(metric_path)

        # Get time
        time = datetime.now()
        time = "{}_{}_{} {}-{}".format(time.year, time.month, time.day, time.hour, time.minute)

        # Save metric log into csv file
        csv_file = '{}_vs_epochs_{}.csv'.format(metric_type, time)
        csv_file_path = os.path.join(metric_path, csv_file)

        print("Saving '{} vs epochs' as a csv file into {} directory\n".format(metric_type, metric_path))
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Epoch'] + ['TRAIN_{}'.format(metric_type)] + ['VAL_{}'.format(metric_type)])
            writer.writerows(zip(epochs_list, train_mean_metric_list, val_mean_metric_list))

        # Plot metric and save into image
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(epochs_list, train_mean_metric_list, label='train', linestyle='--')
        ax.plot(epochs_list, val_mean_metric_list, label='val', linestyle='--')
        
        ax.legend()
        plt.xlabel('epochs')
        plt.ylabel(metric_type)
        
        log_file = '{}_vs_epochs_{}.png'.format(metric_type, time)
        log_file_path = os.path.join(metric_path, log_file)

        print("Plotting '{} vs epochs' and saving as image into {} directory\n".format(metric_type, metric_path))
        plt.savefig(log_file_path)


    def train(self):
        ## CONFIGURATION FOR USING GPU ##
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = "0"


        # TensorFlow session
        with tf.Session(config=config) as sess:

            tf.logging.set_verbosity(tf.logging.INFO)

            # Placeholders
            learning_rate = tf.placeholder(dtype = tf.float32)
            correct_label = tf.placeholder(dtype = tf.float32, shape = (None, None, None, self.num_classes))

            # Define network and training operations 
            image_input, keep_prob, l3, l4, l7   = self.load_vgg(sess)
            layer_output                         = self.layers(l3, l4, l7, self.num_classes)
            logits, train_op, cross_entropy_loss = self.optimize_cross_entropy(layer_output, correct_label, learning_rate, self.num_classes)
            
            predictions_argmax = self.build_predictor(layer_output)
            iou, iou_op        = self.build_metrics(correct_label, 
                                                    predictions_argmax, 
                                                    self.num_classes)

            # Initialize variables 
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # Train the model 
            self.train_nn(sess, train_op, cross_entropy_loss, 
                          image_input, correct_label, keep_prob, learning_rate,
                          iou, iou_op)

            # Do inference to compute IoU on test set and save the output images
            if self.inference_flag:
                self.inference(sess, iou, iou_op, image_input, 
                               correct_label, keep_prob, logits)

            # Save the model
            self.save_model(sess)


    def test(self):
        
        ## CONFIGURATION FOR USING GPU ##
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = "0"

        # TensorFlow session
        with tf.Session(config=config) as sess:
            iou, iou_op, image_input, correct_label, keep_prob, logits = self.restore_model(sess)
            self.inference(sess, iou, iou_op, image_input, 
                           correct_label, keep_prob, logits)


'''
Entry point
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FCN-8s implementation.")

    parser.add_argument("--mode",          type=str,   help="Train or test mode.",
                        default="train")

    parser.add_argument("--epochs",        type=int,   help="Number of epochs we want "
                        "to train the network for.")

    parser.add_argument('--dataset',       type=str,   help='Name of dataset we want to '
                        'train on, or where the test set we want to apply inference on resides.',
                        required=True)

    parser.add_argument('--inference_flag',            help='When set to true, applies inference'
                        'on the test set of the dataset on which we have just trained.',
                        action='store_true')

    parser.add_argument("--learning_rate", type=float, help="Learning rate.", 
                        default=0.00001)

    parser.add_argument("--dropout",       type=float, help="Dropout rate.", 
                        default=0.5)

    parser.add_argument('--batch_size',    type=str,   help='Batch size', default=1)

    parser.add_argument('--num_classes',   type=int,   help='Number of target classes', 
                        default=3)

    parser.add_argument('--image_shape',               help='Image shape (width, height)', 
                        default=(256, 512))

    parser.add_argument('--runs_dir',      type=str,   help='Directory in which to save '
                        'inference output.', 
                        default='runs')

    parser.add_argument('--data_dir',      type=str,   help='Directory where our datasets '
                        'reside.', 
                        default='../data')

    parser.add_argument('--train_gt_subdir',   type=str,  default='gtFine/train')

    parser.add_argument('--train_imgs_subdir', type=str,  default='leftImg8bit/train')

    parser.add_argument('--val_gt_subdir',     type=str,  default='gtFine/val')

    parser.add_argument('--val_imgs_subdir',   type=str,  default='leftImg8bit/val')

    parser.add_argument('--test_gt_subdir',    type=str,  default='gtFine/test')

    parser.add_argument('--test_imgs_subdir',  type=str,  default='leftImg8bit/test')

    parser.add_argument('--model_dir',         type=str,  default='../models/sem_seg')

    parser.add_argument('--logging_dir',       type=str,  default='log')

    args = parser.parse_args()

    # Get the name of the model, either the model to be created or the model 
    # to be used for inference
    if args.mode == 'train': 
        if args.epochs is None:
            parser.error("train mode requires --epochs.")
        model = '{}-Epochs-{}'.format(args.epochs, args.dataset)
    elif args.mode == 'test':
        model = ''
        while len(model) is 0:
            model = input("Enter the name of the model you want to use "
                          "in the format '<epochs>-Epochs-<dataset>' \n--> ")
    args.model = model

    # Convert the arguments into a dictionary for later usage within the class init function
    args_dict = vars(args)

    # Create an FCN object
    fcn = FCN(args_dict)

    if fcn.mode == 'train':
        fcn.train()
    elif fcn.mode == 'test':
        fcn.test()

