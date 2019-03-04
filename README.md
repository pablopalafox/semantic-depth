# SemanticDepth
Deep Learning-based Computer Vision Pipeline to improve Situational Awareness of an Autonomous Vehicle

| | |
|:-------------------------:|:-------------------------:|
|<img alt="test_3" src="/assets/images/test_munich/test_3.png">  |  <img alt="test_3_output" src="/assets/images/test_munich/test_3_output.png">|
|<img alt="test_3_ALL" src="/assets/images/test_munich/test_3_ALL.png">  |  <img alt="test_3_planes" src="/assets/images/test_munich/test_3_planes.png">|


SemanticDepth is a deep learning-based computer vision pipeline that computes the width of the road at a certain depth in front of a car. 

It does so by fusing together two deep learning-based architectures, namely a semantic segmentation network ([fcn8-s](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)) and a monocular depth estimation network ([monodepth](https://github.com/mrharicot/monodepth))

We have two ways of computing the depth of the road at a certain depth. First, a _naive_ way: we compute the pointcloud corresponding to the road in front of us and compute the distance between the furthest point to the left and the furthest point to the right (of this road pointcloud) at a certain depth (depth meaning the direction in front of us).

Second, an _advanced_ distance. Here we additionally extract the pointclouds corresponding to hypothetical left and right fences/walls to each side of the road. Then we fit planes to the road pointcloud and to both the left and right fences. We compute the intersection between the road plane with the left fence plane, and the intersection between the road plane and the right fence plane. We end up with two intersected lines, and we can now decide on a depth at which we wish to compute the width of the road. 

<p align="center">
	<img src="/assets/images/pipeline.png" alt="pipeline">
</p>

This work was done as part of my Semesterarbeit (literally, semester work) at TUM's Chair of Automotive Technology. For more info on the pipeline, check my [thesis](/assets/pdfs/semanticDepthPabloRodriguezPalafox.pdf).

Author: [Pablo Rodriguez Palafox](https://pablorpalafox.github.io/)  
Advisor: [Johannes Betz, M.Sc.](https://www.ftm.mw.tum.de/lehrstuhl/mitarbeiter/fahrdynamik/johannes-betz-m-sc/johannes-betz-m-sc/)   
Head of Chair of Automotive Technology: [Univ.-Prof. Dr.-Ing. Markus Lienkamp](https://www.ftm.mw.tum.de/lehrstuhl/mitarbeiter/lehrstuhlleitung/prof-dr-ing-markus-lienkamp-3/prof-dr-ing-markus-lienkamp-lebenslauf/)  
[Lehrstuhl für Fahrzeugtechnik](https://www.ftm.mw.tum.de/startseite/)   
[Fakultät für Maschinenwesen](https://www.mw.tum.de/startseite/)  
[Technische Universität München](https://www.tum.de/)  



## 1. Requirements (& Installation tips)
This code was tested with Tensorflow 1.0, CUDA 8.0 and Ubuntu 16.04.

First of, install python3-tk.

`$ sudo apt-get install python3-tk`


Git clone this repo and change to the cloned dir:

```bash
$ git clone
$ cd semantic_depth
```

Using virtual environments is always a good idea. We will need to have pip and virtualenv installed.

Install pip and virtualenv:

`$ sudo apt-get install python3-pip python3.5-dev python-virtualenv`


To create a new virtualenv, being inside the root directory of the cloned repository, run the following:

`$ virtualenv --no-site-packages -p python3.5 .venv`

We now have a python3.5 virtual environment (with all packages that you already had installed for your python3.5 installation). Activate the virtualenv like so:

`$ source .venv/bin/activate`

Inside the virtualenv, you can run:

`$ pip install -r requirements.txt`

to get the dependencies needed.



## 2. Datasets

### Datasets for Semantic Segmentation on classes _fence_ and _road_

We labeled 750 [Roborace](https://roborace.com/) images with the classes fence, road and background. For the task of labelling our own images, we used the [cityscapesScripts](https://github.com/mcordts/cityscapesScripts).

We cannot make the whole dataset public, as the original images are property of the [Roborace](https://roborace.com/) competition. A mockup of this dataset can be found [here](data/roborace750_mockup). 

If you would like to get more images, join the [Roborace](https://roborace.com/) competition and you'll get data on which to run our work.

For the semantic segmentation task, another option is that you just train on [Kitti Semantic Segmentation dataset](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015) or on [Cityscapes](https://www.cityscapes-dataset.com/) and then mask out pixels belonging to fences and roads from the images you segment using a trained model on the mentioned datasets. 

### Datasets for Monocular Depth Estimation

[MonoDepth](https://github.com/mrharicot/monodepth), an unsupervised single image depth prediction network that we make use of in our work, can be trained on Kitty or Cityscapes.

We directly use the pre-trained model for Cityscapes, which you can get at the [monodepth](https://github.com/mrharicot/monodepth) repo, at the Models section.

### Munich Test Set

This is a set of 5 images of the streets of Munich on which you can test the whole pipeline. You can find it [here](data/test_images_munich). In section *Test on images*, you can find the commands on how to test our whole pipeline on these images.



## 3. Semantic Segmentation Network

The source files for the semantic segmentation network are under the folder [fcn8s](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf). There you can find an implementation of an [FCN-8s](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) semantic segmenatation architecture.

### To train a new model we need to:

Make sure that your virtulenv is activated. Otherwise, run the following inside the root directory of your project (or wherever you have your virtual environment):

`source .venv/bin/activate`

Then, change directories to 'fcn8s' and execute the 'fcn.py' file to train our FCN-8s implementation on a specified dataset (e.g., roborace750_mockup or Cityscapes) like so:

```bash
$ cd fcn8s
$ python fcn.py --dataset=roborace750_mockup --epochs=100
```


After training is done, the following folders will have been created:

*../models/sem_seg*: contains the model which has been just trained
	
*log*: contains logging info about training:
    - loss vs epochs for training and validation sets
    - IoU vs epochs for training and validation sets

### Pretrained Model for Semantic Segmentation on _fences_ and _road_

Under request at pablo.rodriguez-palafox@tum.de 


### To test a model on the Roborace dataset's test set:

Check that you are inside the [fcn8s](fcn8s) directory.

Within the virtual environment, run the following to inference on the test set of the dataset indicated in the '--dataset' argument by using a previously trained model, which will be asked automatically after running the following command:

`$ python fcn.py --mode=test --dataset=roborace750`

Enter the name of the model you want to use in the format '<epochs>-Epochs-<dataset>', e.g., `100-Epochs-roborace750`


After testing is done, the following folder and files will have appeared in the same folder as the fcn.py file:

*runs*: contains the segmented images
*log/<nameOfTheModelUsed>/iou/test_set_iou_<timestamp>.txt*: contains the IoU metric for each image of the test set
*times.txt*: inference times for each image of the test set



## 4. Monocular Depth Estimation Network (monodepth)
We use the network developed by Godard et al., called [MonoDepth](https://github.com/mrharicot/monodepth) (Copyright © Niantic, Inc. 2018. Patent Pending. All rights reserved.).

### Monodepth model (monocular depth estimation model trained on Cityscapes by [Godard]())

To download the [monodepth model](https://github.com/mrharicot/monodepth) trained on cityscapes by [Godard](https://github.com/mrharicot/monodepth), go to the [monodepth repo](https://github.com/mrharicot/monodepth) or run the following:

```bash
$ cd models
$ sudo chmod +x get_monodepth_model.sh
$ ./get_monodepth_model.sh model_cityscapes ./monodepth/model_cityscapes
``` 


## 5. SemanticDepth - The whole pipeline
SemanticDepth merges together semantic segmentation and monocular depth estimation to compute the distance from the left fence to the right fence in a FormulaE-like circuit. We have also found that by using a semantic segmentation model trained on Roborace images for fence and road classification plus a [monodepth](https://github.com/mrharicot/monodepth) model for disparity estimation, our pipeline generalizes to city environments, like those featured in our [Munich test set](data/test_images_munich)

### Test pipeline on our Munich test set

By running the command below, SemanticDepth will be applied on the [Munich test set](data/test_images_munich) using different focal lengths. By default, the list of focal lengths to try is '[380, 580]'. The reason behind trying different focal lengths is that we are using a monodepth model trained on the Cityscapes dataset, which comprises images with a certain focal lenght. Applying the same model on our own images requires that we tune the focal length so that computing depth from disparity outputs reasonable numbers.

`$ python dist2fence_frame.py --save_data`

Results will be stored inside a newly created folder called *results*. Inside this folder, a folder *380* and a folder *580* will have been created, each containing the results relative to each of the 5 test images on which we have applied the pipeline. Also, a file _data.txt_ will have been generated, where every line refers to a test image except the last line. For every line (every test image), we save the following:

real_distance | dist_naive | dist_advanced | abs(real_distance - dist_naive) | abs(real_distance - dist_advanced)

The last line of this _data.txt_ contains the Mean Absolute Error for the absolute differences between the estimated distance and the real distance (at a depth of x meters - in our experiments, we set x = 10 m)

*\*_output.ply* contains the reconstructed 3D scene, featuring only the road, the walls and the naive and advanced distances (red and green lines) [MeshLab is needed to open a PLY file]

*\*_output.png* features the segmented scene with the computed distances at the top

*\*_output_dips.png* is the disparity map that [monodepth](https://github.com/mrharicot/monodepth) predicts for the given input image

*\*_output_distances.txt* is a plain text file containing the predicted width of the road using both the naive and advanced approaches

*\*_output_times.txt* is a plain text file containing the inference times for each task of the pipeline

The rest of the files can be disregarded. They are only generated for sanity checks.


Note that you can set the --verbose option when running the previous command to get more info during execution:

`$ python dist2fence_frame.py --save_data --verbose`


#### Other functionalites


Note as well that running the python script without any arguments

`$ python dist2fence_frame.py`

will just generate the following files:

*\*_output_distances.txt* is a plain text file containing the predicted width of the road using both the naive and advanced approaches

*\*_output_times.txt* is a plain text file containing the inference times for each task of the pipeline

So no backend info (i.e.., no 3D point clouds that we use behing the scenes to compute distances).

Also, by running the following, SemanticDepth will be applied using the focal length set as param:

`$ python dist2fence_frame.py --fov=360`

Other params:

*--input_frame=<pathToImage>*: If set, the pipeline will only be applied to the indicated image 
*--aproach=both*: If set to _both_, naive and advanced approaches are used

### Run it on the Stuttgart video sequence from Cityscapes

Download the Stuttgart sequence from [Cityscapes](https://www.cityscapes-dataset.com/login/). Extract all the _png_ images from the sequence (or just a subset of the sequence) into *data/stuttgart_video_test*. Then run:

`$ python dist2fence_sequence_of_frames.py --verbose`

By default, the _naive distance_ will be computed, given that the Stuttgart sequence does not have walls/fences at each side of the road, as a Formula-E-like racetrack would, on which to compute our _advanced distance_.

In the *results* folder you will a new folder named *stuttgart_video* containing two other directories, namely *result_sequence_imgs* and *result_sequence_ply*. The former contains the output images with the computed distances written on the frame; the latter contains the masked 3D point cloud on which we compute the road's width at a certain depth.

You can then use the script _create_video_from_frames.py_ inside *utils* to convert the list of images that have been just created (*result_sequence_imgs*) into _mp4_ format.



## 6. License

This work is largely based on the work of [Godard](https://github.com/mrharicot/monodepth), named MonoDepth. MonoDepth license information is stated below:


Copyright © Niantic, Inc. 2018. Patent Pending. All rights reserved.

This Software is licensed under the terms of the UCLB ACP-A Licence which allows for non-commercial use only, the full terms of which are made available in the LICENSE file. For any other use of the software not covered by the terms of this licence, please contact info@uclb.com


From MonoDepth, we use the files *average_gradients.py*, *bilinear_sampler.py*, *monodepth_dataloader.py* and *monodepth_model.py*, which we have not modified in any way. You can find these files in the folder _monodepth_lib_. 

We do use a function (_post_process_disparity_) from the file *monodepth_simple.py* (which can be found in [monodepth](https://github.com/mrharicot/monodepth)'s repository) inside our files *dist2fence.py* and *dist2fence_cityscapes_sequence.py*.

Furthermore, files *fcn8s/fcn.py* and *fcn8s/helper.py* are based on the [FCN-8s implementation by Udacity](https://github.com/udacity/CarND-Semantic-Segmentation), released under the [MIT License](https://opensource.org/licenses/MIT).

The rest of the files not mentioned above in this project are released under a [GPLv3 License](LICENSE).
