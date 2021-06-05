# SemanticDepth

### [Paper](https://www.mdpi.com/1424-8220/19/14/3224/htm) | [Video](https://youtu.be/0yBb6kJ3mgQ)

Fusing Semantic Segmentation and Monocular Depth Estimation for Enabling Autonomous Driving in Roads without Lane Lines


| | |
|:-------------------------:|:-------------------------:|
|<img alt="test_3" src="/assets/images/test_munich/test_3.jpg">  |  <img alt="test_3_output" src="/assets/images/test_munich/test_3_output.jpg">|
|<img alt="test_3_ALL" src="/assets/images/test_munich/test_3_ALL.jpg">  |  <img alt="test_3_planes" src="/assets/images/test_munich/test_3_planes.jpg">|

---

Click on the image below to watch a VIDEO demonstrating the system on Cityscapes:

[![STUTTGART SEQUENCE](http://img.youtube.com/vi/0yBb6kJ3mgQ/0.jpg)](https://youtu.be/0yBb6kJ3mgQ)

---

<a name="intro"></a>
SemanticDepth is a deep learning-based computer vision pipeline that computes the width of the road at a certain depth in front of a car. 

It does so by fusing together two deep learning-based architectures, namely a **semantic segmentation** network ([fcn8-s](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)) and a **monocular depth estimation** network ([monodepth](https://github.com/mrharicot/monodepth)).

We have two ways of computing the width of the road at a certain depth:

* The __road's width__ (rw) itself. This measure is obtained like so. We obtain the point cloud corresponding to the road in front of the camera. Then we compute the distance between the furthest point to the left and the furthest point to the right of this road point cloud at a certain depth. Here, depth means the direction in front of the camera.

* The __fence-to-fence distance__ (f2f). In this approach we additionally extract the point clouds corresponding to left and right fences/walls to each side of the road (assuming they exist). Then we fit planes to the point clouds of the road and to those of the left and right fences. We compute the intersection between the road's plane with the left fence's plane, and the intersection between the road's plane and the right fence's plane. We end up with two intersected lines. We can now decide on a depth at which we wish to compute the width of the road, here meaning the distance between these two intersection lines.

<p align="center">
	<img src="/assets/images/semanticdepth.jpg" alt="pipeline">
</p>


## Citation
If you find our work useful in your research, please consider citing:

	@article{palafox2019semanticdepth,
	  title={Semanticdepth: Fusing semantic segmentation and monocular depth estimation for enabling autonomous driving in roads without lane lines},
	  author={Palafox, Pablo R and Betz, Johannes and Nobis, Felix and Riedl, Konstantin and Lienkamp, Markus},
	  journal={Sensors},
	  volume={19},
	  number={14},
	  pages={3224},
	  year={2019},
	  publisher={Multidisciplinary Digital Publishing Institute}
	}



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

to get the [dependencies](requirements.txt) needed.



## 2. Datasets

### Datasets for Semantic Segmentation on classes _fence_ and _road_

For the semantic segmentation task, we labeled 750 [Roborace](https://roborace.com/) images with the classes fence, road and background. For the task of labelling our own images, we used the [cityscapesScripts](https://github.com/mcordts/cityscapesScripts).

We cannot make the whole dataset public, as the original images are property of the [Roborace](https://roborace.com/) competition. A mockup of this dataset can be found [here](data/roborace750_mockup). It follows the same structure as the Cityscapes dataset. If you would like to get more images, join the [Roborace](https://roborace.com/) competition and you'll get tons of data from the racetracks.

Another option is training on [Cityscapes](https://www.cityscapes-dataset.com/) on the classes _fence_ and _road_ (and _background_). If your goal is participating in the Roborace competition, doing this can get you decent results when running inference on Roborace images.

### Datasets for Monocular Depth Estimation

[MonoDepth](https://github.com/mrharicot/monodepth), an unsupervised single image depth prediction network that we make use of in our work, can be trained on [Kitti](http://www.cvlibs.net/datasets/kitti/eval_depth_all.php) or [Cityscapes](https://www.cityscapes-dataset.com/).

We directly use a model pre-trained on Cityscapes, which you can get at the [monodepth](https://github.com/mrharicot/monodepth) repo, at the Models section. Alternatively, follow the instructions in section [Monodepth model](#monodepth).


### Munich Test Set

This is a set of 5 images of the streets of Munich on which you can test the whole pipeline. You can find it [here](data/test_images_munich). In section [Test SemanticDepth on our Munich test set](#test_pipeline) you can find the commands on how to test our whole pipeline on these images.



## 3. SemanticDepth - The whole pipeline
SemanticDepth merges together [semantic segmentation](#sem_seg) and [monocular depth estimation](#monodepth) to compute the distance from the left fence to the right fence in a Formula E-like circuit (where the Roborace competition takes place). We have also found that by using a semantic segmentation model trained on Roborace images for fence and road classification plus a [monodepth](https://github.com/mrharicot/monodepth) model pre-trained on Cityscapes our pipeline generalizes to city environments, like those featured in our [Munich test set](data/test_images_munich).

<a name="test_pipeline"></a>
### Test SemanticDepth

By running the command below, SemanticDepth will be applied on the [Munich test set](data/test_images_munich) using different focal lengths. By default, the list of focal lengths to try is `[380, 580]`. The reason behind trying different focal lengths is that we are using a [monodepth model trained on the Cityscapes dataset](#monodepth), and Cityscapes comprises images with a certain focal lenght. As the author (Godard) puts it in this [discussion](https://github.com/mrharicot/monodepth/issues/190), the behaviour of monodepth is undefined when applied on images which have different aspect ratio and focal length as those on which the model was trained, since the network really only saw one type of images. Applying the same model on our own images requires that we tune the focal length so that computing depth from disparity outputs reasonable numbers (again, see [discussion on the topic](https://github.com/mrharicot/monodepth/issues/190)).

`$ python semantic_depth.py --save_data`

Results will be stored inside a newly created folder called **results**. Inside this folder, two more directories, namely **380** and **580**, will have been created, each containing the results relative to each of the 5 test images on which we have applied SemanticDepth. Also, a file _data.txt_ will have been generated, where every line refers to a test image except the last line. For every line (every test image), we save the following:

`real_distance|road_width|fence2fence|abs(real_distance-road_width)|abs(real_distance-fence2fence)`

The last line of this _data.txt_ contains the Mean Absolute Error for the absolute differences between the estimated distance and the real distance at a depth of x meters -- in our experiments, we set x = 10 m. We compute the MAE both for the road's width and the fence-to-fence distance (see the [Introduction](#intro) for an explanation on these two approaches).

After having ran the previous python script with the `--save_data` argument set, we can now find the following inside the folders **380** and **580**:

* **\*\_output.ply** contains the reconstructed 3D scene, featuring only the road, the walls and the [road's width and fence-to-fence distance](#intro) (red and green lines, respectively). You can use [MeshLab](http://www.meshlab.net/) to open a PLY file.

* **\*\_output.png** features the segmented scene with the computed distances at the top.

* **\*\_output_dips.png** is the disparity map that [monodepth](https://github.com/mrharicot/monodepth) predicts for the given input image.

* **\*\_output_distances.txt** is a plain text file containing the [road's width and the fence-to-fence distance](#intro).

* **\*\_output_times.txt** is a plain text file containing the inference times for each task of the pipeline.

The rest of the files can be disregarded. They are only generated for sanity checks.


Note that you can set the `--verbose` option when running the previous command to get more info during execution, like so:

`$ python semantic_depth.py --save_data --verbose`


#### Other functionalites


Note as well that running the python script without any arguments

`$ python semantic_depth.py`

will just generate the following files:

* **\*\_output_distances.txt** is a plain text file containing the [road's width and fence-to-fence distance](#intro).

* **\*\_output_times.txt** is a plain text file containing the inference times for each task of the pipeline.

So no backend info (i.e., no 3D point clouds, which are just used in the backend to compute distances).

Also, by running the following, SemanticDepth will be applied using the focal length provided as argument:

`$ python semantic_depth.py --f=360`

Other params:

* `--input_frame=<pathToImage>`: If set, the pipeline will only be applied to the indicated image 
* `--aproach=both`: If set to _both_, the road's width (rw) and the fence-to-fence distance (f2f) are computed. By setting it to _rw_ only the road's width will be computed.
* `--is_city`: Must be set when we want to process an image from Cityscapes. It helps set the correct intrinsic camera parameters).

#### I just want to test the system on a single image!

Simply run:

`python semantic_depth.py --input_frame=media/images/bielefeld_018644.png --save_data --is_city --f=580`

The `--is_city` flag indicates the system that we are processing a Cityscapes frame.

### Test SemanticDepth on the Stuttgart video sequence from Cityscapes

Download the Stuttgart sequence from [Cityscapes](https://www.cityscapes-dataset.com/login/). Extract all the _png_ images from the sequence (or just a subset of the sequence) into *data/stuttgart_video_test*. Then run:

`$ python semantic_depth_cityscapes_sequence.py --verbose`

By default, the _road's width_ will be computed, given that the Stuttgart sequence does not have walls/fences at each side of the road, as a Formula-E-like racetrack would, on which to compute our _fence-to-fence distance_.

In the **results** folder (which will have been created in the root if you didn't have one yet) you will find a new folder named **stuttgart_video** containing two other directories, namely **result_sequence_imgs** and **result_sequence_ply**. The former contains the output images with the computed distances written on the frame; the latter contains the masked 3D point cloud on which we compute the road's width at a certain depth.

You can then use the script [_create_video_from_frames.py_](utils/create_video_from_frames.py) inside **utils** to convert the list of images that have been just created (**result_sequence_imgs**) into _mp4_ format.


<a name="sem_seg"></a>
## 4. Semantic Segmentation Network

The source files for the semantic segmentation network are under the folder [fcn8s](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf). There you can find an implementation of an [FCN-8s](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) semantic segmenatation architecture.

### To train a new model we need to:

* Make sure that your virtulenv is activated. Otherwise, run the following inside the root directory of your project (or wherever you have your virtual environment):

    `source .venv/bin/activate`

* Then, change directories to [fcn8s](fcn8s) and execute the **fcn.py** file to train our FCN-8s implementation on a specified dataset (e.g., roborace750_mockup or Cityscapes) like so:

    ```bash
    $ cd fcn8s
    $ python fcn.py --dataset=roborace750_mockup --epochs=100
    ```


* After training is done, the following folders will have been created:

    - **../models/sem_seg**: contains the model which has been just trained
	
    - **log** (inside [fcn8s](fcn8s)): contains logging info about training:  
        - loss vs epochs for training and validation sets
        - IoU vs epochs for training and validation sets

### Pretrained Model for Semantic Segmentation on _fences_ and _road_

Under request at pablo.rodriguez-palafox@tum.de. See [models/get_sem_seg_models.md](models/get_sem_seg_models.md) for further details on how to get them.


### Test a model on the Roborace dataset's test set:

Check that you are inside the [fcn8s](fcn8s) directory.

Within the virtual environment, execute the following to run inference on the test set of the dataset indicated in the `--dataset` argument by using a previously trained model, which will be asked automatically after running the following command:

`$ python fcn.py --mode=test --dataset=roborace750_mockup`

`Enter the name of the model you want to use in the format <epochs>-Epochs-<dataset>, e.g., 100-Epochs-roborace750`


After testing is done, the following folder and files will have appeared in the same folder as the fcn.py file:

* **runs**: contains the segmented images
* **log/<nameOfTheModelUsed>/iou/test_set_iou_<timestamp>.txt**: contains the IoU metric for each image of the test set
* **times.txt**: inference times for each image of the test set


<a name="monodepth"></a>
## 5. Monocular Depth Estimation Network (monodepth)
We use the network developed by Godard et al., called [MonoDepth](https://github.com/mrharicot/monodepth) (Copyright © Niantic, Inc. 2018. Patent Pending. All rights reserved.).


### Monodepth model (monocular depth estimation model trained on Cityscapes by [Godard](https://github.com/mrharicot/monodepth))


To download the [monodepth model](https://github.com/mrharicot/monodepth) trained on cityscapes by [Godard](https://github.com/mrharicot/monodepth), go to the [monodepth repo](https://github.com/mrharicot/monodepth) or run the following:

```bash
$ cd models
$ sudo chmod +x get_monodepth_model.sh
$ ./get_monodepth_model.sh model_cityscapes ./monodepth/model_cityscapes
``` 



## 6. License

Files [fcn8s/fcn.py](fcn8s/fcn.py) and [fcn8s/helper.py](fcn8s/helper.py) are based on the [FCN-8s implementation by Udacity](https://github.com/udacity/CarND-Semantic-Segmentation), released under the [MIT License](https://opensource.org/licenses/MIT).

The rest of the files in this project are released under a [GPLv3 License](https://www.gnu.org/licenses/gpl-3.0.en.html).

Check the [LICENSE](LICENSE) for a detailed explanation on the licenses under which this work is released.
