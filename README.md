# DeepVOG
<p align="center"> 
<img width="320" height="240" src="ellipsoids.png">
</p>
DeepVOG is a framework for pupil segmentation and gaze estimation based on a fully convolutional neural network. Currently it is available for offline gaze estimation of eye-tracking video clips.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

To run DeepVOG, you need to have a Python distribution (we recommend [Anaconda](https://www.anaconda.com/)) and the following Python packages:

```
numpy >= 1.12
scikit-video >=1.1.0
scikit-image >= 0.14.0
tensorflow-gpu >= 1.12.0
keras >= 2.2.4
urwid
```
As an alternative, you can use our docker image which already includes all the dependencies. The only requirement is a platform installed with nvidia driver and nvidia-docker (or nvidia runtime of docker).
### Installing
A step by step series of examples that tell you how to get DeepVOG running.<br/>
1. Installing from package

```
$ git clone https://github.com/pydsgz/DeepVOG
$ cd ~/DeepVOG/
$ python setup.py install
```
If it happens to be missing some dependencies listed above, you may install them with pip: {br/}
```
$ pip install numpy
$ pip install scikit-video
$ ...
```
2. If you are familiar with docker, you can directly pull our docker image from dockerhub. (For tutorials on docker, see [docker](https://docs.docker.com/install/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker))

```
$ docker run --runtime=nvidia -it --rm yyhhoi/deepvog:v1.0.0 bash
or
$ nvidia-docker run -it --rm yyhho/deepvog:v1.0.0 bash
```

### Usage (Text-based user interface)
DeepVOG comes with a simple text-based user interface (TUI). After installation, you can simply call in python:
```python
import deepvog
tui = deepvog.tui(base_dir) # base_dir is where you put your video data.
tui.run()
```
DeepVOG first fits a 3D eyeball model from a video clip. Base on the eyeball model, it can estimate the gaze direction on other videos. It has no problem that you fit an eyeball model and infer the gaze directions both from the same video clip. For clinical use, users may want to have a separate clip where the subject performed a calibration paradigm, specifically for model fitting. <br/>

For organization of data, it is recommended the base_dir follows the structure below: <br/>
```
/base_dir
    /fitting_dir # which contains video clips for estimating an 3D eyeball model
        /video_1.mp4
        /video_2.mp4
        /...
    /model_dir # which will store the eyeball models fitted by the program
    /inference_dir # which contain video clips on which you want to infer the gaze directions
        /video_3.mp4
        /...
    /results_dir # which will store the gaze results (.csv) inferred based on the 3D eyeball model.
        
```

Result:<br/>
![https://i.imgur.com/0zc13mv.png](https://i.imgur.com/0zc13mv.png)<br/>
From now on, you can follow the instruction within the interface and do offline analysis on your videos.<br/>

For docker user, you may call the command below:<br/>
```
$ docker run --runtime=nvidia -it --rm -v /path_to_your_base_dir:/mnt yyhhoi/deepvog:v1.0.0 bash deepvog
or
$ nvidia-docker run -it --rm -v /path_to_your_base_dir:/mnt yyhhoi/deepvog:v1.0.0 bash deepvog
```

### Usage (As a python module)
For more flexibility, you may import the module directly in python.
```python
import deepvog
model = deepvog.load_DeepVOG() # Load our pretrained deep-learning model
inferer = deepvog.gaze_inferer(model, focal_length, video_shape, sensor_size) # it requires information of your camera's focal length and sensor size, which should be available in product manual. 
inferer.fit("video_1.mp4")
inferer.predict("video_1.mp4", "result_video_1.csv" ) # infer gaze from "video_1.mp4" and output the results into "result_video_1.csv"
```
## Publication and Citation

If you plan to use this work in your research or product, please cite this repository and our publication pre-print on [arXiv](https://arxiv.org/). 

Links to other papers:


## Authors

* **Yiu Yuk Hoi** - *Implementation and validation*
* **Seyed-Ahmad Ahmadi** - *Research study concept*
* **Moustafa Aboulatta** - *Initial work*

## License

This project is licensed under the GNU General Public License v3.0 (GNU GPLv3) License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

We thank our fellow researchers at the German Center for Vertigo and Balance Disorders for help in acquiring data for training and validation of pupil segmentation and gaze estimation. In particular, we would like to thank Theresa Raiser, Dr. Virginia Flanagin and Prof. Dr. Peter zu Eulenburg.
