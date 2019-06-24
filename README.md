# DeepVOG
<p align="center"> 
<img width="320" height="240" src="ellipsoids.png">
</p>
DeepVOG is a framework for pupil segmentation and gaze estimation based on a fully convolutional neural network. Currently it is available for offline gaze estimation of eye-tracking video clips.

## Citation
DeepVOG has been peer-reviewed and accepted as an original article in the Journal of Neuroscience Method (Elsevier). 
The manuscript is available open access and can be downloaded free of charge [here](https://doi.org/10.1016/j.jneumeth.2019.05.016). If you use DeepVOG or some part of the code, please cite (see [bibtex](citations.bib)):

Yiu YH, Aboulatta M, Raiser T, Ophey L, Flanagin VL, zu Eulenburg P, Ahmadi SA. DeepVOG: Open-source Pupil Segmentation and Gaze Estimation in Neuroscience using Deep Learning. Journal of neuroscience methods. vol. 324, 2019, DOI: https://doi.org/10.1016/j.jneumeth.2019.05.016


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

To run DeepVOG, you need to have a Python distribution (we recommend [Anaconda](https://www.anaconda.com/)) and the following Python packages:

```
numpy
scikit-video
scikit-image
tensorflow-gpu
keras
urwid (Not necessary if you do not use the Text-based user interface)
```
As an alternative, you can use our docker image which already includes all the dependencies. The only requirement is a platform installed with nvidia driver and nvidia-docker (or nvidia runtime of docker).

### Installing
A step by step series of examples that tell you how to get DeepVOG running.<br/>
1. Installing from package

```
$ git clone https://github.com/pydsgz/DeepVOG
 (or you can download the files in this repo with your browser)
```
Move to the directory of DeepVOG that you just cloned/downloaded, and type
```
$ python setup.py install
```
If it happens to be missing some dependencies listed above, you may install them with pip: <br/>
```
$ pip install numpy
$ pip install scikit-video
$ ...
```
2. It is highly recommended to run our program in docker. You can directly pull our docker image from dockerhub. (For tutorials on docker, see [docker](https://docs.docker.com/install/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker))

```
$ docker run --runtime=nvidia -it --rm yyhhoi/deepvog:v1.1.1 bash
or
$ nvidia-docker run -it --rm yyhhoi/deepvog:v1.1.1 bash
```
### Usage (Command-line interface)
The CLI allows you to fit/infer single video, or multiple of them by importing a csv table. They can be simply called by:
```
$ python -m deepvog --fit /PATH/video_fit.mp4 /PATH/eyeball_model.json
$ python -m deepvog --infer /PATH/video_infer.mp4 /PATH/eyeball_model.json /PATH/results.csv
$ python -m deepvog --table /PATH/list_of_operations.csv
```
DeepVOG first fits a 3D eyeball model from a video clip. Base on the eyeball model, it estimates the gaze direction on any other videos if the relative position of the eye with respect to the camera remains the same. It has no problem that you fit an eyeball model and infer the gaze directions from the same video clip. However, for clinical use, some users may want to have a more accurate estimate by having a separate fitting clip where the subjects perform a calibration paradigm. <br/>

In addition, you will need to specify your camera parameters such as focal length, if your parameters differ from default values.
```
$ python -m deepvog --fit /PATH/video_fit.mp4 /PATH/eyeball_model.json --flen 12 --vid-shape 240,320 --sensor 3.6,4.8 --batchsize 32 --gpu 0
```
Please refer to [doc/documentation.md](doc/documentation.md) for the meaning of arguments and input/output formats. Alternatively, you can also type `$ python -m deepvog -h` for usage examples.


### Usage (Text-based user interface)
DeepVOG comes with a simple text-based user interface (TUI). After installation, you can simply type in terminal:
```
$ python -m deepvog --tui
```

If it is successful, you should see the interface: <br/>

<p align="center"> 
<img src="https://i.imgur.com/0zc13mv.png">
</p>
From now on, you can follow the instructions within the interface and do offline analysis on your videos.<br/>




### Usage (As a python module)
For more flexibility, you may import the module directly in python.
```python
import deepvog

# Load our pre-trained network
model = deepvog.load_DeepVOG()

# Initialize the class. It requires information of your camera's focal length and sensor size, which should be available in product manual.
inferer = deepvog.gaze_inferer(model, focal_length, video_shape, sensor_size) 

# Fit an eyeball model from "video_1.mp4". The model will be stored as the "inferer" instance's attribute.
inferer.fit("video_1.mp4")

# After fitting, infer gaze from "video_1.mp4" and output the results into "result_video_1.csv"
inferer.predict("video_1.mp4", "result_video_1.csv" )

# Optional

# You may also save the eyeball model to "video_1_mode.json" for subsequent gaze inference
inferer.save_eyeball_model("video_1_model.json") 

# By loading the eyeball model, you don't need to fit the model again with inferer.fit("video_1.mp4")
inferer.load_eyeball_model("video_1_model.json") 

```

## Publication and Citation

If you plan to use this work in your research or product, please cite this repository and our publication pre-print on [arXiv](https://arxiv.org/). 

## Authors

* **Yiu Yuk Hoi** - *Implementation and validation*
* **Seyed-Ahmad Ahmadi** - *Research study concept*
* **Moustafa Aboulatta** - *Initial work*

## Links to other related papers
- [U-Net: Convolutional Networks for Biomedical Image Segmentation
](https://arxiv.org/abs/1505.04597)
- [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)
- [A fully-automatic, temporal approach to single camera, glint-free 3D eye model fitting](https://www.cl.cam.ac.uk/research/rainbow/projects/eyemodelfit/)

## License

This project is licensed under the GNU General Public License v3.0 (GNU GPLv3) License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

We thank our fellow researchers at the German Center for Vertigo and Balance Disorders for help in acquiring data for training and validation of pupil segmentation and gaze estimation. In particular, we would like to thank Theresa Raiser, Dr. Virginia Flanagin and Prof. Dr. Peter zu Eulenburg.

DeepVOG was created under support by the German Federal Ministry of Education and Research (BMBF) in connection with the foundation of the German Center for Vertigo and Balance Disorders (DSGZ) (grant number 01 EO 0901), and a stipend of the Graduate School of Systemic Neurosciences (DFG-GSC 82/3).
