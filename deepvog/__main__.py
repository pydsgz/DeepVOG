import argparse
from argparse import RawTextHelpFormatter


from .jobman import deepvog_jobman_CLI

# Running DeepVOG with command-line parser.

description_text = """

Belows are the examples of usage. Don't forget to set up camera parameters such as focal length, because it varies from equipment to equipment and is necessaray for accuracy.

    Example #1 - Fitting an eyeball model (using default camera parameters)
    python -m deepvog --fit ~/video01.mp4 ~/subject01.json

    Example #2 - Infer gaze (using default camera parameters)
    python -m deepvog --infer ~/video01.mp4 ~/subject01.json ~/subject01_video01.csv

    Example #3 - Setting up necessary parameters (focal length = 12mm, video shape = (240,320), ... etc)
    python -m deepvog --fit ./vid.mp4 ./model.json -f 12 -vs 240 320 -s 3.6 4.8 -b 32 -g 0

Caution: If your video has an aspect ratio (height/width) other than 0.75, please crop it manually until it reaches the required aspect ratio otherwise the gaze estimate will not be accurate. Futher release will include cropping option.

    Example #4 - If you cropped the video from (250,390) to (240,360), keep using the argument "-vs 250 390"
    python -m deepvog --fit ./vid.mp4 ./model.json -f 12 -vs 250 390 -s 3.6 4.8 -b 32 -g 0
"""
fit_help = "Fitting an eyeball model. Call with --fit [video_src_path] [eyeball_model_saving_path]."
infer_help = "Inter video from eyeball model. Call with --infer [video_scr_path] [eyeball_model_path] [results_saving_path]"
ori_vid_shape_help = 'Original and uncropped video shape of your camera output, height and width in pixel. Default = 240 320'
flen_help = 'Focal length of your camera in mm.'
gpu_help = 'GPU device number. Default = 0'
sensor_help = 'Sensor size of your camera digital sensor, height and width in mm. Default = 3.6 4.8'
batch_help = 'Batch size for forward inference. Default = 512.'
parser = argparse.ArgumentParser(description= description_text, formatter_class=RawTextHelpFormatter)
required = parser.add_argument_group('required arguments')
required.add_argument("--fit", help = fit_help, nargs=2, type = str, metavar=("VIDEO_SRC","MODEL_PATH"))
required.add_argument("--infer", help = infer_help, nargs=3, type=str, metavar=("VIDEO_SRC","MODEL_SRC", "RESULTS_PATH"))
parser.add_argument("-f", "--flen", help = flen_help, default = 6, type = float, metavar=("FOCAL_LENGTH"))
parser.add_argument("-g", "--gpu", help = gpu_help, default= "0", type = str, metavar= ("GPU_NUMBER"))
parser.add_argument("-vs" ,"--vidshape", help = ori_vid_shape_help, default= (240, 320), nargs=2, type = int, metavar=("HEIGHT", "WIDTH"))
parser.add_argument("-s", "--sensor", help = sensor_help, default= (3.6, 4.8), nargs=2, type = float, metavar=("HEIGHT", "WIDTH"))
parser.add_argument("-b", "--batchsize", help = batch_help, default= 512, type = int, metavar=("BATCH_SIZE"))

args = parser.parse_args()

if (args.fit is None) and (args.infer is None):
    parser.error("Either --fit or --infer argument is requried")
flen = args.flen
gpu = args.gpu
ori_video_shape = args.vidshape
sensor_size = args.sensor
batch_size = args.batchsize
jobman = deepvog_jobman_CLI(gpu, flen, ori_video_shape, sensor_size, batch_size)


if (args.fit is not None) and (args.infer is None):
    vid_src_fitting, eyemodel_save = args.fit
    jobman.fit(vid_src_fitting, eyemodel_save)
if (args.fit is None) and (args.infer is not None):
    vid_scr_inference, eyemodel_load, result_output = args.infer
    jobman.infer(eyemodel_load, vid_scr_inference, result_output)
if (args.fit is not None) and (args.infer is not None):
    vid_scr_inference, eyemodel_load, result_output = args.infer
    vid_src_fitting, eyemodel_save = args.fit
    jobman.fit(vid_src_fitting, eyemodel_save)
    jobman.infer(eyemodel_load, vid_scr_inference, result_output)
