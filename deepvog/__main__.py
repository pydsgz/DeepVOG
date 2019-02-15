import argparse
from argparse import RawTextHelpFormatter


<<<<<<< HEAD
=======
from .jobman import deepvog_jobman_CLI

>>>>>>> a0984e4600ff68e6d564d554242dffa101e1c0c8
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
<<<<<<< HEAD
    
You can also fit and infer a video from a csv table:

    Example #5
    python -m deepvog --table ./table.csv -f 12 -vs 240 320 -s 3.6 4.8 -b 32 -g 0

The csv file must follow the format below:
    
    operation, fit_vid,         infer_vid,          eyeball_model,  result
    fit      , ./fit_vid.mp4,   ./infer_vid.mp4,    ./model.json,   ./output_result.csv
    infer    , ./fit_vid2.mp4,  ./infer_vid2.mp4,   ./model2.json,  ./output_result2.csv
    both     , ./fit_vid3.mp4,  ./infer_vid3.mp4,   ./model3.json,  ./output_result3.csv
    fit      , ./fit_vid4.mp4,  ./infer_vid4.mp4,   ./model4.json,  ./output_result4.csv
    ...

The "operation" column should contain either fit/infer/both: 
    1. fit: it will fit the video specified by the path "fit_vid" and save the model to the path specified by "eyeball_model". Other columns are irrelevant and can be omited.
    2. infer: it will load the model from the path specified in "eyeball_model", and infer the video from the path specified in "infer_vid", and then save the result to the path specified by "result".
    3. both: first performs "fit", then performs "infer".
If you run it in docker and mount your local directory, you should specify the above paths with relative paths.

"""

fit_help = "Fitting an eyeball model. Call with --fit [video_src_path] [eyeball_model_saving_path]."
infer_help = "Inter video from eyeball model. Call with --infer [video_scr_path] [eyeball_model_path] [results_saving_path]"
table_help = 'Fit or infer videos from a csv table. The column names of the csv must follow a format (see --help description). Call with --table [csv_path]'
=======
"""
fit_help = "Fitting an eyeball model. Call with --fit [video_src_path] [eyeball_model_saving_path]."
infer_help = "Inter video from eyeball model. Call with --infer [video_scr_path] [eyeball_model_path] [results_saving_path]"
>>>>>>> a0984e4600ff68e6d564d554242dffa101e1c0c8
ori_vid_shape_help = 'Original and uncropped video shape of your camera output, height and width in pixel. Default = 240 320'
flen_help = 'Focal length of your camera in mm.'
gpu_help = 'GPU device number. Default = 0'
sensor_help = 'Sensor size of your camera digital sensor, height and width in mm. Default = 3.6 4.8'
batch_help = 'Batch size for forward inference. Default = 512.'
<<<<<<< HEAD


=======
>>>>>>> a0984e4600ff68e6d564d554242dffa101e1c0c8
parser = argparse.ArgumentParser(description= description_text, formatter_class=RawTextHelpFormatter)
required = parser.add_argument_group('required arguments')
required.add_argument("--fit", help = fit_help, nargs=2, type = str, metavar=("VIDEO_SRC","MODEL_PATH"))
required.add_argument("--infer", help = infer_help, nargs=3, type=str, metavar=("VIDEO_SRC","MODEL_SRC", "RESULTS_PATH"))
<<<<<<< HEAD
required.add_argument("--table", help = table_help, type=str, metavar = ("CSV_PATH"))
=======
>>>>>>> a0984e4600ff68e6d564d554242dffa101e1c0c8
parser.add_argument("-f", "--flen", help = flen_help, default = 6, type = float, metavar=("FOCAL_LENGTH"))
parser.add_argument("-g", "--gpu", help = gpu_help, default= "0", type = str, metavar= ("GPU_NUMBER"))
parser.add_argument("-vs" ,"--vidshape", help = ori_vid_shape_help, default= (240, 320), nargs=2, type = int, metavar=("HEIGHT", "WIDTH"))
parser.add_argument("-s", "--sensor", help = sensor_help, default= (3.6, 4.8), nargs=2, type = float, metavar=("HEIGHT", "WIDTH"))
parser.add_argument("-b", "--batchsize", help = batch_help, default= 512, type = int, metavar=("BATCH_SIZE"))

args = parser.parse_args()

<<<<<<< HEAD
if (args.fit is None) and (args.infer is None) and (args.table is None):
    parser.error("Either --fit, --infer or --table argument is requried")
    
    
from .jobman import deepvog_jobman_CLI, deepvog_jobman_table_CLI

=======
if (args.fit is None) and (args.infer is None):
    parser.error("Either --fit or --infer argument is requried")
>>>>>>> a0984e4600ff68e6d564d554242dffa101e1c0c8
flen = args.flen
gpu = args.gpu
ori_video_shape = args.vidshape
sensor_size = args.sensor
batch_size = args.batchsize
<<<<<<< HEAD

if args.table is not None:
    jobman_table = deepvog_jobman_table_CLI(args.table, gpu, flen, ori_video_shape, sensor_size, batch_size)
    jobman_table.run_batch()

else:

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

    
=======
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
>>>>>>> a0984e4600ff68e6d564d554242dffa101e1c0c8
