# Documentation
This guide explains the arguments of the program's command. You can always type `--help` or `-h` to see the arguments.

## Application modes
DeepVOG works with four modes (fit, infer, table, tui). You can only specify **EXACTLY** one mode by using one of the below arguments. Program will not proceed if more than one arguments from belows is entered.

- `--fit`: Fit an eyeball model from a video. It accept two arguments. (1) Path of the video of which you want to fit an eyeball model. (2) Path of the .json file that you want to save your fitted model.
- `--infer`: Infer gaze directions from a video based on a fitted eyeball model. It accepts three arguments. (1) Path of the video of which you want to infer the gaze direction. (2) Path of the .json file of which you stored your fitted eyeball model previously with `--fit` argument. (3) Path of the gaze results that you want to save.
- `--table`: Fit or infer videos from a csv file. It accepts one argument (1) Path of the csv file. The content of csv file must follow a format, see section [Input/output format](#input/output-format).
- `--tui`: Open Text-Based User interface (TUI). It accepts one argument (1) Base directory that you store/will store you videos, eyemodels and output results. The instruction is inside the TUI.

## Camera intrinsic parameters and others
Although the default camera intrinsic parameters are given, they vary largely across different cameras. it is very important that you specify the correct values according to the camera that you used to record the videos, otherwise the gaze results would be different. The camera intrinsic parameters can be found in their production manual which is usually available online in the provider's website.

- `-f` or `--flen`: Focal length of your camera in **mm**. Default: `6`.
- `-g` or `--gpu`: GPU device number. Default: `0`.
- `-vs` or `--vidshape`: Original and uncropped video shape of your camera output, height and width in **pixel**. Default: `(240,320)`.
- `-s` or `--sensor`: Sensor size of your camera digital sensor, height and width in **mm**. Default: `(3.6,4.8)`. 
- `-b` or `--batchsize`: Batch size of video frames for gaze inference. It is recommended to be at least 32. Default: `512`.
- `-v` or `--visualize`: Path of the video you want to store your visualization. Default: `""` (no visualization to save). This function is not yet available with `--table` and `--tui` mode.
- `-m` or `--heatmap`: Showing heatmap in the saved visualization video. This function is not yet available with `--table` and `--tui` mode. 

## Input/output format

### CSV table
With `--table` argument, you enter the "table" mode of DeepVOG, that you can batch fit/infer multiple videos without giving command one by one. The CSV table must follow the format below:

<pre>
    operation, fit_vid,             infer_vid,              eyeball_model,      result
    fit      , /PATH/fit_vid1.mp4,  /PATH/infer_vid1.mp4,   /PATH/model1.json,  /PATH/output_result1.csv
    infer    , /PATH/fit_vid2.mp4,  /PATH/infer_vid2.mp4,   /PATH/model2.json,  /PATH/output_result2.csv
    both     , /PATH/fit_vid3.mp4,  /PATH/infer_vid3.mp4,   /PATH/model3.json,  /PATH/output_result3.csv
    fit      , /PATH/fit_vid4.mp4,  /PATH/infer_vid4.mp4,   /PATH/model4.json,  /PATH/output_result4.csv
    ...
</pre>

1. Delimitor is comma. Leading/trailing spaces do not matter since they will be ignored.
2. The column titles must follow the same order and texts as above, i.e. `operation`, `fit_vid`, `infer_vid`, `eyeball_model`, `result`.
3. In the `operation` column, it contains three options `fit`, `infer` or `both`. 
   * `fit`:  Fit eyeball model from the video specified in `fit_vid`, and save the model to the path specified in `eyeball_model`. Columns `infer_vid` and `result` will be ignored.
   * `infer`: Load the eyeball model from the path specified in `eyeball_model`, infer the gaze from the video specified in `infer_vid` and save the gaze estimation results to the path specified in `result`. Column `fit_vid` will be ignored.
   * `both`: Combining both operations `fit` and `infer`. First fit an eyeball model from the video specified in `fit_vid`, then save it to the path in `eyeball_model`, and use the saved eyeball model to infer gaze direction from the video specified in `infer_vid`, and finally, save the gaze estimation results to the path in `result`.

### Output results
Gaze estimation result is saved in a .csv file, which contains the following information:

1. Pupil centre coordinates on the 2D image plane (pupil2D_x, pupil2D_y) in pixel.

2. Angular eye movement in horizontal/yaw (gaze_x) and in vertical/pitch (gaze_y) in degree.

3. Pupil segmentation confidence: The higher the value, the more confidence the result is. Recommended threshold > 0.96 for high accuracy.

4. Consistence: Whether "Consistent Pupil Estimate" is applied during the gaze estimation, a feature from Swirski and Dodgson (2013). 1 means the eyeball model is used to estimate the gaze, 0 means the gaze direction is obtained by pure unprojection (which is unreliable). It is recommended to filter out gaze direction estimates that has consistence equal to 0.

As a result, your .csv output will store 6 columns of data: `pupil2D_x`, `pupil2D_y`, `gaze_x`, `gaze_y`, `confidence`, `consistence`
