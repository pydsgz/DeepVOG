# Release Notes
DeepVOG v1.1.4 (Date: 31-07-2019)
>
>**Improvements**:
>1. Added `--skip_existed` flag for skipping the operation in `--table` mode if the output file already exists
>2. Added `--skip_errors` flag for skipping the operation in `--table` mode and continue the next video if error is encountered. 
>3. Added `--log_errors` flag for logging the errors and tracebacks in a file for `--table` mode, when error is encountered.
>4. Added `--no_gaze` flag for only pupil segmentation in `--infer` mode.
>5. One more column (`with_gaze`) to fill in the input csv file for `--table` mode. It enables/disable gaze estimation in `--table` mode.
>
>For details, see [doc/documentation.md](doc/documentation.md)
>
>**Removed**:
>1. Text-based User Interface (TUI) is removed.
 
 
DeepVOG v1.1.3 (Date: 24-07-2019)

>**Improvements**:
>1. Added `-v` or `--visualize` tag for visualization of pupil segmentation/gaze output in a new video.
>2. Added `-m` or `--heatmap` tag showing heatmap alongside with the visualized output when `--visualize` is enabled.    
>
>For details, see [doc/documentation.md](doc/documentation.md)
>
>**Bug fixed**:
>1. Fixed the sampling of fitting lines in RANSAC being not adaptive to the number of observations
>2. Fixed the criteria of fitting lines acceptance in RANSAC being too strict for some videos.
 