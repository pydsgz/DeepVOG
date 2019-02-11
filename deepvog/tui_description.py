import urwid



def instruction_listwalker(Button_centered, back_button_callback):
    list_body = []
    div = urwid.Divider()
    title = urwid.Text("Instruction", align="center")
    overview = urwid.AttrMap(Button_centered("Overview"), None, 'reversed')
    overview_text = urwid.Text(
    """
    DeepVOG performs off-line gaze inference in two steps:
        1. Fit an 3D eyeball model
        2. Infer gaze direction based on 3D eyeball model

    The results will then be stored as a .csv file.
    """)

    first_step = urwid.AttrMap(Button_centered("First step: Set up workspace parameters"), None, 'reversed')

    first_step_text = urwid.Text(
    """
    Base directory: It is the directory where you store all of your data. The user-interface will only search for information from this directory

    - Focal length: It is the focal length of your camera that you use to record your video. This information can be available from your camera product manual.

    - Original video shape: It is the shape of your video frame, (height, width) in the unit of pixel. "Original" means if you ever cropped your video to a smaller shape, please use the shape before it was cropped.

    - Sensor size: You can find out the sensor size from your camera product manual. For example, an 1/3-inch CMOS sensor has sensor size (3.6, 4.8) in mm. The wiki page https://en.wikipedia.org/wiki/Image_sensor_format provides the sensor size for different type of sensor.

    - GPU number: The GPU device number. If you have only one GPU, the default number (0) will be fine. 

    - Batch size: Size of the batch that the neural network propagate forward. It is recommended to be larger than 32, and should be a power of 2 (for example, 32, 64, 128, 256...)

    - Directory of videos for model fitting: The folder in your base directory that contains video for model fitting. If the path of that folder is "/base_directory/something/target_folder", then you just need to type "something/target_folder".

    - Directory of fitted 3D eyeball models: The folder in your base directory that the program will store the models after fitting. The program will only search for the models from this directory for gaze inference.

    - Directory of videos for gaze inference: The folder in your base directory that contains videos for gaze inference.

    - Directory of output results: The folder in your base directory where gaze inference results will be stored. The format is ".csv".

    - Save workspace parameters: a json file of "config.json" will be created (or over-written) in your base directory. It contains all information above (except the base directory which you define when you start the user-interface).

    - Load workspace parameters: Load "config.json" from your base directory.
    """
    )

    second_step = urwid.AttrMap(Button_centered("Second step: Model fitting"), None, 'reversed')

    second_step_text = urwid.Text(
    """
    1. Choose "Fit 3D eyeball models" from the main menu.

    2. Select videos for fitting. Then press "Start fitting".
    
    3. Confirm.
    
    4. Fitted models will be stored in "Directory of fitted 3D eyeball models" that you defined in the first step. The model will have the same name as the video used for fitting.
    """
    )

    third_step = urwid.AttrMap(Button_centered("Third step: Model fitting"), None, 'reversed')

    third_step_text = urwid.Text(
    """
    1. Choose "Gaze Inference" from the main menu.
    
    2. There are two options. The first option as implied by the description, searches for video that has the same name as the model's. For example, "video_1.mp4" will be inferred by model "video_1.json". The second option you can specify individually which model and video to infer.
    
    3. If you choose the first option, after confirming the setting is correct, the gaze inference will begin. Results will be stored as .csv in the directory defined in first step.
    
    4. If you chooose the second option, you will enter a "Model Selection" page. You can click into the each model, and specify which videos to be inferred by that model. Then as usual, start and confirm.
    """
    )
    results_explanation = urwid.AttrMap(Button_centered("Explanation of inference results"), None, 'reversed') 

    results_explanation_text = urwid.Text(
    """
    The .csv file contains below information:

    1. Pupil centre coordinates on the 2D image plane (pupil2D_x, pupil2D_y) in pixel

    2. Gaze direction (gaze_x, gaze_y) in degree.

    3. Pupil segmentation confidence: The higher the value, the more confidence the result is. Recommended threshold >0.96 for high accuracy.

    4. Consistence: Whether "Consistent Pupil Estimate" is applied in the model, dubbed by Swirski and Dodgson (2013). 1 means the eyeball model is used to estimate the gaze, 0 means the gaze direction is obtained by pure unprojection (which is unreliable at all). It is recommended to filter out gaze direction estimates that has consistence equal to 0.

    """
    )

    cautions = urwid.AttrMap(Button_centered("Cautions"), None, 'reversed') 

    cautions_text = urwid.Text(
    """
    1. If the shape of your video does not have a aspect ratio of height/width = 0.75, the gaze inference will not be reliable. You will need to crop the video manually, such that the aspect ratio matches (In the next release we will include an option of cropping within the user-interface).
    
    2. Batch size cannot be more than the frames of any video.
    """
    )


    back_button = Button_centered("back")
    urwid.connect_signal(back_button, 'click', back_button_callback)

    list_body = [title, div,
                 overview, div,
                 overview_text, div, 
                 first_step, div, 
                 first_step_text, div, 
                 second_step, div, 
                 second_step_text, div,
                 third_step, div,
                 third_step_text, div,
                 results_explanation, div,
                 results_explanation_text, div,
                 cautions, div,
                 cautions_text, div,
                 urwid.AttrMap(back_button, None, 'reversed')]
    return urwid.ListBox(urwid.SimpleFocusListWalker(list_body))



def aboutus_listwalker(Button_centered, back_button_callback):
    list_body = []
    div = urwid.Divider()
    title = urwid.Text("About us", align="center")

    pub = urwid.AttrMap(Button_centered("Publication and Citation"), None, 'reversed')
    pub_text = urwid.Text(
    """
    If you plan to use this work in your research or product, please cite this repository and our publication pre-print on arXiv.
    """
    )


    authors = urwid.AttrMap(Button_centered("Authors"), None, 'reversed')
    authors_text = urwid.Text(
    """
    Yuk-Hoi Yiu - Implementation and validation

    Seyed-Ahmad Ahmadi - Research study concepts

    Moustafa Aboulatta - Initial work
    """
    )

    license_title = urwid.AttrMap(Button_centered("License"), None, 'reversed')
    license_text = urwid.Text(
    """
    This project is licensed under the GNU General Public License v3.0 (GNU GPLv3) License - see the LICENSE file for details.
    """
    )

    contact = urwid.AttrMap(Button_centered("Contact"), None, 'reversed')
    contact_text = urwid.Text(
    """
    If you have any feedback for us, feel free to visit our GitHub repo: https://github.com/pydsgz/DeepVOG
    """
    )
    acknowledgments = urwid.AttrMap(Button_centered("Acknowledgments"), None, 'reversed')
    acknowledgments_text = urwid.Text(
    """
    We thank our fellow researchers at the German Center for Vertigo and Balance Disorders for help in acquiring data for training and validation of pupil segmentation and gaze estimation. In particular, we would like to thank Theresa Raiser, Dr. Virginia Flanagin and Prof. Dr. Peter zu Eulenburg.
    """
    )

    back_button = Button_centered("back")
    urwid.connect_signal(back_button, 'click', back_button_callback)

    list_body = [title, div,
                 pub, div,
                 pub_text, div,
                 authors, div,
                 authors_text, div, 
                 license_title, div,
                 license_text, div,
                 contact, div,
                 contact_text, div,
                 acknowledgments, div,
                 acknowledgments_text, div,
                 urwid.AttrMap(back_button, None, 'reversed')]
    return urwid.ListBox(urwid.SimpleFocusListWalker(list_body))