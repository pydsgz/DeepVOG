import deepvog
from deepvog.visualisation import Visualizer


def visualize(input_vid_path, output_vid_path, record_path, eyeball_model_path):
    """
    This is a demonstration of the code - how to draw the visualization.
    The Visualizer class inherits from gaze_inferer class. Its .predict() method is overridden with video drawing code.
    The .predict() method is same as the inferer.gaze_inferer.predict() method, except it draws the visualization.
    The visualization output has the following drawings:
        1. Fitted ellipse contour (threshold = 0.5)
        2. Dummy line connecting between eyeball centre and ellipse centre
        3. Projected gaze vector extended from the ellipse centre, pointing to the outside
        4. A small circle at the ellipse centre separating the two lines above (2. and 3.)

    Parameters
    ----------
    input_vid_path : str
        Path of the input video that you want to infer. E.g. "deepvog/test_data/vid.mp4"
    output_vid_path : str
        Path of the output video that you want to write out your drawn frames
    record_path : str
        Path of the .csv results
    eyeball_model_path : str
        Path of the eyeball model .json file

    Returns
    -------
        None
    """

    # Initialization
    model = deepvog.load_DeepVOG()
    viser = Visualizer(model, 6, (240, 320), (3.6, 4.8))
    # viser.fit(input_vid_path, 32)
    # viser.save_eyeball_model(eyeball_model_path)
    viser.load_eyeball_model(eyeball_model_path)

    # (IMPORTANT) Visualization and drawing video happens in the below method
    viser.predict(input_vid_path, record_path, 32, "", output_vid_path)
