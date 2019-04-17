import numpy as np
import skvideo.io as skv
import os
import sys
from keras.models import load_model
from skvideo.utils import rgb2gray
from glob import glob
import keras.backend as K
from tensorflow.python.client import device_lib

# loss function for loadind model
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 0.0000001) / (K.sum(y_true_f) + K.sum(y_pred_f) +  0.0000001)

def dice_coef_multilabel(y_true, y_pred, numLabels=4):
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice


def main(model_path, data_dir, prediction_path):


    # Grab all the input videos
    video_extensions = ["*.pgm", "*.mp4"]
    videos = []
    for extension in video_extensions:
        video_paths = os.path.join(data_dir, extension)
        videos += glob(video_paths)
        
    # load Deep learning model
    model = load_model(model_path, custom_objects = {"dice_coef_multilabel":dice_coef_multilabel})


    # Loop for each video
    for vid in videos:
        # Define video's specific parameters
        vid_name = os.path.split(vid)[1] # with extension
        vid_ext = os.path.splitext(vid_name)[1]
        output_path = prediction_path + os.path.splitext(vid_name)[0] + ".npy"
        vid_reader = skv.FFmpegReader(vid)
        
        num_frames = vid_reader.getShape()[0]
        batch_array = np.zeros((num_frames, 240, 320, 4)).astype(np.uint8)
        print("Current video:", vid)
        print("with num_frames = ", num_frames)
        # Loop for each frame
        for idx, frame in enumerate(vid_reader.nextFrame()):

            # Printing progress
            print("\rNow is at %d/%d" %(idx, num_frames ), end="", flush=True)
            
            # Preprocessing before DL network
            if vid_ext == ".pgm" or vid_name == "4.5mA_0.0ms_180s.mp4" or "max" in vid_name:
                frame = frame[:,36:356,:]
            frame_gray = rgb2gray(frame)/255
            prediction = model.predict(frame_gray)
            batch_array[idx,] = (prediction[0,]*255).astype(np.uint8)
            
        print("\n")
        print("Saving prediction: ", output_path)
        np.save(output_path, batch_array)
        

if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = "test_data/videos"
        model_path = "models/multiple_layers.h5"
        prediction_path = "test_data/predictions/"
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[1])
        main(model_path, data_dir, prediction_path)
    else:
        print("Input the GPU device number as the first argument")
    
