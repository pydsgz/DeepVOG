import skvideo.io as skv
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import load_model
import keras.backend as K
import numpy as np
import os
from .CheckEllipse import computeEllipseConfidence
from .eyefitter import SingleEyeFitter
from .utils import save_json, load_json, convert_vec2angle31
import logging


"""
Ensure video has:
    1. shape (240, 320) by 
        1. resize or 
        2. crop(not yet implemented)
    2. values of float [0,1]
    3. grayscale
    
"""



class gaze_inferer(object):
    def __init__(self, model, flen, ori_video_shape, sensor_size):
        """
        Initialize necessary parameters and load deep_learning model
        
        Args:
            model: Deep learning model that perform image segmentation. Pre-trained model is provided at https://github.com/pydsgz/DeepVOG/model/DeepVOG_model.py, simply by loading load_DeepVOG() with "DeepVOG_weights.h5" in the same directory. If you use your own model, it should take input of grayscale image (m, 240, 320, 3) with value float [0,1] and output (m, 240, 320, 3) with value float [0,1] where (m, 240, 320, 1) is the pupil map.
            
            flen (float): Focal length of camera in mm. You can look it up at the product menu of your camera
            
            ori_video_shape (tuple or list or np.ndarray): Original video shape from your camera, (height, width) in pixel. If you cropped the video before, use the "original" shape but not the cropped shape
            
            sensor_size (tuple or list or np.ndarray): Sensor size of your camera, (height, width) in mm. For 1/3 inch CMOS sensor, it should be (3.6, 4.8). Further reference can be found in https://en.wikipedia.org/wiki/Image_sensor_format and you can look up in your camera product menu
        

        """
        # Assertion of shape
        try:
            assert ((isinstance(flen, int) or isinstance(flen, float)))
            assert (isinstance(ori_video_shape, tuple) or isinstance(ori_video_shape, list) or isinstance(ori_video_shape, np.ndarray))
            assert (isinstance(sensor_size, tuple) or isinstance(sensor_size, list) or isinstance(sensor_size, np.ndarray))
            assert (isinstance(sensor_size, tuple) or isinstance(sensor_size, list) or isinstance(sensor_size, np.ndarray) )
        except AssertionError:
            print("At least one of your arguments does not have correct type")
            raise TypeError
        # logging.basicConfig(level=logging.DEBUG)
        # Parameters dealing with camera and video shape
        self.flen = flen
        self.ori_video_shape, self.sensor_size = np.array(ori_video_shape).squeeze(), np.array(sensor_size).squeeze()
        self.mm2px_scaling = np.linalg.norm(self.ori_video_shape)/np.linalg.norm(self.sensor_size)
        

        self.model = model
        self.confidence_fitting_threshold = 0.96
        self.eyefitter = SingleEyeFitter(focal_length= self.flen * self.mm2px_scaling,
                                         pupil_radius= 2 * self.mm2px_scaling,
                                         initial_eye_z= 50 * self.mm2px_scaling)

    def fit(self, video_src, batch_size = 32, print_prefix=""):
        """
        Fitting an eyeball model from video_src. After calling this method, eyeball model is stored as the attribute of the instance. 
        After fitting, you can either call .save_eyeball_model() to save the model for later use, or directly call .predict() for gaze inference

        Args:
            video_src (str): Path to the video by which you want to fit an eyeball model
            batch_size (int): Batch size for each forward pass in neural network
            print_prefix (str): Printing out identifier in case you need
        """
        video_name_root, ext, vreader, (fitvid_m, fitvid_w, fitvid_h, fitvid_channels), shape_correct, image_scaling_factor = self._get_video_info(video_src)
        
        # Correct eyefitter parameters in accord with the image resizing
        self.eyefitter.focal_length = self.flen*self.mm2px_scaling * image_scaling_factor
        self.eyefitter.pupil_radius = 2 * self.mm2px_scaling * image_scaling_factor

        initial_frame, final_frame = 0 , fitvid_m
        num_frames_fit = final_frame - initial_frame
        # Duration not yet implement#
        final_batch_idx = final_frame - (final_frame % batch_size) 
        X_batch = np.zeros((batch_size, 240, 320, 3))
        X_batch_final = np.zeros((final_frame % batch_size, 240, 320, 3))
        for idx, frame in enumerate(vreader.nextFrame()):

            print("\r%sFitting %s (%d%%)" % (print_prefix, video_name_root+ext, (idx/fitvid_m)*100), end="", flush=True)
            frame_preprocessed = self._preprocess_image(frame, shape_correct)
            mini_batch_idx = idx % batch_size
            if ((mini_batch_idx != 0) and (idx < final_batch_idx)) or (idx == 0):
                X_batch[mini_batch_idx, :, :, :] = frame_preprocessed
            elif ((mini_batch_idx == 0) and (idx < final_batch_idx) or (idx == final_batch_idx)):
                Y_batch = self.model.predict(X_batch)
                self._fitting_batch(Y_batch)
                # self._write_batch(Y_batch) # Just for debugging
                X_batch = np.zeros((batch_size, 240, 320, 3))
                X_batch[mini_batch_idx, :, :, :] = frame_preprocessed
            elif ((idx > final_batch_idx) and (idx != final_frame - 1)):
                X_batch_final[idx - final_batch_idx, :, :, :] = frame_preprocessed
            elif (idx == final_frame - 1):
                print("\r%sFitting %s (100%%)" % (print_prefix, video_name_root+ext), end="", flush=True)
                X_batch_final[idx - final_batch_idx, :, :, :] = frame_preprocessed
                Y_batch = self.model.predict(X_batch_final)
                self._fitting_batch(Y_batch)
                # self._write_batch(Y_batch) # Just for debugging
        logging.debug("\n######### FITTING STARTS ###########")
        _ = self.eyefitter.fit_projected_eye_centre(ransac=True, max_iters = 100, min_distance = 3*num_frames_fit)
        radius, _ = self.eyefitter.estimate_eye_sphere()
        if (self.eyefitter.eye_centre is None) or (self.eyefitter.aver_eye_radius is None):
            logging.error("None 3D model detected, entering python debugging mode")
        else:
            logging.debug("(Model|c,mm) Projected eye centre: {}\n".format(self.eyefitter.eye_centre.squeeze()))
            logging.debug("(Model|c,mm) Eye sphere radius: {}".format(radius/self.mm2px_scaling))
        logging.debug("######### FITTING ENDS ###########\n")
        vreader.close()
        print()

    def predict(self, video_src, output_record, batch_size=32, print_prefix = ""):
        """
        Inferring gaze directions from video_src and write the records (.csv) to output_record. 
        Eyeball model has to be initialized first, either by calling self.fit() method, or by loading it from path with self.load_eyeball_model()
        
        Args:
            video_src (str): Path to the video from which you want to infer the gaze direction
            output_record (str): Path to the .csv file where you want to save the result data (e.g. pupil centre coordinates and gaze estimates)
            batch_size (int): Batch size for each forward pass in neural network
            print_prefix (str): Printing out identifier in case you need
        """
        self._check_eyeball_model_exists()
        self.results_recorder = open(output_record, "w")
        self.results_recorder.write("frame,pupil2D_x,pupil2D_y,gaze_x,gaze_y,confidence,consistence\n")
        video_name_root, ext, vreader, (infervid_m, infervid_w, infervid_h, infervid_channels), shape_correct, image_scaling_factor = self._get_video_info(video_src)
        # Correct eyefitter parameters in accord with the image resizing
        self.eyefitter.focal_length = self.flen*self.mm2px_scaling * image_scaling_factor
        self.eyefitter.pupil_radius = 2 * self.mm2px_scaling * image_scaling_factor
        
        final_batch_size = infervid_m % batch_size
        final_batch_idx = infervid_m - final_batch_size
        X_batch = np.zeros((batch_size, 240, 320, 3))
        X_batch_final = np.zeros((infervid_m % batch_size, 240, 320, 3))
        for idx, frame in enumerate(vreader.nextFrame()):

            print("\r%sInferring %s (%d%%)" % (print_prefix, video_name_root+ext, (idx/infervid_m)*100), end="", flush=True)
            frame_preprocessed = self._preprocess_image(frame, shape_correct)
            mini_batch_idx = idx % batch_size

            # Before reaching the batch size, stack the array
            if ((mini_batch_idx != 0) and (idx < final_batch_idx)) or (idx == 0):
                X_batch[mini_batch_idx, :, :, :] = frame_preprocessed

            # After reaching the batch size, but not the final batch, predict and infer angles
            elif ((mini_batch_idx == 0) and (idx < final_batch_idx) or (idx == final_batch_idx)):
                Y_batch = self.model.predict(X_batch)
                # =============== infer angles by batch here ====================
                positions, gaze_angles, inference_confidence = self._infer_batch(Y_batch, idx-batch_size)
                X_batch = np.zeros((batch_size, 240, 320, 3))
                X_batch[mini_batch_idx, :, :, :] = frame_preprocessed

            # Within the final batch but not yet reaching the last index, stack the array
            elif ((idx > final_batch_idx) and (idx != infervid_m - 1)):
                X_batch_final[idx - final_batch_idx, :, :, :] = frame_preprocessed

            # Within the final batch and reaching the last index, predict and infer angles
            elif (idx == infervid_m - 1):
                print("\r%sInferring %s (100%%)" % (print_prefix, video_name_root+ext), end="", flush=True)
                X_batch_final[idx - final_batch_idx, :, :, :] = frame_preprocessed
                Y_batch = self.model.predict(X_batch_final)
                # =============== infer angles by batch here ====================
                positions, gaze_angles, inference_confidence = self._infer_batch(Y_batch, idx - final_batch_size)
            else:
                import pdb
                pdb.set_trace()
        self.results_recorder.close()
        print()
    def save_eyeball_model(self, path):
        """
        Save eyeball model parameters in json format.
        
        Args:
            path (str): path of the eyeball model file.
        """
        
        if (self.eyefitter.eye_centre is None) or (self.eyefitter.aver_eye_radius is None):
            print("3D eyeball model not found")
            raise
        else:
            save_dict = {"eye_centre": self.eyefitter.eye_centre.tolist(),
                         "aver_eye_radius": self.eyefitter.aver_eye_radius}
            save_json(path, save_dict)
    def load_eyeball_model(self, path):
        """
        Load eyeball model parameters of json format from path.
        
        Args:
            path (str): path of the eyeball model file.
        """
        loaded_dict = load_json(path)
        if (self.eyefitter.eye_centre is None) or (self.eyefitter.aver_eye_radius is None):
            self.eyefitter.eye_centre = np.array(loaded_dict["eye_centre"])
            self.eyefitter.aver_eye_radius = loaded_dict["aver_eye_radius"]
            
        else:
            logging.warning("3D eyeball exists and reloaded")

    def _fitting_batch(self, Y_batch):
        for Y_each in Y_batch:
            pred_each = Y_each[:,:,1]
            _, _, _, _, (_, _, centre, w, h, radian, ellipse_confidence) = self.eyefitter.unproject_single_observation(pred_each)
            
            if (ellipse_confidence > self.confidence_fitting_threshold) and (centre is not None):
                self.eyefitter.add_to_fitting()

    def _infer_batch(self, Y_batch, idx):
        for batch_idx, Y_each in enumerate(Y_batch):
            frame = idx + batch_idx + 1
            pred_each = Y_each[:,:,1]
            _, _, _, _, (_, _, centre, w, h, radian, ellipse_confidence) = self.eyefitter.unproject_single_observation(pred_each)
            if centre is not None:
                p_list, n_list, _, consistence = self.eyefitter.gen_consistent_pupil()
                p1, n1 = p_list[0], n_list[0]
                px, py, pz = p1[0,0], p1[1,0], p1[2,0]
                x, y = convert_vec2angle31(n1)
                positions = (px, py, pz, centre[0], centre[1]) # Pupil 3D positions and 2D projected positions
                gaze_angles = (x, y) # horizontal and vertical gaze angles
                inference_confidence = (ellipse_confidence, consistence)
                self.results_recorder.write("%d,%f,%f,%f,%f,%f,%f\n" % (frame, centre[0], centre[1],
                                                                                 x, y,
                                                                                 ellipse_confidence, consistence))
                
            else:
                positions, gaze_angles, inference_confidence = None, None, None
                self.results_recorder.write("%d,%f,%f,%f,%f,%f,%f\n" % (frame, np.nan, np.nan,
                                                                                np.nan, np.nan,
                                                                                np.nan, np.nan))
        return positions, gaze_angles, inference_confidence

    def _get_video_info(self, video_src):
        video_name_with_ext = os.path.split(video_src)[1]
        video_name_root, ext = os.path.splitext(video_name_with_ext)
        vreader = skv.FFmpegReader(video_src)
        m, w, h, channels = vreader.getShape()
        image_scaling_factor = np.linalg.norm((240,320))/np.linalg.norm((h,w))
        shape_correct = self._inspectVideoShape(w, h)
        return video_name_root, ext, vreader, (m, w, h, channels), shape_correct, image_scaling_factor

    def _check_eyeball_model_exists(self):
        try:
            assert isinstance(self.eyefitter.eye_centre, np.ndarray)
            assert self.eyefitter.eye_centre.shape == (3,1)
            assert self.eyefitter.aver_eye_radius is not None
        except AssertionError as e:
            logging.error("You must initialize 3D eyeball parameters first by fit() function")
            raise e


    @staticmethod
    def _inspectVideoShape(w, h):
        if (w, h) == (240, 320):
            return True
        else:
            return False
    @staticmethod
    def _computeCroppedShape(ori_video_shape, crop_size):
        video = np.zeros(ori_video_shape)
        cropped = video[crop_size[0]:crop_size[1], crop_size[2], crop_size[3]]
        return cropped.shape

    @staticmethod
    def _preprocess_image(img, resizing):
        """
        
        Args:
            img (numpy array): unprocessed image with shape (w, h, 3) and values int [0, 255]
        Returns:
            output_img (numpy array): processed grayscale image with shape ( 240, 320, 1) and values float [0,1]
        """
        output_img = np.zeros((240, 320, 3))
        img = img/255
        img = rgb2gray(img)
        if resizing == True:
            img = resize(img, (240, 320))
        output_img[:,:,:] = img.reshape( 240, 320, 1)
        return output_img
        

    def __del__(self):
        pass
        # self.vwriter.close()    
