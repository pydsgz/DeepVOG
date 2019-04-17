import skvideo.io as skv
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pdb 
from .torsion_lib.Segmentation import getSegmentation_fromDL
from .torsion_lib.draw_ellipse import fit_ellipse
from .torsion_lib.CrossCorrelation import genPolar, findTorsion
from skimage import img_as_float
from skvideo.utils import rgb2gray

class offline_inferer(object):
    def __init__(self, video_path, pred_path):
        
        # Dealing with paths
        self.video_path = video_path
        self.pred_path = pred_path
        self.vid_name_ext = os.path.split(video_path)[1]
        self.vid_name_root, self.vid_ext = os.path.splitext(self.vid_name_ext)

        # Operational parameters
        self.time_display = 150 # range of frame when plotting graph
        mpl.rcParams.update({'font.size': 8})

        # Video data related
        print("\n=================================\nNow inferring video {}".format(self.vid_name_root))
        print("Loading video data and prediction arrays...")
        self.vid_reader = skv.FFmpegReader(video_path)
        self.vid_shape = self.vid_reader.getShape()
        self.predictions = np.load(pred_path)/255 # prediction data should be stored as np.uint8
        self.predictions_masked = np.ma.masked_where(self.predictions < 0.5, self.predictions)

        # Lists/ data recorder
        self.rotation_results = []

    def plot_video(self, output_video_path, output_record_path, update_template = False):
        # Initialise writer and recorder
        self.vid_writer = skv.FFmpegWriter(output_video_path)
        self.record_fh = open(output_record_path, "w")
        self.record_fh.write("frame,rotation\n")

        # Initialise counter
        idx = 0
        # Loop for each frame
        for pred, pred_masked, frame in zip(self.predictions, self.predictions_masked, self.vid_reader.nextFrame()):

            # Print progress
            print("\rNow is at %d/%d" % (idx, self.vid_shape[0]), end="", flush=True)

            # Initialize maps and frames
            frame = img_as_float(frame) # frame ~ (240, 320, 3)
            frame_gray = rgb2gray(frame)[0,:,:,0] # frame_gray ~ (240, 320)
            frame_rgb = np.zeros(frame.shape) # frame_rgb ~ (240, 320, 3)
            frame_rgb[:,:,:] = frame_gray.reshape(frame_gray.shape[0], frame_gray.shape[1], 1)
            useful_map, (pupil_map, _, _, _) = getSegmentation_fromDL(pred)  # useful_map is used for polar transformation and cross-correlation
            _, (pupil_map_masked, iris_map_masked, glints_map_masked, visible_map_masked) = getSegmentation_fromDL(pred_masked)
            rr, _, centre, _, _, _, _, _ = fit_ellipse(pupil_map, 0.5)
            
            
            # Cross-correlation
            if idx == 0 :
                polar_pattern_template, polar_pattern_template_longer, r_template, theta_template, extra_radian = genPolar(frame_gray, useful_map, center = centre, template = True,
                                                                                        filter_sigma = 100, adhist_times = 2)
                rotated_info = (polar_pattern_template, r_template, theta_template)
                rotation = 0
            elif rr is not None:
                # for finding the rotation value and determine if it is needed to update
                rotation, rotated_info , _ = findTorsion(polar_pattern_template_longer, frame_gray, useful_map, center = centre,
                                                        filter_sigma = 100, adhist_times = 2)
                if (update_template == True) and rotation == 0:
                    polar_pattern_template, polar_pattern_template_longer, r_template, theta_template, extra_radian = genPolar(frame_gray, useful_map, center = centre, template = True,
                                                                                                    filter_sigma = 100, adhist_times = 2)

            else:
                rotation, rotated_info = np.nan, None

            self.rotation_results.append(rotation)
            self.record_fh.write("{},{}\n".format(idx, rotation))
            
            # Drawing the frames of visualisation video
            rotation_plot_arr = self._plot_rotation_curve(idx)
            segmented_frame = self._draw_segmented_area( frame_gray, pupil_map_masked, iris_map_masked, glints_map_masked, visible_map_masked)
            polar_transformed_graph_arr = self._plot_polar_transformed_graph((polar_pattern_template, r_template, theta_template), rotated_info, extra_radian)
            frames_to_draw = (frame_rgb, rotation_plot_arr, segmented_frame, polar_transformed_graph_arr)
            final_output = self._build_final_output_frame(frames_to_draw)
            self.vid_writer.writeFrame(final_output)
            idx += 1

        self.delete_handles()
        del self.predictions
        del self.predictions_masked
        
    def _plot_rotation_curve(self, idx, y_lim = (-4, 4)):
        fig, ax = plt.subplots( figsize=(3.2,2.4)) #width, height 
        if idx < self.time_display:
            
            ax.plot(np.arange(0, idx), self.rotation_results[0:idx], color = "b", label = "DeepVOG 3D")
            ax.set_xlim(0,self.time_display)
        else:

            ax.plot(np.arange(idx- self.time_display, idx), self.rotation_results[idx-self.time_display:idx], color = "b", label = "DeepVOG 3D")
            ax.set_xlim(idx-self.time_display, idx)
        ax.legend()
        ax.set_ylim(y_lim[0],y_lim[1])
        ax.set_yticks(np.arange(y_lim[0],y_lim[1]))
        plt.tight_layout()
        fig.canvas.draw()
        w,h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)/255
        buf.shape = (h, w, 3)
        plt.close()
        return buf
    def _draw_segmented_area(self, frame_gray, pupil_map_masked, iris_map_masked, glints_map_masked, visible_map_masked):
        # Plot segmented area
        fig, ax = plt.subplots(figsize=(3.2,2.4))
        ax.imshow(frame_gray, vmax=1, vmin=0, cmap="gray")
        ax.imshow(visible_map_masked, cmap="autumn", vmax=1, vmin=0, alpha = 0.2)
        ax.imshow(iris_map_masked, cmap="GnBu", vmax=1, vmin=0, alpha = 0.2)
        ax.imshow(pupil_map_masked, cmap="hot", vmax=1, vmin=0, alpha = 0.2)
        ax.imshow(glints_map_masked, cmap="OrRd", vmax=1, vmin=0, alpha = 0.2)
        ax.set_axis_off()
        plt.tight_layout()
        fig.canvas.draw()
        w,h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)/255
        buf.shape = (h, w, 3)
        plt.close()
        return buf
    def _plot_polar_transformed_graph(self, template_info, rotated_info, extra_radian ):
        (polar_pattern, r, theta) = template_info
        if rotated_info is not None:
            (polar_pattern_rotated, r_rotated, theta_rotated) = rotated_info
        else:
            polar_pattern_rotated, r_rotated, theta_rotated = np.zeros(polar_pattern.shape), r, theta

        # x axis correction
        theta_longer = np.rad2deg(theta) - np.rad2deg((theta.max()-theta.min() )/2)
        theta_shorter = np.rad2deg(theta_rotated) - np.rad2deg((theta_rotated.max() - theta_rotated.min())/2)
        theta_extra = np.rad2deg(extra_radian)
        
        # Plotting
        fig, ax = plt.subplots(2, figsize=(3.2,2.4))
        # ax[0].imshow(polar_pattern, cmap="gray", extent=(theta_longer.min() - theta_extra, theta_longer.max() + theta_extra, r.max(), r.min()), aspect='auto')
        ax[0].imshow(polar_pattern, cmap="gray", extent=(theta_shorter.min(), theta_shorter.max(), r.max(), r.min()), aspect='auto')
        ax[0].set_title("Template")
        ax[1].imshow(polar_pattern_rotated, cmap="gray", extent=(theta_shorter.min(), theta_shorter.max(), r_rotated.max(), r_rotated.min()), aspect='auto')
        ax[1].set_title("Rotated pattern")
        plt.tight_layout()
        fig.canvas.draw()
        w,h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)/255
        buf.shape = (h, w, 3)
        plt.close()
        return buf
    def _build_final_output_frame(self, frames_to_draw):
        """
        args:
            frames_to_draw: tuple with length 4. Starting from top left corner in clockwise direction.
        """
        height, width = 240, 320
        final_output = np.zeros((height*2, width*2, 3))
        final_output[0:height, 0:width, : ] = frames_to_draw[0]
        final_output[0:height, width:width*2, :] = frames_to_draw[1]
        final_output[height:height*2, 0:width, :] = frames_to_draw[2]
        final_output[height:height*2, width:width*2, :] = frames_to_draw[3]
        final_output = (final_output*255).astype(np.uint8)
        return final_output

    def delete_handles(self):
        self.record_fh.close()
        self.vid_writer.close()
        self.vid_reader.close()
        return

if __name__ == "__main__":
    pass
