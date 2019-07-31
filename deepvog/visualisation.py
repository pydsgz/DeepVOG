from skimage.draw import ellipse_perimeter, line, circle_perimeter, line_aa
import skvideo.io as skv
import numpy as np


def draw_line(output_frame, frame_shape, o, l, color=[255, 0, 0]):
    """

    Parameters
    ----------
    output_frame : numpy.darray
        Video frame to draw the circle. The value of video frame should be of type int [0, 255]
    frame_shape : list or tuple or numpy.darray
        Shape of the frame. For example, (240, 320)
    o : list or tuple or numpy.darray
        Origin of the line, with shape (2,) denoting (x, y).
    l : list or tuple or numpy.darray
        Vector with length. Body of the line. Shape = (2, ), denoting (x, y)
    color : tuple or list or numpy.darray
        RBG colors, e.g. [255, 0, 0] (red color), values of type int [0, 255]

    Returns
    -------
    output frame : numpy.darray
        Frame with the ellipse drawn.
    """
    R, G, B = color
    rr, cc = line(int(np.round(o[0])), int(np.round(o[1])), int(np.round(o[0] + l[0])), int(np.round(o[1] + l[1])))
    rr[rr > int(frame_shape[1]) - 1] = frame_shape[1] - 1
    cc[cc > int(frame_shape[0]) - 1] = frame_shape[0] - 1
    rr[rr < 0] = 0
    cc[cc < 0] = 0
    output_frame[cc, rr, 0] = R
    output_frame[cc, rr, 1] = G
    output_frame[cc, rr, 2] = B
    return output_frame


def draw_ellipse(output_frame, frame_shape, ellipse_info, color=[255, 255, 0]):
    """
    Draw a circle on an image or video frame. Drawing will be discretized.

    Parameters
    ----------
    output_frame : numpy.darray
        Video frame to draw the circle. The value of video frame should be of type int [0, 255]
    frame_shape : list or tuple or numpy.darray
        Shape of the frame. For example, (240, 320)
    ellipse_info : list or tuple
        Information of ellipse parameters. (rr, cc, centre, w, h, radian, ellipse_confidence).
    color : tuple or list or numpy.darray
        RBG colors, e.g. [255, 0, 0] (red color), values of type int [0, 255]

    Returns
    -------
    output frame : numpy.darray
        Frame withe the ellipse drawn.
    """

    R, G, B = color
    (rr, cc, centre, w, h, radian, ellipse_confidence) = ellipse_info
    rr[rr > int(frame_shape[1]) - 1] = frame_shape[1] - 1
    cc[cc > int(frame_shape[0]) - 1] = frame_shape[0] - 1
    rr[rr < 0] = 0
    cc[cc < 0] = 0
    output_frame[cc, rr, 0] = R
    output_frame[cc, rr, 1] = G
    output_frame[cc, rr, 2] = B
    return output_frame


def draw_circle(output_frame, frame_shape, centre, radius, color=[255, 0, 0]):
    """
    Draw a circle on an image or video frame. Drawing will be discretized.

    Parameters
    ----------
    output_frame : numpy.darray
        Video frame to draw the circle. The value of video frame should be of type int [0, 255]
    frame_shape : list or tuple or numpy.darray
        Shape of the frame. For example, (240, 320)
    centre : list or tuple or numpy.darray
        x,y coordinate of the circle centre
    radius : int or float
        Radius of the circle to draw.
    color : tuple or list or numpy.darray
        RBG colors, e.g. [255, 0, 0] (red color), values of type int [0, 255]

    Returns
    -------
    output frame : numpy.darray
        Frame withe the circle drawn.
    """

    R, G, B = color
    rr_p1, cc_p1 = circle_perimeter(int(np.round(centre[0])), int(np.round(centre[1])), radius)
    rr_p1[rr_p1 > int(frame_shape[1]) - 1] = frame_shape[1] - 1
    cc_p1[cc_p1 > int(frame_shape[0]) - 1] = frame_shape[0] - 1
    rr_p1[rr_p1 < 0] = 0
    cc_p1[cc_p1 < 0] = 0
    output_frame[cc_p1, rr_p1, 0] = R
    output_frame[cc_p1, rr_p1, 1] = G
    output_frame[cc_p1, rr_p1, 2] = B
    return output_frame


class VideoManager:
    def __init__(self, vreader, output_record_path="", output_video_path="", heatmap=False):
        # Parameters
        self.vreader = vreader
        self.heatmap = heatmap
        self.output_video_flag = True if output_video_path else False
        self.output_record_flag = True if output_record_path else False
        self.vwriter = skv.FFmpegWriter(output_video_path) if self.output_video_flag else None
        self.results_recorder = open(output_record_path, "w") if self.output_record_flag else None

        # Initialization actions
        self._initialize_results_recorder()

    def write_frame_with_condition(self, vid_frame, pred_each):

        if self.heatmap:
            heatmap_frame = np.zeros((pred_each.shape[0], pred_each.shape[1], 3))  # Shape = (w, h, 3)
            heatmap_frame[:, :, :] = np.around(
                pred_each.reshape(pred_each.shape[0], pred_each.shape[1], 1) * 255).astype(int)
            output_frame = np.concatenate((vid_frame, heatmap_frame), axis=1)
            self.vwriter.writeFrame(output_frame)
        else:
            self.vwriter.writeFrame(vid_frame)

    def write_results(self, frame_id, pupil2D_x, pupil2D_y, gaze_x, gaze_y, confidence, consistence):
        self.results_recorder.write("%d,%f,%f,%f,%f,%f,%f\n" % (frame_id, pupil2D_x, pupil2D_y,
                                                                gaze_x, gaze_y,
                                                                confidence, consistence))

    def _initialize_results_recorder(self):
        if self.output_record_flag:
            self.results_recorder.write("frame,pupil2D_x,pupil2D_y,gaze_x,gaze_y,confidence,consistence\n")

    def __del__(self):
        self.vreader.close()
        if self.vwriter:
            self.vwriter.close()
        if self.results_recorder:
            self.results_recorder.close()
