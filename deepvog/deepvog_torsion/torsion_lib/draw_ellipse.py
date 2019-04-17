import numpy as np
import matplotlib as mpl
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from .bwperim import bwperim
from .ellipses import LSqEllipse #The code is pulled frm https://github.com/bdhammel/least-squares-ellipse-fitting
from skimage.draw import ellipse_perimeter

"""
Extra files required:
    1. bwperim.py
    2. ellipses.py

fit_ellipse() function:

inputs: img, threshold
    -img: images with shape (height, width, channels)
    -threshold: float value (0,1), for determining the threshold of ellipse detection
outputs: rr, cc, center, w,h,radian, ell
    -rr: second index of an array for plotting an ellipse.
    -cc: second index of an array for plotting an ellipse. E.g. img[cc,rr] = 1
    -center: [x-coordinate, y-coordinate] of the ellipse center
    -w: width of the ellipse. It needs to be multiplied by 2 when fitted with matplotlib.patches.ellipse
    -h: height of the ellipse. Same as above for matplotlib.patches plotting
    -radian: orientation of the ellipse in radian unit.
    -ell: drawing ellipse by adding artists to axes object of matplotlib
"""

def isolate_islands(prediction, threshold):
    bw = closing(prediction > threshold , square(3))
    labelled = label(bw)  
    regions_properties = regionprops(labelled)
    max_region_area = 0
    select_region = 0
    for region in regions_properties:
        if region.area > max_region_area:
            max_region_area = region.area
            select_region = region
    output = np.zeros(labelled.shape)
    if select_region == 0:
        return output, bw
    else:
        output[labelled == select_region.label] = 1
        return output, bw

# input: output from bwperim -- 2D image with perimeter of the ellipse = 1
def gen_ellipse_contour_perim(perim, color = "r"): 
    # Vertices
    input_points = np.where(perim == 1)
    if (np.unique(input_points[0]).shape[0]) < 6 or (np.unique(input_points[1]).shape[0]< 6) :
        return None, None, None, None, None, None, None
    else:
        try:
            vertices = np.array([input_points[0], input_points[1]]).T
            # Contour
            fitted = LSqEllipse()
            fitted.fit([vertices[:,1], vertices[:,0]])
            center, w,h, radian = fitted.parameters()
            ell = mpl.patches.Ellipse(xy = [center[0],center[1]], width = w*2, height = h*2, angle = np.rad2deg(radian), fill = False, color = color)
            # Because of the np indexing of y-axis, orientation needs to be minus
            rr, cc = ellipse_perimeter(int(np.round(center[0])), int(np.round(center[1])), int(np.round(w)), int(np.round(h)), -radian)
            return rr, cc, center, w,h, radian, ell
        except:
            return None, None, None, None, None, None, None


# If no ellipse is detected, outputs will be all None
def fit_ellipse(img, threshold, color = "r"):
    isolated_pred, thresholded_closed = isolate_islands(img, threshold = threshold)
    perim_pred = bwperim(isolated_pred)
    ########### edition, masking bwperim_output on the img boundaries as 0 ##############
    perim_pred[0, :] = 0
    perim_pred[perim_pred.shape[0]-1, :] = 0
    perim_pred[:, 0] = 0
    perim_pred[:, perim_pred.shape[1]-1] = 0
    ##################################
    rr, cc, center, w,h,radian, ell = gen_ellipse_contour_perim(perim_pred, color)
    return rr, cc, center, w, h, radian, ell, (thresholded_closed, isolated_pred, perim_pred)