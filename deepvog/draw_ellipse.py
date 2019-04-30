import numpy as np
import matplotlib as mpl
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from .bwperim import bwperim
from .ellipses import LSqEllipse #The code is pulled frm https://github.com/bdhammel/least-squares-ellipse-fitting
from skimage.draw import ellipse_perimeter


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
        return output
    else:
        output[labelled == select_region.label] = 1
        return output

# input: output from bwperim -- 2D image with perimeter of the ellipse = 1
def gen_ellipse_contour_perim(perim, color = "r"): 
    # Vertices
    input_points = np.where(perim == 1)
    if (np.unique(input_points[0]).shape[0]) < 6 or (np.unique(input_points[1]).shape[0]< 6) :
        return None
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
            return (rr, cc, center, w, h, radian, ell)
        except:
            return None

def gen_ellipse_contour_perim_compact(perim): 
    # Vertices
    input_points = np.where(perim == 1)
    if (np.unique(input_points[0]).shape[0]) < 6 or (np.unique(input_points[1]).shape[0]< 6) :
        return None
    else:
        try:
            vertices = np.array([input_points[0], input_points[1]]).T
            # Contour
            fitted = LSqEllipse()
            fitted.fit([vertices[:,1], vertices[:,0]])
            center, w,h, radian = fitted.parameters()
            # Because of the np indexing of y-axis, orientation needs to be minus
            return (center, w,h, radian)
        except:
            return None

def fit_ellipse(img, threshold = 0.5, color = "r", mask=None):

    isolated_pred = isolate_islands(img, threshold = threshold)
    perim_pred = bwperim(isolated_pred)

    # masking eyelid away from bwperim_output. Currently not available in DeepVOG (But will be used in DeepVOG-3D)
    if mask is not None:
        mask_bool = mask < 0.5
        perim_pred[mask_bool] = 0

    # masking bwperim_output on the img boundaries as 0 
    perim_pred[0, :] = 0
    perim_pred[perim_pred.shape[0]-1, :] = 0
    perim_pred[:, 0] = 0
    perim_pred[:, perim_pred.shape[1]-1] = 0
    ellipse_info = gen_ellipse_contour_perim(perim_pred, color)

    return ellipse_info

def fit_ellipse_compact(img, threshold = 0.5, mask=None):
    """Fitting an ellipse to the thresholded pixels which form the largest connected area.

    Args:
        img (2D numpy array): Prediction from the DeepVOG network (240, 320), float [0,1]
        threshold (scalar): thresholding pixels for fitting an ellipse
        mask (2D numpy array): Prediction from DeepVOG-3D network for eyelid region (240, 320), float [0,1].
                                intended for masking away the eyelid such as the fitting is better
    Returns:
        ellipse_info (tuple): A tuple of (center, w, h, radian), center is a list [x-coordinate, y-coordinate] of the ellipse centre. 
                                None is returned if no ellipse can be found.
    """
    isolated_pred = isolate_islands(img, threshold = threshold)
    perim_pred = bwperim(isolated_pred)

    # masking eyelid away from bwperim_output. Currently not available in DeepVOG (But will be used in DeepVOG-3D)
    if mask is not None:
        mask_bool = mask < 0.5
        perim_pred[mask_bool] = 0

    # masking bwperim_output on the img boundaries as 0 
    perim_pred[0, :] = 0
    perim_pred[perim_pred.shape[0]-1, :] = 0
    perim_pred[:, 0] = 0
    perim_pred[:, perim_pred.shape[1]-1] = 0
    
    ellipse_info = gen_ellipse_contour_perim_compact(perim_pred)
    return ellipse_info
