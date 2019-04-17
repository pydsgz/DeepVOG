import numpy as np
import matplotlib as mpl
from skimage.draw import polygon
from .draw_ellipse import fit_ellipse
def checkEllipse(x,y, c1,c2,w,h,radian):
    x_can = (x-c1) * np.cos(radian) + (y - c2) * np.sin(radian)
    y_can = -(x-c1) * np.sin(radian) + (y - c2) * np.cos(radian)
    output = np.power(x_can/w,2) + np.power(y_can/h,2)
    return output

def generateSegmentedMap_ellipse(xx, yy, fitter_class, points, color = "r", draw=True):
    fitter = fitter_class()
    pts_np = np.array(points)
    fitter.fit([pts_np[:,0], pts_np[:,1]])
    center, w,h, radian = fitter.parameters()
    output_map = checkEllipse(xx, yy, center[0], center[1], w, h, radian)
    output_map[output_map<1] = 1
    output_map[output_map>1] = 0
    if draw == True:
        ell = mpl.patches.Ellipse(xy = [center[0],center[1]], width = w*2, height = h*2, angle = np.rad2deg(radian), fill = False, color = color)
        return output_map, ell
    else:
        return output_map
def generateSegmentedMap_polygon(xx, yy, points, color = "r", draw=True):
    pts_np = np.array(points)
    rr,cc = polygon(pts_np[:,1], pts_np[:,0], xx.shape)
    output_map = np.zeros(xx.shape)
    output_map[rr,cc] = 1
    if draw == True:
        ell = mpl.patches.Polygon(pts_np, fill=False)
        return output_map, ell
    else:
        return output_map

# not needed. Conversion from raw labels to trianing/testing data
def getSegmentation_fromJson(img1_path, img2_path, img_shape=(240,320)):
    img1_data = readSingleCoords_json(img1_path)
    img2_data = readSingleCoords_json(img2_path)
    xx, yy = np.meshgrid(np.arange(img_shape[1]), np.arange(img_shape[0]))
    
    def segmentation_part(xx,yy,LSqEllipse, data_dict ):
        pupil_map, pupil_ell = generateSegmentedMap_ellipse(xx, yy, LSqEllipse, data_dict["pupil_boundary"], color = "r")
        iris_map, iris_ell = generateSegmentedMap_ellipse(xx, yy, LSqEllipse, data_dict["iris_boundary"], color = "g")
        glint_1_map, glint_1_ell = generateSegmentedMap_ellipse(xx, yy, LSqEllipse, data_dict["glint_1_boundary"], color = "r")
        glint_2_map, glint_2_ell = generateSegmentedMap_ellipse(xx, yy, LSqEllipse, data_dict["glint_2_boundary"], color = "r")
        visible_map, visible_ell = generateSegmentedMap_polygon(xx, yy, data_dict["visible_boundary"], color = "b")
        useful_map = np.zeros(xx.shape)
        useful_map[(pupil_map == 0) & (iris_map == 1) & (visible_map == 1) & (glint_1_map == 0) & (glint_2_map == 0)] = 1
        return useful_map
    useful_map1 = segmentation_part(xx,yy, LSqEllipse, img1_data)
    useful_map2 = segmentation_part(xx,yy, LSqEllipse, img2_data)
    return useful_map1, useful_map2



def getSegmentation_fromDL(pred, thresholding = True, threshold = 0.5):
    """
    Arguments:
        pred = 3D numpy array ~ (height, width, channels), value's range between 0 and 1 (float). Prediction output of the DL network
        thresholding = Boolean. Whether you threshold the pred array or not.
        threshold = scalar.
    Return:
        useful_map = 2D numpy array. Segmented region of the eye for torsional tracking.
    """

    if thresholding == True:
        pred[pred > threshold] = 1
        pred[pred < threshold] = 0
    img_shape = (pred.shape[0], pred.shape[1])
    useful_map = np.zeros(img_shape)
    pupil_map = pred[:,:,0]
    iris_map = pred[:,:,1]
    combined_glints_map = pred[:,:,2]
    visible_map = pred[:,:,3]
    useful_map[(pupil_map == 0) & (iris_map == 1) & (visible_map == 1) & (combined_glints_map == 0)] = 1
    return useful_map, (pupil_map, iris_map, combined_glints_map, visible_map)