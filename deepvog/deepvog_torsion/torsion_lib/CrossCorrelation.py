import numpy as np
from skimage.filters import gaussian
from skimage.exposure import equalize_adapthist as adhist
from .PolarTransform import polarTransform
from scipy.signal import correlate
from skimage import img_as_float
def guassianMap(img_mean, img_std, useful_map, filter_sigma = 1):
    guassian_map = np.random.normal(img_mean, img_std, useful_map.shape)
    guassian_map[guassian_map < 0] = 0
    guassian_map[guassian_map > 1] = 1
    guassian_map = gaussian(guassian_map, sigma= filter_sigma)
    guassian_map[useful_map==1] = 0
    return guassian_map

def genPolar(img, useful_map, center, template=False, filter_sigma = 1, adhist_times = 2):
    if adhist_times >= 1:
        img_enhanced = img_as_float(adhist(img))
    else:
        img_enhanced = img_as_float(img)
    # print(img_enhanced)
    img_enhanced[useful_map == 0] = 0
    # guassian_map = guassianMap(img.mean(), img.std(), useful_map, filter_sigma = filter_sigma)
    guassian_map = guassianMap(0.5, 0.2, useful_map, filter_sigma = filter_sigma)
    # If no radial/tangential filtering is performed, alter the codes to contain only one polarTransform function to speed up performance
    output_img, r, theta = polarTransform(img_enhanced, np.where(useful_map==1)[::-1], origin=center )
    if adhist_times >= 2:
        kernel_size = None
        if (output_img.shape[0] < 8): # solving the "Division by zero" error in adhist function (kernel_size = 0 if img.shape[?] < 8)
            kernel_size = [1,1]
            if (output_img.shape[1] > 8):
                kernel_size[1] = output_img.shape[1]//8
        output_img = adhist(output_img, kernel_size)
    output_gaussian, r_gaussian, theta_gaussian = polarTransform(guassian_map, np.where(useful_map==1)[::-1], origin=center )
    output = output_img + output_gaussian

    if template == True:
        extra_index, extra_rad = 25*50, np.deg2rad(25)
        output_longer = np.concatenate((output[:,output.shape[1]-extra_index:], output, output[:, 0:extra_index]), axis = 1)
        return output, output_longer, r, theta, extra_rad
    else:
        return output, r, theta
def findTorsion(output_template, img_r, useful_map_r, center,  filter_sigma = 1, adhist_times = 2):
    output_r, r_r, theta_r = genPolar(img_r, useful_map_r, center , filter_sigma = filter_sigma, adhist_times = adhist_times)
    output_coor = correlate(output_template, output_r, mode="same", method = "fft")
    output_mean = output_coor.mean(axis=0)
    max_index = np.where(output_mean == output_mean.max())
    rotation = max_index[0]/50-(180+25)
    return rotation.squeeze(), (output_r, r_r, theta_r), output_coor
