from glob import glob
import os
import sys

import matplotlib as mpl
import numpy as np
from skimage.io import imread

from ellipses import LSqEllipse #The code is pulled from https://github.com/bdhammel/least-squares-ellipse-fitting

# This annotation script is written by Taha Emre

plt = mpl.pyplot


def _get_annotation_path_from_image_path(image_path):
    return os.path.splitext(image_path)[0] + '.txt'




def fit_pupil(image_path, curr_image_number, plot=False, write_annotation=False):
    while True:
        plt.ion()
        fig, ax = plt.subplots()
        img = imread(image_path)
        ax.set_title('({}): {}'.format(curr_image_number, os.path.basename(image_path)))
        ax.imshow(img, cmap='gray')

        key_points = plt.ginput(-1, mouse_pop=2, mouse_stop=3, timeout=-1) #If negative, accumulate clicks until the input is terminated manually.

        if not key_points:
            if write_annotation:
                with open(_get_annotation_path_from_image_path(image_path), 'w+') as f:
                    f.write("closed_eye")
                
                with open(os.path.splitext(image_path)[0]+"_points.txt", 'w+') as f:  #For detecting selected
                    f.write("closed_eye")
            plt.close()
            break

        fitted = LSqEllipse()
        fitted.fit([[x[0] for x in key_points], [x[1] for x in key_points]])
        center_coord, width,height,angle = fitted.parameters()
        axes = np.array([width,height])
        angle = np.rad2deg(angle)

        if write_annotation:
            with open(_get_annotation_path_from_image_path(image_path), 'w+') as f:
                if all([c <= 50 for c in center_coord]):
                    points_str = '-1:-1'
                else:
                    points_str = '{}:{}'.format(center_coord[0], center_coord[1])
                f.write(points_str)
            
            with open(os.path.splitext(image_path)[0]+"_points.txt", 'w+') as f:  #For detecting selected
                for point in key_points:
                    f.write('{}:{}\n'.format(point[0],point[1]))


        if plot:
            ax.annotate('pred center', xy=center_coord, xycoords='data',
                        xytext=(0.2, 0.2), textcoords='figure fraction',
                        arrowprops=dict(arrowstyle="->"), color='y')
            plt.scatter(x=center_coord[0], y=center_coord[1], c='red', marker='x')

            ell = mpl.patches.Ellipse(xy=center_coord, width=axes[0]*2,
                                      height=axes[1]*2, angle=angle, fill=False, color='r')

            ax.add_artist(ell)
            plt.show()
            confirmation_point = plt.ginput(1, timeout=-1, mouse_add=3, mouse_stop=3)
            plt.close()
            if len(confirmation_point) == 0:
                break

def annotate(base_dir='./video_sample'):
    images_paths = sorted(glob(os.path.join(base_dir, '*png')))
    imag_paths = sorted(glob(os.path.join(base_dir,'*jpg')))
    annotation_paths = glob(os.path.join(base_dir, '*txt'))
    i = 1
    for image_path in images_paths:
        if _get_annotation_path_from_image_path(image_path) in annotation_paths:
            continue
        else:
            fit_pupil(image_path, curr_image_number=i, plot=True, write_annotation=True)
            i += 1

    for imag_path in imag_paths:
        if _get_annotation_path_from_image_path(imag_path) in annotation_paths:
            continue
        else:
            fit_pupil(imag_path, curr_image_number=i, plot=True, write_annotation=True)
            i += 1



if __name__ == '__main__':
    if len(sys.argv) > 1:
        annotate(sys.argv[1])
    else:
        annotate()
