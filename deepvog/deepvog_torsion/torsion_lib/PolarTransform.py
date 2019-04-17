import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage
import skimage.io as ski

#%%
def cart2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def index_coords(data, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y
def polarTransform(img, data, origin=None):
    """Reprojects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    x, y = data
    
    if origin is None:
        origin = (nx//2, ny//2)
    x = x - origin[0]
    y = y - origin[1]
#
#    # Determine that the min and max r and theta coords will be...
#    x, y = index_coords(data, origin=origin)
    
    r, theta = cart2polar(x, y)

    # Make a regular (in polar space) grid based on the min and max r & theta
    ny = int(np.round(r.max()-r.min()))
    nx = int(np.round(360*(1/0.02))) # resolution = 0.02 degree
    
    r_i = np.linspace(r.min(), r.max(), ny)
    theta_i = np.linspace(0, np.pi*2, nx) 
#    theta_i = np.linspace(theta.min(), theta.max(), nx)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    xi, yi = polar2cart(r_grid, theta_grid)
    xi += origin[0] # We need to shift the origin back to 
    yi += origin[1] # back to the lower-left corner...
    xi, yi = xi.flatten(), yi.flatten()
    coords = np.vstack((xi, yi)) # (map_coordinates requires a 2xn array)
    # Reproject each band individually and the restack
    # (uses less memory than reprojection the 3-dimensional array in one step)
#    bands = []
#    print("data.T shape = ", data.T.shape)
#    for band in data.T:
#        zi = sp.ndimage.map_coordinates(band, coords, order=1)
#        bands.append(zi.reshape((nx, ny)))
#    output = np.dstack(bands)
    transformed = sp.ndimage.map_coordinates(img, coords[::-1], order=1)
    output = transformed.reshape(ny,nx)
    return output, r_i, theta_i