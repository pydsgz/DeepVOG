import numpy as np
import json


def convert_vec2angle31(n1):
    """
    Inputs:
        n1 = numpy array with shape (3,1)
    """
    assert n1.shape == (3,1)
    n1 = n1/np.linalg.norm(n1)
    n1_x, n1_y, n1_z_abs = n1[0,0], n1[1,0], np.abs(n1[2,0])
    # x-augulation            
    if n1_x > 0:
        x_angle = np.arctan(n1_z_abs/n1_x)
    else:
        x_angle = np.pi - np.arctan(n1_z_abs/np.abs(n1_x))
    # y-angulation
    if n1_y > 0:
        y_angle = np.arctan(n1_z_abs/n1_y)
    else:
        y_angle = np.pi - np.arctan(n1_z_abs/np.abs(n1_y))
    x_angle = np.rad2deg(x_angle)
    y_angle = np.rad2deg(y_angle)
    return [x_angle, y_angle]

def save_json(path, save_dict):
    json_str = json.dumps(save_dict, indent=4)
    with open(path, "w") as fh:
        fh.write(json_str)
        
def load_json(path):
    with open(path, "r+") as fh:
        json_str = fh.read()
    return json.loads(json_str)

if __name__ == "__main__":
    pass
