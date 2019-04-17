from glob import glob
import os
import numpy as np
import json

def readSingleCoords_original(path, name):
    """
    Inputs: 
        path = path of the data
        name = root name of the .txt file (*_iris_points.txt and *_pupil_points.txt)
    Ouputs:
        Tuple object (a,b):
            a = numpy array of iris boundary points, with shape (number_of_samples, dimension=2). Dimension = (x,y)
            b = numpy array of pupil boundary points. Shape is the same as a
    """
    
    
    iris_fh = open(os.path.join(path, name+"_iris_points.txt"))
    pupil_fh = open(os.path.join(path, name+"_pupil_points.txt"))
    
    iris_points_list = []
    pupil_points_list = []
    for iris_line in iris_fh:
        if len(iris_line) > 0:
            iris_xy = iris_line.split(":")
            iris_xy = tuple(map(lambda x: float(x.strip()), iris_xy))
            iris_points_list.append(iris_xy)
    for pupil_line in pupil_fh:
        if len(pupil_line) > 0:
            pupil_xy = pupil_line.split(":")
            pupil_xy = tuple(map(lambda x: float(x.strip()), pupil_xy))
            pupil_points_list.append(pupil_xy)
    
    iris_fh.close()
    pupil_fh.close()
    return (np.array(iris_points_list), np.array(pupil_points_list))

def readSingleCoords_json(path):
    with open(path, "r+") as fh:
        dict_str = fh.read()
    return json.loads(dict_str)
    
def readTorsionalAngle_fromMax(path):
    data_dict = readSingleCoords_json(path)
    data_name = np.array(data_dict["DataNames"])
    data_data = np.array(data_dict["Data"])
    
    torsional_index = np.where(data_name == 'LeftEyeMarkerTorsion ')[0][0]
    time_index = np.where(data_name == 'Time                 ')[0][0]
    torsional_angle = np.array(data_data[:, torsional_index])
    data_time = np.array(data_data[:, time_index])
    
    return data_dict, data_data, data_name, data_time, torsional_angle
    
    
if __name__ == "__main__":
    pass