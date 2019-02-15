import os
from .model.DeepVOG_model import load_DeepVOG
from .inferer import gaze_inferer
from ast import literal_eval
from .utils import csv_reader

class deepvog_jobman_CLI(object):
    def __init__(self, gpu_num, flen, ori_video_shape, sensor_size, batch_size):
        """
        
        Args:
            gpu_num (str)
            flen (float)
            ori_video_shape (tuple)
            sensor_size (tuple)
            batch_size (int)
        
        """
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num
        self.model = load_DeepVOG()
        self.flen = flen
        self.ori_video_shape = ori_video_shape
        self.sensor_size = sensor_size
        self.batch_size = batch_size
        
    def fit(self, vid_path, output_json_path, print_prefix=""):
        inferer = gaze_inferer(self.model, self.flen, self.ori_video_shape, self.sensor_size)
        inferer.fit(vid_path, batch_size = self.batch_size, print_prefix=print_prefix)
        inferer.save_eyeball_model(output_json_path) 

    def infer(self, eyeball_model_path, video_scr, record_path, print_prefix=""):
        inferer = gaze_inferer(self.model, self.flen, self.ori_video_shape, self.sensor_size)
        inferer.load_eyeball_model(eyeball_model_path)
        inferer.predict( video_scr, record_path, batch_size=self.batch_size, print_prefix=print_prefix)

class deepvog_jobman_table_CLI(deepvog_jobman_CLI):
    def __init__(self, csv_path, gpu_num, flen, ori_video_shape, sensor_size, batch_size):
        self.csv_dict = csv_reader(csv_path)
        super(deepvog_jobman_table_CLI, self).__init__( gpu_num, flen, ori_video_shape, sensor_size, batch_size)
    def run_batch(self):
        num_operations = len(self.csv_dict['operation'])
        operation_counts = dict()
        for i in range(num_operations):
            current_operation = self.csv_dict['operation'][i]
            operation_counts[current_operation] = operation_counts.get(current_operation, 0) + 1
        
        
        print("Total number of operations = %d"% (num_operations))
        print("     - Fit    %d/%d " % (operation_counts.get("fit", 0), num_operations))
        print("     - Infer  %d/%d " % (operation_counts.get("infer", 0), num_operations))
        print("     - Both   %d/%d " % (operation_counts.get("both", 0), num_operations))
        for i in range(num_operations):
            current_operation = self.csv_dict['operation'][i]
            progress = '%d/%d ' % (i+1, num_operations)
            if current_operation == "fit":
                self.fit(self.csv_dict['fit_vid'][i], self.csv_dict['eyeball_model'][i], print_prefix = progress)
            elif current_operation == "infer":
                self.infer(self.csv_dict['eyeball_model'][i], self.csv_dict['infer_vid'][i], self.csv_dict['result'][i], print_prefix = progress)
            elif current_operation == "both":
                self.fit(self.csv_dict['fit_vid'][i], self.csv_dict['eyeball_model'][i], print_prefix = progress)
                self.infer(self.csv_dict['eyeball_model'][i], self.csv_dict['infer_vid'][i], self.csv_dict['result'][i], print_prefix = progress)
class deepvog_jobman_TUI(deepvog_jobman_CLI):
    def __init__(self, gpu_num, flen, ori_video_shape, sensor_size, batch_size):
        """
        Arguments are parsed from TUI. Therefore, all of them are in type (str). Compared to CLI, additional conversion is required.
        Also, infer() method deals with filenames automatically as you won't specify it in TUI
        
        """
        
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_num)
        self.model = load_DeepVOG()
        self.flen = float(flen)
        self.ori_video_shape = literal_eval(ori_video_shape)
        self.sensor_size = literal_eval(sensor_size)
        self.batch_size = int(batch_size)
    def infer(self, eyeball_model_path, video_scr, record_dir, print_prefix=""):
        video_name_root = os.path.splitext(os.path.split(video_scr)[1])[0]
        eyeball_model_name_root = os.path.splitext(os.path.split(eyeball_model_path)[1])[0]
        record_name = "fit-{}_infer-{}.csv".format(eyeball_model_name_root, video_name_root)
        record_path = os.path.join(record_dir, record_name)
        inferer = gaze_inferer(self.model, self.flen, self.ori_video_shape, self.sensor_size)
        inferer.load_eyeball_model(eyeball_model_path)
        inferer.predict( video_scr, record_path, batch_size=self.batch_size, print_prefix=print_prefix)
