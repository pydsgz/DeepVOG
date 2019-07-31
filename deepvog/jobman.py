import os
import traceback
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
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
        self.model = load_DeepVOG()
        self.flen = float(flen)
        self.ori_video_shape = ori_video_shape
        self.sensor_size = sensor_size
        self.batch_size = batch_size

    def fit(self, vid_path, output_json_path, output_video_path="", heatmap=False,
            print_prefix=""):

        inferer = gaze_inferer(self.model, self.flen, self.ori_video_shape, self.sensor_size)
        inferer.process(video_src=vid_path, mode="Fit", batch_size=self.batch_size, output_video_path=output_video_path,
                        heatmap=heatmap, print_prefix=print_prefix)
        inferer.save_eyeball_model(output_json_path)

    def infer(self, vid_path, eyeball_model_path, output_record_path, output_video_path="", heatmap=False,
              infer_gaze_flag=True, print_prefix=""):
        if isinstance(infer_gaze_flag, str):
            try:
                infer_gaze_flag = int(infer_gaze_flag)
            except ValueError:
                infer_gaze_flag = False

        inferer = gaze_inferer(self.model, self.flen, self.ori_video_shape, self.sensor_size,
                               infer_gaze_flag=infer_gaze_flag)
        if infer_gaze_flag:
            inferer.load_eyeball_model(eyeball_model_path)
        inferer.process(video_src=vid_path, mode="Infer", batch_size=self.batch_size,
                        output_record_path=output_record_path, output_video_path=output_video_path, heatmap=heatmap,
                        print_prefix=print_prefix)


class deepvog_jobman_table_CLI(deepvog_jobman_CLI):
    def __init__(self, csv_path, gpu_num, flen, ori_video_shape, sensor_size, batch_size,
                 skip_errors=False, skip_existed=False, error_log_path=""):
        self.csv_dict = csv_reader(csv_path)
        self.skip_errors, self.skip_existed, self.error_log_path = skip_errors, skip_existed, error_log_path
        super(deepvog_jobman_table_CLI, self).__init__(gpu_num, flen, ori_video_shape, sensor_size, batch_size)
        self._initialize_error_log()

    def run_batch(self):
        num_operations = len(self.csv_dict['operation'])
        operation_counts = dict()
        for i in range(num_operations):
            current_operation = self.csv_dict['operation'][i]
            operation_counts[current_operation] = operation_counts.get(current_operation, 0) + 1

        print("Total number of operations = %d" % (num_operations))
        print("     - Fit    %d/%d " % (operation_counts.get("fit", 0), num_operations))
        print("     - Infer  %d/%d " % (operation_counts.get("infer", 0), num_operations))
        print("     - Both   %d/%d " % (operation_counts.get("both", 0), num_operations))
        for i in range(num_operations):
            try:
                current_operation = self.csv_dict['operation'][i]
                progress = '%d/%d ' % (i + 1, num_operations)

                # Skip file if the output already existed
                if self.skip_existed:
                    output_existed, concerned_path = self._check_if_output_existed(current_operation, i)
                    if output_existed:
                        print("Video %d skipped (operation %s),\nbecause %s is/are found" % (i+1,
                                                                                             current_operation,
                                                                                             str(concerned_path)))
                        continue

                # Actions for each operation type
                if current_operation == "fit":
                    self.fit(vid_path=self.csv_dict['fit_vid'][i],
                             output_json_path=self.csv_dict['eyeball_model'][i],
                             print_prefix=progress)
                elif current_operation == "infer":
                    self.infer(vid_path=self.csv_dict['infer_vid'][i],
                               eyeball_model_path=self.csv_dict['eyeball_model'][i],
                               output_record_path=self.csv_dict['result'][i],
                               infer_gaze_flag=self.csv_dict["with_gaze"][i],
                               print_prefix=progress)
                elif current_operation == "both":
                    self.fit(vid_path=self.csv_dict['fit_vid'][i],
                             output_json_path=self.csv_dict['eyeball_model'][i],
                             print_prefix=progress)
                    self.infer(vid_path=self.csv_dict['infer_vid'][i],
                               eyeball_model_path=self.csv_dict['eyeball_model'][i],
                               output_record_path=self.csv_dict['result'][i],
                               infer_gaze_flag=self.csv_dict["with_gaze"][i],
                               print_prefix=progress)
            except Exception:
                existed, paths = self._check_if_output_existed(current_operation, i)
                if existed:
                    for output_path in paths:
                        os.remove(output_path)
                if self.skip_errors:
                    try:
                        print("\nError encountered. Video {} skipped.".format(i + 1))
                        if self.error_log_path:
                            self._log_error(i)
                        continue
                    except Exception:
                        print("Error encountered when logging the error.")
                        traceback.print_exc()
                        continue
                else:
                    traceback.print_exc()
                    raise

    def _initialize_error_log(self):
        if self.error_log_path:
            # This function mainly serves to create and overwrite existing logs
            # (plus add some meta information for inspection)
            with open(self.error_log_path, "w") as fh:
                opening_msg = "DEEPVOG ERROR LOG\n"
                content = "Configurations:\nFLEN:{}\nOriginal video shape:{}\nSensor size:{}\nBatch size:{}\n".format(
                    self.flen,
                    self.ori_video_shape,
                    self.sensor_size,
                    self.batch_size
                )
                fh.write(opening_msg + content)

    def _log_error(self, i):

        with open(self.error_log_path, "a") as fh:
            msg = ["{}: {}(delimiter)".format(k, v[i]) for k, v in self.csv_dict.items()]
            msg_pretty = "".join(msg).replace("(delimiter)", "\n")
            line_separator = "\n" + "="*30 + "\n"
            fh.write(line_separator + msg_pretty)
            traceback.print_exc(file=fh)
        print("Error logged in {}".format(self.error_log_path))

    def _check_if_output_existed(self, current_operation, vid_idx):
        output_existed = False
        concerned_path = ""
        if current_operation == "fit":
            # For "fit" mode, skip if eyeball model already exists
            eyeball_model_path = self.csv_dict['eyeball_model'][vid_idx]
            output_existed = os.path.isfile(eyeball_model_path)
            concerned_path = (eyeball_model_path, )
        elif current_operation == "infer":
            # For "infer" mode, skip if gaze record already exists
            output_record_path = self.csv_dict['result'][vid_idx]
            output_existed = os.path.isfile(output_record_path)
            concerned_path = (output_record_path, )
        elif current_operation == "both":
            # For "both" mode, skip if both eyeball AND gaze record already exist
            eyeball_model_path = self.csv_dict['eyeball_model'][vid_idx]
            output_record_path = self.csv_dict['result'][vid_idx]
            eyeball_existed = os.path.isfile(eyeball_model_path)
            record_existed = os.path.isfile(output_record_path)
            if eyeball_existed and record_existed:
                output_existed = True
                concerned_path = (eyeball_model_path, output_record_path)
        return output_existed, concerned_path
