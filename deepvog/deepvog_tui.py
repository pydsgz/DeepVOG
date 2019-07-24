import urwid
import numpy
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tell tensorflow to shut up!
import sys
import json
from .model.DeepVOG_model import load_DeepVOG
from .inferer import gaze_inferer
from .utils import save_json, load_json
from .tui_description import instruction_listwalker, aboutus_listwalker
from .jobman import deepvog_jobman_TUI as deepvog_jobman
from glob import glob
from ast import literal_eval

import pdb
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings


class Button_centered(urwid.Button):
    def __init__(self, label, on_press=None, user_data=None):
        super(Button_centered, self).__init__(label, on_press=None, user_data=None)
        self._label.align = 'center'


class EditionStorer(object):
    def __init__(self):
        self.Editors = dict()

    def set_editor(self, key, label, align, attr_map=None, focus_map=None):
        self.Editors[key] = urwid.AttrMap(urwid.Edit(label, align=align), attr_map, focus_map)

    def get_editor(self, key):
        return self.Editors[key]


class CheckboxStorer(object):
    def __init__(self):
        self.boxes = dict()

    def set_box(self, key, label, attr_map=None, focus_map=None, state=False):
        self.boxes[key] = self.boxes.get(key, urwid.AttrMap(urwid.CheckBox(label, state=state), attr_map, focus_map))

    def get_box(self, key):
        return self.boxes[key]


class deepvog_tui(object):
    def __init__(self, base_dir):
        # Parameters and directories of a session
        self.base_dir = base_dir
        self.flen = str(6)
        self.ori_video_shape = str((240, 320))
        self.sensor_size = str((3.6, 4.8))
        self.GPU_number = str(0)
        self.batch_size = str(512)
        self.fitting_dir = ""
        self.eyeballmodel_dir = ""
        self.inference_dir = ""
        self.results_dir = ""

        # Fitting/Inference parameters
        self.selected_fitting_vids = dict()
        self.execution_code = "exit"
        self.inference_dict = dict()

        # urwid setting
        self.main_interface_title = "DeepVOG"
        self.main_menu_choices = ["Set parameters and directories",
                                  "Fit 3D eyeball models",
                                  "Gaze inference",
                                  "Instruction",
                                  "About us",
                                  "Exit"]
        self.palette = [('reversed', 'standout', '')]
        self.main_widget = None
        self.main_loop = None
        self.fitting_checkboxes = CheckboxStorer()
        self.model_checkboxes = CheckboxStorer()
        self.inference_checkboxes_dict = dict()

        self.editors = EditionStorer()
        self.editors.set_editor("flen", "", "right", focus_map="reversed")
        # self.editors.get_editor("flen").original_widget.edit_text = self.flen

        self.editors.set_editor("ori_video_shape", "", "right", focus_map="reversed")
        # self.editors.get_editor("ori_video_shape").original_widget.edit_text = self.ori_video_shape

        self.editors.set_editor("sensor_size", "", "right", focus_map="reversed")
        # self.editors.get_editor("sensor_size").original_widget.edit_text= self.sensor_size

        self.editors.set_editor("GPU", "", "right", focus_map="reversed")
        # self.editors.get_editor("GPU").original_widget.edit_text = str(0)

        self.editors.set_editor("batch_size", "", "right", focus_map="reversed")
        # self.editors.get_editor("batch_size").original_widget.edit_text = str(512)

        self.editors.set_editor("fitting_dir", "", "right", focus_map="reversed")
        self.editors.set_editor("eyeballmodel_dir", "", "right", focus_map="reversed")
        self.editors.set_editor("inference_dir", "", "right", focus_map="reversed")
        self.editors.set_editor("results_dir", "", "right", focus_map="reversed")
        self.update_edit_from_params()

    # Main menu
    def _main_menu(self, title, choices):
        body = [urwid.Text(title, align="center"), urwid.Divider(), urwid.Text("Main menu", align="center"),
                urwid.Divider()]

        button_dict = dict()

        for choice in choices:
            button_dict[choice] = urwid.Button(choice)
            body.append(urwid.AttrMap(button_dict[choice], None, focus_map='reversed'))
        urwid.connect_signal(button_dict["Exit"], 'click', self.exit_program)
        urwid.connect_signal(button_dict["Set parameters and directories"], 'click',
                             self.onClick_set_parameters_from_main)
        urwid.connect_signal(button_dict["Fit 3D eyeball models"], 'click', self.onClick_fit_models_from_main)
        urwid.connect_signal(button_dict["Gaze inference"], 'click', self.onClick_gaze_inference_options_from_main)
        urwid.connect_signal(button_dict["Instruction"], 'click', self.onClick_instructions)
        urwid.connect_signal(button_dict["About us"], 'click', self.onClick_aboutus)
        return urwid.ListBox(urwid.SimpleFocusListWalker(body))

    # "Set parameters" page
    def onClick_set_parameters_from_main(self, button):
        # Title and divider
        title_set_params = urwid.Text("Set parameters and directories\nYour base directory is {}".format(self.base_dir),
                                      align="center")
        div = urwid.Divider()

        # Ask and answer
        ask_flen = urwid.Text('Focal length of the camera in mm:\n', align="left")
        answer_flen = self.editors.get_editor("flen")

        ask_ori_video_shape = urwid.Text('Original video shape (height,width) in pixel:\n', align="left")
        answer_ori_video_shape = self.editors.get_editor("ori_video_shape")

        ask_sensor_size = urwid.Text('Sensor size (height,width) in mm:\n', align="left")
        answer_sensor_size = self.editors.get_editor("sensor_size")

        ask_GPU = urwid.Text('GPU number:\n', align="left")
        answer_GPU = self.editors.get_editor("GPU")

        ask_batch = urwid.Text('Batch size:\n', align="left")
        answer_batch = self.editors.get_editor("batch_size")

        ask_fitting_dir = urwid.Text('Directory of videos for model fitting:\n', align="left")
        answer_fitting_dir = self.editors.get_editor("fitting_dir")

        ask_eyeballmodel_dir = urwid.Text('Directory of fitted 3D eyeball models:\n', align="left")
        answer_eyeballmodel_dir = self.editors.get_editor("eyeballmodel_dir")

        ask_inference_dir = urwid.Text('Directory of videos for gaze inference:\n', align="left")
        answer_inference_dir = self.editors.get_editor("inference_dir")

        ask_results_dir = urwid.Text('Directory of output results:\n', align="left")
        answer_results_dir = self.editors.get_editor("results_dir")

        # Buttons for save/back
        save_button = Button_centered("Save workspace parameters")
        urwid.connect_signal(save_button, 'click', self.onClick_save_params)

        load_button = Button_centered("Load workspace parameters")
        urwid.connect_signal(load_button, 'click', self.onClick_load_params)

        back_button = Button_centered("Back")
        urwid.connect_signal(back_button, 'click', self.onClick_back_from_params)

        # Constructing piles and columns
        col_flen = urwid.Columns([ask_flen, answer_flen])
        col_ori_video_shape = urwid.Columns([ask_ori_video_shape, answer_ori_video_shape])
        col_sensor_size = urwid.Columns([ask_sensor_size, answer_sensor_size])
        col_GPU = urwid.Columns([ask_GPU, answer_GPU])
        col_batch = urwid.Columns([ask_batch, answer_batch])
        col_fitting_dir = urwid.Columns([ask_fitting_dir, answer_fitting_dir])
        col_eyeballmodel_dir = urwid.Columns([ask_eyeballmodel_dir, answer_eyeballmodel_dir])
        col_inference_dir = urwid.Columns([ask_inference_dir, answer_inference_dir])
        col_results_dir = urwid.Columns([ask_results_dir, answer_results_dir])

        all_piles = urwid.Pile(
            [col_flen, col_ori_video_shape, col_sensor_size, col_GPU, col_batch, col_fitting_dir, col_eyeballmodel_dir,
             col_inference_dir, col_results_dir])

        whole_fill = urwid.Filler(urwid.Pile([title_set_params,
                                              div,
                                              all_piles,
                                              urwid.AttrMap(save_button, None, 'reversed'),
                                              urwid.AttrMap(load_button, None, 'reversed'),
                                              urwid.AttrMap(back_button, None, 'reversed')]))

        # Set the interface
        self.main_widget.original_widget = whole_fill

    # Fitting models page (select files for fitting)
    def onClick_fit_models_from_main(self, button):
        title_fit_models = urwid.Text("Fit 3D eyeball models\nVideos from: {}".format(self.get_fitting_dir()),
                                      align="center")
        div = urwid.Divider()
        vids_paths, vids_names = self.grab_paths_and_names(self.get_fitting_dir())
        fitting_list_body = [title_fit_models, div]

        # select all checkboxes
        self.fitting_checkboxes.set_box("select all", "select all", focus_map='reversed')
        urwid.connect_signal(self.fitting_checkboxes.get_box("select all").original_widget, 'change',
                             self.onChange_fitting_selectall, (vids_names, vids_paths))
        fitting_list_body.append(self.fitting_checkboxes.get_box("select all"))

        # Start button
        start_button = Button_centered("Start fitting")
        urwid.connect_signal(start_button, 'click', self.onClick_start_from_fitting)

        # Back button
        back_button = Button_centered("Back and save")
        urwid.connect_signal(back_button, 'click', self.onClick_back_to_main)

        # Check box for all videos
        for vid_path, vid_name in zip(vids_paths, vids_names):
            self.fitting_checkboxes.set_box(vid_name, vid_name, focus_map='reversed')
            urwid.connect_signal(self.fitting_checkboxes.get_box(vid_name).original_widget, 'change',
                                 self.onChange_fitting_checkbox, (vid_name, vid_path))
            fitting_list_body.append(self.fitting_checkboxes.get_box(vid_name))

        fitting_list_body.append(urwid.AttrMap(start_button, None, 'reversed'))
        fitting_list_body.append(urwid.AttrMap(back_button, None, 'reversed'))

        fitting_list_walker = urwid.ListBox(urwid.SimpleFocusListWalker(fitting_list_body))
        self.main_widget.original_widget = fitting_list_walker

        return fitting_list_walker

    # Page: Options for gaze inference
    def onClick_gaze_inference_options_from_main(self, button):
        list_body = []

        # Tittle
        title_inference_options = urwid.Text("Gaze inference options\nVideos from: {}".format(self.get_inference_dir()),
                                             align="center")
        div = urwid.Divider()
        list_body.append(title_inference_options)
        list_body.append(div)

        # Buttons
        match_with_name_button = urwid.Button("Infer all videos that have the same file names with the models")
        urwid.connect_signal(match_with_name_button, 'click', self.onClick_inferALL_from_gazeInference_options)
        specify_models_button = urwid.Button("Specify videos to infer for each eyeball model")
        urwid.connect_signal(specify_models_button, 'click', self.onClick_inferSpecific_from_gazeInference_options)
        list_body.append(urwid.AttrMap(match_with_name_button, None, 'reversed'))
        list_body.append(div)
        list_body.append(urwid.AttrMap(specify_models_button, None, 'reversed'))

        # Back button
        back_button = Button_centered("Back")
        urwid.connect_signal(back_button, 'click', self.onClick_back_to_main)
        list_body.append(urwid.AttrMap(back_button, None, 'reversed'))

        inference_options_list_walker = urwid.ListBox(urwid.SimpleFocusListWalker(list_body))
        self.main_widget.original_widget = inference_options_list_walker

    # Clicking "back and save" in "Set up parameters" page
    def onClick_back_from_params(self, button):
        # Save parameters as attributes
        self.update_params_from_edit()
        self.main_widget.original_widget = self._main_menu(self.main_interface_title, self.main_menu_choices)

    # Page: Confirmation for fitting 3D eyeballs
    def onClick_start_from_fitting(self, button):
        title = urwid.Text(
            "(Confirm before you start)\nEyeball models will be stored in {}".format(self.get_models_dir()),
            align="center")
        div = urwid.Divider()
        confirmation_page_list = [title, div]
        for vid_name in self.selected_fitting_vids.keys():
            confirmation_page_list.append(urwid.Text(self.selected_fitting_vids[vid_name], align='left'))
        # Start button
        start_button = Button_centered("Confirm")
        urwid.connect_signal(start_button, 'click', self.onClick_start_fitting)

        # Back button
        back_button = Button_centered("Back")
        urwid.connect_signal(back_button, 'click', self.onClick_fit_models_from_main)

        confirmation_page_list.append(urwid.AttrMap(start_button, None, 'reversed'))
        confirmation_page_list.append(urwid.AttrMap(back_button, None, 'reversed'))
        confirmation_page_list_walker = urwid.ListBox(urwid.SimpleFocusListWalker(confirmation_page_list))
        self.main_widget.original_widget = confirmation_page_list_walker

    # Page: Confirmation for "Infer all" option
    def onClick_inferALL_from_gazeInference_options(self, button):
        list_body = []

        # Title
        title = urwid.Text("(Confirm before you start)", align="center")
        div = urwid.Divider()
        list_body.append(title)
        list_body.append(div)

        # Title for each column
        model_column_title = urwid.Text("Models:", align="center")
        infervid_column_title = urwid.Text("Videos to infer:", align="center")

        column_titles = urwid.Columns([model_column_title, infervid_column_title], 1)
        list_body.append(column_titles)
        list_body.append(div)
        # Model_videos pairs
        models_paths, models_names = self.grab_paths_and_names(self.get_models_dir(), ".json")
        infer_vids_paths, infer_vids_names = self.grab_paths_and_names(self.get_inference_dir())
        # model_vid_pairs_dict = dict()
        self.update_all_models_and_infervids((models_paths, models_names), (infer_vids_paths, infer_vids_names),
                                             match_names=True)

        for model_idx, key in enumerate(self.inference_dict.keys()):
            button_model = urwid.Button(str(model_idx) + ". " + key)
            text_appender = ""
            for infer_vid_path in self.inference_dict[key]["infer_vids_paths"]:
                infer_vid_name = os.path.split(infer_vid_path)[1]
                text_appender += " - " + infer_vid_name + "\n"
            text_infer_vid = urwid.Text(text_appender)
            model_vid_pair = urwid.Columns([button_model, text_infer_vid], 1)
            list_body.append(model_vid_pair)
            # model_vid_pairs_dict[key] = model_vid_pair

        # Start button
        start_button = Button_centered("Confirm")
        urwid.connect_signal(start_button, 'click', self.onClick_start_inferall)

        # Back button
        back_button = Button_centered("Back")
        urwid.connect_signal(back_button, 'click', self.onClick_gaze_inference_options_from_main)

        # List appending
        # for key in model_vid_pairs_dict.keys():
        #     list_body.append(model_vid_pairs_dict[key])

        list_body.append(urwid.AttrMap(start_button, None, 'reversed'))
        list_body.append(urwid.AttrMap(back_button, None, 'reversed'))

        list_walker = urwid.ListBox(urwid.SimpleFocusListWalker(list_body))
        self.main_widget.original_widget = list_walker

    # Page: Model selection
    def onClick_inferSpecific_from_gazeInference_options(self, button):

        models_paths, models_names = self.grab_paths_and_names(self.get_models_dir(), ".json")
        infer_vids_paths, infer_vids_names = self.grab_paths_and_names(self.get_inference_dir())
        # Title and list body
        title = urwid.Text("Model Selection\n Models from: {}".format(self.get_models_dir()), align="center")
        div = urwid.Divider()
        list_body = [title, div]

        # Checkbox "Select all" for models
        self.model_checkboxes.set_box("select all", "select all", focus_map='reversed')
        urwid.connect_signal(self.model_checkboxes.get_box("select all").original_widget, 'change',
                             self.onChange_model_selectall)
        list_body.append(self.model_checkboxes.get_box("select all"))

        # Checkboxes for all models and inference videos (not shown)
        for model_path, model_name in zip(models_paths, models_names):
            self.model_checkboxes.set_box(model_name, model_name, focus_map="reversed")
            urwid.connect_signal(self.model_checkboxes.get_box(model_name).original_widget, 'change',
                                 self.onChange_select_model_specific, model_name)
            list_body.append(self.model_checkboxes.get_box(model_name))
            self.inference_checkboxes_dict[model_name] = CheckboxStorer()
            self.inference_checkboxes_dict[model_name].set_box("select all", "select all", focus_map="reversed")
            urwid.connect_signal(self.inference_checkboxes_dict[model_name].get_box("select all").original_widget,
                                 'change', self.onChange_model_select_all_vids, model_name)
            for infer_vid_name in infer_vids_names:
                self.inference_checkboxes_dict[model_name].set_box(infer_vid_name, infer_vid_name, focus_map="reversed")
                urwid.connect_signal(self.inference_checkboxes_dict[model_name].get_box(infer_vid_name).original_widget,
                                     'change', self.onChange_model_select_vid, (model_name, infer_vid_name))

        # Update all the checkboxes according to self.inference_dict()
        self.update_model_checkboxes()
        # self.update_infer_vids_checkboxes()
        # Start button
        start_button = Button_centered("Start")
        urwid.connect_signal(start_button, 'click', self.onClick_start_from_inferSpecific)
        list_body.append(urwid.AttrMap(start_button, None, 'reversed'))

        # Back button
        back_button = Button_centered("Back")
        urwid.connect_signal(back_button, 'click', self.onClick_gaze_inference_options_from_main)
        list_body.append(urwid.AttrMap(back_button, None, 'reversed'))

        list_walker = urwid.ListBox(urwid.SimpleFocusListWalker(list_body))
        self.main_widget.original_widget = list_walker

    def onClick_start_from_inferSpecific(self, button):
        list_body = []

        # Title
        title = urwid.Text("(Confirm before you start)", align="center")
        div = urwid.Divider()
        list_body.append(title)
        list_body.append(div)

        # Title for each column
        model_column_title = urwid.Text("Models:", align="center")
        infervid_column_title = urwid.Text("Videos to infer:", align="center")

        column_titles = urwid.Columns([model_column_title, infervid_column_title], 1)
        list_body.append(column_titles)
        list_body.append(div)
        # Model_videos pairs
        model_vid_pairs_dict = dict()
        for model_idx, key in enumerate(self.inference_dict.keys()):
            button_model = urwid.Button(str(model_idx) + ". " + key)
            text_appender = ""
            for infer_vid_path in self.inference_dict[key]["infer_vids_paths"]:
                infer_vid_name = os.path.split(infer_vid_path)[1]
                text_appender += " - " + infer_vid_name + "\n"
            text_infer_vid = urwid.Text(text_appender)
            model_vid_pair = urwid.Columns([button_model, text_infer_vid], 1)
            model_vid_pairs_dict[key] = model_vid_pair

        # Start button
        start_button = Button_centered("Confirm")
        urwid.connect_signal(start_button, 'click', self.onClick_start_inferall)

        # Back button
        back_button = Button_centered("Back")
        urwid.connect_signal(back_button, 'click', self.onClick_inferSpecific_from_gazeInference_options)

        # List appending
        for key in model_vid_pairs_dict.keys():
            list_body.append(model_vid_pairs_dict[key])

        list_body.append(urwid.AttrMap(start_button, None, 'reversed'))
        list_body.append(urwid.AttrMap(back_button, None, 'reversed'))

        list_walker = urwid.ListBox(urwid.SimpleFocusListWalker(list_body))
        self.main_widget.original_widget = list_walker

    def onClick_back_to_main(self, button):
        self.main_widget.original_widget = self._main_menu(self.main_interface_title, self.main_menu_choices)

    def onChange_fitting_selectall(self, check_box, new_state, vid_data):
        (vids_names, vids_paths) = vid_data
        if new_state == True:
            for key in self.fitting_checkboxes.boxes.keys():
                self.fitting_checkboxes.get_box(key).original_widget.set_state(True, do_callback=False)
            for vid_name, vid_path in zip(vids_names, vids_paths):
                self.selected_fitting_vids[vid_name] = vid_path
        else:
            for key in self.fitting_checkboxes.boxes.keys():
                self.fitting_checkboxes.get_box(key).original_widget.set_state(False, do_callback=False)
            for vid_name, vid_path in zip(vids_names, vids_paths):
                if self.selected_fitting_vids.get(vid_name) is not None:
                    self.selected_fitting_vids.pop(vid_name)

    def onChange_fitting_checkbox(self, check_box, new_state, vid_data):
        (vid_name, vid_path) = vid_data
        if new_state == True:
            self.selected_fitting_vids[vid_name] = vid_path
        if new_state == False:
            if self.selected_fitting_vids.get(vid_name) is not None:
                self.selected_fitting_vids.pop(vid_name)

    def onChange_model_selectall(self, check_box, new_state):
        models_paths, models_names = self.grab_paths_and_names(self.get_models_dir(), ".json")
        infer_vids_paths, infer_vids_names = self.grab_paths_and_names(self.get_inference_dir())
        if new_state == True:
            self.update_all_models_and_infervids((models_paths, models_names), (infer_vids_paths, infer_vids_names))
            self.update_model_checkboxes()
        if new_state == False:
            self.inference_dict = dict()
            self.update_model_checkboxes()

    def onChange_select_model_specific(self, check_box, new_state, model_name):
        models_paths, models_names = self.grab_paths_and_names(self.get_models_dir(), ".json")
        infer_vids_paths, infer_vids_names = self.grab_paths_and_names(self.get_inference_dir())
        list_body = []

        # Title and divider
        title = urwid.Text("Select videos for {}".format(model_name), align="center")
        div = urwid.Divider()
        list_body.append(title)
        list_body.append(div)

        # Checkboxes
        list_body.append(self.inference_checkboxes_dict[model_name].get_box("select all"))
        for box_key in self.inference_checkboxes_dict[model_name].boxes.keys():
            if box_key == "select all":
                continue
            list_body.append(self.inference_checkboxes_dict[model_name].get_box(box_key))

        # Back button
        back_button = Button_centered("Back")
        urwid.connect_signal(back_button, 'click', self.onClick_inferSpecific_from_gazeInference_options)
        list_body.append(urwid.AttrMap(back_button, None, 'reversed'))

        # Update checkboxes
        self.update_infer_vids_checkboxes()

        # List walkier
        list_walker = urwid.ListBox(urwid.SimpleFocusListWalker(list_body))
        self.main_widget.original_widget = list_walker

    def onChange_model_select_all_vids(self, check_box, new_state, model_name):
        if new_state == True:
            for infer_vid_name in self.inference_checkboxes_dict[model_name].boxes.keys():
                if infer_vid_name == "select all":
                    continue
                self.inference_checkboxes_dict[model_name].get_box(infer_vid_name).original_widget.set_state(True,
                                                                                                             do_callback=True)
        if new_state == False:
            for infer_vid_name in self.inference_checkboxes_dict[model_name].boxes.keys():
                if infer_vid_name == "select all":
                    continue
                self.inference_checkboxes_dict[model_name].get_box(infer_vid_name).original_widget.set_state(False,
                                                                                                             do_callback=True)

    def onChange_model_select_vid(self, check_box, new_state, model_vid_info):
        (model_name, infer_vid_name) = model_vid_info
        model_path = os.path.join(self.get_models_dir(), model_name)
        infer_vid_path = os.path.join(self.get_inference_dir(), infer_vid_name)
        self.inference_dict[model_name] = self.inference_dict.get(model_name,
                                                                  {"model_path": model_path, "infer_vids_paths": []})

        if new_state == True:
            if infer_vid_path not in self.inference_dict[model_name]["infer_vids_paths"]:
                self.inference_dict[model_name]["infer_vids_paths"].append(infer_vid_path)
        else:
            if infer_vid_path in self.inference_dict[model_name]["infer_vids_paths"]:
                popping_idx = self.inference_dict[model_name]["infer_vids_paths"].index(infer_vid_path)
                self.inference_dict[model_name]["infer_vids_paths"].pop(popping_idx)
                if len(self.inference_dict[model_name]["infer_vids_paths"]) == 0:
                    self.inference_dict.pop(model_name)

    def execution_outsideTUI(self):

        if self.execution_code == "exit":
            print("\nProgram exited.")
        if self.execution_code == "fit":
            print("\nFitting starts")
            self.deepVOG_fitting()
            input("Press 'Enter' to go back to main interface")
            self.run_tui()
        if self.execution_code == "infer":
            print("\nInference starts")
            self.deepVOG_inference()
            self.inference_dict = dict()
            input("Press 'Enter' to go back to main interface")
            self.run_tui()
        if self.execution_code == "debug":
            pdb.set_trace()

    def deepVOG_fitting(self):
        jobman = deepvog_jobman(self.GPU_number, self.flen, self.ori_video_shape, self.sensor_size, self.batch_size)
        number_all_vids = len(self.selected_fitting_vids)
        for vid_idx, vid_name in enumerate(self.selected_fitting_vids.keys()):
            vid_path = self.selected_fitting_vids[vid_name]
            vid_name_root = os.path.splitext(os.path.split(vid_path)[1])[0]
            output_json_path = os.path.join(self.base_dir, self.eyeballmodel_dir, vid_name_root + ".json")
            jobman.fit(vid_path=vid_path, output_json_path=output_json_path,
                       print_prefix="- {}/{} of all videos: ".format(vid_idx + 1, number_all_vids))

    def deepVOG_inference(self):
        number_all_vids = 0
        for key in self.inference_dict.keys():
            number_all_vids += len(self.inference_dict[key]["infer_vids_paths"])
        jobman = deepvog_jobman(self.GPU_number, self.flen, self.ori_video_shape, self.sensor_size, self.batch_size)
        i = 0
        for model_idx, key in enumerate(self.inference_dict.keys()):
            eyemodel_path = self.inference_dict[key]["model_path"]
            print("Using eyeball model ({}/{}): {}".format(model_idx + 1, len(self.inference_dict), eyemodel_path))
            for infer_vid_path in self.inference_dict[key]["infer_vids_paths"]:
                jobman.infer(eyemodel_path, infer_vid_path, self.get_results_dir(),
                             print_prefix="- {}/{} of all videos: ".format(i + 1, number_all_vids))
                i += 1

    def run_tui(self):
        self.main_widget = urwid.Padding(self._main_menu(self.main_interface_title, self.main_menu_choices), left=2,
                                         right=2)
        self.main_loop = urwid.Overlay(self.main_widget, urwid.SolidFill(u"\N{MEDIUM SHADE}"),
                                       align='center', width=('relative', 60), valign='middle', height=('relative', 60),
                                       min_width=20, min_height=9)

        self.loop_process = urwid.MainLoop(self.main_loop, self.palette)
        self.loop_process.run()

        self.execution_outsideTUI()

    def update_all_models_and_infervids(self, models_info, infer_vids_indo, match_names=False):
        (models_paths, models_names) = models_info
        (infer_vids_paths, infer_vids_names) = infer_vids_indo
        self.inference_dict = dict()
        i = 0
        for model_path, model_name in zip(models_paths, models_names):
            self.inference_dict[model_name] = {"model_path": model_path,
                                               "infer_vids_paths": []}
            for infer_vid_path, infer_vid_name in zip(infer_vids_paths, infer_vids_names):
                if match_names == True:
                    if (os.path.splitext(model_name)[0] == os.path.splitext(infer_vid_name)[0]):
                        if infer_vid_path not in self.inference_dict[model_name]["infer_vids_paths"]:
                            self.inference_dict[model_name]["infer_vids_paths"].append(infer_vid_path)
                else:
                    if infer_vid_path not in self.inference_dict[model_name]["infer_vids_paths"]:
                        self.inference_dict[model_name]["infer_vids_paths"].append(infer_vid_path)

    # Update the checkboxes for model in "inferSpecific" page. Called every time when the page is entered
    def update_model_checkboxes(self):
        # If the dictionary is emtpy, jsut set all checkboxes as zero
        if len(self.inference_dict.keys()) == 0:
            for current_key in self.model_checkboxes.boxes.keys():
                self.model_checkboxes.get_box(current_key).original_widget.set_state(False, do_callback=False)
        else:
            for current_key in self.model_checkboxes.boxes.keys():  # "select all included"
                if current_key == "select all":
                    num_selected_model = 0
                    num_all_models = len(self.model_checkboxes.boxes.keys()) - 1  # number of total model

                    for selected_model_key in self.inference_dict.keys():
                        if len(self.inference_dict[selected_model_key]["infer_vids_paths"]) == (
                                len(self.inference_checkboxes_dict[selected_model_key].boxes.keys()) - 1):
                            num_selected_model += 1
                    if num_selected_model == num_all_models:
                        self.model_checkboxes.get_box(current_key).original_widget.set_state(True, do_callback=False)
                    else:
                        self.model_checkboxes.get_box(current_key).original_widget.set_state(False, do_callback=False)

                else:
                    if current_key not in self.inference_dict.keys():
                        self.model_checkboxes.get_box(current_key).original_widget.set_state(False, do_callback=False)
                    elif len(self.inference_dict[current_key]["infer_vids_paths"]) > 0:
                        self.model_checkboxes.get_box(current_key).original_widget.set_state(True, do_callback=False)
                    else:
                        self.model_checkboxes.get_box(current_key).original_widget.set_state(False, do_callback=False)

    def update_infer_vids_checkboxes(self):
        # If the dictionary is emtpy, jsut set all checkboxes as zero
        if len(self.inference_dict.keys()) == 0:
            for current_model_key in self.inference_checkboxes_dict.keys():
                for current_infer_vid_key in self.inference_checkboxes_dict[current_model_key].boxes.keys():
                    self.inference_checkboxes_dict[current_model_key].get_box(
                        current_infer_vid_key).original_widget.set_state(False, do_callback=False)

        else:
            for current_model_key in self.inference_checkboxes_dict.keys():

                if current_model_key == "select all":
                    continue
                for infer_vid_name_layer1 in self.inference_checkboxes_dict[current_model_key].boxes.keys():
                    # Check if the checkbox "select all" should be checked or not
                    if infer_vid_name_layer1 == "select all":
                        all_same = True
                        for infer_vid_name_layer2 in self.inference_checkboxes_dict[current_model_key].boxes.keys():
                            if infer_vid_name_layer2 == "select all":
                                continue
                            else:
                                if current_model_key not in self.inference_dict.keys():
                                    all_same = False
                                elif os.path.join(self.get_inference_dir(), infer_vid_name_layer2) not in \
                                        self.inference_dict[current_model_key]["infer_vids_paths"]:
                                    all_same = False
                        if all_same == True:
                            self.inference_checkboxes_dict[current_model_key].get_box(
                                "select all").original_widget.set_state(True, do_callback=False)
                        else:
                            self.inference_checkboxes_dict[current_model_key].get_box(
                                "select all").original_widget.set_state(False, do_callback=False)
                    # For all other checkbox of videos
                    else:
                        if current_model_key not in self.inference_dict.keys():
                            self.inference_checkboxes_dict[current_model_key].get_box(
                                infer_vid_name_layer1).original_widget.set_state(False, do_callback=False)
                        elif os.path.join(self.get_inference_dir(), infer_vid_name_layer1) in \
                                self.inference_dict[current_model_key]["infer_vids_paths"]:
                            self.inference_checkboxes_dict[current_model_key].get_box(
                                infer_vid_name_layer1).original_widget.set_state(True, do_callback=False)
                        else:
                            self.inference_checkboxes_dict[current_model_key].get_box(
                                infer_vid_name_layer1).original_widget.set_state(False, do_callback=False)

    def update_params_from_edit(self):
        self.flen = self.editors.get_editor("flen").original_widget.edit_text
        self.ori_video_shape = self.editors.get_editor("ori_video_shape").original_widget.edit_text
        self.sensor_size = self.editors.get_editor("sensor_size").original_widget.edit_text
        self.GPU_number = self.editors.get_editor("GPU").original_widget.edit_text
        self.batch_size = self.editors.get_editor("batch_size").original_widget.edit_text
        self.fitting_dir = self.editors.get_editor("fitting_dir").original_widget.edit_text
        self.eyeballmodel_dir = self.editors.get_editor("eyeballmodel_dir").original_widget.edit_text
        self.inference_dir = self.editors.get_editor("inference_dir").original_widget.edit_text
        self.results_dir = self.editors.get_editor("results_dir").original_widget.edit_text

    def update_edit_from_params(self):
        self.editors.get_editor("flen").original_widget.edit_text = self.flen
        self.editors.get_editor("ori_video_shape").original_widget.edit_text = self.ori_video_shape
        self.editors.get_editor("sensor_size").original_widget.edit_text = self.sensor_size
        self.editors.get_editor("GPU").original_widget.edit_text = self.GPU_number
        self.editors.get_editor("batch_size").original_widget.edit_text = self.batch_size
        self.editors.get_editor("fitting_dir").original_widget.edit_text = self.fitting_dir
        self.editors.get_editor("eyeballmodel_dir").original_widget.edit_text = self.eyeballmodel_dir
        self.editors.get_editor("inference_dir").original_widget.edit_text = self.inference_dir
        self.editors.get_editor("results_dir").original_widget.edit_text = self.results_dir

    # Get paths/directories
    def get_fitting_dir(self):
        return os.path.join(self.base_dir, self.fitting_dir)

    def get_models_dir(self):
        return os.path.join(self.base_dir, self.eyeballmodel_dir)

    def get_inference_dir(self):
        return os.path.join(self.base_dir, self.inference_dir)

    def get_results_dir(self):
        return os.path.join(self.base_dir, self.results_dir)

    def onClick_save_params(self, button):
        self.update_params_from_edit()
        params_dict = {"focal length": self.flen,
                       "original video shape": self.ori_video_shape,
                       "sensor size": self.sensor_size,
                       "GPU number": self.GPU_number,
                       "batch size": self.batch_size,
                       "fitting dir": self.fitting_dir,
                       "eyeball model dir": self.eyeballmodel_dir,
                       "inference dir": self.inference_dir,
                       "results dir": self.results_dir}
        save_path = os.path.join(self.base_dir, "config.json")
        save_json(save_path, params_dict)
        self.onClick_back_from_params(0)

    def onClick_load_params(self, button):
        try:
            load_path = os.path.join(self.base_dir, "config.json")
            params_dict = load_json(load_path)
            self.flen = params_dict["focal length"]
            self.ori_video_shape = params_dict["original video shape"]
            self.sensor_size = params_dict["sensor size"]
            self.GPU_number = params_dict["GPU number"]
            self.batch_size = params_dict["batch size"]
            self.fitting_dir = params_dict["fitting dir"]
            self.eyeballmodel_dir = params_dict["eyeball model dir"]
            self.inference_dir = params_dict["inference dir"]
            self.results_dir = params_dict["results dir"]
            self.update_edit_from_params()
        except FileNotFoundError:
            pass

    def onClick_instructions(self, button):
        list_walker = instruction_listwalker(Button_centered, self.onClick_back_to_main)
        self.main_widget.original_widget = list_walker

    def onClick_aboutus(self, button):
        list_walker = aboutus_listwalker(Button_centered, self.onClick_back_to_main)
        self.main_widget.original_widget = list_walker

    # After confirmation, start fitting outside of tui.
    def onClick_start_fitting(self, button):
        self.execution_code = "fit"
        raise urwid.ExitMainLoop()

    # After confirmation, start infer all videos outside of tui
    def onClick_start_inferall(self, button):
        self.execution_code = "infer"
        raise urwid.ExitMainLoop()

    def exit_program(self, button):
        self.execution_code = "exit"
        raise urwid.ExitMainLoop

    @staticmethod
    def grab_paths_and_names(target_dir, extensions=[""]):
        """
        Please add dot to the extension, for example, extensions = [".mp4", ".avi", ".pgm"]
        """
        # Grap all paths
        all_paths = []
        for extension in extensions:
            paths = glob(os.path.join(target_dir, "*" + extension))
            all_paths += paths
        all_paths = sorted(all_paths)

        all_names = list(map(lambda x: os.path.split(x)[1], all_paths))
        return all_paths, all_names


if __name__ == "__main__":
    if len(sys.argv) > 1:
        tui = deepvog_tui(str(sys.argv[1]))
        tui.run_tui()
