import build_folder, dataset_rename, EvaluationForMCSSim, GTtime2index
import os

path = r"Y:\PYL\work_space\DSOL-realsense\MCS-dataset\forest-manual\rgb"
#fx fy cx cy baseline
# intrinsics = "640 640 319.5 239.5 0.07"
intrinsics = "700 700 350 300 0.05"

build_folder.build_folder_function(path)
dataset_rename.dataset_rename_function(path, intrinsics)

parent_path = os.path.abspath(os.path.join(path, os.pardir))
EvaluationForMCSSim.EvaluationForMCSSim_function(parent_path)


# base_path = "/home/ps/work_space/DSOL-realsense"
# output_path = "/home/ps/work_space/dsol_ws/src/dsol-main/save"

# time2index.time2index_function(base_path,output_path)