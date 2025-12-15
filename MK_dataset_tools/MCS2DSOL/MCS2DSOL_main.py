import build_folder, dataset_rename, EvaluationForMCSSim, time2index

path = "/home/ps/work_space/DSOL-realsense/qqm_new_forest"
#fx fy cx cy baseline
intrinsics = "640 640 319.5 239.5 0.07"

build_folder.build_folder_function(path)
dataset_rename.dataset_rename_function(path, intrinsics)
EvaluationForMCSSim.EvaluationForMCSSim_function(path)


# base_path = "/home/ps/work_space/DSOL-realsense"
# output_path = "/home/ps/work_space/dsol_ws/src/dsol-main/save"

# time2index.time2index_function(base_path,output_path)