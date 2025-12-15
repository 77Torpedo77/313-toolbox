import os
import re


def rename_dataset(dataset_path):
    """
    Rename files in `dataset_path` so that numeric parts become 6-digit zero-padded.

    Rules:
    - If filename stem is purely digits (e.g. "1.png"), convert to "000001.png".
    - If filename ends with digits (e.g. "img_15.png"), pad the trailing digits to 6 digits: "000015.png".
    - Other filenames (no trailing digits) are left unchanged.

    The function performs a two-phase rename (to temporary names, then to final names)
    to avoid collisions when target names already exist in the same folder.
    """
    files = os.listdir(dataset_path)

    # 仅判断一次样式：若首个含数字的文件名为纯数字，则认为目录为纯数字样式；否则认为为带前缀样式
    mode = None  # 'pure' or 'strip_prefix'
    for filename in files:
        stem, _ = os.path.splitext(filename)
        if re.search(r"\d+", stem):
            mode = 'pure' if stem.isdigit() else 'strip_prefix'
            break
    if mode is None:
        return

    # 直接重命名，不做临时名或冲突检测（简化版）
    for filename in files:
        stem, ext = os.path.splitext(filename)
        if mode == 'pure':
            if not stem.isdigit():
                continue
            new_name = stem.zfill(6) + ext
        else:
            m = re.findall(r"(\d+)", stem)
            if not m:
                continue
            new_name = f"{int(m[-1]):06d}" + ext

        if new_name == filename:
            continue
        os.rename(os.path.join(dataset_path, filename), os.path.join(dataset_path, new_name))
        print(f"Renamed: {filename} -> {new_name}")

def main():
    #用于将group0-x中的0和x重命名为infra1和infra2，并将其中的图片重命名为6位数格式，直接修改路径运行即可，每次只会修改路径指向的那一个文件夹。
    """
    Main function to rename dataset subfolders and create a calibration file.
    This script processes a dataset directory by renaming specific subfolders 
    and generating a calibration file. The dataset directory is expected to 
    contain subfolders named numerically (e.g., "0", "1", "2", etc.). The 
    folder "0" is renamed to "infra1", and the first other numeric folder 
    (e.g., "1") is renamed to "infra2". After renaming, the function calls 
    `rename_dataset` on both renamed folders and creates a calibration file 
    named `calib.txt` in the dataset directory.
    Steps:
    1. Rename the "0" folder to "infra1".
    2. Rename the first numeric folder (other than "0") to "infra2".
    3. Call `rename_dataset` on both "infra1" and "infra2".
    4. Create a calibration file `calib.txt` with predefined content.
    Note:
        - Ensure the `dataset_path` variable is set to the correct dataset 
          directory path before running the script.
        - The script assumes the dataset directory exists and contains the 
          expected folder structure.
    Raises:
        FileNotFoundError: If the specified dataset directory does not exist.
    Outputs:
        - Renamed subfolders in the dataset directory.
        - A calibration file `calib.txt` in the dataset directory.
    """
    # 直接在此处设置您的路径
    base_dir = "/home/ps/work_space/DSOL-realsense/manual_V100_E3"
    intrinsics = "700 600 350 300 0.5"
    group_folders = [f for f in os.listdir(base_dir) if f.startswith("group0-")]
    for group_folder in group_folders:
        dataset_path = os.path.join(base_dir, group_folder)
        print(f"Processing: {dataset_path}")
        left_path = dataset_path + "/0"
        right_path = None
        for subfolder in os.listdir(dataset_path):
            if subfolder.isdigit() and subfolder != "0":
                right_path = os.path.join(dataset_path, subfolder)
                break
        if not os.path.exists(dataset_path):
            print(f"Error: Directory {dataset_path} does not exist")
            continue

        os.rename(left_path, dataset_path + "/infra1")
        os.rename(right_path, dataset_path + "/infra2")

        infra1_path = dataset_path + "/infra1"
        infra2_path = dataset_path + "/infra2"

        rename_dataset(infra1_path)
        rename_dataset(infra2_path)

        calib_file_path = os.path.join(dataset_path, "calib.txt")
        with open(calib_file_path, "w", encoding="utf-8") as calib_file:
            calib_file.write(intrinsics)
        print(f"Created calibration file at: {calib_file_path}")
    # 处理完所有group0-x文件夹后退出
    return

def dataset_rename_function(path,intrinsics):
    # 直接在此处设置您的路径
    base_dir = path
    group_folders = [f for f in os.listdir(base_dir) if f.startswith("group0-")]
    for group_folder in group_folders:
        dataset_path = os.path.join(base_dir, group_folder)
        print(f"Processing: {dataset_path}")
        left_path = dataset_path + "/0"
        right_path = None
        for subfolder in os.listdir(dataset_path):
            if subfolder.isdigit() and subfolder != "0":
                right_path = os.path.join(dataset_path, subfolder)
                break
        if not os.path.exists(dataset_path):
            print(f"Error: Directory {dataset_path} does not exist")
            continue

        os.rename(left_path, dataset_path + "/infra1")
        os.rename(right_path, dataset_path + "/infra2")

        infra1_path = dataset_path + "/infra1"
        infra2_path = dataset_path + "/infra2"

        rename_dataset(infra1_path)
        rename_dataset(infra2_path)

        calib_file_path = os.path.join(dataset_path, "calib.txt")
        with open(calib_file_path, "w", encoding="utf-8") as calib_file:
            calib_file.write(intrinsics)
        print(f"Created calibration file at: {calib_file_path}")
    # 处理完所有group0-x文件夹后退出
    return

if __name__ == "__main__":
    main()