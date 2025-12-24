import MK_EuRoC_DIR
import MK_EuRoc_TimeStamps
import os

# 在这里直接设置您的数据目录路径
DATA_DIR = r"D:\testMK\rgb"  # 修改为您的实际路径
# 源目录名称
SRC_DIR_CAM0 = "0"  # 修改为cam0的源目录名称
SRC_DIR_CAM1 = "3"  # 修改为cam1的源目录名称


# 把下面的值改为你要统计的图片文件夹路径和输出文件路径。
img_count_path = os.path.join(DATA_DIR, SRC_DIR_CAM0) # 统计图片数量，用于生成timestamp.txt文件
timestamp_out_path = os.path.join(DATA_DIR, "TimeStamps.txt")  # 输出文件路径

MK_EuRoc_TimeStamps.mk_euroc_timestamps(img_count_path,timestamp_out_path)

MK_EuRoC_DIR.mk_euroc_dir(DATA_DIR,SRC_DIR_CAM0,SRC_DIR_CAM1)

