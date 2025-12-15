#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import sys
from pathlib import Path



def ensure_dir_exists(directory):
    """确保目录存在，如果不存在则创建"""
    os.makedirs(directory, exist_ok=True)
    print(f"确保目录存在: {directory}")

def move_images(src_dir, dst_dir):
    """将源目录中的图片文件剪切到目标目录"""
    # 确保目标目录存在
    ensure_dir_exists(dst_dir)
    
    # 常见图片扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # 获取所有文件
    files = os.listdir(src_dir)
    moved_count = 0
    
    for file in files:
        src_path = os.path.join(src_dir, file)
        
        # 检查是否为文件且是图片文件
        if os.path.isfile(src_path) and any(file.lower().endswith(ext) for ext in image_extensions):
            dst_path = os.path.join(dst_dir, file)
            shutil.move(src_path, dst_path)
            moved_count += 1
    
    print(f"已从 {src_dir} 移动 {moved_count} 个图片文件到 {dst_dir}")
    return moved_count

def mk_euroc_dir(DATA_DIR,SRC_DIR_CAM0,SRC_DIR_CAM1):

    # 打印工作目录和数据目录
    print(f"当前工作目录: {os.getcwd()}")
    print(f"数据目录: {DATA_DIR}")
    
    # 源目录和目标目录
    src_dir_0 = os.path.join(DATA_DIR, SRC_DIR_CAM0)
    src_dir_1 = os.path.join(DATA_DIR, SRC_DIR_CAM1)
    
    dst_dir_cam0 = os.path.join(DATA_DIR, "mav0", "cam0", "data")
    dst_dir_cam1 = os.path.join(DATA_DIR, "mav0", "cam1", "data")
    
    moved_cam0 = 0
    moved_cam1 = 0
    
    # 检查源目录是否存在
    if not os.path.exists(src_dir_0):
        print(f"警告: 源目录不存在: {src_dir_0}")
        print(f"将自动创建目录: {src_dir_0}")
        os.makedirs(src_dir_0, exist_ok=True)
    else:
        # 创建目标目录（如果不存在）
        ensure_dir_exists(dst_dir_cam0)
        # 移动图片
        moved_cam0 = move_images(src_dir_0, dst_dir_cam0)
    
    if not os.path.exists(src_dir_1):
        print(f"警告: 源目录不存在: {src_dir_1}")
        print(f"将自动创建目录: {src_dir_1}")
        os.makedirs(src_dir_1, exist_ok=True)
    else:
        # 创建目标目录（如果不存在）
        ensure_dir_exists(dst_dir_cam1)
        # 移动图片
        moved_cam1 = move_images(src_dir_1, dst_dir_cam1)
    
    print(f"总共移动了 {moved_cam0 + moved_cam1} 个图片文件")
    if moved_cam0 > 0:
        print(f"- 从 {src_dir_0} 到 {dst_dir_cam0}: {moved_cam0} 个文件")
    if moved_cam1 > 0:
        print(f"- 从 {src_dir_1} 到 {dst_dir_cam1}: {moved_cam1} 个文件")
    
    return 0

def main():
    # 在这里直接设置您的数据目录路径
    DATA_DIR = r"E:\SLAM\DSOL-realsense\qqm_new\qqm_new_forest"  # 修改为您的实际路径

    # 源目录名称
    SRC_DIR_CAM0 = "0"  # 修改为cam0的源目录名称
    SRC_DIR_CAM1 = "3"  # 修改为cam1的源目录名称
    
    # 打印工作目录和数据目录
    print(f"当前工作目录: {os.getcwd()}")
    print(f"数据目录: {DATA_DIR}")
    
    # 源目录和目标目录
    src_dir_0 = os.path.join(DATA_DIR, SRC_DIR_CAM0)
    src_dir_1 = os.path.join(DATA_DIR, SRC_DIR_CAM1)
    
    dst_dir_cam0 = os.path.join(DATA_DIR, "mav0", "cam0", "data")
    dst_dir_cam1 = os.path.join(DATA_DIR, "mav0", "cam1", "data")
    
    moved_cam0 = 0
    moved_cam1 = 0
    
    # 检查源目录是否存在
    if not os.path.exists(src_dir_0):
        print(f"警告: 源目录不存在: {src_dir_0}")
        print(f"将自动创建目录: {src_dir_0}")
        os.makedirs(src_dir_0, exist_ok=True)
    else:
        # 创建目标目录（如果不存在）
        ensure_dir_exists(dst_dir_cam0)
        # 移动图片
        moved_cam0 = move_images(src_dir_0, dst_dir_cam0)
    
    if not os.path.exists(src_dir_1):
        print(f"警告: 源目录不存在: {src_dir_1}")
        print(f"将自动创建目录: {src_dir_1}")
        os.makedirs(src_dir_1, exist_ok=True)
    else:
        # 创建目标目录（如果不存在）
        ensure_dir_exists(dst_dir_cam1)
        # 移动图片
        moved_cam1 = move_images(src_dir_1, dst_dir_cam1)
    
    print(f"总共移动了 {moved_cam0 + moved_cam1} 个图片文件")
    if moved_cam0 > 0:
        print(f"- 从 {src_dir_0} 到 {dst_dir_cam0}: {moved_cam0} 个文件")
    if moved_cam1 > 0:
        print(f"- 从 {src_dir_1} 到 {dst_dir_cam1}: {moved_cam1} 个文件")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
