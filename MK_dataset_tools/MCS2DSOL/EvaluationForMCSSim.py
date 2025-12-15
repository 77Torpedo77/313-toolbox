import numpy as np
from transforms3d.quaternions import mat2quat

import tkinter as tk
from tkinter import filedialog

import pandas as pd

from scipy.spatial.transform import Rotation

# Methods = ['ORB-SLAM2','Stereo_DSO','R3D3']
Methods = []

def se3_to_tum(se3, timestamp):
    # 从旋转矩阵提取四元数
    quat = mat2quat(se3[:3, :3])
    # 提取平移和四元数
    tx, ty, tz = se3[:3, 3]
    qw, qx, qy, qz = quat
    # 返回TUM格式的字符串
    return f"{timestamp} {tx} {ty} {tz} {qw} {qx} {qy} {qz}"

def tum_to_se3(pos, quat):
    # 从四元数创建旋转矩阵
    rot = Rotation.from_quat(quat).as_matrix()
    # 创建SE3矩阵 (4x4矩阵)
    se3_matrix = np.eye(4)
    se3_matrix[:3, :3] = rot
    se3_matrix[:3, 3] = pos
    return se3_matrix

def EvaluationForMCSSim_function(path):
    # 替换文件夹选择为手动指定路径
    folder_selected = path
    print("使用的文件夹：", folder_selected)
    # 读取多相机系统参数文件
    with open(folder_selected + '/MCS_parameter.csv', 'r',encoding="GB2312") as file:
        MCS = []
        lines = file.readlines()
        MCS_type = int(lines[0].strip('\n').split(',')[1])  # 多相机系统类型
        subcamera_num = len(lines) - 2  # 子相机数量
        for i in range(subcamera_num):
            focal_length = float(lines[i + 2].strip('\n').split(',')[1])
            film_width = float(lines[i + 2].strip('\n').split(',')[2])
            resolution = np.array(lines[i + 2].strip('\n').split(',')[3:5], dtype=int)
            focal_y = focal_x = focal_length / (film_width / resolution[0])
            principal_point_x = (resolution[0] - 1) / 2
            principal_point_y = (resolution[1] - 1) / 2
            axis_angle_subcamera_to_MCS = np.array(lines[i + 2].strip('\n').split(',')[5:8], dtype=float)
            extrinsic_matrix = np.eye(4)
            extrinsic_matrix[:3, :3] = Rotation.from_rotvec(axis_angle_subcamera_to_MCS).as_matrix()
            extrinsic_matrix[:3, -1] = np.array(lines[i + 2].strip('\n').split(',')[8:11], dtype=float) / 1000
            subcamera = {'resolution': resolution,
                         "intrinsic_matrix": np.array(
                             [[focal_x, 0.0, principal_point_x], [0.0, focal_y, principal_point_y], [0, 0, 1]]),
                         "extrinsic_matrix": extrinsic_matrix}
            MCS.append(subcamera)
    with open(folder_selected + '/attitude.csv', 'r',encoding="GB2312") as file:
        steps = []
        lines = file.readlines()
        steps_num = len(lines) - 2
        time_stamps = np.zeros(steps_num)
        for i in range(steps_num):
            time_stamp = float(lines[i + 2].strip('\n').split(',')[1])
            euler_platform_yaw = float(lines[i + 2].strip('\n').split(',')[7])
            euler_platform_pitch = float(lines[i + 2].strip('\n').split(',')[5])
            euler_platform_roll = float(lines[i + 2].strip('\n').split(',')[6])
            rotation_platform1 = Rotation.from_euler('ZXY', [-euler_platform_roll, -euler_platform_pitch, -euler_platform_yaw],
                                              degrees=True).as_matrix()
            euler_platform_to_MCS_yaw = float(lines[i + 2].strip('\n').split(',')[8])
            euler_platform_to_MCS_pitch = float(lines[i + 2].strip('\n').split(',')[9])
            euler_platform_to_MCS_roll = float(lines[i + 2].strip('\n').split(',')[10])
            rotation_platform2 = Rotation.from_euler('ZXY', [-euler_platform_to_MCS_roll, -euler_platform_to_MCS_pitch,
                                                      -euler_platform_to_MCS_yaw], degrees=True).as_matrix()
            rotation_platform = np.matmul(rotation_platform2, rotation_platform1)
            ENU = np.array(lines[i + 2].strip('\n').split(',')[11:14], dtype=float)
            offset_platform = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).dot(ENU)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_platform.transpose()
            transform_matrix[:3, -1] = offset_platform
            steps.append({'time_stamp': time_stamp, 'transform_matrix': transform_matrix})
            time_stamps[i] = time_stamp

    if 'ORB-SLAM2' in Methods:
        data = np.genfromtxt(folder_selected + '/ORB-SLAM2_trajectory.txt', delimiter=' ', missing_values='NaN')
        time_stamps_ORB2 = data[:,0]
        steps_ORB2 = []
        for i in range(len(time_stamps_ORB2)):
            # 解析TUM数据
            pos = np.array(data[i,1:4])
            quat = np.array(data[i,4:])
            steps_ORB2.append({'time_stamp': time_stamps_ORB2[i], 'transform_matrix': tum_to_se3(pos, quat)})

    if 'Stereo_DSO' in Methods:
        data = np.genfromtxt(folder_selected + '/Stereo_DSO_trajectory.txt', delimiter=' ', missing_values='NaN')
        time_stamps_SDSO = time_stamps[data[:,0].astype(int)]
        steps_SDSO = []
        for i in range(len(time_stamps_SDSO)):
            se3 = np.eye(4)
            se3[0, :] = data[i, 2:6]
            se3[1, :] = data[i, 6:10]
            se3[2, :] = data[i, 10:14]
            steps_SDSO.append({'time_stamp': time_stamps_SDSO[i], 'transform_matrix': se3})

    if 'R3D3' in Methods:
        with open(folder_selected + '/R3D3_trajectory.csv', 'r') as file:
            steps_R3D3 = []
            lines = file.readlines()
            steps_num = len(lines) - 1
            time_stamps_R3D3 = np.zeros(steps_num)
            for i in range(steps_num):
                temp = lines[i + 1].strip('\n').split(',')[0]
                time_stamps_R3D3[i] = time_stamps[int(temp.rsplit('_', 1)[-1].lstrip('_'))]
                pos = np.array(lines[i + 1].strip('\n').split(',')[1:4])
                quat = np.array(lines[i + 1].strip('\n').split(',')[4:])
                steps_R3D3.append({'time_stamp': time_stamps_R3D3[i], 'transform_matrix': tum_to_se3(pos, quat)})

    left_camera = 0  # 设置左相机的序号

    init_transform_matrix = steps[0].get('transform_matrix')
    pose_origin = np.matmul(init_transform_matrix, MCS[left_camera].get('extrinsic_matrix'))
    # 输出轨迹真值到txt文件（TUM格式）
    with open(folder_selected + '/Ground_truth.txt', "w") as file:
        for step in range(len(steps)):
            true_pose = np.matmul(steps[step].get('transform_matrix'), MCS[left_camera].get('extrinsic_matrix'))
            pose = np.matmul(np.linalg.inv(pose_origin), true_pose)
            file.write(se3_to_tum(pose, steps[step].get("time_stamp")) + "\n")
    print('Ground truth trajectory is OK!')

    if 'ORB-SLAM2' in Methods:
        init_frame = np.where(np.isclose(time_stamps, time_stamps_ORB2[0], atol=1e-4))[0][0]
        pose_origin = np.matmul(np.linalg.inv(np.matmul(steps[0].get('transform_matrix'), MCS[left_camera].get('extrinsic_matrix'))),
                                np.matmul(steps[init_frame].get('transform_matrix'), MCS[left_camera].get('extrinsic_matrix')))
        # 输出轨迹真值到txt文件（TUM格式）
        with open(folder_selected + '/ORB-SLAM2.txt', "w") as file:
            for step in range(len(steps_ORB2)):
                true_pose = steps_ORB2[step].get('transform_matrix')
                pose = np.matmul(pose_origin, true_pose)
                file.write(se3_to_tum(pose, steps_ORB2[step].get("time_stamp")) + "\n")
        print('ORB-SLAM2 trajectory is OK!')

    if 'Stereo_DSO' in Methods:
        init_frame = np.where(np.isclose(time_stamps, time_stamps_SDSO[0], atol=1e-4))[0][0]
        pose_origin = np.matmul(np.linalg.inv(np.matmul(steps[0].get('transform_matrix'), MCS[left_camera].get('extrinsic_matrix'))),
                                np.matmul(steps[init_frame].get('transform_matrix'), MCS[left_camera].get('extrinsic_matrix')))
        # 输出轨迹真值到txt文件（TUM格式）
        with open(folder_selected + '/Stereo_DSO.txt', "w") as file:
            for step in range(len(steps_SDSO)):
                true_pose = steps_SDSO[step].get('transform_matrix')
                pose = np.matmul(pose_origin, true_pose)
                file.write(se3_to_tum(pose, steps_SDSO[step].get("time_stamp")) + "\n")
        print('steps_SDSO trajectory is OK!')

    if 'R3D3' in Methods:
        # 输出轨迹真值到txt文件（TUM格式）
        with open(folder_selected + '/R3D3.txt', "w") as file:
            for step in range(len(steps_R3D3)):
                true_pose = steps_R3D3[step].get('transform_matrix')
                pose = np.matmul(np.linalg.inv(MCS[left_camera].get('extrinsic_matrix')), true_pose)
                file.write(se3_to_tum(pose, steps_R3D3[step].get("time_stamp")) + "\n")
        print('steps_R3D3 trajectory is OK!')

if __name__ == '__main__':
    EvaluationForMCSSim_function("/home/ps/work_space/DSOL-realsense/manual_V100_E3")