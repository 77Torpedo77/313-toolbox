#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
import glob

def delete_pfm_files(root_path):
	"""
	删除root_path及其所有子文件夹下的.pfm文件
	"""
	for dirpath, dirnames, filenames in os.walk(root_path):
		for filename in filenames:
			if filename.lower().endswith('.pfm'):
				file_path = os.path.join(dirpath, filename)
				try:
					os.remove(file_path)
					print(f"Deleted: {file_path}")
				except Exception as e:
					print(f"Failed to delete {file_path}: {e}")

def get_numeric_folders(root_path):
	"""
	获取root_path下所有以数字命名的子文件夹，返回排序后的列表
	"""
	numeric_folders = []
	for name in os.listdir(root_path):
		folder_path = os.path.join(root_path, name)
		if os.path.isdir(folder_path) and name.isdigit():
			numeric_folders.append(name)
	numeric_folders.sort(key=lambda x: int(x))
	return numeric_folders

def create_group_folders(root_path, numeric_folders):
	"""
	创建group0-x文件夹，并将0和x文件夹内容复制进去
	"""
	if '0' not in numeric_folders:
		print("No folder named '0' found. Abort group creation.")
		return
	for name in numeric_folders:
		if name == '0':
			continue
		group_name = f"group0-{name}"
		group_path = os.path.join(root_path, group_name)
		if os.path.exists(group_path):
			shutil.rmtree(group_path)
		os.makedirs(group_path)
		# 复制0文件夹内容
		src0 = os.path.join(root_path, '0')
		dst0 = os.path.join(group_path, '0')
		shutil.copytree(src0, dst0)
		# 复制x文件夹内容
		srcx = os.path.join(root_path, name)
		dstx = os.path.join(group_path, name)
		shutil.copytree(srcx, dstx)
		print(f"Created {group_name} with 0 and {name}")

def main():
	# 直接在此处填写目标根目录路径
	root_path = "/home/ps/work_space/DSOL-realsense/U_L100R50_V10_sea"  # TODO: 修改为你的目标路径

	if not os.path.isdir(root_path):
		print(f"{root_path} 不是有效的文件夹路径")
		return

	# print("删除所有.pfm文件...")
	# delete_pfm_files(root_path)

	print("查找数字命名的子文件夹...")
	numeric_folders = get_numeric_folders(root_path)
	print(f"找到: {numeric_folders}")

	print("创建group0-x文件夹...")
	create_group_folders(root_path, numeric_folders)

	print("完成！")

def build_folder_function(path):
	# 直接在此处填写目标根目录路径
	root_path = path

	if not os.path.isdir(root_path):
		print(f"{root_path} 不是有效的文件夹路径")
		return

	# print("删除所有.pfm文件...")
	# delete_pfm_files(root_path)

	print("查找数字命名的子文件夹...")
	numeric_folders = get_numeric_folders(root_path)
	print(f"找到: {numeric_folders}")

	print("创建group0-x文件夹...")
	create_group_folders(root_path, numeric_folders)

	print("完成！")

if __name__ == "__main__":
	main()
