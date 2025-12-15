"""
生成 TimeStamps.txt 的小工具。

定义两个变量：
- `img_count_path`：要统计图片数量的目录（Path 对象或字符串）。
- `timestamp_out_path`：输出 TimeStamps.txt 的路径（Path 对象或字符串）。

输出文件每行一个数字，从 1 开始，直到图片数量（包含）。

也支持命令行参数：
	python MK_EuRoc_TimeStamps.py --img-dir /path/to/images --out TimeStamps.txt

默认只统计常见图片扩展名：jpg/jpeg/png/bmp/tif/tiff
"""

from pathlib import Path
import argparse
from typing import Iterable

# ---- 用户可修改的默认路径 ----
# 把下面的值改为你要统计的图片文件夹路径和输出文件路径。
img_count_path = Path(r"E:\SLAM\DSOL-realsense\qqm_new_dso_Euroc\qqm_new_beach\mav0\cam0\data")  # 默认当前目录，修改为你的图片目录，例如 Path(r"D:\dataset\images")
timestamp_out_path = Path(r"E:\SLAM\DSOL-realsense\qqm_new_dso_Euroc\qqm_new_beach\TimeStamps.txt")  # 默认输出到当前工作目录


def is_image_file(p: Path, exts: Iterable[str] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")) -> bool:
	return p.is_file() and p.suffix.lower() in exts


def count_images_in_dir(dirpath: Path) -> int:
	if not dirpath.exists():
		raise FileNotFoundError(f"图片目录不存在: {dirpath}")
	if not dirpath.is_dir():
		raise NotADirectoryError(f"提供的路径不是目录: {dirpath}")
	# 递归或非递归？此处只统计顶层文件，如需递归请修改为 rglob
	files = [p for p in dirpath.iterdir() if is_image_file(p)]
	return len(files)


def write_timestamps(out_path: Path, count: int) -> None:
	out_path = out_path if isinstance(out_path, Path) else Path(out_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	# 使用 newline='\n' 强制 Unix 风格换行，避免在 Windows 上被转换为 CRLF
	with out_path.open("w", encoding="utf-8", newline="\n") as f:
		for i in range(0, count):
			f.write(f"{i}\n")


def mk_euroc_timestamps(img_count_path,timestamp_out_path):
	# ---- 用户可修改的默认路径 ----
	# 把下面的值改为你要统计的图片文件夹路径和输出文件路径。
	img_count_path = Path(img_count_path)  # 默认当前目录，修改为你的图片目录，例如 Path(r"D:\dataset\images")
	timestamp_out_path = Path(timestamp_out_path)  # 默认输出到当前工作目录

	img_dir = img_count_path
	out_file = timestamp_out_path

	try:
		count = count_images_in_dir(img_dir)
	except Exception as e:
		print(f"错误：{e}")
		return 1

	write_timestamps(out_file, count)
	print(f"已写入 {out_file}，共 {count} 条时间戳（从1到{count}）。")
	return 0


def main():
	# ---- 用户可修改的默认路径 ----
	# 把下面的值改为你要统计的图片文件夹路径和输出文件路径。
	img_count_path = Path(r"E:\SLAM\DSOL-realsense\qqm_new_dso_Euroc\qqm_new_beach\mav0\cam0\data")  # 默认当前目录，修改为你的图片目录，例如 Path(r"D:\dataset\images")
	timestamp_out_path = Path(r"E:\SLAM\DSOL-realsense\qqm_new_dso_Euroc\qqm_new_beach\TimeStamps.txt")  # 默认输出到当前工作目录

	img_dir = img_count_path
	out_file = timestamp_out_path

	try:
		count = count_images_in_dir(img_dir)
	except Exception as e:
		print(f"错误：{e}")
		return 1

	write_timestamps(out_file, count)
	print(f"已写入 {out_file}，共 {count} 条时间戳（从1到{count}）。")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

