import os
import shutil


def time2index_function(base_path,output_path):

    base_path = base_path
    output_path1 = output_path

    # 确保输出目录存在
    os.makedirs(output_path1, exist_ok=True)

    # 遍历 base_path 下的所有子文件夹
    for folder_name in os.listdir(base_path):
        input_path = os.path.join(base_path, folder_name)
        if os.path.isdir(input_path):  # 确保是文件夹
            input_file = os.path.join(input_path, "Ground_truth.txt")
            if os.path.exists(input_file):  # 确保 Ground_truth.txt 存在
                output_file = os.path.join(output_path1, "GT_" + folder_name + ".txt")

                with open(input_file, "r") as infile, open(output_file, "w") as outfile:
                    for row_number, line in enumerate(infile, start=1):
                        columns = line.strip().split()
                        if len(columns) == 8:  # 确保文件有 8 列
                            columns[0] = str(row_number)  # 用行号替换第一列
                            outfile.write(" ".join(columns) + "\n")
                        else:
                            print(f"Skipping line with unexpected column count: {line.strip()}")

                # 提示已生成 GT 文件
                print(f"GT 文件已生成: {output_file}")

                # 复制输出文件到对应的子文件夹
                dest_in_folder = os.path.join(input_path, "GT_" + folder_name + ".txt")
                shutil.copy(output_file, dest_in_folder)
                # 提示已复制回原子文件夹
                print(f"已复制 GT 文件到: {dest_in_folder}")
            else:
                print(f"Ground_truth.txt not found in {input_path}")

if __name__ == "__main__":
    time2index_function("/home/ps/work_space/DSOL-realsense","/home/ps/work_space/dsol_ws/src/dsol-main/save")