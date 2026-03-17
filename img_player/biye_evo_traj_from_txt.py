"""
自动化 evo 轨迹评估脚本
========================
功能：
  1. 自动对多条轨迹执行 evo_ape / evo_rpe 评估，将结果保存为 zip
  2. 自动使用 evo_res 汇总展示所有评估结果（含对比图表）
  3. 支持自定义图例名称（rename_dict）
  4. 支持在 APE 和 RPE 之间切换

使用方法：
  修改下方【用户配置区】中的参数，然后直接运行本脚本即可。
"""

import subprocess
import os
import sys
import shutil
from evo.tools import file_interface

# 自动检测 evo 可执行文件所在的 Scripts 目录
_SCRIPTS_DIR = os.path.join(os.path.dirname(sys.executable), "")

def _evo_exe(name):
    """获取 evo 可执行文件的完整路径"""
    exe = os.path.join(os.path.dirname(sys.executable), name + ".exe")
    if os.path.exists(exe):
        return exe
    # fallback: 直接使用命令名（依赖 PATH）
    return name

# ============================================================
# 【用户配置区】 - 在此修改评估参数
# ============================================================

# ---- 评估指标 ----
# "ape" = 绝对位姿误差, "rpe" = 相对位姿误差
METRIC = "ape"

# ---- RPE 专用参数（仅 METRIC="rpe" 时生效）----
DELTA = 1                # 相对位姿的间隔
DELTA_UNIT = "f"         # 间隔单位: "f"=frames, "d"=degrees, "r"=radians, "m"=meters

# ---- 目标测试目录 ----
# 将在此目录下查找轨迹文件、真值文件，并在该目录下生成 temp 文件夹
# TARGET_DIR = r"D:\Py-Project\myevo\biyelunwen\chap03\kitti09"
TARGET_DIR = r"D:\Py-Project\myevo\biyelunwen\chap04\forest-manul"
# ---- 轨迹格式 ----
# 支持 "tum", "kitti", "euroc" 等 evo 支持的格式
TRAJ_FORMAT = "tum"

# ---- 真值文件 ----
# 真值文件需位于 TARGET_DIR 目录下
GT_FILE = "Ground_truth.txt"

# ---- 待评估的轨迹文件列表 ----
# 如果列表为空 []，则自动扫描 TARGET_DIR 下所有 .txt 文件（排除 GT_FILE）
# 如果列表非空，则只评估列表中指定的文件
TRAJ_FILES = []

# ---- 图例重命名字典 ----
# key: 轨迹文件名, value: 显示在图例中的名称
# 如果某个文件不在字典中，则使用去掉 .txt 后缀的文件名
RENAME_DICT = {
    "DSOL.txt":         "DSOL",
    "Ours.txt":         "Ours",
    "ORB-SLAM3.txt":    "ORB-SLAM3",
    "VINS-Fusion.txt":  "VINS-Fusion",
}

# ---- 评估选项 ----
POSE_RELATION = "trans_part"   # 位姿关系: "trans_part", "angle_deg", "angle_rad", "full" 等
ALIGN = True                   # 是否启用 Umeyama 对齐 (-a)
ALIGN_AND_SCALE = False        # 是否启用对齐+缩放 (-as)，启用时 ALIGN 将被忽略
PLOT_MODE = "xz"               # 绘图模式: "xy", "xz", "yz", "xyz"

# ---- 图表保存 ----
SAVE_PLOT = True               # 是否将图表保存为 PDF
PLOT_FILENAME = "plot.pdf"     # 图表保存的文件名（将存在 TARGET_DIR 目录下）
SAVE_ERROR_RESULT = True       # 是否将 evo_res 误差统计保存为 error_result.md
ERROR_RESULT_FILENAME = "error_result.md"  # 误差结果文件名（与 PLOT_FILENAME 同目录）

# ---- 临时目录 ----
TEMP_DIR = "temp"              # 在 TARGET_DIR 下生成，存放中间 zip 文件的目录名单

# ============================================================
# 【执行逻辑】 - 通常不需要修改以下内容
# ============================================================

def get_legend_name(filename):
    """获取轨迹的图例名称"""
    if filename in RENAME_DICT:
        return RENAME_DICT[filename]
    return os.path.splitext(filename)[0]


def build_eval_cmd(gt_path, traj_path, save_path):
    """构建 evo_ape 或 evo_rpe 命令"""
    if METRIC == "ape":
        cmd = [_evo_exe("evo_ape"), TRAJ_FORMAT, gt_path, traj_path]
    elif METRIC == "rpe":
        cmd = [_evo_exe("evo_rpe"), TRAJ_FORMAT, gt_path, traj_path,
               "-d", str(DELTA), "-u", DELTA_UNIT]
    else:
        raise ValueError(f"不支持的 METRIC: {METRIC}，请使用 'ape' 或 'rpe'")

    cmd.extend(["-r", POSE_RELATION])

    if ALIGN_AND_SCALE:
        cmd.append("-as")
    elif ALIGN:
        cmd.append("-a")

    cmd.extend(["--save_results", save_path])
    return cmd


def _save_error_result_md(md_path: str, evo_output: str):
    """将 evo_res 的输出保存为 Markdown 文件"""
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metric_upper = METRIC.upper()

    # 构造对齐说明行（与 evo 输出保持一致）
    if ALIGN_AND_SCALE:
        align_note = "with Sim(3) Umeyama alignment"
    elif ALIGN:
        align_note = "with SE(3) Umeyama alignment"
    else:
        align_note = "without alignment"

    lines = [
        f"# EVO Error Result",
        f"",
        f"**生成时间**: {now}  ",
        f"**指标**: {metric_upper} w.r.t. `{POSE_RELATION}` ({align_note})  ",
        f"**数据目录**: `{TARGET_DIR}`  ",
        f"",
        f"## 输出结果",
        f"",
        f"```",
        evo_output.rstrip(),
        f"```",
        f"",
    ]

    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] 误差结果已保存: {md_path}")


def main():

    target_dir = os.path.abspath(TARGET_DIR)
    if not os.path.exists(target_dir):
        print(f"[ERROR] 目标测试目录不存在: {target_dir}")
        return
        
    # 切换相对路径到目标目录以防止 evo 在错误的地方查找
    os.chdir(target_dir)

    # 1. 创建/清理临时目录
    temp_path = os.path.join(target_dir, TEMP_DIR)
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.makedirs(temp_path)
    print(f"[INFO] 临时目录已创建: {temp_path}")

    # 2. 检查真值文件
    gt_path = os.path.join(target_dir, GT_FILE)
    if not os.path.exists(gt_path):
        print(f"[ERROR] 真值文件不存在: {gt_path}")
        return

    # 3. 确定待评估的轨迹文件列表
    traj_files = TRAJ_FILES
    if not traj_files:
        # 自动扫描 TARGET_DIR 下所有 .txt 文件（排除真值文件）
        traj_files = sorted([
            f for f in os.listdir(target_dir)
            if f.endswith('.txt') and f != GT_FILE
        ])
        print(f"[INFO] 自动发现 {len(traj_files)} 个轨迹文件: {', '.join(traj_files)}")

    # 4. 逐个评估轨迹文件
    zip_files = []
    metric_upper = METRIC.upper()
    print(f"\n{'='*60}")
    print(f"  开始 {metric_upper} 评估 (pose_relation={POSE_RELATION})")
    print(f"{'='*60}\n")

    for traj_file in traj_files:
        traj_path = os.path.join(target_dir, traj_file)
        if not os.path.exists(traj_path):
            print(f"[WARN] 轨迹文件不存在，跳过: {traj_file}")
            continue

        zip_name = os.path.splitext(traj_file)[0] + ".zip"
        save_path = os.path.join(temp_path, zip_name)

        cmd = build_eval_cmd(gt_path, traj_path, save_path)
        print(f"[EVAL] {traj_file}")
        print(f"       CMD: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=target_dir)
            if result.returncode == 0:
                print(f"       ✓ 评估完成 -> {zip_name}")
                if result.stdout.strip():
                    # 打印统计信息（缩进显示）
                    for line in result.stdout.strip().split('\n'):
                        print(f"         {line}")
                zip_files.append(save_path)
            else:
                print(f"       ✗ 评估失败:")
                print(f"         {result.stderr.strip()}")
        except Exception as e:
            print(f"       ✗ 异常: {e}")

        print()

    if not zip_files:
        print("[ERROR] 没有成功评估任何轨迹文件，退出。")
        return

    # 4. 重命名图例
    print(f"{'='*60}")
    print(f"  重命名图例")
    print(f"{'='*60}\n")

    for zip_file in zip_files:
        basename = os.path.basename(zip_file).replace('.zip', '.txt')
        legend_name = get_legend_name(basename)
        try:
            res = file_interface.load_res_file(zip_file)
            res.info["est_name"] = legend_name
            file_interface.save_res_file(zip_file, res)
            print(f"  {basename} -> \"{legend_name}\"")
        except Exception as e:
            print(f"  [WARN] 重命名失败 {basename}: {e}")

    # 5. 调用 evo_res 汇总展示
    print(f"\n{'='*60}")
    print(f"  evo_res 汇总展示")
    print(f"{'='*60}\n")

    cmd_res = [_evo_exe("evo_res")] + zip_files + ["--plot"]
    if SAVE_PLOT:
        plot_path = os.path.join(target_dir, PLOT_FILENAME)
        # 预先删除旧文件，避免 evo_res 弹出交互式覆盖确认提示导致程序卡住
        if os.path.exists(plot_path):
            os.remove(plot_path)
            print(f"[INFO] 已删除旧文件: {plot_path}")
        cmd_res.extend(["--save_plot", plot_path])
        
    print(f"[CMD] {' '.join(cmd_res)}\n")

    try:
        res_result = subprocess.run(
            cmd_res, check=True, cwd=target_dir,
            capture_output=True, text=True
        )
        # 打印 evo_res 输出到控制台
        if res_result.stdout:
            print(res_result.stdout)
        if res_result.stderr:
            print(res_result.stderr)

        # 将误差统计结果保存为 error_result.md
        if SAVE_ERROR_RESULT:
            # 确定保存路径：与 plot.pdf 同目录
            plot_dir = os.path.dirname(os.path.join(target_dir, PLOT_FILENAME))
            error_md_path = os.path.join(plot_dir, ERROR_RESULT_FILENAME)
            # 合并 stdout 和 stderr（evo_res 有时将表格写到 stderr）
            combined_output = (res_result.stdout or "") + (res_result.stderr or "")
            _save_error_result_md(error_md_path, combined_output)

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] evo_res 执行失败: {e}")
        if e.output:
            print(e.output)
    except Exception as e:
        print(f"[ERROR] 异常: {e}")


if __name__ == "__main__":
    main()
