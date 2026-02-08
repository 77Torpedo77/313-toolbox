"""
打包脚本 - 将 generate_tex.py 打包为独立可执行文件
使用 PyInstaller 创建无需 Python 环境即可运行的程序
"""

import subprocess
import sys
from pathlib import Path


def main():
    # 检查 PyInstaller 是否已安装
    try:
        import PyInstaller
        print(f"✅ PyInstaller 版本: {PyInstaller.__version__}")
    except ImportError:
        print("📦 正在安装 PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # 项目路径
    project_dir = Path(__file__).parent
    main_script = project_dir / "generate_tex.py"
    scripts_dir = project_dir / "scripts"
    
    # PyInstaller 命令
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onedir",                    # 打包为文件夹（比 --onefile 快）
        "--name", "generate_tex",      # 输出名称
        "--add-data", f"{scripts_dir};scripts",  # 包含 scripts 目录
        "--hidden-import", "fitz",     # 确保 PyMuPDF 被包含
        "--hidden-import", "pymupdf",
        "--clean",                     # 清理临时文件
        "--noconfirm",                 # 覆盖已有输出
        str(main_script)
    ]
    
    print("\n🔨 开始打包...")
    print(f"   命令: {' '.join(cmd)}\n")
    
    subprocess.run(cmd, check=True)
    
    print("\n" + "=" * 50)
    print("✅ 打包完成！")
    print(f"   输出目录: {project_dir / 'dist' / 'generate_tex'}")
    print("\n📋 使用说明:")
    print("   1. 将 dist/generate_tex 整个文件夹复制到目标电脑")
    print("   2. 在目标电脑上运行: generate_tex.exe")
    print("   3. 确保 data/paper/ 目录与 exe 在同一位置")
    print("=" * 50)


if __name__ == "__main__":
    main()
