"""
PDF 首页提取模块
提取每个 PDF 的首页并保存为无空格文件名的新 PDF
"""

import re
from pathlib import Path
from typing import List, Tuple

# 尝试导入 PyMuPDF
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


def sanitize_filename(filename: str) -> str:
    """
    清理文件名，仅移除空格和 LaTeX 不兼容的特殊字符，保留中文
    
    Args:
        filename: 原始文件名
    
    Returns:
        清理后的文件名（保留中英文、数字、连字符和下划线）
    """
    # 移除 .pdf 扩展名
    name = filename.rsplit('.', 1)[0] if filename.endswith('.pdf') else filename
    
    # 只替换空格和特殊字符为下划线（保留中文、英文、数字、连字符）
    # 保留: 中文(\u4e00-\u9fff)、英文(a-zA-Z)、数字(0-9)、连字符(-)、下划线(_)
    sanitized = re.sub(r'[^\u4e00-\u9fffa-zA-Z0-9\-_]', '_', name)
    
    # 合并连续的下划线
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # 移除首尾下划线
    sanitized = sanitized.strip('_')
    
    # 确保文件名不为空
    if not sanitized:
        sanitized = "unnamed"
    
    return sanitized + '.pdf'


def extract_first_page(input_pdf: Path, output_pdf: Path) -> bool:
    """
    使用 PyMuPDF 提取 PDF 首页
    
    Args:
        input_pdf: 输入 PDF 路径
        output_pdf: 输出 PDF 路径
    
    Returns:
        是否成功
    """
    if not HAS_PYMUPDF:
        raise ImportError("需要安装 PyMuPDF: pip install PyMuPDF")
    
    try:
        # 打开源 PDF
        src_doc = fitz.open(str(input_pdf))
        
        # 创建新 PDF，只包含第一页
        dst_doc = fitz.open()
        dst_doc.insert_pdf(src_doc, from_page=0, to_page=0)
        
        # 保存
        dst_doc.save(str(output_pdf))
        
        src_doc.close()
        dst_doc.close()
        
        return True
    except Exception as e:
        print(f"  ❌ 提取失败: {input_pdf.name}: {e}")
        return False


def batch_extract(input_dir: Path, output_dir: Path) -> List[Tuple[Path, Path, str]]:
    """
    批量提取 PDF 首页
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
    
    Returns:
        [(原始路径, 输出路径, 清理后文件名), ...] 列表
    """
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    pdf_files = sorted(input_dir.glob("*.pdf"))
    
    for pdf_file in pdf_files:
        sanitized_name = sanitize_filename(pdf_file.name)
        output_path = output_dir / sanitized_name
        
        if extract_first_page(pdf_file, output_path):
            results.append((pdf_file, output_path, sanitized_name))
            print(f"  ✅ {pdf_file.name} → {sanitized_name}")
    
    return results


if __name__ == "__main__":
    if not HAS_PYMUPDF:
        print("❌ 需要安装 PyMuPDF:")
        print("   pip install PyMuPDF")
        exit(1)
    
    # 测试提取
    project_dir = Path(__file__).parent.parent
    input_dir = project_dir / "data" / "paper"
    output_dir = project_dir / "output" / "first_pages"
    
    print(f"📂 输入目录: {input_dir}")
    print(f"📂 输出目录: {output_dir}")
    print()
    
    results = batch_extract(input_dir, output_dir)
    print(f"\n✅ 完成！共提取 {len(results)} 个 PDF 首页")
