"""
PDF 文件名标题解析模块
从 PDF 文件名中提取序号、类型和标题

文件名格式: "序号-类型-标题.pdf"
例如: "10-SCI-Feature incremental learning with causality.pdf"
"""

import re
from pathlib import Path
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class PaperInfo:
    """论文信息数据类"""
    index: int          # 序号
    paper_type: str     # 类型 (SCI/EI等)
    title: str          # 标题
    filename: str       # 原始文件名
    filepath: Path      # 完整路径


def parse_filename(filename: str) -> Tuple[int, str, str]:
    """
    从 PDF 文件名解析序号、类型和标题
    
    Args:
        filename: PDF 文件名，如 "10-SCI-Feature incremental...pdf"
    
    Returns:
        (序号, 类型, 标题) 元组
    """
    # 移除 .pdf 扩展名
    name = filename.rsplit('.', 1)[0] if filename.endswith('.pdf') else filename
    
    # 匹配格式: 数字-类型-标题（类型支持中英文）
    pattern = r'^(\d+)-([^-]+)-(.+)$'
    match = re.match(pattern, name)
    
    if match:
        index = int(match.group(1))
        paper_type = match.group(2)
        title = match.group(3)
        return index, paper_type, title
    else:
        # 如果无法解析，使用文件名作为标题
        return 0, "Unknown", name


def get_all_papers(paper_dir: Path) -> List[PaperInfo]:
    """
    获取目录下所有 PDF 论文信息，按序号排序
    
    Args:
        paper_dir: PDF 论文目录路径
    
    Returns:
        PaperInfo 列表，按序号升序排列
    """
    papers = []
    
    for pdf_file in paper_dir.glob("*.pdf"):
        index, paper_type, title = parse_filename(pdf_file.name)
        papers.append(PaperInfo(
            index=index,
            paper_type=paper_type,
            title=title,
            filename=pdf_file.name,
            filepath=pdf_file
        ))
    
    # 按序号排序（序号相同时按文件名排序，确保一致性）
    papers.sort(key=lambda p: (p.index, p.filename))
    return papers


if __name__ == "__main__":
    # 测试解析功能
    test_dir = Path(r"D:\Py-Project\mk-jiang-latex\data\paper")
    papers = get_all_papers(test_dir)
    
    print(f"共找到 {len(papers)} 篇论文\n")
    for p in papers[:5]:  # 只打印前5条
        print(f"[{p.index}] {p.paper_type}: {p.title}")
