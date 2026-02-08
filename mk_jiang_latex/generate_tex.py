"""
LaTeX 文档生成主脚本
从 PDF 论文集自动生成 LaTeX 文档
每页包含文章标题，下方嵌入对应 PDF 的首页
"""

from pathlib import Path
import sys

# 添加 scripts 目录到路径
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from parse_pdf_title import get_all_papers, PaperInfo
from extract_first_pages import extract_first_page, sanitize_filename, HAS_PYMUPDF


def escape_latex(text: str) -> str:
    """转义 LaTeX 特殊字符"""
    special_chars = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for char, escaped in special_chars.items():
        text = text.replace(char, escaped)
    return text


def generate_latex_document(papers: list, output_path: Path, first_pages_dir: Path):
    """
    生成 LaTeX 文档
    
    Args:
        papers: PaperInfo 列表
        output_path: 输出 .tex 文件路径
        first_pages_dir: 首页 PDF 目录
    """
    # 计算首页目录相对于 LaTeX 文件的路径
    rel_dir = first_pages_dir.relative_to(output_path.parent)
    
    latex_content = r"""\documentclass[a4paper,12pt]{article}

% 页面设置 - 减小边距以容纳更多内容
\usepackage[top=1.5cm, bottom=1.5cm, left=2cm, right=2cm]{geometry}

% PDF/图片嵌入
\usepackage{graphicx}
\usepackage{pdfpages}  % 用于追加完整PDF页面

% 中文支持
\usepackage{ctex}

% 标题格式 - 紧凑标题，无自动编号
\usepackage{titlesec}
\titleformat{\section}{\Large\bfseries}{}{0pt}{}
\titleformat{\subsection}{\large\bfseries}{}{0pt}{}
\titleformat{\subsubsection}{\normalsize\bfseries\centering}{}{0pt}{}
\titlespacing*{\section}{0pt}{0pt}{0.5ex}       % 防止 section 前产生空白页
\titlespacing*{\subsection}{0pt}{0pt}{0.5ex}    % 防止 subsection 前产生空白页
\titlespacing*{\subsubsection}{0pt}{0.5ex}{0.5ex}

% 禁用自动编号（目录直接显示标题）
\setcounter{secnumdepth}{0}

% 目录格式 - 取消换行时的悬挂缩进（保留层级缩进）
\usepackage{tocloft}
\setlength{\cftsubsubsecnumwidth}{0pt}  % 编号宽度为0，取消悬挂缩进
\renewcommand{\cftsubsubsecfont}{\normalfont}  % 条目字体
\renewcommand{\cftsubsubsecpagefont}{\normalfont}  % 页码字体

% 文件名处理（兼容旧版 TeX Live 如 2018）
\usepackage{grffile}

% 超链接和书签
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    unicode=true,
    bookmarks=true,
    bookmarksnumbered=false,
    bookmarksopen=true,
    pdfstartview=FitH,
}

% 书签包 - 解决书签问题
\usepackage{bookmark}

% 去除段落缩进
\setlength{\parindent}{0pt}

\begin{document}

% 目录使用罗马数字页码
\pagenumbering{roman}
\renewcommand{\contentsname}{\hspace*{\fill}目~录\hspace*{\fill}}
\tableofcontents
\newpage

% 正文使用阿拉伯数字页码
\pagenumbering{arabic}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 【使用说明】如需追加完整PDF（不仅是首页），在对应论文后添加以下代码：
%
% 方法1: 使用 \includegraphics 逐页追加（格式与首页一致）
%   \begin{center}
%   \includegraphics[width=\textwidth,height=0.85\textheight,keepaspectratio,page=2]{data/paper/原始文件名.pdf}
%   \end{center}
%   \newpage
%
% 方法2: 使用 \includepdf 批量追加（一次追加多页）
%   \includepdf[pages={2-},scale=0.85,pagecommand={\thispagestyle{plain}},offset=0 -1cm]{data/paper/原始文件名.pdf}
%
% 参数说明:
%   page=N          - 指定第N页（\includegraphics用）
%   pages={2-}      - 第2页到最后（\includepdf用）
%   pages={2-5}     - 第2-5页
%   pages={2,4,6}   - 仅第2、4、6页
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"""
    
    # 为每篇论文生成一页
    for paper in papers:
        # 处理标题：将下划线替换为空格
        display_title = paper.title.replace('_', ' ')
        title_escaped = escape_latex(display_title)
        
        # 使用清理后的文件名（无空格）
        sanitized_name = sanitize_filename(paper.filename)
        pdf_path = f"{rel_dir.as_posix()}/{sanitized_name}"
        
        # 标题和 PDF 在同一页，PDF 缩放到页面宽度
        # 标题格式: "序号.标题"，使用三级标题(subsubsection)便于用户手动添加一级、二级标题
        latex_content += f"""% ========== [{paper.index}] {paper.paper_type} ==========
\\subsubsection{{{paper.index}.{title_escaped}}}
\\vspace{{0.5em}}
\\begin{{center}}
\\includegraphics[width=\\textwidth,height=0.85\\textheight,keepaspectratio]{{{pdf_path}}}
\\end{{center}}
\\newpage

"""
    
    latex_content += r"\end{document}"
    
    # 写入文件
    output_path.write_text(latex_content, encoding='utf-8')
    print(f"✅ LaTeX 文档已生成: {output_path}")


def main():
    # 检查 PyMuPDF
    if not HAS_PYMUPDF:
        print("❌ 需要安装 PyMuPDF 来提取 PDF 首页:")
        print("   pip install PyMuPDF")
        return
    
    # 配置路径 - 兼容 PyInstaller 打包后的情况
    # PyInstaller 打包后 __file__ 指向 _internal 目录，需要用 sys.executable 获取 exe 位置
    if getattr(sys, 'frozen', False):
        # 打包后的 exe
        project_dir = Path(sys.executable).parent
    else:
        # 直接运行 .py 文件
        project_dir = Path(__file__).parent
    
    paper_dir = project_dir / "data" / "paper"
    first_pages_dir = project_dir / "output" / "first_pages"
    output_tex = project_dir / "papers.tex"
    
    # 创建输出目录
    first_pages_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有论文
    print("📖 正在解析 PDF 文件...")
    papers = get_all_papers(paper_dir)
    print(f"   找到 {len(papers)} 篇论文")
    
    # 提取首页
    print("\n📄 正在提取 PDF 首页...")
    extracted_count = 0
    for paper in papers:
        sanitized_name = sanitize_filename(paper.filename)
        output_path = first_pages_dir / sanitized_name
        
        # 如果已存在则跳过
        if output_path.exists():
            extracted_count += 1
            continue
        
        if extract_first_page(paper.filepath, output_path):
            extracted_count += 1
            print(f"   ✅ [{paper.index}] {paper.title[:40]}...")
    
    print(f"   共 {extracted_count} 个首页准备就绪")
    
    # 生成 LaTeX
    print("\n📝 正在生成 LaTeX 文档...")
    generate_latex_document(papers, output_tex, first_pages_dir)
    
    print("\n🎉 完成！请使用 XeLaTeX 编译生成的文档:")
    print(f"   xelatex {output_tex.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n" + "=" * 50)
        input("按 Enter 键退出...")

