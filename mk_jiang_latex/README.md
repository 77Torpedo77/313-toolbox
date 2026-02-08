# PDF 论文集 LaTeX 生成工具

本项目用于自动将大量 PDF 论文合并为一个 LaTeX 文档，每篇论文占两页：
1. 包含论文标题（和原始 PDF 首页）
2. 支持 PDF 书签导航

## 环境依赖

### Python 环境
- Python 3.6+
- 依赖库：`PyMuPDF` (用于提取 PDF 首页)

```bash
pip install PyMuPDF
```

### LaTeX 环境
- TeX Live 2018 或更高版本 (推荐使用 XeLaTeX 编译)
- 依赖包：
  - `ctex` (中文支持)
  - `geometry` (页面设置)
  - `graphicx` (图片/PDF 插入)
  - `titlesec` (标题格式)
  - `hyperref` (超链接)
  - `bookmark` (书签)
  - `grffile` (文件名处理，增强旧版 TeX 兼容性)

## 使用方法

1. **准备数据**
   - 将 PDF 论文放入 `data/paper/` 目录
   - 文件名格式建议：`序号-类型-标题.pdf` (例如 `10-SCI-Title.pdf`)

2. **生成 LaTeX**
   运行脚本，自动提取首页并生成代码：
   ```bash
   python generate_tex.py
   ```
   *注意：脚本会自动将 PDF 首页提取到 `output/first_pages/` 并清理文件名中的空格和特殊字符。*

3. **编译文档**
   使用 XeLaTeX 编译生成的 `papers.tex`：
   ```bash
   xelatex papers.tex
   ```
   *建议编译两次以生成正确的目录和书签。*

## 兼容性说明 (TeX Live 2018)

本项目已针对 **TeX Live 2018** 进行兼容性优化：
- **文件名处理**：Python 脚本会自动清理文件名（移除空格和生僻字符），避免旧版 LaTeX 处理复杂文件名时的错误。
- **包支持**：显式引入了 `grffile` 包，确保在旧版 TeX Live 中能正确解析文件名中的多重扩展名。
- **书签**：使用 `bookmark` 包替代默认书签生成，兼容性更好。
