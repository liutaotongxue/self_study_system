"""PDF 文本抽取 —— Phase 2-W1-2。

只负责"PDF → list[Page]"这一层,不管章节、不管 chunking。
其他模块在此基础上做章节检测和切块。
"""
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class Page:
    """单页 PDF 抽取结果。"""
    pdf_page: int       # 1-based PDF 物理页号(给 chapter_detect 用)
    text: str           # 抽取后的原始文本


def extract_pages(pdf_path: str | Path) -> list[Page]:
    """打开 PDF,逐页抽文本。

    返回列表,长度 = PDF 总页数。文本是 pymupdf 默认 get_text() 输出,
    保留换行。不做清洗(页眉页脚等噪声留给下一层处理,以便文本清洗策略
    可以独立 tune,不影响这层接口稳定性)。

    没有 OCR fallback —— Phase 2 假设教材 PDF 有文本层。
    """
    doc = pymupdf.open(str(pdf_path))
    try:
        return [
            Page(pdf_page=i + 1, text=doc[i].get_text())
            for i in range(doc.page_count)
        ]
    finally:
        doc.close()


def join_pages_text(pages: list[Page], start_pdf_page: int, end_pdf_page: int) -> str:
    """把指定 PDF 页范围(包含两端)的文本拼接成一段。

    pages 是 extract_pages 的输出,已经按 pdf_page 顺序排好。
    边界容错:end < start 或越界时返回空字符串。
    """
    if end_pdf_page < start_pdf_page or start_pdf_page < 1:
        return ""
    parts = [
        p.text for p in pages
        if start_pdf_page <= p.pdf_page <= end_pdf_page
    ]
    return "\n".join(parts)
