"""文本切块 —— Phase 2-W1-4.

输入:section 的 joined text(从 pdf.py 的 join_pages_text 来)
输出:list[str] 切好的 chunk 文本

Phase 2-W1 简化:
  - 不做 per-chunk page tracking(每个 chunk 共用 section 的 page 范围)
  - 只做最小文本清洗(ligature 归一化 + 处理排版断词)
  - 其他噪声(页眉/页脚/脚注)留给 Phase 2 后续
"""
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# LaTeX 排版常见 ligature 字符 → ASCII
LIGATURE_MAP = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
}


def normalize_text(text: str) -> str:
    """Phase 2-W1 最小清洗:
      1. 替换 ligature 字符
      2. 合并因排版断行的连字符词(e.g. 'environ-\\nment' → 'environment')
         只处理小写字母对,避免误伤 'U.S.-China' 这种合法连字符
    """
    for old, new in LIGATURE_MAP.items():
        text = text.replace(old, new)
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)
    return text


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """切块,优先在段落/句子边界处切。

    返回 list[str]。空 text 或空白 text 返回空列表。
    """
    if not text or not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text)
