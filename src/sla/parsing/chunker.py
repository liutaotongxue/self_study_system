"""Text chunking.

Input:  joined section text (from pdf.py join_pages_text).
Output: list[str] of chunked text.

Simplifying assumptions:
  - No per-chunk page tracking (every chunk shares the section's page range).
  - Minimal text cleanup only (ligature normalization + line-break dehyphenation).
  - Other noise (headers/footers/footnotes) is left for later stages.
"""
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# Common LaTeX ligature characters -> ASCII equivalents.
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
    """Minimal cleanup:
      1. Replace ligature characters.
      2. Re-join words broken by typographic line-breaks
         (e.g. 'environ-\\nment' -> 'environment'). Only matches lowercase
         pairs so legitimate hyphens like 'U.S.-China' are preserved.
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
    """Split text into chunks, preferring paragraph / sentence boundaries.

    Returns list[str]. Empty or whitespace-only text returns an empty list.
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
