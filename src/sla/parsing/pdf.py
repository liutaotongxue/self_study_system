"""PDF text extraction.

This module is only responsible for "PDF -> list[Page]". It does not
handle chapter detection or chunking — those live in other modules
that build on this layer.
"""
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class Page:
    """Result of extracting a single PDF page."""
    pdf_page: int       # 1-based physical page number (used by chapter_detect)
    text: str           # Raw extracted text


def extract_pages(pdf_path: str | Path) -> list[Page]:
    """Open the PDF and extract text page by page.

    Returns a list whose length equals the PDF page count. Text is the
    default pymupdf get_text() output (newlines preserved). No cleanup
    here — header/footer noise is left to the next layer so the
    cleanup strategy can be tuned independently without breaking this
    interface.

    No OCR fallback — this assumes the PDF has a text layer.
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
    """Concatenate the text from a PDF page range (inclusive on both ends).

    pages is the output of extract_pages, already sorted by pdf_page.
    Returns an empty string for boundary errors (end < start or out of range).
    """
    if end_pdf_page < start_pdf_page or start_pdf_page < 1:
        return ""
    parts = [
        p.text for p in pages
        if start_pdf_page <= p.pdf_page <= end_pdf_page
    ]
    return "\n".join(parts)
