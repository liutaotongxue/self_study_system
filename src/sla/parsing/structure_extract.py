"""S2 human-anchored structure extraction (vision path).

User picks the TOC pages -> render to PNG -> single
with_structured_output(Pydantic).invoke vision-LLM call -> write
document_structure. Not an agent and does not touch the graph harness.

Vision input beats text-layer input on OCR-degraded scans
(2026-05 probe: doc3 text layer had half-broken titles whereas the
vision path produced 366 clean entries with monotonic page numbers).

Provider: S4 OCR already uses Gemini 2.5-flash; S2 TOC is the same
vision task and switched to Gemini in 2026-05, so fork users only need
GOOGLE_API_KEY to run the first 4 steps (upload -> mark TOC -> mark
content -> see chunks). ANTHROPIC_API_KEY is only required for note +
KG generation.

Load-bearing rule: chapter_id = "ch" + section_id is derived
deterministically in Python — we never trust the LLM for this (S3
status page LEFT JOIN spine; S4 writes Chunk.chapter_id with the same
derivation). A section_id that is not dotted integers (chapter title
rows / "chapter overview" / exercises / references / preface /
appendix / unnumbered material) cannot form a stable join key — we
skip-and-count rather than fabricating a bad key that would pollute
the spine.
"""
import base64
import re

import pymupdf
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from pydantic import BaseModel, Field

from sla.config import settings
from sla.harness.prompts import STRUCTURE_EXTRACT_SYSTEM
from sla.models.domain import Document, DocumentStructure

_SID_RE = re.compile(r"^\d+(\.\d+)*$")
_RENDER_ZOOM = 1.8           # ~144 DPI; readable for CJK TOC pages

# safety_settings = BLOCK_NONE: OCR is over the user's own legitimate
# textbook, so we don't accept false-positive truncation. Same policy
# as content_extract — kept as 4 lines in each module rather than
# importing across modules, to avoid implicit coupling.
_GEMINI_SAFETY_NONE = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


class StructureItem(BaseModel):
    # After the human-anchored pivot, content pages are entered manually
    # (p=50-65), so the TOC-extracted page numbers have no consumer.
    # Dropping book_page_start: (1) no reader in the data flow,
    # (2) JSON payload shrinks ~30% per item, easing LLM output truncation
    # on large-TOC books (the OutputParserException seen on doc 1 in 2026-05).
    section_id: str = Field(
        description="目录里的原始编号串,如 '1' / '1.3' / '6.1.1';"
                    "几级照抄不规整化;确无编号则空字符串 ''")
    title: str = Field(
        description="该条标题,教材原文语言,不翻译不改写")


class StructureExtraction(BaseModel):
    sections: list[StructureItem] = Field(
        description="全书章节条目,按目录出现顺序")


def derive_chapter_id(section_id: str) -> str | None:
    """Load-bearing deterministic derivation. Non dotted-integer numbering
    returns None — the caller should skip-and-count. The resulting format
    must match Chunk.chapter_id and backend _ck (routes_domain:216), and
    S4 must use this same derivation."""
    s = (section_id or "").strip()
    if not _SID_RE.match(s):
        return None
    return "ch" + s


def parse_toc_payload(payload: str) -> tuple[list[int], bool]:
    """Parse 'toc:7-10' / 'toc:7,8,9' / 'toc:7-10,15' (optional trailing
    '!force') into (sorted unique 1-based PDF pages, force).
    Raises ValueError on malformed input (loud)."""
    s = (payload or "").strip()
    force = s.endswith("!force")
    if force:
        s = s[: -len("!force")]
    if not s.startswith("toc:"):
        raise ValueError(f"非法 extract_toc 载荷 {payload!r}(应形如 'toc:7-10')")
    spec = s[len("toc:"):].strip()
    pages: set[int] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            if "-" in tok:
                a_s, b_s = tok.split("-", 1)
                a, b = int(a_s), int(b_s)
                if a < 1 or b < a:
                    raise ValueError
                pages.update(range(a, b + 1))
            else:
                v = int(tok)
                if v < 1:
                    raise ValueError
                pages.add(v)
        except ValueError:
            raise ValueError(
                f"非法页范围片段 {tok!r}(应为正整数或 a-b 且 a<=b)") from None
    if not pages:
        raise ValueError("未解析出任何 TOC 页")
    return sorted(pages), force


def render_pages_to_b64(file_path: str, pages: list[int]) -> list[str]:
    """Render the given 1-based PDF pages to PNG -> base64.
    zoom=1.8 ~ 144 DPI. Reuses the existing pymupdf dependency
    (also used by sla.parsing.pdf) — no new dependency."""
    pdf = pymupdf.open(file_path)
    try:
        if max(pages) > pdf.page_count:
            raise ValueError(
                f"请求的 TOC 页最大 {max(pages)} 超出 PDF 总页 {pdf.page_count}")
        out: list[str] = []
        for pg in pages:
            page = pdf[pg - 1]
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(_RENDER_ZOOM, _RENDER_ZOOM))
            out.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
        return out
    finally:
        pdf.close()


def extract_and_persist(
    db, document_id: int, pages: list[int], force: bool,
    *, model_name: str = "gemini-2.5-flash", max_tokens: int = 32000,
) -> dict:
    """Render TOC pages -> single structured-output vision-LLM call ->
    write document_structure rows.

    Idempotency: if rows already exist and not force -> skip (no render,
    no LLM call, loud). With force -> delete every document_structure row
    for this document first, then write.

    Returns a summary dict for job.detail.
    """
    doc = db.get(Document, document_id)
    if doc is None or not doc.file_path:
        raise ValueError(f"document {document_id} 不存在或无 file_path")

    existing = (
        db.query(DocumentStructure)
        .filter(DocumentStructure.document_id == document_id)
        .count()
    )
    if existing and not force:
        return {"result": "skipped_existing", "existing": existing}

    images_b64 = render_pages_to_b64(doc.file_path, pages)

    content: list = [{
        "type": "text",
        "text": (f"以下是该教材的目录页图像(共 {len(pages)} 页,"
                 f"PDF 物理页 {pages[0]}-{pages[-1]})。"
                 f"请按 system 规则抽取全书章节结构。"),
    }]
    for b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": f"data:image/png;base64,{b64}",
        })

    model = ChatGoogleGenerativeAI(
        model=model_name,
        max_output_tokens=max_tokens,
        google_api_key=settings.google_api_key,
        safety_settings=_GEMINI_SAFETY_NONE,
    )
    structured = model.with_structured_output(StructureExtraction)
    result: StructureExtraction = structured.invoke([
        SystemMessage(content=STRUCTURE_EXTRACT_SYSTEM),
        HumanMessage(content=content),
    ])

    if force and existing:
        db.query(DocumentStructure).filter(
            DocumentStructure.document_id == document_id
        ).delete(synchronize_session=False)

    seen: set[str] = set()
    inserted = unkeyed = dups = 0
    for it in result.sections:
        cid = derive_chapter_id(it.section_id)
        if cid is None:
            unkeyed += 1
            continue
        if cid in seen:                       # UNIQUE(doc, chapter_id) collision guard, loud
            dups += 1
            continue
        seen.add(cid)
        db.add(DocumentStructure(
            document_id=document_id,
            section_id=it.section_id.strip(),
            chapter_id=cid,
            title=(it.title or "").strip(),
            book_page_start=0,    # Field retained (NOT NULL); always 0 — no reader, no side effect
            source="llm",
        ))
        inserted += 1
    db.commit()
    return {
        "result": "extracted", "inserted": inserted,
        "unkeyed": unkeyed, "dups_dropped": dups,
        "source_pages": pages,
    }
