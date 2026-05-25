"""S4 human-anchored content extraction (vision path).

User picks the content pages of a section -> render to PNG -> single
vision-LLM call returns plain body text -> existing
chunker.split_into_chunks -> write Chunk.

Mirrors the single-shot pattern of S2 structure_extract but the output
is unstructured text (model.invoke().content), reusing render_pages_to_b64.

Payload format: GenerationJob.chapter_id carries
"ch1.2|p=50-65[!force]". Built by the route, parsed by
run_generation, this module operates on the parsed
(ch_real, pages, force) triple. The /chapters route's active dict
uses _ch_base to strip the |p= suffix back to the real chapter.

Load-bearing rule: Chunk.chapter_id is written as-is (equal to
document_structure.chapter_id). NEVER derive or transform it here —
derivation only happens once, in S2 derive_chapter_id.
"""
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)

from sla.config import settings
from sla.harness.prompts import CONTENT_EXTRACT_SYSTEM
from sla.models.domain import Chunk, Document, DocumentStructure
from sla.parsing.chunker import normalize_text, split_into_chunks
from sla.parsing.structure_extract import render_pages_to_b64

_PAYLOAD_RE = re.compile(r"^(ch\d+(?:\.\d+)*)\|p=(.+?)(!force)?$")

# safety_settings = BLOCK_NONE everywhere: the product intent is to OCR
# the user's own legitimate textbook, so we don't accept false-positive
# truncation. Being able to set this explicitly (vs Anthropic's opaque
# filter) is a structural advantage of Gemini for this task.
_GEMINI_SAFETY_NONE = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


def parse_content_payload(payload: str) -> tuple[str, list[int], bool]:
    """Parse 'ch1.2|p=50-65[!force]' / 'ch6.1.1|p=200,202-205' into
    (chapter_id, sorted unique 1-based PDF pages, force).
    Raises ValueError on malformed input.
    """
    s = (payload or "").strip()
    m = _PAYLOAD_RE.match(s)
    if not m:
        raise ValueError(
            f"非法 extract_content 载荷 {payload!r}"
            f"(应形如 'ch1.2|p=50-65[!force]')")
    ch_id, spec, force_tag = m.group(1), m.group(2), m.group(3)
    force = bool(force_tag)
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
        raise ValueError("未解析出任何页")
    return ch_id, sorted(pages), force


def _extract_text_from_result(content) -> str:
    """AIMessage.content from ChatAnthropic.invoke can be either str or
    list[dict] (depends on version/response). Concatenate all text blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
            elif isinstance(c, str):
                parts.append(c)
        return "".join(parts)
    return str(content)


def extract_and_chunk(
    db, document_id: int, chapter_id: str, pages: list[int], force: bool,
    *, model_name: str = "gemini-2.5-flash", max_tokens: int = 8000,
) -> dict:
    # Default model gemini-2.5-flash, ratified 2026-05-21
    # (see memory: vision-ocr-provider-choice). Full diagnostic chain:
    #   - sonnet-4-6 / opus-4-7 hit Anthropic server-side content filter
    #     (APIStatusError "Output blocked by content filtering policy",
    #     stop_reason=None)
    #   - haiku-4-5 passes with a single page + minimal prompt, but hits
    #     the same filter as soon as we add either multi-page input or
    #     the full CONTENT_EXTRACT_SYSTEM — not a general fix
    #   - gemini-2.5-flash passes in every scenario:
    #     finish_reason=STOP, safety_ratings=[], LaTeX math preserved
    #     ($, \hat, \frac, \sum, \sqrt), clean Chinese output,
    #     section-boundary filtering works (stops at next "1.2" header),
    #     cost ~$0.002/section (1/25 of sonnet).
    # Key advantage: Gemini's safety_settings can be set to BLOCK_NONE
    # explicitly; Anthropic's filter is opaque and not user-configurable.
    """Render the given PDF pages -> single vision-LLM call (plain text)
    -> chunker -> write Chunk rows.

    Idempotency: if Chunks already exist and not force -> skip (no render,
    no LLM call, loud return). With force -> delete all existing
    (document_id, chapter_id) Chunks before writing.

    Returns a summary dict for job.detail.
    """
    doc = db.get(Document, document_id)
    if doc is None or not doc.file_path:
        raise ValueError(f"document {document_id} 不存在或无 file_path")
    # Verify the document has a document_structure row for this chapter_id
    # (defense against S3 missing a check).
    has_struct = (
        db.query(DocumentStructure)
        .filter(DocumentStructure.document_id == document_id,
                DocumentStructure.chapter_id == chapter_id)
        .first() is not None
    )
    if not has_struct:
        raise ValueError(
            f"document {document_id} 无 chapter_id={chapter_id!r} 的结构行;"
            "先在书库点📑标注目录")

    existing = (
        db.query(Chunk)
        .filter(Chunk.document_id == document_id,
                Chunk.chapter_id == chapter_id)
        .count()
    )
    if existing and not force:
        return {"result": "skipped_existing", "existing": existing}

    images_b64 = render_pages_to_b64(doc.file_path, pages)

    # Gemini wants image content as image_url (data URL); Anthropic wants
    # source_type=base64. The LangChain abstraction adapts to each provider,
    # so we just use the conventional form for the target provider.
    content: list = [{
        "type": "text",
        "text": (f"以下是该教材【{chapter_id}】的内容页图像(共 {len(pages)} 页,"
                 f"PDF 物理页 {pages[0]}-{pages[-1]})。"
                 f"请按 system 规则还原干净正文文本。"),
    }]
    for b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": f"data:image/png;base64,{b64}",
        })

    model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.google_api_key,
        max_output_tokens=max_tokens,
        temperature=0,
        safety_settings=_GEMINI_SAFETY_NONE,
    )
    # Plain-text SystemMessage (per provider-portability-preference: avoid
    # going deeper into Anthropic-native multi-block cache_control, which
    # Gemini does not support).
    result = model.invoke([
        SystemMessage(content=CONTENT_EXTRACT_SYSTEM),
        HumanMessage(content=content),
    ])
    raw = _extract_text_from_result(result.content)
    cleaned = normalize_text(raw)
    texts = split_into_chunks(cleaned)
    if not texts:
        raise ValueError(
            f"chunker 切不出任何 chunk(LLM 返回 {len(raw)} 字符);"
            "可能是 OCR 输出过短 / 页范围错")

    if force and existing:
        db.query(Chunk).filter(
            Chunk.document_id == document_id,
            Chunk.chapter_id == chapter_id,
        ).delete(synchronize_session=False)

    pg_start, pg_end = pages[0], pages[-1]
    for txt in texts:
        db.add(Chunk(
            document_id=document_id,
            chapter_id=chapter_id,
            page_start=pg_start,
            page_end=pg_end,
            content=txt,
        ))
    db.commit()
    return {
        "result": "extracted", "inserted": len(texts),
        "chars": len(cleaned),
        "source_pages": pages,
    }
