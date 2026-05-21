"""S2 human-anchored 结构抽取(视觉路径):人选目录页 → 渲染为 PNG →
一发视觉结构化 LLM 调用 → 写 document_structure。【非 agent、不碰
graph】,镜像 kg.py 的 ChatAnthropic.with_structured_output(Pydantic).invoke
单发范式,但以图像消息替代文本(2026-05 路径B探针实测:视觉在 OCR 烂书
上显著胜文本层 —— doc3 文本层标题半毁→视觉 366 条全洁、页码单增,
$0.26/书 TOC、~4 分钟级延迟)。

承重:chapter_id = "ch" + section_id 在 Python 确定性派生,不信 LLM
(S3 状态页 LEFT JOIN 命脉;S4 写 Chunk.chapter_id 必须用【同一派生】)。
section_id 非点分整数(章标题行/本章概要/习题/参考文献/前言/附录/无
编号)→ 不可形成稳定 join 键 → 跳过并计数,不硬塞坏键污染脊。
"""
import base64
import re

import pymupdf
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from sla.config import settings
from sla.harness.prompts import STRUCTURE_EXTRACT_SYSTEM
from sla.models.domain import Document, DocumentStructure

_SID_RE = re.compile(r"^\d+(\.\d+)*$")
_RENDER_ZOOM = 1.8           # ≈ 144 DPI;Anthropic 自动下采样到 ~1.15MP/页,CJK 够认


class StructureItem(BaseModel):
    section_id: str = Field(
        description="目录里的原始编号串,如 '1' / '1.3' / '6.1.1';"
                    "几级照抄不规整化;确无编号则空字符串 ''")
    title: str = Field(
        description="该条标题,教材原文语言,不翻译不改写")
    book_page_start: int = Field(
        description="该条在目录里印的起始页码(整数);确无则 0")


class StructureExtraction(BaseModel):
    sections: list[StructureItem] = Field(
        description="全书章节条目,按目录出现顺序")


def derive_chapter_id(section_id: str) -> str | None:
    """承重确定性派生。非点分整数编号 → None(调用方跳过+计数)。
    与 Chunk.chapter_id / 后端 _ck(routes_domain:216)同形;S4 必须同此。"""
    s = (section_id or "").strip()
    if not _SID_RE.match(s):
        return None
    return "ch" + s


def parse_toc_payload(payload: str) -> tuple[list[int], bool]:
    """'toc:7-10' / 'toc:7,8,9' / 'toc:7-10,15'(+尾缀 '!force')
    → (sorted unique 1-based PDF 页, force)。非法即 ValueError(loud)。"""
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
    """渲染指定 PDF 页(1-based)为 PNG → base64。zoom=1.8 ≈ 144DPI。
    项目既有 pymupdf 依赖(sla.parsing.pdf 也用它);零新依赖。"""
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
    *, model_name: str = "claude-sonnet-4-6", max_tokens: int = 16000,
) -> dict:
    """渲染 TOC 页 → 单发视觉结构化 LLM → 写 document_structure。
    幂等:已有行且 ¬force → 跳过(不渲染不调 LLM,省钱,loud);
    force → 先删该 doc 全部 document_structure 再写。
    返回 summary dict 供 job.detail。"""
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
            "type": "image",
            "source_type": "base64",
            "data": b64,
            "mime_type": "image/png",
        })

    model = ChatAnthropic(
        model=model_name,
        max_tokens=max_tokens,
        api_key=settings.anthropic_api_key,
    )
    structured = model.with_structured_output(StructureExtraction)
    # cache_control:system prompt 跨调用 5min 内命中(图像内容不缓存,
    # 仅 system 文本块;同书重抽 / 多本相邻时仍省 system 部分)
    result: StructureExtraction = structured.invoke([
        SystemMessage(content=[{
            "type": "text",
            "text": STRUCTURE_EXTRACT_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }]),
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
        if cid in seen:                       # uq(doc,chapter_id) 防撞 + loud
            dups += 1
            continue
        seen.add(cid)
        db.add(DocumentStructure(
            document_id=document_id,
            section_id=it.section_id.strip(),
            chapter_id=cid,
            title=(it.title or "").strip(),
            book_page_start=it.book_page_start or 0,
            source="llm",
        ))
        inserted += 1
    db.commit()
    return {
        "result": "extracted", "inserted": inserted,
        "unkeyed": unkeyed, "dups_dropped": dups,
        "source_pages": pages,
    }
