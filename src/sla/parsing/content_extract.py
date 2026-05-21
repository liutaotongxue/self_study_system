"""S4 human-anchored 内容抽取(视觉路径):人选章节内容页 → 渲染 PNG →
单发视觉 LLM 出【纯文本正文】→ 既有 chunker.split_into_chunks → 写 Chunk。
镜像 S2 的 structure_extract 单发模式,但出文本非结构化(model.invoke
取 .content),复用 render_pages_to_b64。

载荷:GenerationJob.chapter_id 字段承载 "ch1.2|p=50-65[!force]";路由建,
run_generation parse,本模块用解出来的 (ch_real, pages, force)。
/chapters route 的 active dict 用 _ch_base 剥离 |p= 后缀回真章。

承重:Chunk.chapter_id 原样写入(等于 document_structure.chapter_id),
绝不在此环节做任何派生/转换(派生只在 S2 的 derive_chapter_id 一处)。
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

# safety_settings 全 BLOCK_NONE:产品意图是 OCR 自己合法教材,不接受任何
# 误判截断。Gemini 这点 vs Anthropic 黑盒不可调是结构优势。
_GEMINI_SAFETY_NONE = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


def parse_content_payload(payload: str) -> tuple[str, list[int], bool]:
    """'ch1.2|p=50-65[!force]' / 'ch6.1.1|p=200,202-205'
    → (chapter_id, sorted unique 1-based PDF 页, force)。非法即 ValueError。"""
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
    """ChatAnthropic.invoke 返回的 AIMessage.content 可能是 str 也可能是 list[dict]
    (取决于版本/响应)。取出全部 text 块拼起来。"""
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
    # 默认 gemini-2.5-flash(2026-05-21 ratify,见 memory:vision-ocr-provider-choice):
    # 完整诊断链:
    #   - sonnet-4-6/opus-4-7 撞 Anthropic 服务端 content filter(APIStatusError
    #     "Output blocked by content filtering policy",stop_reason=None)
    #   - haiku-4-5 单页+简 prompt 过,但【多页 或 完整 CONTENT_EXTRACT_SYSTEM】
    #     任一条件就撞同样的 filter(haiku 不是普适解)
    #   - gemini-2.5-flash 全场景过:finish_reason=STOP、safety_ratings=[]、
    #     LaTeX 数学完整($,\hat,\frac,\sum,\sqrt 全在)、中文洁净、节边界过滤
    #     生效(看到 1.2 标题自动截止)、成本 ~$0.002/节(1/25 sonnet)
    # 关键:Gemini safety_settings 可显式 BLOCK_NONE,Anthropic 黑盒不可调。
    """渲染指定 PDF 页 → 单发视觉 LLM(纯文本)→ chunker → 写 Chunk。
    幂等:已有 Chunk 且 ¬force → 跳过(不渲染不调 LLM,省钱,loud);
    force → 先删该 (doc, chapter_id) 全部 Chunk 再写。
    返回 summary dict 供 job.detail。"""
    doc = db.get(Document, document_id)
    if doc is None or not doc.file_path:
        raise ValueError(f"document {document_id} 不存在或无 file_path")
    # 校验:该 doc 真有此 chapter_id 的 document_structure 行(防 S3 漏校验)
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

    # Gemini 图像 content 用 image_url(data URL)形;Anthropic 用 source_type=base64;
    # langchain 抽象层对各 provider 自动适配,我们按目标 provider 用其惯例形即可。
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
    # SystemMessage plain text(per [[provider-portability-preference]]:
    # 不深耕 Anthropic-native multi-block cache_control;Gemini 不支持该形)。
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
