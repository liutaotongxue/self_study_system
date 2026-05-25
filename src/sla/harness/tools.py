"""LangGraph harness toolset: the agent uses these 4 tools to read chunks and produce notes/questions.

Each function is decorated with @tool. After decoration:
  - The function's docstring is sent to Claude as the tool description, so it must clearly state
    "when to use it, what the args are, what it returns" (written for Claude to read, not Python readers).
  - The function signature's type hints are auto-derived into JSON Schema; Claude uses that to construct args.
  - Call form changes from fn(x) to fn.invoke({"x": ...}).
"""
import json
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, BeforeValidator, Field

from sla.db import SessionLocal
from sla.models.domain import Chunk, Document, Note, Question
from sla.parsing.pdf import extract_pages


# Type alias: document_id is an InjectedToolArg -- Claude does not see this field in the schema;
# it is injected by policy_aware_tool_node from state["document_id"] before invoking the tool.
# None means "do not filter by document", preserving backward compatibility (fixture / single-document scenarios).
InjectedDocumentId = Annotated[int | None, InjectedToolArg]


def _parse_if_json_string(v):
    """Defensive parse: Claude occasionally serializes a nested structure into a JSON string;
    BeforeValidator restores the string into a list before pydantic validation. If Claude's
    generated JSON itself has a bug, this raises JSONDecodeError -- that is not BeforeValidator's
    fault; the real fix is to make the schema explicit about nested structures (see QuestionItem)."""
    if isinstance(v, str):
        return json.loads(v)
    return v


class QuestionItem(BaseModel):
    """单道思考题。schema 里有明确的 properties,Claude 看到后会构造
    proper nested object,而不是 fallback 到 JSON 字符串化。"""

    content: str = Field(description="题目正文,开放性思考题(不是选择/填空)")
    difficulty: int | None = Field(default=None, description="难度 1-5")
    tags: list[str] | None = Field(default=None, description="主题标签数组,如 ['policy','value']")


@tool
def list_chunks(chapter_id: str, document_id: InjectedDocumentId = None) -> str:
    """列出某一章节下所有 chunks 的元数据(chunk_id、页码、内容预览)。

    在开始阅读一个章节前,必须先调用此工具了解这一节有多少段、每段大概讲什么,
    然后再决定按什么顺序用 read_chunk 读取。

    返回 JSON 数组字符串,每项字段:
      - chunk_id:整数,用于传给 read_chunk
      - page_start / page_end:页码范围
      - preview:content 前 120 字符,用于判断这段大概在讲什么

    chunks 按 chunk_id 升序。
    """
    db = SessionLocal()
    try:
        q = db.query(Chunk).filter(Chunk.chapter_id == chapter_id)
        # document_id is injected from state by policy_aware_tool_node; filters out cross-document chapter_id collisions
        if document_id is not None:
            q = q.filter(Chunk.document_id == document_id)
        chunks = q.order_by(Chunk.id).all()
        result = [
            {
                "chunk_id": c.id,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "preview": c.content[:120],
            }
            for c in chunks
        ]
        return json.dumps(result, ensure_ascii=False)
    finally:
        db.close()


@tool
def read_chunk(chunk_id: int) -> str:
    """读取指定 chunk_id 的完整内容,用于详细阅读单段文本。

    必须先用 list_chunks 拿到可用的 chunk_id 列表后再调用本工具,不要凭空猜测 id。
    返回该 chunk 的完整 content 字符串。若 id 不存在,返回 "(chunk {id} not found)"。
    """
    db = SessionLocal()
    try:
        # db.get(Cls, pk) is the SQLAlchemy 2.0 recommended form for primary-key lookup
        chunk = db.get(Chunk, chunk_id)
        if chunk is None:
            return f"(chunk {chunk_id} not found)"
        return chunk.content
    finally:
        db.close()


@tool
def save_note(
    chapter_id: str,
    content_md: str,
    document_id: InjectedDocumentId = None,
) -> str:
    """为指定章节保存一份 Markdown 学习笔记到数据库。

    必须在读完该章节所有 chunks 之后再调用,不要边读边写。
    笔记要 Markdown 格式,体现这一节的核心概念、它们之间的关系、以及作者的关键论证。
    一个章节调用一次即可,不要分多次保存。

    返回 JSON 字符串:{"ok": true, "note_id": <int>}。
    """
    db = SessionLocal()
    try:
        # Resolve document_id: use InjectedToolArg when provided (accurate for multi-document scenarios),
        # otherwise reverse-lookup (single-document / backward-compatible fixture scenarios)
        if document_id is not None:
            doc_id = document_id
        else:
            any_chunk = (
                db.query(Chunk)
                .filter(Chunk.chapter_id == chapter_id)
                .first()
            )
            if any_chunk is None:
                return json.dumps({"ok": False, "error": f"chapter {chapter_id} not found"})
            doc_id = any_chunk.document_id

        note = Note(
            document_id=doc_id,
            chapter_id=chapter_id,
            content_md=content_md,
        )
        db.add(note)
        db.commit()
        # Explicit refresh after commit, ensuring note.id is populated
        db.refresh(note)
        # Return JSON: upstream (policy_aware_tool_node) parses note_id and writes the Artifact row
        return json.dumps({"ok": True, "note_id": note.id}, ensure_ascii=False)
    finally:
        db.close()


@tool
def save_questions(
    chapter_id: str,
    items: Annotated[list[QuestionItem], BeforeValidator(_parse_if_json_string)],
    document_id: InjectedDocumentId = None,
) -> str:
    """为指定章节批量保存思考题到数据库。

    必须在 save_note 之后调用。
    items 是题目数组,直接传 JSON array(不要序列化成字符串),每项字段:
      - content (必填,string):题目正文,开放性思考题,不是选择/填空
      - difficulty (可选,integer 1-5):难度估计
      - tags (可选,string array):主题标签

    一次保存 3-5 道题,覆盖最重要的几个概念。
    返回 JSON 字符串:{"ok": true, "question_ids": [<int>, ...]}。
    """
    db = SessionLocal()
    try:
        # Same as save_note: document_id wins, reverse-lookup when absent
        if document_id is not None:
            doc_id = document_id
        else:
            any_chunk = (
                db.query(Chunk)
                .filter(Chunk.chapter_id == chapter_id)
                .first()
            )
            if any_chunk is None:
                return json.dumps({"ok": False, "error": f"chapter {chapter_id} not found"})
            doc_id = any_chunk.document_id

        # Add all rows then commit once: faster, and semantically "this batch of questions" as one unit
        saved_qs: list[Question] = []
        for it in items:
            q = Question(
                document_id=doc_id,
                chapter_id=chapter_id,
                # items is list[QuestionItem], access by attribute; BeforeValidator catches str input
                content=it.content,
                difficulty=it.difficulty,
                # tags column is JSON type; pass a Python list directly and SQLAlchemy serializes
                tags=it.tags,
                # source defaults to "agent", status defaults to "unattempted"; do not set explicitly
            )
            db.add(q)
            saved_qs.append(q)
        # Flush first: SQLAlchemy fills back the autoincrement id on each q. commit also flushes, but
        # commit's default expire-on-commit invalidates these instances, so reading q.id triggers N refresh queries
        db.flush()
        question_ids = [q.id for q in saved_qs]
        db.commit()
        # Return JSON: upstream (policy_aware_tool_node) parses question_ids and writes Artifact rows
        return json.dumps({"ok": True, "question_ids": question_ids}, ensure_ascii=False)
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# 1b: structure-extraction subagent toolset (independent of ALL_TOOLS; save_note etc.
# are truly absent from its model schema -- build_graph(tools=) only binds these 3,
# rather than blocking at runtime)
# --------------------------------------------------------------------------- #

@tool
def read_page_range(start: int, end: int, document_id: InjectedDocumentId = None) -> str:
    """读取 PDF 第 start..end 页(1-based,含两端)纯文本,用于看前言/目录(TOC)
    以抽章节结构。仅允许读前 N 页(N 由 policy 限,超出会被拒;正文锚定请用
    probe_body)。返回各页文本,以 '--- p.{n} ---' 分隔。"""
    if document_id is None:
        return "(error: no document_id; 结构 Task 必须带 Task.document_id)"
    db = SessionLocal()
    try:
        doc = db.get(Document, document_id)
        if doc is None or not doc.file_path:
            return f"(error: document {document_id} 不存在或无 file_path)"
        fp = doc.file_path
    finally:
        db.close()
    # Re-extract O(total pages) every call; max_steps is bounded so no caching yet (deliberate, optimize later)
    pages = extract_pages(fp)
    sel = [p for p in pages if start <= p.pdf_page <= end]
    if not sel:
        return f"(range {start}..{end} 无页;PDF 共 {len(pages)} 页)"
    return "\n".join(f"--- p.{p.pdf_page} ---\n{p.text}" for p in sel)


@tool
def probe_body(token: str, document_id: InjectedDocumentId = None) -> str:
    """在【正文全书页域,不受前 N 页限制】定位 token 首现的 PDF 页,用于推断
    book页→PDF页 的 offset。token 例:'Chapter 5' 或节号 '5.1'。返回命中
    [{pdf_page, line}] 列表。这是你提议 offset 的【证据】,不是判定 —— 另有
    独立确定性 gate 复核你,不要自证。"""
    if document_id is None:
        return "(error: no document_id)"
    db = SessionLocal()
    try:
        doc = db.get(Document, document_id)
        if doc is None or not doc.file_path:
            return f"(error: document {document_id} 不存在或无 file_path)"
        fp = doc.file_path
    finally:
        db.close()
    pages = extract_pages(fp)
    hits = []
    for p in pages:
        for line in p.text.splitlines():
            if token in line:
                hits.append({"pdf_page": p.pdf_page, "line": line.strip()[:160]})
                break
    if not hits:
        return f"(token {token!r} 在 {len(pages)} 页中未命中)"
    return json.dumps(hits, ensure_ascii=False)


class SectionItem(BaseModel):
    """单个小节(7 字段契约)。显式 properties → Claude 构造 proper nested
    object,而非 fallback 空调用(踩 tools.py:35-37 / save_questions 已验教训:
    list[dict] 无结构 schema 会让 Claude 填不进 → ValidationError 死循环)。"""

    section_id: str = Field(description="原 TOC 编号,如 '1.3'(两级十进制 N.M)")
    chapter_id: str = Field(description="DB 键 = 'ch'+section_id,如 'ch1.3'")
    title: str = Field(description="该节标题")
    book_page_start: int = Field(description="该节起始【书印刷页号】(TOC 上的页码)")
    book_page_end: int = Field(description="该节结束书印刷页号(下一节起始-1)")
    pdf_page_start: int = Field(description="该节起始【PDF 物理页号】= book_page_start + offset")
    pdf_page_end: int = Field(description="该节结束 PDF 物理页号 = book_page_end + offset")


@tool
def propose_structure(
    sections: Annotated[list[SectionItem], BeforeValidator(_parse_if_json_string)],
) -> str:
    """提交最终章节结构(单次,提交即结束本任务)。sections 是小节数组,
    直接传 JSON array of objects(不要序列化成字符串),每项 7 字段见
    SectionItem。仅支持两级十进制编号 N.M。本工具【不写库、不校验】——
    verbatim 提议由 runner 持久化进 ToolCall.input,另有独立 gate 逐字读它
    做 7 字段校验后才冻结。返回 {"ok":true,"n":<count>}。"""
    n = len(sections) if isinstance(sections, list) else 0
    return json.dumps({"ok": True, "n": n}, ensure_ascii=False)


# Convenience import for other modules: from sla.harness.tools import ALL_TOOLS
ALL_TOOLS = [list_chunks, read_chunk, save_note, save_questions]
STRUCTURE_TOOLS = [read_page_range, probe_body, propose_structure]
# name -> tool object: single source of truth for bind (graph.build_graph) and dispatch
# (policy_aware_tool_node), preventing the schema-bind and runtime-whitelist from forking
TOOL_REGISTRY = {t.name: t for t in ALL_TOOLS + STRUCTURE_TOOLS}
