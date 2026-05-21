"""LangGraph harness 的工具集:agent 通过这 4 个工具读 chunks、产出 note/question。

每个函数都用 @tool 装饰。装饰后:
  - 函数的 docstring 会作为 tool description 发给 Claude,所以 docstring 要写清
    "什么时候用、参数是什么、返回什么"(给 Claude 看,不是给 Python 读者看)。
  - 函数签名的 type hints 会被自动推导成 JSON Schema,Claude 据此构造 args。
  - 调用方式从 fn(x) 变成 fn.invoke({"x": ...})。
"""
import json
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, BeforeValidator, Field

from sla.db import SessionLocal
from sla.models.domain import Chunk, Document, Note, Question
from sla.parsing.pdf import extract_pages


# Type alias:document_id 是 InjectedToolArg —— Claude 看不到 schema 里这个字段,
# 由 policy_aware_tool_node 在调用工具前从 state["document_id"] 注入。
# None 表示"不按 document 过滤",保持向后兼容(fixture / 单文档场景)。
InjectedDocumentId = Annotated[int | None, InjectedToolArg]


def _parse_if_json_string(v):
    """防御性 parse:Claude 偶尔会把嵌套结构序列化成 JSON 字符串发过来,
    BeforeValidator 在 pydantic 校验前先把字符串还原成 list。如果 Claude
    生成的 JSON 本身有 bug,这里会抛 JSONDecodeError —— 不算 BeforeValidator
    的锅,真正的修法是让 schema 里有明确的嵌套结构(见 QuestionItem)。"""
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
        # document_id 由 policy_aware_tool_node 从 state 注入,过滤跨文档同 chapter_id 的混淆
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
        # db.get(Cls, pk) 是 SQLAlchemy 2.0 主键查询的推荐写法
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
        # 解析 document_id:若 InjectedToolArg 给了就直接用(多文档场景准确),
        # 否则反查(单文档/向后兼容 fixture 场景)
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
        # commit 后显式 refresh,确保 note.id 已填回
        db.refresh(note)
        # 返回 JSON:上层(policy_aware_tool_node)解析 note_id 写 Artifact 行
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
        # 同 save_note: document_id 优先,缺时反查
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

        # 一次性 add 多条,只 commit 一次:更快,且语义上"这一批题"作为整体
        saved_qs: list[Question] = []
        for it in items:
            q = Question(
                document_id=doc_id,
                chapter_id=chapter_id,
                # items 是 list[QuestionItem],按属性访问;BeforeValidator 兜住 str 输入
                content=it.content,
                difficulty=it.difficulty,
                # tags 字段是 JSON 类型,直接传 Python list,SQLAlchemy 自动序列化
                tags=it.tags,
                # source 默认 "agent"、status 默认 "unattempted",不显式赋值
            )
            db.add(q)
            saved_qs.append(q)
        # 先 flush:让 SQLAlchemy 把自增 id 填回 q 对象。commit 也会 flush,但
        # commit 之后默认会 expire 这些 instance,再读 q.id 会触发 N 次 refresh 查询
        db.flush()
        question_ids = [q.id for q in saved_qs]
        db.commit()
        # 返回 JSON:上层(policy_aware_tool_node)解析 question_ids 写 Artifact 行
        return json.dumps({"ok": True, "question_ids": question_ids}, ensure_ascii=False)
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# 1b:结构抽取 subagent 工具集(独立于 ALL_TOOLS;save_note 等对它从模型 schema
# 真缺席 —— build_graph(tools=) 只 bind 这 3 个,不是 runtime 才拦)
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
    # 每次重抽 O(总页);max_steps 有界故先不缓存(优化属后续,刻意不提前抽象)
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


# 方便其他模块统一导入:from sla.harness.tools import ALL_TOOLS
ALL_TOOLS = [list_chunks, read_chunk, save_note, save_questions]
STRUCTURE_TOOLS = [read_page_range, probe_body, propose_structure]
# name → tool 对象:bind(graph.build_graph) 与 dispatch(policy_aware_tool_node)
# 的单一真源,杜绝 schema-bind 与 runtime-白名单再分叉
TOOL_REGISTRY = {t.name: t for t in ALL_TOOLS + STRUCTURE_TOOLS}
