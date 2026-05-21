"""Policy 校验机制:工具执行前的权限/配额守门。

核心设计约定(Phase 1B-1 设计讨论的产物):

  - 检查函数返回 `str | None`:None=通过,字符串=违规原因
  - 工具完全无感:tools.py 里的 @tool 函数不知道 Policy 存在,schema 干净
  - Policy 通过 state["policy"] 注入,不污染 Claude 视角
  - 配额计数(artifacts_created)由本 node 维护,工具不感知
  - max_steps 不在这里实现,改用 LangGraph 内置 recursion_limit
    (caller 在 graph.invoke 时传 config={"recursion_limit": policy.max_steps * 2})
  - denied ToolMessage 只声明事实和边界,不教 Claude 该做什么——
    Claude 训练里有"tool failed → 改参数"模式,过度建议反而误导
"""
import json

from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

from sla.db import SessionLocal
from sla.harness.tools import ALL_TOOLS, TOOL_REGISTRY
from sla.models.domain import Chunk
from sla.models.runtime import Artifact

# 注:本模块不 import AgentState(避免 graph.py ↔ policy.py 循环依赖)。
# policy_aware_tool_node 的 state 参数不加类型注解,LangGraph 会从
# StateGraph(AgentState) 声明里自动推导节点的 state 类型。


# 工具名 → tool 对象,policy_aware_tool_node 派发用。
# 用 TOOL_REGISTRY(ALL_TOOLS + STRUCTURE_TOOLS 超集):原 4 个解析不变,
# 结构工具可被 dispatch(必需);超集对现学习任务零回归。
TOOLS_BY_NAME = dict(TOOL_REGISTRY)


# --------------------------------------------------------------------------- #
# Policy 数据类
# --------------------------------------------------------------------------- #

class Policy(BaseModel):
    """单个 task 的执行策略。Phase 1B-2 会从 Task 表加载;Phase 1B-1 测试时硬编。"""

    read_scope: list[str] = Field(
        description="允许读取的 chapter_id 列表",
    )
    max_steps: int = Field(
        default=20,
        description="最大循环步数;graph 调用方应转换为 recursion_limit = max_steps * 2",
    )
    allowed_tools: list[str] = Field(
        default_factory=lambda: [t.name for t in ALL_TOOLS],
        description="工具白名单,默认包含全部 4 个工具",
    )
    max_artifacts: dict[str, int] = Field(
        default_factory=lambda: {"note": 1, "question_batch": 1},
        description="每类 artifact 的最大产出数",
    )
    max_pages: int | None = Field(
        default=None,
        description=(
            "1b 结构抽取:read_page_range 可读的最大 PDF 页号(前 N 页)。"
            "None=不限(现学习任务无此键 → 默认 None → 字节零回归)。"
            "probe_body 刻意【不】受此限(否则 offset 锚不了、1c (B) 做不出)"
        ),
    )
    max_tokens: int | None = Field(
        default=None,
        description=(
            "模型单次响应最大输出 token。None → build_graph 默认 4096"
            "(现学习任务无此键 → None → 仍 4096 → 字节零回归)。"
            "结构抽取须设高(propose_structure 一次吐全部小节,~80 节 "
            "JSON ≫ 4096 否则 tool 调用被截断 → 空 args 死循环)。"
            "上界非成本下界:只按实际生成 token 计费,设高不浪费"
        ),
    )


# --------------------------------------------------------------------------- #
# Denied ToolMessage 构造
# --------------------------------------------------------------------------- #

def make_denied_message(tc: dict, reason: str) -> ToolMessage:
    """格式: 'Denied by policy. {reason}'

    只声明事实(被拒原因 + 边界),不附 'Try X' 建议——Claude 自己根据 reason 推断
    比我们硬编 hint 更可靠,而且不会误导。
    """
    return ToolMessage(
        content=f"Denied by policy. {reason}",
        tool_call_id=tc["id"],
    )


# --------------------------------------------------------------------------- #
# 工具特定检查函数
#
# 约定:返回 str(违规原因) 或 None(通过)。需要 DB 时函数内部开 session,
# 不依赖外部传 db,签名一致便于 dispatch 表统一管理。
# --------------------------------------------------------------------------- #

def _check_list_chunks(tc: dict, policy: Policy, state) -> str | None:
    chapter_id = tc["args"].get("chapter_id")
    if chapter_id not in policy.read_scope:
        return (
            f"chapter_id='{chapter_id}' is not in read_scope. "
            f"Allowed chapters: {policy.read_scope}"
        )
    return None


def _check_read_chunk(tc: dict, policy: Policy, state) -> str | None:
    """需要 DB 查询:chunk 属于哪个 chapter,该 chapter 是否在 read_scope。"""
    chunk_id = tc["args"].get("chunk_id")
    db = SessionLocal()
    try:
        chunk = db.get(Chunk, chunk_id)
        if chunk is None:
            return f"chunk_id={chunk_id} does not exist."
        if chunk.chapter_id not in policy.read_scope:
            return (
                f"chunk_id={chunk_id} belongs to chapter '{chunk.chapter_id}', "
                f"not in read_scope. Allowed chapters: {policy.read_scope}"
            )
    finally:
        db.close()
    return None


def _check_save_note(tc: dict, policy: Policy, state) -> str | None:
    chapter_id = tc["args"].get("chapter_id")
    if chapter_id not in policy.read_scope:
        return (
            f"chapter_id='{chapter_id}' is not in read_scope. "
            f"Allowed chapters: {policy.read_scope}"
        )
    artifacts = state.get("artifacts_created", {})
    note_max = policy.max_artifacts.get("note", 0)
    if artifacts.get("note", 0) >= note_max:
        return (
            f"Note quota exceeded: {artifacts.get('note', 0)} note(s) already created, "
            f"max allowed: {note_max}."
        )
    return None


def _check_save_questions(tc: dict, policy: Policy, state) -> str | None:
    chapter_id = tc["args"].get("chapter_id")
    if chapter_id not in policy.read_scope:
        return (
            f"chapter_id='{chapter_id}' is not in read_scope. "
            f"Allowed chapters: {policy.read_scope}"
        )
    artifacts = state.get("artifacts_created", {})
    qb_max = policy.max_artifacts.get("question_batch", 0)
    if artifacts.get("question_batch", 0) >= qb_max:
        return (
            f"Question batch quota exceeded: "
            f"{artifacts.get('question_batch', 0)} batch(es) already created, "
            f"max allowed: {qb_max}."
        )
    return None


def _check_read_page_range(tc: dict, policy: Policy, state) -> str | None:
    """1b:read_page_range 夹死前 N 页(policy.max_pages)。max_pages=None 不限
    (现学习任务无此工具、也无此键 → 永不走到这)。probe_body 无此 check,
    故全书页域可达 —— Correction(2):否则 offset 锚不了、1c (B) 结构上做不出。"""
    mp = policy.max_pages
    if mp is None:
        return None
    start = tc["args"].get("start")
    end = tc["args"].get("end")
    if start is None or end is None:
        return None  # 缺参交给 pydantic / 工具自身
    if start < 1:
        return f"read_page_range start={start} must be >= 1."
    if end > mp:
        return (
            f"read_page_range end={end} exceeds max_pages={mp}. "
            f"Only the first {mp} pages are readable for structure extraction; "
            f"use probe_body for body-page anchoring beyond that."
        )
    return None


# 工具名 → 检查函数。加新工具时只改这张表,policy_aware_tool_node 本身不动。
# probe_body / propose_structure 刻意【无】条目:只受白名单门禁,
# probe_body 须全书页域可达(Correction 2),propose_structure 校验属 1c。
CHECKS_BY_TOOL = {
    "list_chunks": _check_list_chunks,
    "read_chunk": _check_read_chunk,
    "save_note": _check_save_note,
    "save_questions": _check_save_questions,
    "read_page_range": _check_read_page_range,
}


# --------------------------------------------------------------------------- #
# Artifact 写入(Phase 1B-2)
#
# save_note / save_questions 成功执行后,policy_aware_tool_node 顺手写一行
# Artifact 链接 Run.id → Note.id 或 Question.id。这是"Run 产出了哪些东西"
# 的显式索引,不依赖时间扫,Phase 1B-4 (Eval) 直接用。
#
# 设计:
#   - 工具 schema 完全不变(Claude 看到的工具签名干净)
#   - state["run_id"] = None 时静默跳过(smoke 脚本场景)
#   - 解析工具返回的 JSON 失败也静默跳过(工具本身没成功,无 Artifact 可链)
# --------------------------------------------------------------------------- #

def _write_artifact_for_save_note(run_id: int | None, result_str: str) -> None:
    """从 save_note 的 JSON 返回提取 note_id,写一行 Artifact。"""
    if run_id is None:
        return
    try:
        parsed = json.loads(str(result_str))
    except (json.JSONDecodeError, TypeError):
        return
    note_id = parsed.get("note_id") if parsed.get("ok") else None
    if note_id is None:
        return
    db = SessionLocal()
    try:
        db.add(Artifact(
            run_id=run_id,
            kind="note",
            ref_table="note",
            ref_id=note_id,
        ))
        db.commit()
    finally:
        db.close()


def _write_artifacts_for_save_questions(run_id: int | None, result_str: str) -> None:
    """从 save_questions 的 JSON 返回提取 question_ids,每个 id 写一行 Artifact。"""
    if run_id is None:
        return
    try:
        parsed = json.loads(str(result_str))
    except (json.JSONDecodeError, TypeError):
        return
    qids = parsed.get("question_ids", []) if parsed.get("ok") else []
    if not qids:
        return
    db = SessionLocal()
    try:
        for qid in qids:
            db.add(Artifact(
                run_id=run_id,
                kind="question",
                ref_table="question",
                ref_id=qid,
            ))
        db.commit()
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Policy-aware tool node:替换原 ToolNode(ALL_TOOLS) 的位置
# --------------------------------------------------------------------------- #

def policy_aware_tool_node(state) -> dict:
    """LangGraph 节点:工具执行前 policy 校验,执行后更新配额计数 + 写 Artifact。

    流程(对当前 AIMessage 里的每个 tool_call):
      1. 白名单检查:tc.name 在 policy.allowed_tools?
      2. 工具特定检查:read_scope / max_artifacts(通过 CHECKS_BY_TOOL 分发)
      3. 任一不过 → 构造 denied ToolMessage,不调工具,continue
      4. 通过 → 实际调工具,构造正常 ToolMessage
      5. save_note / save_questions 成功后:
         - 更新 artifacts_created 计数
         - 若 state["run_id"] 非 None,写一行 Artifact 链接(Phase 1B-2)

    返回 state 增量:
      {"messages": [ToolMessage...], "artifacts_created": {...}}
    """
    last_msg = state["messages"][-1]
    policy: Policy = state["policy"]
    # 拷贝一份再修改,避免就地改 state(不安全且违反 LangGraph 节点契约)
    artifacts = dict(state.get("artifacts_created", {}))
    tool_messages: list[ToolMessage] = []

    for tc in last_msg.tool_calls:
        name = tc["name"]

        # ---- 1. 白名单 ----
        if name not in policy.allowed_tools:
            tool_messages.append(make_denied_message(
                tc, f"Tool '{name}' is not in allowed_tools: {policy.allowed_tools}",
            ))
            continue

        # ---- 2. 工具特定检查 ----
        check_fn = CHECKS_BY_TOOL.get(name)
        if check_fn is not None:
            violation = check_fn(tc, policy, state)
            if violation:
                tool_messages.append(make_denied_message(tc, violation))
                continue

        # ---- 3. 通过,实际调工具 ----
        tool = TOOLS_BY_NAME[name]
        # 把 state 里的 document_id 注入 tc args(InjectedToolArg 模式)
        # —— Claude 看不到这个字段,我们这里补上让 tools 能按 document 过滤
        tool_args = dict(tc["args"])
        if "document_id" not in tool_args:
            tool_args["document_id"] = state.get("document_id")
        try:
            result = tool.invoke(tool_args)
        except Exception as e:
            # 工具运行时异常(非 policy 拒绝):标 status=error 让 Claude 区分
            tool_messages.append(ToolMessage(
                content=f"Tool execution failed: {type(e).__name__}: {e}",
                tool_call_id=tc["id"],
                status="error",
            ))
            continue

        tool_messages.append(ToolMessage(
            content=str(result),
            tool_call_id=tc["id"],
        ))

        # ---- 4. 配额计数 + Artifact 写入(只在成功执行 write 工具后) ----
        run_id = state.get("run_id")
        if name == "save_note":
            artifacts["note"] = artifacts.get("note", 0) + 1
            _write_artifact_for_save_note(run_id, result)
        elif name == "save_questions":
            artifacts["question_batch"] = artifacts.get("question_batch", 0) + 1
            _write_artifacts_for_save_questions(run_id, result)

    return {"messages": tool_messages, "artifacts_created": artifacts}
