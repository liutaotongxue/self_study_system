"""Policy check mechanism: permission / quota gatekeeping before tool execution.

Core design conventions (output of Phase 1B-1 design discussion):

  - Check functions return `str | None`: None = pass, string = violation reason
  - Tools are completely unaware: @tool functions in tools.py do not know Policy exists; schema stays clean
  - Policy is injected via state["policy"]; does not pollute Claude's view
  - Quota counts (artifacts_created) are maintained by this node; tools do not see them
  - max_steps is not implemented here; uses LangGraph's built-in recursion_limit instead
    (caller passes config={"recursion_limit": policy.max_steps * 2} on graph.invoke)
  - The denied ToolMessage only states the fact and the boundary; does not teach Claude what to do --
    Claude's training has a "tool failed -> change args" pattern; over-prescribing actually misleads
"""
import json

from langchain_core.messages import ToolMessage
from pydantic import BaseModel, Field

from sla.db import SessionLocal
from sla.harness.tools import ALL_TOOLS, TOOL_REGISTRY
from sla.models.domain import Chunk
from sla.models.runtime import Artifact

# Note: this module does not import AgentState (avoids a graph.py <-> policy.py circular dependency).
# The state parameter of policy_aware_tool_node has no type annotation; LangGraph infers the node's
# state type from the StateGraph(AgentState) declaration.


# Tool name -> tool object, used by policy_aware_tool_node for dispatch.
# Use TOOL_REGISTRY (superset of ALL_TOOLS + STRUCTURE_TOOLS): original 4 still resolve unchanged,
# structure tools are dispatchable (required); the superset is zero-regression for current learning tasks.
TOOLS_BY_NAME = dict(TOOL_REGISTRY)


# --------------------------------------------------------------------------- #
# Policy dataclass
# --------------------------------------------------------------------------- #

class Policy(BaseModel):
    """Execution policy for a single task. Phase 1B-2 loads it from the Task table; Phase 1B-1 hardcoded for testing."""

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
# Denied ToolMessage construction
# --------------------------------------------------------------------------- #

def make_denied_message(tc: dict, reason: str) -> ToolMessage:
    """Format: 'Denied by policy. {reason}'

    States only the fact (denial reason + boundary); does not attach a 'Try X' suggestion -- letting
    Claude infer from the reason itself is more reliable than our hardcoded hint, and avoids misleading it.
    """
    return ToolMessage(
        content=f"Denied by policy. {reason}",
        tool_call_id=tc["id"],
    )


# --------------------------------------------------------------------------- #
# Tool-specific check functions
#
# Convention: return str (violation reason) or None (pass). When DB access is needed, the function
# opens its own session and does not depend on an externally passed db; signatures are uniform for
# the dispatch table.
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
    """Needs a DB query: which chapter the chunk belongs to, and whether that chapter is in read_scope."""
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
    """1b: read_page_range is clamped to the first N pages (policy.max_pages). max_pages=None means unbounded
    (current learning tasks have neither this tool nor this key -> never reach here). probe_body has no such check,
    so it can reach the full-book page range -- Correction(2): otherwise the offset cannot be anchored and 1c (B) is structurally infeasible."""
    mp = policy.max_pages
    if mp is None:
        return None
    start = tc["args"].get("start")
    end = tc["args"].get("end")
    if start is None or end is None:
        return None  # Missing args are left to pydantic / the tool itself
    if start < 1:
        return f"read_page_range start={start} must be >= 1."
    if end > mp:
        return (
            f"read_page_range end={end} exceeds max_pages={mp}. "
            f"Only the first {mp} pages are readable for structure extraction; "
            f"use probe_body for body-page anchoring beyond that."
        )
    return None


# Tool name -> check function. To add a new tool, only modify this table; policy_aware_tool_node itself is unchanged.
# probe_body / propose_structure deliberately have NO entries: gated only by the whitelist;
# probe_body must reach the full-book page range (Correction 2), and propose_structure validation lives in 1c.
CHECKS_BY_TOOL = {
    "list_chunks": _check_list_chunks,
    "read_chunk": _check_read_chunk,
    "save_note": _check_save_note,
    "save_questions": _check_save_questions,
    "read_page_range": _check_read_page_range,
}


# --------------------------------------------------------------------------- #
# Artifact writes (Phase 1B-2)
#
# After save_note / save_questions executes successfully, policy_aware_tool_node opportunistically
# writes one Artifact row linking Run.id -> Note.id or Question.id. This is the explicit index for
# "what this Run produced"; it does not rely on a time-based scan, and Phase 1B-4 (Eval) uses it directly.
#
# Design:
#   - Tool schemas are completely unchanged (the signature Claude sees stays clean)
#   - If state["run_id"] is None, silently skip (smoke-script scenario)
#   - If parsing the tool's JSON return fails, also silently skip (the tool itself did not succeed; no Artifact to link)
# --------------------------------------------------------------------------- #

def _write_artifact_for_save_note(run_id: int | None, result_str: str) -> None:
    """Extract note_id from save_note's JSON return, write one Artifact row."""
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
    """Extract question_ids from save_questions' JSON return; write one Artifact row per id."""
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
# Policy-aware tool node: replaces the original ToolNode(ALL_TOOLS) position
# --------------------------------------------------------------------------- #

def policy_aware_tool_node(state) -> dict:
    """LangGraph node: policy-check before tool execution, then update quota counts + write Artifact after.

    Flow (for each tool_call in the current AIMessage):
      1. Whitelist check: is tc.name in policy.allowed_tools?
      2. Tool-specific check: read_scope / max_artifacts (dispatched via CHECKS_BY_TOOL)
      3. Either fails -> build a denied ToolMessage, do not invoke the tool, continue
      4. Pass -> actually invoke the tool, build a normal ToolMessage
      5. After save_note / save_questions succeed:
         - update artifacts_created count
         - if state["run_id"] is not None, write one Artifact link row (Phase 1B-2)

    Returns the state delta:
      {"messages": [ToolMessage...], "artifacts_created": {...}}
    """
    last_msg = state["messages"][-1]
    policy: Policy = state["policy"]
    # Copy then mutate; avoid in-place state modification (unsafe + violates LangGraph node contract)
    artifacts = dict(state.get("artifacts_created", {}))
    tool_messages: list[ToolMessage] = []

    for tc in last_msg.tool_calls:
        name = tc["name"]

        # ---- 1. Whitelist ----
        if name not in policy.allowed_tools:
            tool_messages.append(make_denied_message(
                tc, f"Tool '{name}' is not in allowed_tools: {policy.allowed_tools}",
            ))
            continue

        # ---- 2. Tool-specific check ----
        check_fn = CHECKS_BY_TOOL.get(name)
        if check_fn is not None:
            violation = check_fn(tc, policy, state)
            if violation:
                tool_messages.append(make_denied_message(tc, violation))
                continue

        # ---- 3. Passed; actually invoke the tool ----
        tool = TOOLS_BY_NAME[name]
        # Inject document_id from state into tc args (InjectedToolArg pattern)
        # -- Claude cannot see this field; we add it so tools can filter by document
        tool_args = dict(tc["args"])
        if "document_id" not in tool_args:
            tool_args["document_id"] = state.get("document_id")
        try:
            result = tool.invoke(tool_args)
        except Exception as e:
            # Tool runtime exception (not a policy denial): mark status=error so Claude can distinguish
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

        # ---- 4. Quota count + Artifact write (only after a successful write-tool execution) ----
        run_id = state.get("run_id")
        if name == "save_note":
            artifacts["note"] = artifacts.get("note", 0) + 1
            _write_artifact_for_save_note(run_id, result)
        elif name == "save_questions":
            artifacts["question_batch"] = artifacts.get("question_batch", 0) + 1
            _write_artifacts_for_save_questions(run_id, result)

    return {"messages": tool_messages, "artifacts_created": artifacts}
