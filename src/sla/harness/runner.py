"""Phase 1B-2: end-to-end Task -> Run execution entry point.

run_task(task_id) -> run_id:
  1. Load Task -> build Policy -> create Run(status='running', started_at=now)
  2. graph.stream runs the task; each chunk is translated into Step / ToolCall / ToolResult rows
     - agent node -> 1 Step row + N ToolCall rows (one per AIMessage.tool_calls)
     - tools node -> 1 ToolResult row per ToolMessage (linked via tool_call_id)
  3. policy_aware_tool_node writes Artifact rows itself when save_note / save_questions succeed
     (not handled here; see policy.py)
  4. Exception classification:
     - GraphRecursionError -> status='policy_halted'
     - Other exceptions -> status='failed' (includes traceback)
     - Normal completion -> status='completed'  (v4-agreed string, not 'succeeded')
  5. Write Run.ended_at + Run.error, close the transaction

Not idempotent: each run creates a new Run row (running the same Task multiple times is legal).
"""
import json
import traceback
from datetime import datetime

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.errors import GraphRecursionError

from sla.db import SessionLocal
from sla.harness.graph import build_graph
from sla.harness.policy import Policy
from sla.harness.tools import TOOL_REGISTRY
from sla.models.domain import Document, DocumentStructure
from sla.models.runtime import Artifact, Run, Step, Task, ToolCall, ToolResult


def _build_outline(document_id: int) -> str | None:
    """S5: full-book outline text (includes 3+ levels, info-complete). Helps the learning agent
    locate the current chapter within the book. Returns None when there are no DS rows
    (legacy doc / new book without TOC marked); the caller skips injection accordingly.
    """
    db = SessionLocal()
    try:
        doc = db.get(Document, document_id)
        rows = (db.query(DocumentStructure.section_id,
                         DocumentStructure.chapter_id,
                         DocumentStructure.title)
                .filter(DocumentStructure.document_id == document_id).all())
    finally:
        db.close()
    if not rows:
        return None

    def _ck(cid):
        try:
            return [int(x) for x in cid[2:].split(".")]
        except Exception:
            return [99]

    items = sorted(rows, key=lambda r: _ck(r[1]))
    lines = [f"# 全书大纲:{(doc and doc.title) or '本书'}",
             "(供你定位本章在整体中的位置;不必逐项详读)",
             ""]
    for sid, cid, ti in items:
        depth = cid.count(".")            # ch1=0 / ch1.2=1 / ch1.2.3=2
        lines.append(f"{'  ' * depth}{sid or '?'} {ti or ''}")
    return "\n".join(lines)


def run_task(task_id: int) -> int:
    """End-to-end execution of a Task; returns run_id.

    Caller:
      from sla.harness.runner import run_task
      run_id = run_task(task_id=1)
    """
    # ---------- Load Task, create Run(status='running') ----------
    db = SessionLocal()
    try:
        task = db.get(Task, task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")

        # Deserialize policy JSON -> Policy object (field names must match the Policy class; seed_task.py already aligned)
        policy = Policy(**task.policy)

        run = Run(
            task_id=task_id,
            status="running",
            started_at=datetime.utcnow(),
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        run_id = run.id

        # Copy task fields out before closing the session; graph uses them below
        system_prompt = task.system_prompt
        user_prompt = task.user_prompt
    finally:
        db.close()

    # Copy task.document_id before closing the session (system/user_prompt already copied above)
    db2 = SessionLocal()
    try:
        task = db2.get(Task, task_id)
        task_document_id = task.document_id if task else None
    finally:
        db2.close()

    # ---------- Build initial state ----------
    # run_id is for policy_aware_tool_node to write Artifact rows
    # document_id is for the tools' InjectedToolArg, filtering cross-document same chapter_id
    # cache_control enables Anthropic prompt caching (hits when system + tools are resent each turn)
    # S5: full-book outline injected as a second SystemMessage (plain text portable form,
    # per [[provider-portability-preference]] -- use portable LangChain abstractions, do not
    # deepen Anthropic-native multi-block cache_control; the first system keeps its original
    # form to avoid expanding S5 scope). No DS rows (legacy doc / TOC not marked) -> outline_text
    # is None and is not injected (legacy zero regression).
    outline_text = _build_outline(task_document_id) if task_document_id else None
    initial = {
        "messages": [
            SystemMessage(content=[{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }]),
            *([SystemMessage(content=outline_text)] if outline_text else []),
            HumanMessage(content=user_prompt),
        ],
        "policy": policy,
        "artifacts_created": {},
        "run_id": run_id,
        "document_id": task_document_id,
    }

    # ---------- Run graph, translate stream -> Step/ToolCall/ToolResult ----------
    # Weld A: bind set = policy.allowed_tools (single source of truth, schema follows whitelist).
    # Current learning Task seed allowed_tools is exactly 4 -> resolves back to original ALL_TOOLS -> byte-identical bind;
    # this is a runner-path semantic shift (not a no-op); regression smoke must assert learning tasks still have exactly 4 tools.
    # terminal_tool: only structure tasks (allowed_tools contains propose_structure) terminate structurally;
    # hard backstop is still the _stream_and_persist recursion_limit=max_steps*2 below.
    # max_tokens: policy.max_tokens or 4096 -- learning task has no such key -> None -> 4096 (byte-identical zero regression);
    # structure task must set it high (propose_structure outputs ~80 sections at once; 4096 truncates the tool call -> empty args infinite loop)
    graph = build_graph(
        max_tokens=(policy.max_tokens or 4096),
        tools=[TOOL_REGISTRY[n] for n in policy.allowed_tools],
        terminal_tool=(
            "propose_structure"
            if "propose_structure" in policy.allowed_tools
            else None
        ),
    )
    status = "completed"
    error_text: str | None = None

    try:
        _stream_and_persist(graph, initial, run_id, policy)
    except GraphRecursionError as e:
        # max_steps exceeded, interrupted by LangGraph
        status = "policy_halted"
        error_text = f"GraphRecursionError (max_steps={policy.max_steps} exceeded): {e}"
    except Exception as e:
        status = "failed"
        error_text = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    # ---------- finalize Run ----------
    db = SessionLocal()
    try:
        run = db.get(Run, run_id)
        run.status = status
        run.ended_at = datetime.utcnow()
        run.error = error_text
        db.commit()
    finally:
        db.close()

    return run_id


# --------------------------------------------------------------------------- #
# Internal: stream -> Step/ToolCall/ToolResult translation
# --------------------------------------------------------------------------- #

def _stream_and_persist(graph, initial_state: dict, run_id: int, policy: Policy):
    """Run graph.stream and translate each chunk into Step/ToolCall/ToolResult rows.

    Design:
      - Each agent node update -> 1 Step row (idx incremented) + N ToolCall rows
      - Each tools node update -> 1 ToolResult row per ToolMessage
      - pending_tool_calls dict: look up ToolCall.id from anthropic_tool_use_id
        (because ToolMessage links via tool_call_id, while DB ToolResult uses ToolCall.id)
    """
    step_idx = 0
    pending_tool_calls: dict[str, int] = {}  # anthropic_tool_use_id -> ToolCall.id

    db = SessionLocal()
    try:
        for chunk in graph.stream(
            initial_state,
            stream_mode="updates",
            config={"recursion_limit": policy.max_steps * 2},
        ):
            for node_name, update in chunk.items():
                msgs = update.get("messages", [])
                if not msgs:
                    continue

                if node_name == "agent":
                    _persist_agent_step(db, run_id, step_idx, msgs, pending_tool_calls)
                    step_idx += 1
                elif node_name == "tools":
                    _persist_tool_results(db, run_id, msgs, pending_tool_calls)
    finally:
        db.close()


def _persist_agent_step(
    db,
    run_id: int,
    step_idx: int,
    msgs: list,
    pending_tool_calls: dict[str, int],
):
    """Write 1 Step row + N ToolCall rows."""
    ai_msg = next((m for m in msgs if isinstance(m, AIMessage)), None)
    if ai_msg is None:
        return

    step = Step(
        run_id=run_id,
        idx=step_idx,
        # Phase 1B-2 does not yet persist model_input/output (redundant + data size large).
        # To actually inspect per-step messages, read the ToolCall + ToolResult chain; revisit in Phase 2
        model_input=None,
        model_output=None,
    )
    db.add(step)
    db.flush()  # Obtain step.id for ToolCall.step_id

    for tc in ai_msg.tool_calls or []:
        tc_row = ToolCall(
            step_id=step.id,
            anthropic_tool_use_id=tc["id"],
            name=tc["name"],
            input=tc["args"],
        )
        db.add(tc_row)
        db.flush()
        pending_tool_calls[tc["id"]] = tc_row.id
        # Note: the verbatim proposal lives in tc_row.input (= tc["args"]); captured here verbatim, single writer.
        # The structure_proposal pointer Artifact is NOT written here -- see _persist_tool_results
        # (we need ToolResult status=ok to know success; not yet known at call time -> moved to that path).

    db.commit()


def _persist_tool_results(
    db, run_id: int, msgs: list, pending_tool_calls: dict[str, int]
):
    """Write 1 ToolResult row per ToolMessage; and on a successful propose_structure call,
    write the [unique] structure_proposal pointer Artifact (success-gated, see below)."""
    for tm in msgs:
        if not isinstance(tm, ToolMessage):
            continue

        tc_id = pending_tool_calls.pop(tm.tool_call_id, None)
        if tc_id is None:
            # Should not happen: every ToolMessage must correspond to a ToolCall created in the prior agent step.
            # If it does occur, the langgraph version / our logic has a bug; silently skip for now so a single
            # anomaly does not blow up the whole Run
            continue

        content_str = str(tm.content)
        if content_str.startswith("Denied by policy."):
            status_val = "denied"
            reason = content_str
            content_val = None
        elif getattr(tm, "status", None) == "error":
            status_val = "error"
            reason = content_str
            content_val = None
        else:
            status_val = "ok"
            reason = None
            # content is a JSON column. Tool return may be plain text or a JSON string;
            # on parse success store dict/list, on failure keep the raw str
            try:
                content_val = json.loads(content_str)
            except (json.JSONDecodeError, TypeError):
                content_val = content_str

        db.add(ToolResult(
            tool_call_id=tc_id,
            status=status_val,
            content=content_val,
            reason=reason,
        ))

        # fork1=(a) fix (test-surfaced): the structure_proposal pointer Artifact is written
        # [here] -- only when propose_structure [succeeds], coupling success + single delivery
        # (terminal_tool -> END guarantees termination right after success -> exactly one per Run). The verbatim proposal
        # is still captured verbatim by _persist_agent_step's ToolCall.input (unchanged); 1c reads
        # that ToolCall.input verbatim via Artifact.ref_id (same discipline as chapter_detect:188).
        if status_val == "ok":
            tc_row = db.get(ToolCall, tc_id)
            if tc_row is not None and tc_row.name == "propose_structure":
                db.add(Artifact(
                    run_id=run_id,
                    kind="structure_proposal",
                    ref_table="tool_call",
                    ref_id=tc_id,
                ))
    db.commit()
