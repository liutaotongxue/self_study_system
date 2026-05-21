"""Phase 1B-2:Task → Run 端到端执行入口。

run_task(task_id) -> run_id:
  1. 读 Task → 构 Policy → 创建 Run(status='running', started_at=now)
  2. graph.stream 跑任务,每个 chunk 翻译成 Step / ToolCall / ToolResult 行
     - agent 节点 → 1 行 Step + N 行 ToolCall (按 AIMessage.tool_calls)
     - tools 节点 → 给每个 ToolMessage 写 1 行 ToolResult (按 tool_call_id 链回)
  3. policy_aware_tool_node 在 save_note / save_questions 成功后自己写 Artifact
     (不在这里管,见 policy.py)
  4. 异常分类:
     - GraphRecursionError → status='policy_halted'
     - 其他异常 → status='failed' (含 traceback)
     - 正常完成 → status='completed'  (v4 约定的字符串,不是 'succeeded')
  5. 写 Run.ended_at + Run.error,事务关闭

不幂等:每次跑创建新 Run 行(同 Task 多次 run 是合法的)。
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
    """S5:全书大纲文本(含 3+ 级,info-complete)。供学习 agent 定位本章在全书
    的位置。无 DS 行(legacy doc / 新书未标目录)返 None;调用方据此不注入。
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
    """端到端跑一个 Task,返回 run_id。

    调用方:
      from sla.harness.runner import run_task
      run_id = run_task(task_id=1)
    """
    # ---------- 加载 Task,创建 Run(status='running') ----------
    db = SessionLocal()
    try:
        task = db.get(Task, task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")

        # 反序列化 policy JSON → Policy 对象(字段名要和 Policy 类一致,seed_task.py 已对齐)
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

        # 在 session 关闭前把 task 字段拷出来,后面 graph 用
        system_prompt = task.system_prompt
        user_prompt = task.user_prompt
    finally:
        db.close()

    # 在 session 关闭前再拷一个 task.document_id (上面已拷 system/user_prompt)
    db2 = SessionLocal()
    try:
        task = db2.get(Task, task_id)
        task_document_id = task.document_id if task else None
    finally:
        db2.close()

    # ---------- 构造初始 state ----------
    # run_id 给 policy_aware_tool_node 写 Artifact 用
    # document_id 给工具 InjectedToolArg 过滤跨文档同 chapter_id 用
    # cache_control 启用 Anthropic prompt caching(每轮重发 system + tools 时命中)
    # S5:全书大纲作为第二条 SystemMessage 注入(plain text portable form,
    # per [[provider-portability-preference]] —— 用可移植 LangChain 抽象,不
    # 深耕 Anthropic-native multi-block cache_control;第一条 system 保留原
    # 形态以免扩 S5 scope)。无 DS 行(legacy doc / 未标目录)→ outline_text
    # 为 None 不注入(legacy 零回归)。
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

    # ---------- 跑 graph,翻译 stream → Step/ToolCall/ToolResult ----------
    # Weld A:bind 集 = policy.allowed_tools(单一真源,schema 随白名单)。
    # 现学习 Task seed allowed_tools 恰 4 → 解析回原 ALL_TOOLS → 字节同 bind;
    # 这是 runner 路径语义位移(非 no-op),回归冒烟须断言学习任务仍恰 4 工具。
    # terminal_tool:仅结构任务(allowed_tools 含 propose_structure)结构性终止;
    # 硬兜底仍是下方 _stream_and_persist 的 recursion_limit=max_steps*2。
    # max_tokens:policy.max_tokens or 4096 —— 学习任务无此键→None→4096(字节零回归);
    # 结构任务须高(propose_structure 一次吐 ~80 节,4096 会截断 tool 调用→空 args 死循环)
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
        # max_steps 超限被 LangGraph 中断
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
# 内部:stream → Step/ToolCall/ToolResult 翻译
# --------------------------------------------------------------------------- #

def _stream_and_persist(graph, initial_state: dict, run_id: int, policy: Policy):
    """跑 graph.stream,把每个 chunk 翻译成 Step/ToolCall/ToolResult 行。

    设计:
      - 每个 agent 节点 update → 1 行 Step (idx 递增) + N 行 ToolCall
      - 每个 tools 节点 update → 给每个 ToolMessage 写 1 行 ToolResult
      - pending_tool_calls 字典:从 anthropic_tool_use_id 找回 ToolCall.id
        (因为 ToolMessage 用 tool_call_id 链回,而 DB ToolResult 用 ToolCall.id)
    """
    step_idx = 0
    pending_tool_calls: dict[str, int] = {}  # anthropic_tool_use_id → ToolCall.id

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
    """写 1 行 Step + N 行 ToolCall。"""
    ai_msg = next((m for m in msgs if isinstance(m, AIMessage)), None)
    if ai_msg is None:
        return

    step = Step(
        run_id=run_id,
        idx=step_idx,
        # Phase 1B-2 暂不持久化 model_input/output(冗余 + 数据量大)。
        # 真要看每步 messages,可以读 ToolCall + ToolResult 链回构;Phase 2 再评估
        model_input=None,
        model_output=None,
    )
    db.add(step)
    db.flush()  # 拿到 step.id 用于 ToolCall.step_id

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
        # 注:verbatim 提议就在 tc_row.input(= tc["args"]),此处逐字捕获、单写者。
        # structure_proposal 指针 Artifact 不在此写 —— 见 _persist_tool_results
        # (须等 ToolResult status=ok 才知是否成功,call 时还不知 → 移到那)。

    db.commit()


def _persist_tool_results(
    db, run_id: int, msgs: list, pending_tool_calls: dict[str, int]
):
    """给每个 ToolMessage 写 1 行 ToolResult;并在 propose_structure 成功时
    写【唯一】structure_proposal 指针 Artifact(success-gated,见下)。"""
    for tm in msgs:
        if not isinstance(tm, ToolMessage):
            continue

        tc_id = pending_tool_calls.pop(tm.tool_call_id, None)
        if tc_id is None:
            # 不应该发生:每个 ToolMessage 都该对应上一步 agent 创建的 ToolCall。
            # 真出现说明 langgraph 版本/我们的逻辑有 bug,先 silent skip,以免一行
            # 异常炸掉整个 Run
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
            # content 是 JSON 列。tool 返回可能是 plain text 或 JSON 字符串,
            # parse 成功存 dict/list,失败保留原 str
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

        # fork1=(a) 修正(test-surfaced):structure_proposal 指针 Artifact 写在
        # 【此处】—— 只在 propose_structure【成功】时写,绑 success+单交付
        # (terminal_tool→END 保证成功后即终止 → 全 Run 恰一条)。verbatim 提议
        # 仍由 _persist_agent_step 的 ToolCall.input 逐字捕获(未变);1c 按
        # Artifact.ref_id 取该 ToolCall.input 逐字读(同 chapter_detect:188 纪律)。
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
