"""Phase 1B-4:规则段 Eval。

rule_evaluate(run_id) → 跑 6 条 rule,落 EvalResult,返回 eval_result_id。

设计要点:
  - 数据源全部从 v4 表读:Run / Artifact / Note / Question / ToolResult
    (不依赖 LangGraph 的 checkpoint;Phase 1B-2 的 sync_state_to_db 就是为这一步铺路)
  - 每条 rule 产出 {rule, status('pass'/'fail'), actual, expected, suggestion?}
  - rule_details JSON 存 {"checks": [...rule records...]}
  - rule_passed = all(status == 'pass')
  - 失败 rule 的 suggestion 字段措辞**面向 Claude**——Phase 2 可直接作为反馈回注

不幂等:同一 run 多次评估,每次新建 EvalResult 行(便于跨时间对比阈值调整效果)。
"""
from sla.db import SessionLocal
from sla.models.domain import Note, Question
from sla.models.runtime import Artifact, EvalResult, Run, Step, ToolCall, ToolResult


# --------------------- rule thresholds (集中放,方便调) ---------------------
NOTE_MIN_LEN = 800
Q_MIN, Q_MAX = 3, 5


def rule_evaluate(run_id: int) -> int:
    """对一个 Run 跑全部 rule,落 EvalResult 表,返回 eval_result.id。"""
    db = SessionLocal()
    try:
        run = db.get(Run, run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")

        # ---------- 收集数据 ----------
        artifacts = db.query(Artifact).filter(Artifact.run_id == run_id).all()
        note_refs = [a.ref_id for a in artifacts if a.kind == "note"]
        q_refs = [a.ref_id for a in artifacts if a.kind == "question"]

        notes: list[Note] = [n for n in (db.get(Note, i) for i in note_refs) if n is not None]
        questions: list[Question] = [
            q for q in (db.get(Question, i) for i in q_refs) if q is not None
        ]

        tool_results = (
            db.query(ToolResult)
            .join(ToolCall, ToolResult.tool_call_id == ToolCall.id)
            .join(Step, ToolCall.step_id == Step.id)
            .filter(Step.run_id == run_id)
            .all()
        )

        # ---------- 跑 rule ----------
        checks: list[dict] = []
        checks.append(_check_run_status(run))
        checks.append(_check_note_count(notes))
        if notes:
            checks.append(_check_note_length(notes[0]))
        checks.append(_check_question_count(questions))
        if questions:
            checks.append(_check_questions_well_formed(questions))
        checks.append(_check_all_tools_ok(tool_results))

        # ---------- 汇总写表 ----------
        rule_passed = all(c["status"] == "pass" for c in checks)
        evr = EvalResult(
            run_id=run_id,
            rule_passed=rule_passed,
            rule_details={"checks": checks},
            llm_score=None,         # Phase 1B-6 LLM judge 填
            llm_rationale=None,
        )
        db.add(evr)
        db.commit()
        db.refresh(evr)
        return evr.id
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# 各 rule 实现:返回统一 dict 结构
#
# 约定:
#   {
#     "rule":   "rule_name",
#     "status": "pass" | "fail",
#     "actual": ...,
#     ...各 rule 自己的字段...,
#     "suggestion": "面向 Claude 的可执行建议"或 None
#   }
#
# suggestion 字段措辞要让 Claude 看得懂——Phase 2 会把失败 rule 的 suggestion
# 喂回 agent 做自我纠错,所以语义要清楚、动作要可执行。
# --------------------------------------------------------------------------- #

def _check_run_status(run: Run) -> dict:
    ok = run.status == "completed"

    # 按 status 分类给 suggestion。用描述性措辞,不用 'retry' 这种 agent 做不到的动词。
    if ok:
        suggestion = None
    elif run.status == "policy_halted":
        suggestion = (
            "The agent was interrupted by the recursion limit before finishing. "
            "Either max_steps is too small for this task's expected length, or the agent "
            "was making redundant calls. Inspect the trace to see what step it reached."
        )
    elif run.status == "failed":
        suggestion = (
            "The agent terminated with an exception (see Run.error for the trace). "
            "Common causes: API errors, schema mismatch in tool input/output, or a bug "
            "in policy_aware_tool_node."
        )
    elif run.status == "running":
        suggestion = (
            "The run is still in progress. Wait for it to finish before evaluating."
        )
    else:
        suggestion = f"Run has unexpected status={run.status!r}; check Run.error."

    return {
        "rule": "run_completed",
        "status": "pass" if ok else "fail",
        "actual": run.status,
        "expected": "completed",
        "suggestion": suggestion,
    }


def _check_note_count(notes: list[Note]) -> dict:
    ok = len(notes) == 1
    return {
        "rule": "note_count_eq_1",
        "status": "pass" if ok else "fail",
        "actual": len(notes),
        "expected": 1,
        "suggestion": (
            None if ok else
            "Produce exactly one Markdown note after reading all chunks. "
            f"Currently {len(notes)} note(s) recorded for this run."
        ),
    }


def _check_note_length(note: Note) -> dict:
    n = len(note.content_md)
    ok = n >= NOTE_MIN_LEN
    return {
        "rule": f"note_length_min_{NOTE_MIN_LEN}",
        "status": "pass" if ok else "fail",
        "actual": n,
        "expected_min": NOTE_MIN_LEN,
        "suggestion": (
            None if ok else
            f"Note is too short ({n} chars, need >={NOTE_MIN_LEN}). "
            "Expand the note to cover all key concepts in the chapter "
            "(definitions, relationships, author's main arguments)."
        ),
    }


def _check_question_count(questions: list[Question]) -> dict:
    n = len(questions)
    ok = Q_MIN <= n <= Q_MAX
    return {
        "rule": f"question_count_in_{Q_MIN}_{Q_MAX}",
        "status": "pass" if ok else "fail",
        "actual": n,
        "expected_min": Q_MIN,
        "expected_max": Q_MAX,
        "suggestion": (
            None if ok else
            f"Got {n} questions, expected between {Q_MIN} and {Q_MAX}. "
            "Aim for a balanced set covering the chapter's main concepts."
        ),
    }


def _check_questions_well_formed(questions: list[Question]) -> dict:
    bad_ids = [
        q.id for q in questions
        if (not q.content) or (q.tags is None) or (len(q.tags) == 0)
    ]
    ok = not bad_ids
    return {
        "rule": "all_questions_have_content_and_tags",
        "status": "pass" if ok else "fail",
        "actual": f"{len(questions) - len(bad_ids)}/{len(questions)} well-formed",
        "bad_question_ids": bad_ids or None,
        "suggestion": (
            None if ok else
            f"Questions {bad_ids} are missing content or tags. "
            "Each question must have a non-empty content string AND a non-empty tags array."
        ),
    }


def _check_all_tools_ok(tool_results: list[ToolResult]) -> dict:
    bad = [r for r in tool_results if r.status != "ok"]
    ok = not bad
    statuses = sorted({r.status for r in bad}) if bad else []
    return {
        "rule": "all_tool_results_ok",
        "status": "pass" if ok else "fail",
        "actual": f"{len(tool_results) - len(bad)}/{len(tool_results)} ok",
        "non_ok_statuses": statuses or None,
        "suggestion": (
            None if ok else
            f"Some tool calls returned non-ok status ({statuses}). "
            "Check ToolResult.reason for each failure; common causes are policy denial "
            "(adjust your tool arguments to fit read_scope/quotas) or runtime errors."
        ),
    }
