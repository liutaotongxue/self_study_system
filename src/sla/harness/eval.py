"""Phase 1B-4: rule-based Eval.

rule_evaluate(run_id) -> runs 6 rules, persists an EvalResult, returns eval_result_id.

Design notes:
  - Data source is entirely v4 tables: Run / Artifact / Note / Question / ToolResult
    (does not rely on LangGraph's checkpoint; Phase 1B-2's sync_state_to_db paves the way for this)
  - Each rule produces {rule, status('pass'/'fail'), actual, expected, suggestion?}
  - rule_details JSON stores {"checks": [...rule records...]}
  - rule_passed = all(status == 'pass')
  - The suggestion field on failing rules is phrased **for Claude** -- Phase 2 can feed it back directly

Not idempotent: evaluating the same run multiple times creates a new EvalResult row each time
(useful for comparing threshold-tuning effects over time).
"""
from sla.db import SessionLocal
from sla.models.domain import Note, Question
from sla.models.runtime import Artifact, EvalResult, Run, Step, ToolCall, ToolResult


# --------------------- rule thresholds (centralized for easy tuning) ---------------------
NOTE_MIN_LEN = 800
Q_MIN, Q_MAX = 3, 5


def rule_evaluate(run_id: int) -> int:
    """Run all rules against one Run, persist into the EvalResult table, return eval_result.id."""
    db = SessionLocal()
    try:
        run = db.get(Run, run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")

        # ---------- Collect data ----------
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

        # ---------- Run rules ----------
        checks: list[dict] = []
        checks.append(_check_run_status(run))
        checks.append(_check_note_count(notes))
        if notes:
            checks.append(_check_note_length(notes[0]))
        checks.append(_check_question_count(questions))
        if questions:
            checks.append(_check_questions_well_formed(questions))
        checks.append(_check_all_tools_ok(tool_results))

        # ---------- Aggregate and write ----------
        rule_passed = all(c["status"] == "pass" for c in checks)
        evr = EvalResult(
            run_id=run_id,
            rule_passed=rule_passed,
            rule_details={"checks": checks},
            llm_score=None,         # Filled by Phase 1B-6 LLM judge
            llm_rationale=None,
        )
        db.add(evr)
        db.commit()
        db.refresh(evr)
        return evr.id
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Per-rule implementations: return a unified dict structure
#
# Convention:
#   {
#     "rule":   "rule_name",
#     "status": "pass" | "fail",
#     "actual": ...,
#     ...rule-specific fields...,
#     "suggestion": "actionable advice phrased for Claude" or None
#   }
#
# The suggestion field must be Claude-readable -- Phase 2 will feed the suggestion for failing
# rules back to the agent for self-correction, so the wording must be clear and the action executable.
# --------------------------------------------------------------------------- #

def _check_run_status(run: Run) -> dict:
    ok = run.status == "completed"

    # Branch suggestions by status. Use descriptive wording; avoid verbs like 'retry' that the agent cannot act on.
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
