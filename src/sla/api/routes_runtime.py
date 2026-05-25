"""Harness runtime endpoints.

Phase 1A wrote all GETs (read-only); Phase 1B-5 added 2 POSTs (trigger run + trigger eval)
plus fixed an ordering bug in GET eval (must return the latest when there are multiple EvalResult rows).
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from sla.db import get_db
from sla.harness.eval import rule_evaluate
from sla.harness.kg_export import to_native_json
from sla.harness.runner import run_task
from sla.models.runtime import Artifact, EvalResult, Run, Step, Task

router = APIRouter()


# ---------- Task ----------

@router.get("/tasks")
def list_tasks(db: Session = Depends(get_db)):
    rows = db.query(Task).order_by(Task.id).all()
    return [
        {"id": t.id, "kind": t.kind, "title": t.title, "status": t.status, "document_id": t.document_id}
        for t in rows
    ]


@router.get("/tasks/{task_id}")
def get_task(task_id: int, db: Session = Depends(get_db)):
    t = db.get(Task, task_id)
    if not t:
        raise HTTPException(404, "task not found")
    return {
        "id": t.id, "kind": t.kind, "title": t.title, "description": t.description,
        "system_prompt": t.system_prompt, "user_prompt": t.user_prompt,
        "policy": t.policy, "status": t.status, "document_id": t.document_id,
    }


# ---------- Run ----------

@router.get("/runs/{run_id}")
def get_run(run_id: int, db: Session = Depends(get_db)):
    r = db.get(Run, run_id)
    if not r:
        raise HTTPException(404, "run not found")
    return {
        "id": r.id, "task_id": r.task_id, "status": r.status,
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "ended_at": r.ended_at.isoformat() if r.ended_at else None,
        "error": r.error,
    }


@router.get("/runs/{run_id}/trace")
def get_run_trace(run_id: int, db: Session = Depends(get_db)):
    """Full trace: run + all steps + each step's tool_calls + corresponding tool_results."""
    r = db.get(Run, run_id)
    if not r:
        raise HTTPException(404, "run not found")

    steps_payload = []
    for s in r.steps:
        tcs = []
        for tc in s.tool_calls:
            tr = tc.result
            tcs.append({
                "id": tc.id, "name": tc.name, "input": tc.input,
                "anthropic_tool_use_id": tc.anthropic_tool_use_id,
                "result": (
                    {"status": tr.status, "content": tr.content, "reason": tr.reason}
                    if tr else None
                ),
            })
        steps_payload.append({
            "id": s.id, "idx": s.idx,
            "model_input": s.model_input, "model_output": s.model_output,
            "tool_calls": tcs,
        })

    return {
        "run": {
            "id": r.id, "task_id": r.task_id, "status": r.status,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "ended_at": r.ended_at.isoformat() if r.ended_at else None,
            "error": r.error,
        },
        "steps": steps_payload,
    }


@router.get("/runs/{run_id}/artifacts")
def list_run_artifacts(run_id: int, db: Session = Depends(get_db)):
    rows = db.query(Artifact).filter(Artifact.run_id == run_id).order_by(Artifact.id).all()
    return [
        {"id": a.id, "kind": a.kind, "ref_table": a.ref_table, "ref_id": a.ref_id}
        for a in rows
    ]


@router.get("/runs/{run_id}/eval")
def get_run_eval(run_id: int, db: Session = Depends(get_db)):
    # The same Run may be rule_evaluate'd multiple times (re-eval after threshold/prompt tweaks); return the latest
    e = (
        db.query(EvalResult)
        .filter(EvalResult.run_id == run_id)
        .order_by(EvalResult.id.desc())
        .first()
    )
    if not e:
        return None
    return {
        "id": e.id, "run_id": e.run_id,
        "rule_passed": e.rule_passed, "rule_details": e.rule_details,
        "llm_score": e.llm_score, "llm_rationale": e.llm_rationale,
        "created_at": e.created_at.isoformat(),
    }


# ---------- Phase 1B-5: trigger actions ----------

@router.post("/tasks/{task_id}/runs", status_code=201)
def create_run(task_id: int, db: Session = Depends(get_db)):
    """Synchronously trigger one Run. Phase 1B blocks until it completes (~30-60 sec, ~$0.1).

    Phase 2 will switch to async (return 202 + run_id, run in background).
    """
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(404, "task not found")

    # run_task opens/closes its own session internally; does not reuse the dep-injected db here
    run_id = run_task(task_id=task_id)

    # run_task wrote the Run row in another session; this db may have a cached snapshot, so expire first
    db.expire_all()
    r = db.get(Run, run_id)
    return {
        "id": r.id, "task_id": r.task_id, "status": r.status,
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "ended_at": r.ended_at.isoformat() if r.ended_at else None,
        "error": r.error,
    }


@router.post("/runs/{run_id}/eval", status_code=201)
def create_run_eval(run_id: int, db: Session = Depends(get_db)):
    """Trigger one rule-based evaluation, create a new EvalResult row, return that row."""
    r = db.get(Run, run_id)
    if r is None:
        raise HTTPException(404, "run not found")

    eval_id = rule_evaluate(run_id=run_id)

    db.expire_all()
    e = db.get(EvalResult, eval_id)
    return {
        "id": e.id, "run_id": e.run_id,
        "rule_passed": e.rule_passed, "rule_details": e.rule_details,
        "llm_score": e.llm_score, "llm_rationale": e.llm_rationale,
        "created_at": e.created_at.isoformat(),
    }


# ---------- Phase 3-A temporary: local KG viewer ----------

@router.get("/viewer", response_class=FileResponse)
def kg_viewer():
    """Local KG visualization (three-column layout + vis.js).
    After running uvicorn, open http://localhost:8000/viewer in the browser to view the graph.
    document_id is currently hardcoded = 2 (Sutton & Barto); change to a path param in Phase 3 when productizing.
    """
    from pathlib import Path
    viewer_path = Path(__file__).parent.parent.parent.parent / "web" / "index.html"
    return FileResponse(viewer_path, media_type="text/html")


# ---------- Phase 2-W3-7: KG endpoint ----------

@router.get("/documents/{document_id}/kg")
def get_document_kg(document_id: int):
    """Return the document's KG -- native schema (3 node types + 4 edge types).

    No lossy adaptation. Consumers (any viewer / Anki / Notion / etc.) adapt to this format;
    we do not compromise the format for any single downstream.
    See src/sla/harness/kg_export.py:to_native_json docstring for the detailed schema.
    """
    payload = to_native_json(document_id)
    # Only "document does not exist" should 404 (to_native_json returns document_title=None in that case);
    # an empty KG is a normal state (S2 done labeling TOC, S4 not yet labeled content) -> 200 + empty arrays,
    # letting the frontend run the viewer main flow + empty-state banner, no longer falling back to the
    # chap-fs side path (that side path is retired after the S4 pivot).
    if payload["document_title"] is None:
        raise HTTPException(404, f"document_id={document_id} not found")
    return payload
