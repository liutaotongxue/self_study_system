"""Harness 运行时 endpoints。

Phase 1A 写了所有 GET(只读),Phase 1B-5 加 2 个 POST(触发 run + 触发 eval)
+ 修了一个 GET eval 的排序 bug(有多条 EvalResult 时需要返回最新)。
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
    """完整 trace:run + 所有 steps + 每 step 的 tool_calls + 对应 tool_results。"""
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
    # 同一 Run 可能多次 rule_evaluate(改阈值/调 prompt 后重评),返回最新一条
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


# ---------- Phase 1B-5: 触发动作 ----------

@router.post("/tasks/{task_id}/runs", status_code=201)
def create_run(task_id: int, db: Session = Depends(get_db)):
    """同步触发一次 Run。Phase 1B 阻塞等跑完(~30-60 秒,~$0.1)。

    Phase 2 改异步(返回 202 + run_id,后台跑)。
    """
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(404, "task not found")

    # run_task 内部自己开关 session,不复用这里 dep-injected 的 db
    run_id = run_task(task_id=task_id)

    # run_task 在别的 session 里写了 Run 行,这里的 db 可能有缓存快照,先 expire
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
    """触发一次规则评估,创建新 EvalResult 行,返回该行。"""
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


# ---------- Phase 3-A 临时:本地 KG viewer ----------

@router.get("/viewer", response_class=FileResponse)
def kg_viewer():
    """本地 KG 可视化(三栏布局 + vis.js)。
    跑 uvicorn 后浏览器开 http://localhost:8000/viewer 看图。
    document_id 当前 hardcode = 2(Sutton & Barto),Phase 3 真做产品时改成 path param。
    """
    from pathlib import Path
    viewer_path = Path(__file__).parent.parent.parent.parent / "web" / "index.html"
    return FileResponse(viewer_path, media_type="text/html")


# ---------- Phase 2-W3-7: KG endpoint ----------

@router.get("/documents/{document_id}/kg")
def get_document_kg(document_id: int):
    """返回 document 的 KG —— 原生 schema(3 node types + 4 edge types)。

    无 lossy 适配。消费者(任何 viewer/Anki/Notion 等)按这个格式适配,
    我们不为单一 downstream 做格式妥协。
    详细 schema 见 src/sla/harness/kg_export.py:to_native_json 的 docstring。
    """
    payload = to_native_json(document_id)
    # 仅 "document 不存在" 才 404(to_native_json 此时返 document_title=None);
    # 空 KG 是正常态(S2 标完目录、S4 尚未标内容)→ 200 + 空数组,让前端走
    # viewer 主流程 + 空态横幅,不再退化到 chap-fs 旁路(S4 pivot 后该旁路废)。
    if payload["document_title"] is None:
        raise HTTPException(404, f"document_id={document_id} not found")
    return payload
