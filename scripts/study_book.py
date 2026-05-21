"""Phase 2-W2-1:多章节 orchestrator —— 自动学完某 document 的某章全部节。

跑法:
  python scripts/study_book.py --document-id 2 --chapter 1
    # 自动学完 document=2 的 ch1.1..ch1.8

  python scripts/study_book.py --document-id 2 --chapter 1 --sections ch1.3,ch1.4
    # 只学指定的几节

成本:每节 ~$0.10-0.15;一章 8 节 ~$0.80-1.00。
时间:每节 30-60 秒,串行跑,8 节约 5-8 分钟。

流程:
  1. discover_chapters:从 DB 找 (document_id, chapter prefix) 下所有 chapter_id
  2. 逐节(--sections 没指定就是全部):
     - seed_task_for_chapter:幂等创建 Task
     - run_task:跑完一个 Run + 落 Step/ToolCall/Artifact
     - rule_evaluate:6 条 rule,落 EvalResult
  3. 末尾汇总:几节通过、失败原因

不并发(Phase 2 单用户场景,串行就够)。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sla.config import settings  # noqa: E402

from sla.db import SessionLocal  # noqa: E402
from sla.harness.eval import rule_evaluate  # noqa: E402
from sla.harness.prompts import LEARNING_AGENT_SYSTEM  # noqa: E402
from sla.harness.runner import run_task  # noqa: E402
from sla.models.domain import Chunk  # noqa: E402
from sla.models.runtime import EvalResult, Run, Task  # noqa: E402


def discover_chapters(document_id: int, chapter_prefix: str) -> list[str]:
    """从 chunk 表 distinct 出 (document, chapter_prefix) 下所有 chapter_id。

    e.g. discover_chapters(2, '1') → ['ch1.1', 'ch1.2', ..., 'ch1.8']
    """
    db = SessionLocal()
    try:
        rows = (
            db.query(Chunk.chapter_id)
            .filter(Chunk.document_id == document_id)
            .filter(Chunk.chapter_id.like(f"ch{chapter_prefix}.%"))
            .distinct()
            .order_by(Chunk.chapter_id)
            .all()
        )
        return [r[0] for r in rows]
    finally:
        db.close()


def seed_task_for_chapter(document_id: int, chapter_id: str) -> int:
    """幂等地为给定章节创建 Task,返回 task_id。

    Title 唯一(同 document_id + chapter_id 一对一),所以重跑不重复建。
    """
    db = SessionLocal()
    try:
        title = f"Study {chapter_id} of document {document_id}"
        existing = db.query(Task).filter(Task.title == title).first()
        if existing:
            return existing.id
        task = Task(
            document_id=document_id,
            kind="study_chapter",
            title=title,
            description=f"Study chapter {chapter_id} from document {document_id}.",
            system_prompt=LEARNING_AGENT_SYSTEM,
            user_prompt=(
                f"请阅读 chapter_id={chapter_id} 这一章,完成学习笔记和思考题。"
            ),
            policy={
                # read_scope 只允许本节,policy_aware_tool_node 阻止越界
                "read_scope": [chapter_id],
                # 长章节可能需要更多 step(ch1.7 有 44 chunks)
                "max_steps": 50,
                "allowed_tools": [
                    "list_chunks", "read_chunk", "save_note", "save_questions",
                ],
                "max_artifacts": {"note": 1, "question_batch": 1},
            },
            status="pending",
        )
        db.add(task)
        db.commit()
        db.refresh(task)
        return task.id
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(prog="study_book")
    parser.add_argument("--document-id", type=int, required=True, help="目标 document.id")
    parser.add_argument(
        "--chapter", default="1",
        help="章节前缀(e.g. '1' 匹配 ch1.x;默认 '1')",
    )
    parser.add_argument(
        "--sections", default=None,
        help="逗号分隔的具体 chapter_id(覆盖 chapter 自动发现)",
    )
    args = parser.parse_args()

    if not settings.anthropic_api_key:
        print("FAIL: ANTHROPIC_API_KEY is empty", file=sys.stderr)
        sys.exit(1)

    # ---------- 1. 选要学的 sections ----------
    if args.sections:
        sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    else:
        sections = discover_chapters(args.document_id, args.chapter)
    if not sections:
        print(
            f"FAIL: no sections found for document_id={args.document_id} chapter={args.chapter!r}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"=== will study {len(sections)} section(s) in chapter {args.chapter!r} ===")
    for s in sections:
        print(f"  {s}")

    # ---------- 2. 逐节跑 ----------
    results: list[dict] = []
    for i, chapter_id in enumerate(sections, 1):
        print(f"\n{'=' * 60}\n[{i}/{len(sections)}] {chapter_id}\n{'=' * 60}")
        task_id = seed_task_for_chapter(args.document_id, chapter_id)
        print(f"  task_id={task_id}, calling run_task...")
        try:
            run_id = run_task(task_id=task_id)
        except Exception as e:
            print(f"  run_task FAILED: {type(e).__name__}: {e}")
            results.append({
                "chapter_id": chapter_id,
                "task_id": task_id,
                "run_id": None,
                "status": "exception",
                "eval_id": None,
                "rule_passed": False,
                "failed_rules": [f"run_task threw: {type(e).__name__}"],
            })
            continue

        eval_id = rule_evaluate(run_id=run_id)

        # 读 status 和 eval 结果
        db = SessionLocal()
        try:
            run = db.get(Run, run_id)
            evr = db.get(EvalResult, eval_id)
            failed_rules = [
                c["rule"] for c in evr.rule_details["checks"]
                if c["status"] == "fail"
            ]
            results.append({
                "chapter_id": chapter_id,
                "task_id": task_id,
                "run_id": run_id,
                "status": run.status,
                "eval_id": eval_id,
                "rule_passed": evr.rule_passed,
                "failed_rules": failed_rules,
            })
            mark = "[PASS]" if evr.rule_passed else "[FAIL]"
            print(f"  → run.status={run.status}  eval={mark}  failed={failed_rules}")
        finally:
            db.close()

    # ---------- 3. 汇总 ----------
    print(f"\n\n{'=' * 60}\nSummary\n{'=' * 60}")
    completed = sum(1 for r in results if r["status"] == "completed")
    rule_passed = sum(1 for r in results if r["rule_passed"])
    print(f"  sections studied:   {len(results)}")
    print(f"  Run.status=completed: {completed}/{len(results)}")
    print(f"  EvalResult.rule_passed: {rule_passed}/{len(results)}")
    print()
    for r in results:
        mark = "[PASS]" if r["rule_passed"] else "[FAIL]"
        print(
            f"  {mark} {r['chapter_id']:8s}  task={r['task_id']:3d}  "
            f"run={str(r['run_id']):>5s}  status={r['status']}"
        )
        if r["failed_rules"]:
            print(f"          failed_rules: {r['failed_rules']}")

    print(f"\n[ok] study_book done")


if __name__ == "__main__":
    main()
