"""P3b runner:UI 触发的章节 生成/重建。被 routes_generation Popen detached 起。

  python scripts/run_generation.py --job-id J [--dry-run]

generate: study_book(--sections 锁单章) → build_kg(--notes 锁单章) → backfill(整 doc)
rebuild : 跳 study_book;build_kg(--notes) → backfill
chapter_id 用【精确等值 ==】查 note ids(非前缀 LIKE,否则 ch3.1 误吞 ch3.10
等,跨章累加灾难从查询后门复活)。backfill 整 doc(幂等,无单章义)。
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sla.db import SessionLocal  # noqa: E402
from sla.models.domain import Document, Domain, Note  # noqa: E402
from sla.models.runtime import GenerationJob  # noqa: E402

ROOT = Path(__file__).parent.parent
PY = sys.executable


def _set(db, job, **kw):
    for k, v in kw.items():
        setattr(job, k, v)
    db.commit()


def _step(db, job, name, argv, dry):
    _set(db, job, step=name)
    if dry:
        import time
        time.sleep(1)
        return
    r = subprocess.run([PY, *argv], cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-800:]
        _set(db, job, status="failed",
             detail=f"{name} rc={r.returncode}\n{tail}", ended_at=datetime.utcnow())
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(prog="run_generation")
    p.add_argument("--job-id", type=int, required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    db = SessionLocal()
    try:
        job = db.get(GenerationJob, a.job_id)
        if job is None:
            print(f"FAIL: generation job {a.job_id} not found", file=sys.stderr)
            sys.exit(1)
        _set(db, job, status="running", pid=os.getpid())

        doc, ch, mode = job.document_id, job.chapter_id, job.mode

        if mode == "ingest":
            d = db.get(Document, doc)
            if d is None or not d.file_path:
                _set(db, job, status="failed",
                     detail=f"document {doc} 无 file_path,无法增量 ingest",
                     ended_at=datetime.utcnow())
                sys.exit(1)
            dom = db.get(Domain, d.domain_id)
            _step(db, job, "ingest",
                  ["scripts/ingest_pdf.py", "--pdf", d.file_path,
                   "--domain", dom.name if dom else "", "--chapter", ch], a.dry_run)
            d.status = "ready"   # P2:ingest 成功 → 标记已处理(整本/单章都置;§32 残差:严格='跑过 ingest')
            _set(db, job, status="done", step=None, ended_at=datetime.utcnow())
            print(f"[run_generation] job {a.job_id} (ingest ch{ch}) done")
            return

        if mode == "extract_toc":
            # human-anchored S2:人选目录页 → 一发结构化 LLM(非 agent,
            # 不 import graph)→ 写 document_structure。ch 形如 "toc:7-10[!force]"
            # 不可走下面的 ch[2:] 解析,故早返,放 ingest 之后、major 之前。
            d = db.get(Document, doc)
            if d is None or not d.file_path:
                _set(db, job, status="failed",
                     detail=f"document {doc} 无 file_path,无法抽结构",
                     ended_at=datetime.utcnow())
                sys.exit(1)
            from sla.parsing.structure_extract import (  # lazy:langchain 重
                extract_and_persist,
                parse_toc_payload,
            )
            try:
                tpages, tforce = parse_toc_payload(ch)
                _set(db, job, step="extract")
                r = extract_and_persist(db, doc, tpages, tforce)
            except Exception as e:
                _set(db, job, status="failed",
                     detail=f"extract_toc: {type(e).__name__}: {e}"[:800],
                     ended_at=datetime.utcnow())
                sys.exit(1)
            if r.get("result") == "skipped_existing":
                detail = (f"已有 {r['existing']} 行结构未覆盖;"
                          f"重抽请勾强制重抽(force)")
            else:
                detail = (f"抽出 {r['inserted']} 节"
                          f"(跳无编号 {r['unkeyed']},去重 {r['dups_dropped']})"
                          f";源页 {tpages};请到状态页核对")
            _set(db, job, status="done", step=None,
                 detail=detail, ended_at=datetime.utcnow())
            print(f"[run_generation] job {a.job_id} (extract_toc) done: {detail}")
            return

        if mode == "extract_content":
            # human-anchored S4:人选该节内容页 → 单发视觉 OCR + 切块写 Chunk
            # ch 形如 "ch1.2|p=50-65[!force]";不可走 ch[2:] 解析,故早返。
            d = db.get(Document, doc)
            if d is None or not d.file_path:
                _set(db, job, status="failed",
                     detail=f"document {doc} 无 file_path,无法抽内容",
                     ended_at=datetime.utcnow())
                sys.exit(1)
            from sla.parsing.content_extract import (  # lazy:langchain 重
                extract_and_chunk,
                parse_content_payload,
            )
            try:
                ch_real, cpages, cforce = parse_content_payload(ch)
                _set(db, job, step="ocr")
                r = extract_and_chunk(db, doc, ch_real, cpages, cforce)
            except Exception as e:
                _set(db, job, status="failed",
                     detail=f"extract_content: {type(e).__name__}: {e}"[:800],
                     ended_at=datetime.utcnow())
                sys.exit(1)
            if r.get("result") == "skipped_existing":
                detail = (f"{ch_real} 已有 {r['existing']} chunks 未覆盖;"
                          f"重抽请勾强制重抽(force)")
            else:
                detail = (f"{ch_real} OCR + 切块出 {r['inserted']} chunks"
                          f"({r['chars']} 字符);源页 {cpages}")
            _set(db, job, status="done", step=None,
                 detail=detail, ended_at=datetime.utcnow())
            print(f"[run_generation] job {a.job_id} "
                  f"(extract_content {ch_real}) done: {detail}")
            return

        major = ch[2:].split(".")[0] if ch.startswith("ch") else ch.split(".")[0]

        if mode == "generate":
            _step(db, job, "study_book",
                  ["scripts/study_book.py", "--document-id", str(doc),
                   "--chapter", major, "--sections", ch], a.dry_run)

        # chapter_id 精确等值(== 非 LIKE;后门也堵死,不让跨章从查询复活)
        note_ids = []
        if not a.dry_run:
            note_ids = [
                str(n.id) for n in db.query(Note).filter(
                    Note.document_id == doc, Note.chapter_id == ch).all()
            ]
            if not note_ids:
                _set(db, job, status="failed",
                     detail=f"no Note for (doc={doc}, chapter_id={ch}) — study_book 未产出?",
                     ended_at=datetime.utcnow())
                sys.exit(1)

        # --clear:删该 chapter 现有 KG 后再抽,防多次 rebuild 累积。generate
        # 路径首次跑也 --clear(no-op 若无旧)是 idempotency 防御;rebuild 路径必须。
        _step(db, job, "build_kg",
              ["scripts/build_kg.py", "--document-id", str(doc),
               "--notes", ",".join(note_ids), "--clear"], a.dry_run)
        _step(db, job, "backfill",
              ["scripts/backfill_kg_anchor.py", "--document-id", str(doc),
               "--use-embedding-fallback", "--use-cross-note-fallback"], a.dry_run)

        _set(db, job, status="done", step=None, ended_at=datetime.utcnow())
        print(f"[run_generation] job {a.job_id} ({mode} {ch}) done")
    finally:
        db.close()


if __name__ == "__main__":
    main()
