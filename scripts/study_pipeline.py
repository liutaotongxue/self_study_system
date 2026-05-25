"""End-to-end one-command pipeline: ingest -> study_book -> build_kg -> backfill.

  python scripts/study_pipeline.py --pdf ~/Books/X.pdf --domain "RL" --chapter 4

Design notes:

  - document_id is not copy-pasted by the user. After ingest, look up the
    document by Document.file_path == abspath(--pdf) (same dedup key as
    ingest itself — deterministic, read-only, no stdout parsing).

  - Each step runs as a subprocess and its returncode is checked. If any
    gate fails (Eval-1 chapter detection / O4 note_ref integrity) the
    chain halts immediately, reports which step failed, and does not
    proceed on dirty data (exit 1).

  - study_book is NOT idempotent: seed_task is idempotent by title, but
    Notes are produced by the agent's save_note tool — one per run, with
    no (doc, chapter) upsert. Re-running accumulates rather than replaces
    (observed: doc1/ch1.3 grew to 7 Notes). A one-command wrapper that
    silently re-runs would corrupt the signal and add anchor ambiguity in
    the viewer. So step 2 runs an accumulation pre-check: STOP (exit 3)
    if Notes already exist; pass --allow-accumulate to explicitly bypass.
    build_kg upsert is first-wins idempotent (verified by O4), so it is
    not pre-checked. Properly fixing study_book re-run semantics
    (replace/version/skip) touches 6 acceptance points + O4 chap_note —
    deferred as a product decision; the orchestrator's job here is just
    to refuse to silently bite during self-use runs.

  - v1 only handles a single chapter. --chapter all is rejected explicitly:
    study_book/build_kg's discovery is LIKE "ch{X}.%", so passing "all"
    becomes "chall.%" which matches 0 rows -> study_book.py:126 loud
    sys.exit(1). The orchestrator exits earlier (exit 2) at step 1 to
    avoid the expensive whole-book ingest followed by a chain halt at
    step 2 with chunks present but no Notes. Cross-chapter "all" support
    is a separate feature.

  - Backfill is invoked with --use-embedding-fallback and
    --use-cross-note-fallback hard-coded as defaults (without them anchor
    coverage drops to ~89%; easy to forget).

  - At the end, probe :8000 and print the exact next-step URL.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sla.db import SessionLocal
from sla.models.domain import Document, Note

PY = sys.executable
ROOT = Path(__file__).parent.parent


def run(step_name, argv):
    print(f"\n{'='*60}\n[pipeline] {step_name}\n{'='*60}", flush=True)
    r = subprocess.run([PY, *argv], cwd=ROOT)
    if r.returncode != 0:
        print(f"\n[pipeline] STOP — {step_name} 失败(returncode={r.returncode})。"
              f"链中止,不续跑(不在坏数据上接力)。", file=sys.stderr)
        sys.exit(1)


def resolve_document_id(pdf_abs: str) -> int:
    db = SessionLocal()
    try:
        d = db.query(Document).filter(Document.file_path == pdf_abs).first()
        if d is None:
            print(f"[pipeline] STOP — ingest 后查不到 Document(file_path={pdf_abs})",
                  file=sys.stderr)
            sys.exit(1)
        return d.id
    finally:
        db.close()


def check_note_accumulation(doc_id: int, chapter: str, allow: bool):
    db = SessionLocal()
    try:
        n = db.query(Note).filter(
            Note.document_id == doc_id,
            Note.chapter_id.like(f"ch{chapter}.%")).count()
    finally:
        db.close()
    if n and not allow:
        print(f"[pipeline] STOP — doc={doc_id} ch{chapter} 已有 {n} 个 Note。"
              f"study_book 非幂等:继续将累加第 {n+1} 个不替换"
              f"(build_kg 幂等,不受影响)。污染自用信号 + viewer 锚点歧义。"
              f" --allow-accumulate 放行 / Ctrl-C / 先清。", file=sys.stderr)
        sys.exit(3)
    if n:
        print(f"[pipeline] WARN 累加模式:doc={doc_id} ch{chapter} 现有 {n} Note,将 +1")


def main():
    p = argparse.ArgumentParser(prog="study_pipeline")
    p.add_argument("--pdf", required=True)
    p.add_argument("--domain", required=True)
    p.add_argument("--chapter", required=True, help='单章如 "4"(不支持 "all")')
    p.add_argument("--force", action="store_true", help="透传 ingest:重写已有 chunks")
    p.add_argument("--sections", default=None, help='透传 study_book:如 "ch4.3,ch4.4"')
    p.add_argument("--allow-accumulate", action="store_true",
                   help="显式放行 study_book 非幂等累加(否则已有 Note 即 STOP)")
    p.add_argument("--skip-backfill", action="store_true")
    a = p.parse_args()

    if a.chapter.strip().lower() == "all":
        print("[pipeline] v1 拒 --chapter all(显式限制,见 docstring:避免全书"
              "ingest + step2 loud-fail 半态)", file=sys.stderr)
        sys.exit(2)
    pdf_abs = str(Path(a.pdf).expanduser().resolve())
    # --domain 折叠空白:防 shell 换行/多空格 paste artifact 静默造垃圾 Domain
    # (ingest_pdf 按 exact-name 建 Domain 无 normalize;入口工具职责挡在此,
    #  不碰 Eval-1-gated 的 ingest_pdf;ingest 裸跑同洞 = O9 类 latent/named)。
    domain = " ".join(a.domain.split())

    # 1. ingest(内含 Eval-1 章检测 gate;idempotent,先跑安全)
    ing = ["scripts/ingest_pdf.py", "--pdf", pdf_abs, "--domain", domain,
           "--chapter", a.chapter] + (["--force"] if a.force else [])
    run("1/4 ingest_pdf", ing)

    doc_id = resolve_document_id(pdf_abs)
    print(f"[pipeline] document_id={doc_id}(自动解析,无需肉眼抄)")

    # 非幂等预检:在 step2(study_book,唯一非幂等产 Note 步)之前
    check_note_accumulation(doc_id, a.chapter, a.allow_accumulate)

    # 2. study_book(Notes + Questions)
    sb = ["scripts/study_book.py", "--document-id", str(doc_id), "--chapter", a.chapter]
    if a.sections:
        sb += ["--sections", a.sections]
    run("2/4 study_book", sb)

    # 3. build_kg(upsert first-wins,幂等)
    run("3/4 build_kg",
        ["scripts/build_kg.py", "--document-id", str(doc_id), "--chapter", a.chapter])

    # 4. backfill(烤死好默认;内含 O4 note_ref gate)
    if a.skip_backfill:
        print("[pipeline] 跳过 backfill(--skip-backfill)")
    else:
        run("4/4 backfill_kg_anchor",
            ["scripts/backfill_kg_anchor.py", "--document-id", str(doc_id),
             "--use-embedding-fallback", "--use-cross-note-fallback"])

    # 末尾:下一步明确化
    # 注:/viewer 返 404 时 urlopen 抛 HTTPError → 误判"未起";v1 可接受
    #     (仅"下一步"提示,误报低害,非 blocker)。
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:8000/viewer", timeout=1)
        up = True
    except Exception:
        up = False
    print(f"\n{'='*60}\n[pipeline] 完成 document_id={doc_id} chapter={a.chapter}")
    if up:
        print("  viewer 在跑 → http://localhost:8000/viewer 切到该 chapter")
    else:
        print("  下一步:uvicorn sla.api.app:app --reload"
              "  然后开 http://localhost:8000/viewer")


if __name__ == "__main__":
    main()
