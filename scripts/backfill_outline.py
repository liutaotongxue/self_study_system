"""只解析原书 TOC → 写 Document.parsed_outline(零碰 chunk、零 O1/O2 gate)。

为何独立于 ingest_pdf(承重安全,非可读性):重跑 `ingest_pdf --chapter all`
会(a)跑 whole-book O1/O2 gate,未验章节 TOC 触雷即 sys.exit、parsed_outline
连坐没写成;(b)对尚无 chunk 的章全量灌入(非预期大写入 + 新 gate 风险)。
本脚本只 extract_pages → parse_toc_text → merge 写 parsed_outline,零这两个耦合。

  python scripts/backfill_outline.py --document-id 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sla.db import SessionLocal  # noqa: E402
from sla.models.domain import Document  # noqa: E402
from sla.parsing.chapter_detect import extract_toc_text, parse_toc_text  # noqa: E402
from sla.parsing.pdf import extract_pages  # noqa: E402


def main():
    p = argparse.ArgumentParser(prog="backfill_outline")
    p.add_argument("--document-id", type=int, required=True)
    a = p.parse_args()

    db = SessionLocal()
    try:
        doc = db.get(Document, a.document_id)
        if doc is None:
            print(f"FAIL: document {a.document_id} not found", file=sys.stderr)
            sys.exit(1)
        if not doc.file_path:
            print(f"FAIL: document {a.document_id} 无 file_path", file=sys.stderr)
            sys.exit(1)

        pdf_path = Path(doc.file_path).expanduser()
        if not pdf_path.exists():
            print(f"FAIL: PDF not found at {pdf_path}", file=sys.stderr)
            sys.exit(1)

        print(f"=== reading TOC from {pdf_path} ===")
        pages = extract_pages(pdf_path)
        entries = parse_toc_text(extract_toc_text(pages))
        outline = {f"ch{e['section_id']}": e["title"]
                   for e in entries if e.get("title")}
        if not outline:
            print("FAIL: 解析不出任何 TOC 标题(TOC 排版异常?)", file=sys.stderr)
            sys.exit(1)

        merged = dict(doc.parsed_outline or {})
        before = len(merged)
        merged.update(outline)
        doc.parsed_outline = merged
        db.commit()

        chap_lvl = sum(1 for k in outline if "." not in k)
        print(f"[ok] parsed_outline 写入 document={doc.id}: {len(outline)} 条"
              f"(章级 {chap_lvl} / 节级 {len(outline) - chap_lvl})"
              f";原 {before} → 现 {len(merged)}")
        for k in sorted(outline):
            print(f"  {k}: {outline[k]}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
