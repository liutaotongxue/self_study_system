"""Ingest a PDF into the database (Document + Chunks).

Usage:
  python scripts/ingest_pdf.py --pdf /path/to/book.pdf --domain "Reinforcement Learning" --chapter 1

Arguments:
  --pdf      <path>             Path to the PDF file
  --domain   <name>             Domain name (reused if exists, created otherwise)
  --chapter  <num|"all">        Parse a single chapter (1, 2, ...) or "all"
                                for the whole book (default: 1)
  --force                       Overwrite existing chunks for the same
                                (document_id, chapter_id)

Idempotency:
  - Domain deduped by name
  - Document deduped by file_path (re-running on the same PDF reuses the
    same document row)
  - Chunk deduped by (document_id, chapter_id); skipped by default if
    present, deleted-then-rewritten with --force

Output:
  Prints the created/reused domain_id and document_id, plus the number
  of chunks generated per section.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sla.db import SessionLocal  # noqa: E402
from sla.models.domain import Chunk, Document, Domain  # noqa: E402
from sla.parsing.chapter_detect import (  # noqa: E402
    SectionDetectionError,
    detect_sections,
    extract_toc_text,
    parse_toc_text,
    validate_sections,
)
from sla.parsing.chunker import normalize_text, split_into_chunks  # noqa: E402
from sla.parsing.pdf import extract_pages, join_pages_text  # noqa: E402


def main():
    parser = argparse.ArgumentParser(prog="ingest_pdf")
    parser.add_argument("--pdf", required=True, help="PDF 文件路径")
    parser.add_argument("--domain", required=True, help="Domain 名称")
    parser.add_argument(
        "--chapter", default="1",
        help='只解析某章("1" / "2" / ...)或 "all" 解全本(默认 "1")',
    )
    parser.add_argument(
        "--force", action="store_true",
        help="同 (document_id, chapter_id) 已有 chunks 时,删了再写",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        print(f"FAIL: PDF not found at {pdf_path}", file=sys.stderr)
        sys.exit(1)

    print(f"=== reading {pdf_path} ===")
    pages = extract_pages(pdf_path)
    print(f"  total pages: {len(pages)}")

    chapter_filter = None if args.chapter == "all" else args.chapter
    sections = detect_sections(pages, chapter_filter)
    if not sections:
        print(f"FAIL: no sections for chapter={args.chapter!r}",
              file=sys.stderr)
        sys.exit(1)

    try:
        validate_sections(sections, pages)
    except SectionDetectionError as e:
        print(f"FAIL section detection gate:\n{e}", file=sys.stderr)
        sys.exit(1)
    print(f"  detected sections: {len(sections)}")
    for s in sections:
        print(f"    {s.chapter_id:8s} {s.title!r} (book p.{s.book_page_start}-{s.book_page_end})")

    db = SessionLocal()
    try:
        # ---------- Domain 幂等 ----------
        domain = db.query(Domain).filter(Domain.name == args.domain).first()
        if domain:
            print(f"\n  reused domain id={domain.id} name={args.domain!r}")
        else:
            domain = Domain(name=args.domain)
            db.add(domain)
            db.commit()
            db.refresh(domain)
            print(f"\n  created domain id={domain.id} name={args.domain!r}")

        # ---------- Document 幂等(按 file_path) ----------
        file_path_str = str(pdf_path)
        document = (
            db.query(Document)
            .filter(Document.file_path == file_path_str)
            .first()
        )
        if document:
            print(f"  reused document id={document.id} title={document.title!r}")
        else:
            document = Document(
                domain_id=domain.id,
                title=pdf_path.stem,
                file_path=file_path_str,
                total_pages=len(pages),
                status="ready",
            )
            db.add(document)
            db.commit()
            db.refresh(document)
            print(f"  created document id={document.id} title={document.title!r}")

        # ---------- 逐节切 Chunk ----------
        total_chunks = 0
        for s in sections:
            existing = (
                db.query(Chunk)
                .filter(
                    Chunk.document_id == document.id,
                    Chunk.chapter_id == s.chapter_id,
                )
                .count()
            )
            if existing > 0:
                if not args.force:
                    print(
                        f"  skip {s.chapter_id}: {existing} chunks already exist "
                        "(use --force to rewrite)"
                    )
                    continue
                db.query(Chunk).filter(
                    Chunk.document_id == document.id,
                    Chunk.chapter_id == s.chapter_id,
                ).delete()
                print(f"  --force: deleted {existing} stale chunks for {s.chapter_id}")

            raw = join_pages_text(pages, s.pdf_page_start, s.pdf_page_end)
            cleaned = normalize_text(raw)
            texts = split_into_chunks(cleaned)
            if not texts:
                print(f"  WARN: {s.chapter_id} 切不出 chunk,跳过")
                continue

            for txt in texts:
                db.add(Chunk(
                    document_id=document.id,
                    chapter_id=s.chapter_id,
                    page_start=s.book_page_start,
                    page_end=s.book_page_end,
                    content=txt,
                ))
            print(f"  {s.chapter_id}: {len(texts)} chunks (book p.{s.book_page_start}-{s.book_page_end})")
            total_chunks += len(texts)

        # parsed_outline:全本 TOC 真章/节名(merge 累加;章级有则有、无则无;
        # 独立于 chunk skip;非关键 → 解析失败只 WARN 不阻断已验过的 ingest)
        try:
            _entries = parse_toc_text(extract_toc_text(pages))
            _outline = {f"ch{e['section_id']}": e["title"]
                        for e in _entries if e.get("title")}
            if _outline:
                _merged = dict(document.parsed_outline or {})
                _merged.update(_outline)
                document.parsed_outline = _merged
                print(f"  parsed_outline merged: +{len(_outline)} TOC titles "
                      f"→ total {len(_merged)}")
        except Exception as _e:
            print(f"  WARN: parsed_outline 未写入(TOC 解析失败:{_e}),不阻断 ingest")

        db.commit()
        print(f"\n[ok] ingest done: domain={domain.id} document={document.id} chunks_written={total_chunks}")
        print(f"  run_task 时使用 document_id={document.id}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
