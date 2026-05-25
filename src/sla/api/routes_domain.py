"""Application-layer read-only endpoints (Phase 1A scope)."""
import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from sqlalchemy import func
from sqlalchemy.orm import Session

from sla.api.routes_generation import UPLOAD_DIR
from sla.db import get_db
from sla.models.domain import (
    Chunk,
    Document,
    DocumentStructure,
    Domain,
    Note,
    Question,
)
from sla.models.kg import KGEdge, KGNode
from sla.models.runtime import GenerationJob, Task, reconcile_generation_jobs

router = APIRouter()


# ---------- Domain ----------

@router.get("/domains")
def list_domains(db: Session = Depends(get_db)):
    rows = db.query(Domain).order_by(Domain.id).all()
    return [{"id": d.id, "name": d.name, "description": d.description} for d in rows]


@router.get("/domains/{domain_id}")
def get_domain(domain_id: int, db: Session = Depends(get_db)):
    d = db.get(Domain, domain_id)
    if not d:
        raise HTTPException(404, "domain not found")
    return {"id": d.id, "name": d.name, "description": d.description}


# ---------- Document ----------

@router.get("/documents")
def list_documents(domain_id: int | None = None, db: Session = Depends(get_db)):
    q = db.query(Document)
    if domain_id is not None:
        q = q.filter(Document.domain_id == domain_id)
    rows = q.order_by(Document.id).all()
    # has_structure: whether the doc has had its TOC annotated (document_structure has rows).
    # Single aggregate query to avoid N+1; frontend uses this in place of technical metadata like "page count / status / id".
    ds_counts = dict(
        db.query(DocumentStructure.document_id,
                 func.count(DocumentStructure.id))
        .group_by(DocumentStructure.document_id).all()
    )
    return [
        {"id": d.id, "domain_id": d.domain_id, "title": d.title,
         "total_pages": d.total_pages, "status": d.status,
         "has_structure": ds_counts.get(d.id, 0) > 0}
        for d in rows
    ]


@router.get("/documents/{document_id}")
def get_document(document_id: int, db: Session = Depends(get_db)):
    d = db.get(Document, document_id)
    if not d:
        raise HTTPException(404, "document not found")
    return {
        "id": d.id, "domain_id": d.domain_id, "title": d.title, "file_path": d.file_path,
        "total_pages": d.total_pages, "parsed_outline": d.parsed_outline, "status": d.status,
    }


@router.get("/documents/{document_id}/pdf")
def get_document_pdf(document_id: int, db: Session = Depends(get_db)):
    """P4: for source-text comparison, embed the original PDF. Path comes from DB rather than user input -> no traversal."""
    d = db.get(Document, document_id)
    if not d:
        raise HTTPException(404, "document not found")
    if not d.file_path or not Path(d.file_path).is_file():
        raise HTTPException(404, f"PDF 文件不在盘上({d.file_path})")
    return FileResponse(d.file_path, media_type="application/pdf")


@router.delete("/documents/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Delete book + cascade-delete all its learning content (destructive, irreversible).

    Explicit ordered delete (not relying on SQLite FK): kg_edge -> kg_node (FK->note.id, so must precede note)
    -> note/question/chunk/generation_job -> task.document_id=NULL (detach agent trajectories,
    do not deep-delete run/step/...) -> document. PDF is deleted only if, after resolving, it lives in UPLOAD_DIR
    (our copy); external user files (e.g. /Books/...) are preserved. If a task is running, refuse delete (409)
    to prevent the child process from writing to already-deleted rows.
    """
    reconcile_generation_jobs(db)
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(404, "document not found")
    active = (
        db.query(GenerationJob)
        .filter(GenerationJob.document_id == document_id,
                GenerationJob.status.in_(("queued", "running")))
        .first()
    )
    if active is not None:
        raise HTTPException(
            409,
            f"该书有处理任务进行中(job {active.id}:{active.mode});结束或失败后再删",
        )

    node_ids = [r[0] for r in db.query(KGNode.id)
                .filter(KGNode.document_id == document_id).all()]
    n_edges = 0
    if node_ids:
        n_edges = (
            db.query(KGEdge)
            .filter((KGEdge.source_node_id.in_(node_ids)) |
                    (KGEdge.target_node_id.in_(node_ids)))
            .delete(synchronize_session=False)
        )
    n_nodes = (db.query(KGNode).filter(KGNode.document_id == document_id)
               .delete(synchronize_session=False))
    n_q = (db.query(Question).filter(Question.document_id == document_id)
           .delete(synchronize_session=False))
    n_note = (db.query(Note).filter(Note.document_id == document_id)
              .delete(synchronize_session=False))
    n_chunk = (db.query(Chunk).filter(Chunk.document_id == document_id)
               .delete(synchronize_session=False))
    # Task #61 closeout: after 1c writes real rows, deleting the book must cascade-delete document_structure, else orphans.
    # FK->document, no FK->note -> ordering is free; placed here (adjacent to chunk, same doc-scoped content).
    n_struct = (db.query(DocumentStructure)
                .filter(DocumentStructure.document_id == document_id)
                .delete(synchronize_session=False))
    n_job = (db.query(GenerationJob)
             .filter(GenerationJob.document_id == document_id)
             .delete(synchronize_session=False))
    db.query(Task).filter(Task.document_id == document_id).update(
        {Task.document_id: None}, synchronize_session=False)

    file_removed = False
    if doc.file_path:
        try:
            p = Path(doc.file_path).resolve()
            if p.is_relative_to(UPLOAD_DIR.resolve()):
                p.unlink(missing_ok=True)
                file_removed = True
        except (OSError, ValueError):
            file_removed = False

    db.delete(doc)
    db.commit()
    return {
        "deleted": {
            "document_id": document_id,
            "kg_edges": n_edges, "kg_nodes": n_nodes, "questions": n_q,
            "notes": n_note, "chunks": n_chunk,
            "document_structure": n_struct, "generation_jobs": n_job,
        },
        "file_removed": file_removed,
    }


@router.get("/documents/{document_id}/chunks")
def list_chunks(document_id: int, chapter_id: str | None = None, db: Session = Depends(get_db)):
    q = db.query(Chunk).filter(Chunk.document_id == document_id)
    if chapter_id:
        q = q.filter(Chunk.chapter_id == chapter_id)
    rows = q.order_by(Chunk.id).all()
    return [
        {
            "id": c.id, "chapter_id": c.chapter_id,
            "page_start": c.page_start, "page_end": c.page_end,
            "preview": c.content[:120],
        }
        for c in rows
    ]


@router.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: int, db: Session = Depends(get_db)):
    c = db.get(Chunk, chunk_id)
    if not c:
        raise HTTPException(404, "chunk not found")
    return {
        "id": c.id, "document_id": c.document_id, "chapter_id": c.chapter_id,
        "page_start": c.page_start, "page_end": c.page_end, "content": c.content,
    }


# ---------- Note / Question ----------

@router.get("/documents/{document_id}/notes")
def list_notes(document_id: int, chapter_id: str | None = None, db: Session = Depends(get_db)):
    q = db.query(Note).filter(Note.document_id == document_id)
    if chapter_id:
        q = q.filter(Note.chapter_id == chapter_id)
    rows = q.order_by(Note.id.desc()).all()
    # Per-chapter physical start page (the page_start the user entered when marking content pages); used for PDF page-jump in the cmp panel
    page_map = dict(
        db.query(Chunk.chapter_id, func.min(Chunk.page_start))
        .filter(Chunk.document_id == document_id, Chunk.page_start.isnot(None))
        .group_by(Chunk.chapter_id).all()
    )
    return [
        {
            "id": n.id, "chapter_id": n.chapter_id, "content_md": n.content_md,
            "created_at": n.created_at.isoformat(),
            "page_start": page_map.get(n.chapter_id),
        }
        for n in rows
    ]


_HEADING_RE = re.compile(r"^#{1,4}\s+(.+?)\s*$", re.MULTILINE)


def _extract_heading(content_md: str) -> str:
    """Take the first markdown `#` heading text, strip leading numbering; fallback empty string."""
    m = _HEADING_RE.search(content_md or "")
    if not m:
        return ""
    h = m.group(1).strip()
    # Strip common "1.1 " / "1.1.2 " / "Section 1 " prefixes to shorten the filename
    h = re.sub(r"^[\d.]+\s+", "", h)
    return h


@router.get("/documents/{document_id}/notes-export")
def export_notes(document_id: int, db: Session = Depends(get_db)):
    """Plan-C export: all notes for the whole doc -> zip (one .md per section + manifest.json).
    Filename ch{id}_{heading-slug}.md; if heading is missing, plain chapter_id.md."""
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, f"document {document_id} 不存在")

    from sla.harness.kg import slugify   # lazy: kg.py top-level imports langchain (heavy)

    notes = (
        db.query(Note)
        .filter(Note.document_id == document_id)
        .order_by(Note.chapter_id)
        .all()
    )
    if not notes:
        raise HTTPException(404, f"document {document_id} 无任何 note 可导出")

    doc_slug = slugify(doc.title or f"document-{document_id}")
    root = doc_slug  # root directory inside the zip

    manifest = {
        "document_id": document_id,
        "document_title": doc.title,
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "note_count": len(notes),
        "files": [],
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        seen_names: set[str] = set()
        for n in notes:
            heading = _extract_heading(n.content_md)
            head_slug = slugify(heading) if heading else ""
            base = f"{n.chapter_id}_{head_slug}" if head_slug else n.chapter_id
            name = f"{base}.md"
            # Prevent collision (duplicates in same chapter / empty extracted heading): append -2/-3 suffix
            i = 2
            while name in seen_names:
                name = f"{base}-{i}.md"
                i += 1
            seen_names.add(name)

            zf.writestr(f"{root}/{name}", n.content_md or "")
            manifest["files"].append({
                "file": name,
                "chapter_id": n.chapter_id,
                "heading": heading,
                "chars": len(n.content_md or ""),
            })

        zf.writestr(f"{root}/manifest.json",
                    json.dumps(manifest, ensure_ascii=False, indent=2))

    # starlette headers must be latin-1 safe: filename= uses an ASCII fallback,
    # filename*=UTF-8'' percent-encoded allows non-ASCII (modern browsers all recognize filename*)
    download_name = f"{doc_slug}-notes.zip"
    ascii_slug = (
        "".join(c for c in doc_slug if c.isascii() and (c.isalnum() or c in "-_"))
        or f"document-{document_id}"
    )
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition":
                f'attachment; filename="{ascii_slug}-notes.zip"; '
                f"filename*=UTF-8''{quote(download_name)}",
        },
    )


@router.get("/documents/{document_id}/questions")
def list_questions(document_id: int, chapter_id: str | None = None, db: Session = Depends(get_db)):
    q = db.query(Question).filter(Question.document_id == document_id)
    if chapter_id:
        q = q.filter(Question.chapter_id == chapter_id)
    rows = q.order_by(Question.id).all()
    return [
        {
            "id": x.id, "chapter_id": x.chapter_id, "content": x.content, "answer": x.answer,
            "difficulty": x.difficulty, "tags": x.tags, "source": x.source, "status": x.status,
        }
        for x in rows
    ]


@router.get("/documents/{document_id}/chapters")
def list_chapter_status(document_id: int, db: Session = Depends(get_db)):
    """Per-chapter generation state. S3 pivot: spine = document_structure (if present); otherwise fall back
    to Chunk.chapter_id distinct (legacy doc compatibility, zero regression).
    state: needs_content (DS has this chapter but no chunk) | generate | rebuild_kg | done.
    active_job = running job_id | null.
    P3b: on-read reconcile clears deadlocks; preserves array shape; adding title/has_chunk fields does not break the contract."""
    reconcile_generation_jobs(db)

    def _ck(c):
        try:
            return [int(x) for x in c[2:].split(".")]
        except Exception:
            return [99]

    # Spine: DS present -> use DS (with title); DS absent -> fall back to Chunk distinct (legacy)
    ds_rows = (db.query(DocumentStructure.chapter_id, DocumentStructure.title)
               .filter(DocumentStructure.document_id == document_id).all())
    if ds_rows:
        spine: list[tuple[str, str | None]] = [(cid, ti) for cid, ti in ds_rows]
    else:
        spine = [(r[0], None) for r in db.query(Chunk.chapter_id)
                 .filter(Chunk.document_id == document_id).distinct() if r[0]]
    # S3 actionable rule: keep only chapter/section levels (ch1 / ch1.2); level 3+ (ch1.2.3 / ch6.1.1)
    # are sub-parts of a section; if S4 also marks content per sub-part it would duplicate-extract with the
    # parent section -> hide them. document_structure still stores all levels (lossless, S5 outline can use them).
    spine = [(cid, ti) for cid, ti in spine if cid.count(".") <= 1]
    spine.sort(key=lambda x: _ck(x[0]))

    chunk_ch = {r[0] for r in db.query(Chunk.chapter_id)
                .filter(Chunk.document_id == document_id).distinct() if r[0]}
    note_ch = {r[0] for r in db.query(Note.chapter_id)
               .filter(Note.document_id == document_id).distinct() if r[0]}
    kg_ch = {r[0] for r in db.query(KGNode.chapter_id)
             .filter(KGNode.document_id == document_id).distinct() if r[0]}
    def _ch_base(s):
        # S4 encodes the page range into chapter_id ("ch1.2|p=50-65"); the active dict needs to
        # strip back to the true chapter to match the spine row; unchanged when there's no "|".
        return s.split("|", 1)[0]
    active = {
        _ch_base(j.chapter_id): j.id
        for j in db.query(GenerationJob).filter(
            GenerationJob.document_id == document_id,
            GenerationJob.status.in_(("queued", "running")),
        ).all()
    }

    out = []
    for ch, ti in spine:
        has_chunk = ch in chunk_ch
        hn, hk = ch in note_ch, ch in kg_ch
        if not has_chunk:
            state = "needs_content"
        elif not hn:
            state = "generate"
        elif not hk:
            state = "rebuild_kg"
        else:
            state = "done"
        out.append({
            "chapter_id": ch, "title": ti, "has_chunk": has_chunk,
            "has_note": hn, "has_kg": hk,
            "state": state, "active_job": active.get(ch),
        })
    return out
