"""P3b: UI-triggered chapter generate/rebuild endpoints (subprocess + job-status + polling)."""
import base64
import subprocess
import sys
from pathlib import Path

import pymupdf
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from sla.db import get_db
from sla.models.domain import Document, Domain
from sla.models.runtime import GenerationJob, reconcile_generation_jobs

router = APIRouter()
ROOT = Path(__file__).parent.parent.parent.parent
UPLOAD_DIR = ROOT / "uploads"


def _start(document_id: int, chapter_id: str, mode: str, db: Session):
    reconcile_generation_jobs(db)   # P3-4: clear deadlocks before checking the lock
    active = (
        db.query(GenerationJob)
        .filter(GenerationJob.status.in_(("queued", "running")))
        .first()
    )
    if active is not None:
        raise HTTPException(
            409,
            f"已有生成任务进行中(job {active.id}:{active.mode} {active.chapter_id})"
            f";一次一个,等它结束或失败后再试",
        )
    job = GenerationJob(
        document_id=document_id, chapter_id=chapter_id, mode=mode, status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    # D2: independent session/process group -> uvicorn --reload / restart does not kill the generation subprocess.
    # POSIX uses start_new_session=True (setsid); Windows uses CREATE_NEW_PROCESS_GROUP.
    # Cannot pass both: start_new_session is silently ignored on Win, the process falls back to inheriting,
    # and a uvicorn restart will kill the child along with it.
    detach_kwargs = (
        {"start_new_session": True}
        if sys.platform != "win32"
        else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    )
    subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "run_generation.py"),
         "--job-id", str(job.id)],
        cwd=str(ROOT), **detach_kwargs,
    )
    return {"job_id": job.id, "status": "queued", "mode": mode, "chapter_id": chapter_id}


@router.post("/documents/{document_id}/chapters/{chapter_id}/generate", status_code=202)
def start_generate(document_id: int, chapter_id: str, db: Session = Depends(get_db)):
    return _start(document_id, chapter_id, "generate", db)


@router.post("/documents/{document_id}/chapters/{chapter_id}/rebuild", status_code=202)
def start_rebuild(document_id: int, chapter_id: str, db: Session = Depends(get_db)):
    return _start(document_id, chapter_id, "rebuild", db)


@router.post("/documents/{document_id}/ingest-chapter", status_code=202)
def start_ingest(document_id: int, chapter: int, db: Session = Depends(get_db)):
    # Incrementally ingest a single chapter: the chapter_id field carries the chapter number string ("4") -> run_generation ingest branch
    if chapter < 1:
        raise HTTPException(400, "chapter 必须为正整数")
    return _start(document_id, str(chapter), "ingest", db)


@router.post("/documents/{document_id}/extract-structure", status_code=202)
def start_extract_structure(
    document_id: int, toc: str, force: bool = False,
    db: Session = Depends(get_db),
):
    # human-anchored S2: human picks the TOC pages -> one-shot LLM extracts the whole-book structure (not an agent).
    # The chapter_id field carries "toc:<page-range>[!force]" (reusing the ingest string-payload precedent;
    # mode=String(16) so name is "extract_toc"=11 chars) -> run_generation extract_toc branch.
    if not toc.strip():
        raise HTTPException(400, "toc 页范围必填,如 7-10 或 7,8,9")
    payload = f"toc:{toc.strip()}" + ("!force" if force else "")
    return _start(document_id, payload, "extract_toc", db)


@router.post("/documents/{document_id}/chapters/{chapter_id}/extract-content",
             status_code=202)
def start_extract_content(
    document_id: int, chapter_id: str, pages: str, force: bool = False,
    db: Session = Depends(get_db),
):
    # human-anchored S4: human picks the section's content pages -> one-shot vision OCR + chunking writes Chunk.
    # The chapter_id field carries "{ch}|p={pages}[!force]" -> run_generation extract_content branch.
    # The /chapters route uses _ch_base to strip the |p= suffix back to the true chapter, to match the spine.
    # mode="extract_content"=15 chars, fits String(16).
    if not pages.strip():
        raise HTTPException(400, "pages 必填,如 50-65 或 50,52-55")
    if not chapter_id.startswith("ch"):
        raise HTTPException(400, f"非法 chapter_id {chapter_id!r}(应形如 ch1.2)")
    payload = f"{chapter_id}|p={pages.strip()}" + ("!force" if force else "")
    return _start(document_id, payload, "extract_content", db)


def _parse_page_spec(s: str) -> list[int]:
    """'50-65' / '50,52-55' / '50' -> sorted unique 1-based pages.
    Same semantics as structure_extract.parse_toc_payload's internal page parsing, without the toc: prefix."""
    pages: set[int] = set()
    for tok in s.replace(" ", "").split(","):
        if not tok:
            continue
        if "-" in tok:
            a_s, b_s = tok.split("-", 1)
            try:
                a, b = int(a_s), int(b_s)
            except ValueError:
                raise ValueError(f"非法页范围 {tok!r}") from None
            if a < 1 or b < a:
                raise ValueError(f"非法页范围 {tok!r}(应 a<=b 且 a>=1)")
            pages.update(range(a, b + 1))
        else:
            try:
                v = int(tok)
            except ValueError:
                raise ValueError(f"非法页 {tok!r}") from None
            if v < 1:
                raise ValueError(f"非法页 {tok!r}(应 >=1)")
            pages.add(v)
    if not pages:
        raise ValueError("空页规范")
    return sorted(pages)


@router.get("/documents/{document_id}/page-thumbs")
def get_page_thumbs(
    document_id: int, pages: str,
    db: Session = Depends(get_db),
):
    """Plan-C $0 preview endpoint: page range -> JPEG q=75 zoom=0.8 thumbnail b64.
    Used by the frontend confirmation modal before paid OCR, to avoid content-filter / wrong-page pitfalls."""
    if not pages.strip():
        raise HTTPException(400, "pages 必填,如 50-65 或 50,52-55")
    try:
        pgs = _parse_page_spec(pages.strip())
    except ValueError as e:
        raise HTTPException(400, str(e))

    doc = db.get(Document, document_id)
    if doc is None or not doc.file_path:
        raise HTTPException(404, f"document {document_id} 不存在或无 file_path")

    pdf = pymupdf.open(doc.file_path)
    try:
        if max(pgs) > pdf.page_count:
            raise HTTPException(
                400, f"页 {max(pgs)} 超出 PDF 总页 {pdf.page_count}",
            )
        mtx = pymupdf.Matrix(0.8, 0.8)   # ~64 DPI, sufficient to read section titles / first words of paragraphs
        out = []
        for pg in pgs:
            pix = pdf[pg - 1].get_pixmap(matrix=mtx)
            jpg = pix.tobytes("jpeg", jpg_quality=75)
            out.append({"page": pg,
                        "data": base64.b64encode(jpg).decode("ascii")})
    finally:
        pdf.close()

    return JSONResponse(
        {"document_id": document_id, "pages": out},
        headers={"Cache-Control": "private, max-age=600"},
    )


@router.post("/upload", status_code=202)
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """P2: web upload of a new book = registration only (write to disk + create Document, status='uploaded'); no parsing.

    Decoupled design: upload does not start a job -> no _start -> no 409 -> structurally dissolves the empty-shell-book problem.
    The book then appears in the library; the user clicks the title to enter the viewer, extracts the TOC via "chapter state"
    -> marks content pages -> generates: a human-anchored section-level workflow. 'uploaded' is an explicit normal state, not an error.
    """
    name = Path(file.filename or "").name              # strip directory components -> no path traversal
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "只接受 .pdf 文件")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / name
    data = await file.read()
    if not data:
        raise HTTPException(400, "空文件")
    dest.write_bytes(data)
    dom = db.query(Domain).filter(Domain.name == "Uploaded").first()
    if dom is None:
        dom = Domain(name="Uploaded")
        db.add(dom)
        db.commit()
        db.refresh(dom)
    doc = db.query(Document).filter(Document.file_path == str(dest)).first()
    if doc is None:
        doc = Document(domain_id=dom.id, title=Path(name).stem,
                        file_path=str(dest), total_pages=0, status="uploaded")
        db.add(doc)
        db.commit()
        db.refresh(doc)
    return {"document_id": doc.id, "title": doc.title, "status": doc.status}


@router.get("/generation-jobs/{job_id}")
def get_generation_job(job_id: int, db: Session = Depends(get_db)):
    reconcile_generation_jobs(db)
    j = db.get(GenerationJob, job_id)
    if j is None:
        raise HTTPException(404, "generation job not found")
    return {
        "id": j.id, "document_id": j.document_id, "chapter_id": j.chapter_id,
        "mode": j.mode, "status": j.status, "step": j.step, "detail": j.detail,
        "created_at": j.created_at.isoformat() if j.created_at else None,
        "ended_at": j.ended_at.isoformat() if j.ended_at else None,
    }
