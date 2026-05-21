"""P3b:UI 触发章节 生成/重建 端点(subprocess + job-status + 轮询)。"""
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
    reconcile_generation_jobs(db)   # P3-4:先清死锁再判锁
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
    # D2:start_new_session=True 起独立会话 → uvicorn --reload / 重启不杀生成子进程
    subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "run_generation.py"),
         "--job-id", str(job.id)],
        cwd=str(ROOT), start_new_session=True,
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
    # 增量 ingest 单章:chapter_id 字段承载章号串("4")→ run_generation ingest 分支
    if chapter < 1:
        raise HTTPException(400, "chapter 必须为正整数")
    return _start(document_id, str(chapter), "ingest", db)


@router.post("/documents/{document_id}/extract-structure", status_code=202)
def start_extract_structure(
    document_id: int, toc: str, force: bool = False,
    db: Session = Depends(get_db),
):
    # human-anchored S2:人选目录页 → 一发 LLM 抽全书结构(非 agent)。
    # chapter_id 字段承载 "toc:<页范围>[!force]"(复用 ingest 串载荷先例,
    # mode=String(16) 故名 "extract_toc"=11)→ run_generation extract_toc 分支。
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
    # human-anchored S4:人选该节内容页 → 一发视觉 OCR + 切块写 Chunk。
    # chapter_id 字段承载 "{ch}|p={pages}[!force]" → run_generation extract_content 分支。
    # /chapters route 用 _ch_base 剥离 |p= 后缀回真章,以匹配 spine。
    # mode="extract_content"=15 字符 ✓ String(16)。
    if not pages.strip():
        raise HTTPException(400, "pages 必填,如 50-65 或 50,52-55")
    if not chapter_id.startswith("ch"):
        raise HTTPException(400, f"非法 chapter_id {chapter_id!r}(应形如 ch1.2)")
    payload = f"{chapter_id}|p={pages.strip()}" + ("!force" if force else "")
    return _start(document_id, payload, "extract_content", db)


def _parse_page_spec(s: str) -> list[int]:
    """'50-65' / '50,52-55' / '50' → sorted unique 1-based 页。
    与 structure_extract.parse_toc_payload 内部页解析同语义,无 toc: 前缀。"""
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
    """C-方案 $0 预览端点:页范围 → JPEG q=75 zoom=0.8 缩略图 b64。
    付费 OCR 前给前端确认 modal 用,免 content-filter/选错页踩坑。"""
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
        mtx = pymupdf.Matrix(0.8, 0.8)   # ~64 DPI,看节标题/段首字够用
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
    """P2:网页上传新书 = 仅登记(存盘 + 建 Document,status='uploaded'),不解析。

    解耦设计:upload 不起 job → 无 _start → 无 409 → 结构性消解空壳书问题。
    书随即出现在书库显「未处理」;用户点「处理本书」才触发整本 ingest
    (POST /documents/{id}/process)。'uploaded' 是显式正常态,非错误。
    """
    name = Path(file.filename or "").name              # 去目录成分 → 无路径穿越
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


@router.post("/documents/{document_id}/process", status_code=202)
def process_document(document_id: int, db: Session = Depends(get_db)):
    """P2:对已上传未处理的书触发整本 ingest(--chapter all),复用 _start。

    409(已有 job 在跑)benign:Document upload 时已建,稍后重点即可、无空壳。
    run_generation ingest 分支成功后置 doc.status='ready'。
    """
    return _start(document_id, "all", "ingest", db)


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
