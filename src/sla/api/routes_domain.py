"""应用层只读 endpoints(Phase 1A 范围)。"""
import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
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
    from sqlalchemy import func
    q = db.query(Document)
    if domain_id is not None:
        q = q.filter(Document.domain_id == domain_id)
    rows = q.order_by(Document.id).all()
    # has_structure:该 doc 是否已标注目录(document_structure 有行)。
    # 一次聚合查避免 N+1;前端用之代替"页数·status·id" 这类技术元数据。
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
    """P4:原文对照用,内嵌原 PDF。路径取自 DB 非用户输入 → 无穿越。"""
    d = db.get(Document, document_id)
    if not d:
        raise HTTPException(404, "document not found")
    if not d.file_path or not Path(d.file_path).is_file():
        raise HTTPException(404, f"PDF 文件不在盘上({d.file_path})")
    return FileResponse(d.file_path, media_type="application/pdf")


@router.delete("/documents/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db)):
    """删书 + 级联删其全部学习内容(破坏性、不可逆)。

    显式有序删(不靠 SQLite FK):kg_edge → kg_node(FK→note.id,故必先于 note)
    → note/question/chunk/generation_job → task.document_id=NULL(脱钩 agent 轨迹,
    不深删 run/step/…)→ document。PDF 仅当解析后在 UPLOAD_DIR 内(我们的拷贝)
    才删;外部用户文件(如 /Books/…)保留。运行中任务则拒删(409),防子进程写已删行。
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
    # Task #61 核销:1c 写真行后,删该书必须级联删 document_structure 否则 orphan。
    # FK→document、无 FK→note → 排序自由,放此处(紧邻 chunk,同 doc-scoped 内容)。
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
    return [
        {"id": n.id, "chapter_id": n.chapter_id, "content_md": n.content_md, "created_at": n.created_at.isoformat()}
        for n in rows
    ]


_HEADING_RE = re.compile(r"^#{1,4}\s+(.+?)\s*$", re.MULTILINE)


def _extract_heading(content_md: str) -> str:
    """取 markdown 第一个 # 标题文本去掉前导编号,fallback 空串。"""
    m = _HEADING_RE.search(content_md or "")
    if not m:
        return ""
    h = m.group(1).strip()
    # 去掉常见 "1.1 " / "1.1.2 " / "第1节 " 前缀,让文件名更短
    h = re.sub(r"^[\d.]+\s+", "", h)
    return h


@router.get("/documents/{document_id}/notes-export")
def export_notes(document_id: int, db: Session = Depends(get_db)):
    """C-方案导出:整本 doc 全部 note → zip(一节一 .md + manifest.json)。
    文件名 ch{id}_{heading-slug}.md;heading 缺失则纯 chapter_id.md。"""
    doc = db.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, f"document {document_id} 不存在")

    from sla.harness.kg import slugify   # lazy:kg.py 顶层 import langchain 重

    notes = (
        db.query(Note)
        .filter(Note.document_id == document_id)
        .order_by(Note.chapter_id)
        .all()
    )
    if not notes:
        raise HTTPException(404, f"document {document_id} 无任何 note 可导出")

    doc_slug = slugify(doc.title or f"document-{document_id}")
    root = doc_slug  # zip 内根目录

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
            # 防 collision(同章重复 / heading 抽空):带 -2/-3 后缀
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

    # starlette headers 必须 latin-1 安全:filename= 用 ASCII fallback,
    # filename*=UTF-8'' percent-encode 才允许中文(现代浏览器都识别 filename*)
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
    """每章生成状态。S3 pivot:脊=document_structure(若有);否则回退
    Chunk.chapter_id distinct(legacy doc 兼容,零回归)。
    state: needs_content(DS 有此章但无 chunk) | generate | rebuild_kg | done。
    active_job=running job_id|null。
    P3b:on-read reconcile 清死锁;保数组 shape,加 title/has_chunk 字段不破契约。"""
    reconcile_generation_jobs(db)

    def _ck(c):
        try:
            return [int(x) for x in c[2:].split(".")]
        except Exception:
            return [99]

    # 脊:DS 有 → 用 DS(带 title);DS 无 → 回退 Chunk distinct(legacy)
    ds_rows = (db.query(DocumentStructure.chapter_id, DocumentStructure.title)
               .filter(DocumentStructure.document_id == document_id).all())
    if ds_rows:
        spine: list[tuple[str, str | None]] = [(cid, ti) for cid, ti in ds_rows]
    else:
        spine = [(r[0], None) for r in db.query(Chunk.chapter_id)
                 .filter(Chunk.document_id == document_id).distinct() if r[0]]
    # S3 actionable rule:只保留章/节两级(ch1 / ch1.2);三级+(ch1.2.3 / ch6.1.1)
    # 是节的子部分,若 S4 也按子部分标内容会与父节重复抽 → 隐藏。
    # document_structure 仍存全级(信息无损,S5 大纲可用)。
    spine = [(cid, ti) for cid, ti in spine if cid.count(".") <= 1]
    spine.sort(key=lambda x: _ck(x[0]))

    chunk_ch = {r[0] for r in db.query(Chunk.chapter_id)
                .filter(Chunk.document_id == document_id).distinct() if r[0]}
    note_ch = {r[0] for r in db.query(Note.chapter_id)
               .filter(Note.document_id == document_id).distinct() if r[0]}
    kg_ch = {r[0] for r in db.query(KGNode.chapter_id)
             .filter(KGNode.document_id == document_id).distinct() if r[0]}
    def _ch_base(s):
        # S4 把页范围编码在 chapter_id("ch1.2|p=50-65"),active dict 要剥
        # 回真章以匹配 spine 行;无 "|" 时不变。
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
