"""Batch KG extraction orchestrator with idempotent upsert into kg_node / kg_edge.

Usage:
  python scripts/build_kg.py --document-id 2 --chapter 1
  python scripts/build_kg.py --document-id 2 --chapter 1 --notes 11,12,13   # Only the given Notes
  python scripts/build_kg.py --document-id 2 --chapter 1 --dry-run         # Extract, don't write

Per-Note flow (serial):
  1. extract_kg_from_note(note_id, context_chapter_depth=3) — feeds context as dedup hint
  2. Upsert KGNode for each concept:
     - Look up by slug(label) within the document (cross-chapter merge by name)
     - Found → reuse, do not update description/type (first-wins, simple/predictable)
     - Not found → create with external_id = f'{doc}_{chap}_{slug}'
  3. Upsert KGEdge for each relation:
     - Resolve source/target via label_to_node map
     - Composite UNIQUE (source, target, type) prevents duplicates; skip on conflict
  4. Commit per Note.

No concurrency (single-user; serial is fine, ~4 min for 8 chapters).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy.exc import IntegrityError  # noqa: E402

from sla.config import settings  # noqa: E402

from sla.db import SessionLocal  # noqa: E402
from sla.harness.kg import build_external_id, extract_kg_from_note, slugify  # noqa: E402
from sla.models.domain import Note  # noqa: E402
from sla.models.kg import KGEdge, KGNode  # noqa: E402


# Edge 类型 → 是否有向
DIRECTED_BY_TYPE = {
    "requires": True,
    "example_of": True,
    "related_to": False,
    "contrasts_with": False,
}


def upsert_kg_node(
    db,
    *,
    label: str,
    node_type: str,
    description: str | None,
    document_id: int,
    chapter_id: str,
    note_id: int,
) -> KGNode:
    """按 slug(label) 跨章合并;找到不更新(first-wins)。"""
    slug = slugify(label)
    # 跨章查同 document 内是否已有同 slug 的节点
    existing = (
        db.query(KGNode)
        .filter(KGNode.document_id == document_id)
        .filter(KGNode.external_id.like(f"{document_id}\\_%\\_{slug}", escape="\\"))
        .first()
    )
    if existing is not None:
        return existing

    n = KGNode(
        external_id=build_external_id(document_id, chapter_id, label),
        type=node_type,
        label=label,
        description=description,
        document_id=document_id,
        chapter_id=chapter_id,
        note_ref_id=note_id,
    )
    db.add(n)
    db.flush()
    return n


def upsert_kg_edge(
    db,
    *,
    source_node_id: int,
    target_node_id: int,
    edge_type: str,
    notes: str | None,
) -> KGEdge | None:
    """(source, target, type) 复合 UNIQUE,已有就 skip。

    自环(source == target)直接拒绝。
    """
    if source_node_id == target_node_id:
        return None

    directed = DIRECTED_BY_TYPE.get(edge_type, True)
    existing = (
        db.query(KGEdge)
        .filter(
            KGEdge.source_node_id == source_node_id,
            KGEdge.target_node_id == target_node_id,
            KGEdge.type == edge_type,
        )
        .first()
    )
    if existing is not None:
        return existing

    # 无向边:source/target 反向也算同一条
    if not directed:
        rev = (
            db.query(KGEdge)
            .filter(
                KGEdge.source_node_id == target_node_id,
                KGEdge.target_node_id == source_node_id,
                KGEdge.type == edge_type,
            )
            .first()
        )
        if rev is not None:
            return rev

    e = KGEdge(
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        type=edge_type,
        directed=directed,
        notes=notes,
    )
    db.add(e)
    db.flush()
    return e


def process_note(
    db,
    note: Note,
    *,
    dry_run: bool = False,
) -> tuple[int, int, int, int]:
    """对单个 Note 做抽取 + upsert。返回 (new_nodes, reused_nodes, new_edges, skipped_edges)。"""
    result = extract_kg_from_note(note.id, context_chapter_depth=3)

    if dry_run:
        print(f"  [dry-run] {len(result.concepts)} concepts, {len(result.relations)} relations")
        for c in result.concepts:
            print(f"    [{c.type:7s}] {c.label!r}")
        return (0, 0, 0, 0)

    # ---- 1. upsert nodes,记录 label → node 映射 ----
    label_to_node: dict[str, KGNode] = {}
    new_nodes = 0
    reused_nodes = 0
    for c in result.concepts:
        before_count = db.query(KGNode).filter(KGNode.document_id == note.document_id).count()
        node = upsert_kg_node(
            db,
            label=c.label,
            node_type=c.type,
            description=c.description,
            document_id=note.document_id,
            chapter_id=note.chapter_id,
            note_id=note.id,
        )
        label_to_node[c.label] = node
        after_count = db.query(KGNode).filter(KGNode.document_id == note.document_id).count()
        if after_count > before_count:
            new_nodes += 1
        else:
            reused_nodes += 1

    # ---- 2. upsert edges ----
    new_edges = 0
    skipped_edges = 0
    for r in result.relations:
        src = label_to_node.get(r.source_label)
        tgt = label_to_node.get(r.target_label)
        if src is None or tgt is None:
            # W3-4 本来就会 drop 引用未知 label 的 relations,这里再兜底
            skipped_edges += 1
            continue

        before_count = db.query(KGEdge).count()
        edge = upsert_kg_edge(
            db,
            source_node_id=src.id,
            target_node_id=tgt.id,
            edge_type=r.type,
            notes=r.notes,
        )
        if edge is None:
            skipped_edges += 1  # 自环被拒
            continue
        after_count = db.query(KGEdge).count()
        if after_count > before_count:
            new_edges += 1
        else:
            skipped_edges += 1

    db.commit()
    return (new_nodes, reused_nodes, new_edges, skipped_edges)


def _clear_chapter_kg(db, document_id: int, chapter_id: str) -> tuple[int, int]:
    """删该 (doc, chapter) 范围内所有 KGNode + 引用它们的 KGEdge。
    rebuild 路径必须用之,否则旧抽取的节点会与新抽取累积(已踩雷 2026-05-21:
    用户中文 prompt 修后 rebuild,旧英文节点未删 → 中英重复双胞胎共 20 个)。
    先删 edge(双 endpoint 引用 KGNode)再删 node。"""
    node_ids = [n.id for n in db.query(KGNode).filter(
        KGNode.document_id == document_id,
        KGNode.chapter_id == chapter_id,
    ).all()]
    if not node_ids:
        return 0, 0
    n_edges = db.query(KGEdge).filter(
        (KGEdge.source_node_id.in_(node_ids))
        | (KGEdge.target_node_id.in_(node_ids))
    ).delete(synchronize_session=False)
    n_nodes = db.query(KGNode).filter(
        KGNode.id.in_(node_ids)
    ).delete(synchronize_session=False)
    db.commit()
    return n_nodes, n_edges


def main():
    parser = argparse.ArgumentParser(prog="build_kg")
    parser.add_argument("--document-id", type=int, required=True)
    parser.add_argument(
        "--chapter", default=None,
        help="章节前缀如 '1' 匹配 ch1.*;不指定则全 document 所有 Notes",
    )
    parser.add_argument(
        "--notes", default=None,
        help="指定 note id 逗号列表(覆盖 --chapter)",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="抽前清涉及 chapter 的现有 KGNode + Edge。rebuild 路径必加,"
             "否则新旧节点累积(尤其语言切换时中英重复)。",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只抽取打印,不写库",
    )
    args = parser.parse_args()

    if not settings.anthropic_api_key:
        print("FAIL: ANTHROPIC_API_KEY is empty", file=sys.stderr)
        sys.exit(1)

    # 选 Notes
    db = SessionLocal()
    try:
        if args.notes:
            ids = [int(x) for x in args.notes.split(",") if x.strip()]
            notes = (
                db.query(Note)
                .filter(Note.document_id == args.document_id, Note.id.in_(ids))
                .order_by(Note.chapter_id)
                .all()
            )
        else:
            q = db.query(Note).filter(Note.document_id == args.document_id)
            if args.chapter:
                q = q.filter(Note.chapter_id.like(f"ch{args.chapter}.%"))
            notes = q.order_by(Note.chapter_id).all()

        if not notes:
            print(
                f"FAIL: no Notes for document_id={args.document_id} "
                f"chapter={args.chapter!r} notes={args.notes!r}",
                file=sys.stderr,
            )
            sys.exit(1)

        print(f"=== build_kg: {len(notes)} Notes to process ===")
        for n in notes:
            print(f"  Note id={n.id}  chapter={n.chapter_id}  length={len(n.content_md)}")

        if args.clear and not args.dry_run:
            cleared_chapters: set[str] = set()
            for note in notes:
                if note.chapter_id in cleared_chapters:
                    continue
                n_nodes, n_edges = _clear_chapter_kg(
                    db, args.document_id, note.chapter_id)
                cleared_chapters.add(note.chapter_id)
                print(f"  [--clear] {note.chapter_id}: 删 {n_nodes} nodes + "
                      f"{n_edges} edges")

        total_new_nodes = 0
        total_reused = 0
        total_new_edges = 0
        total_skipped = 0
        t_start = time.time()

        for i, note in enumerate(notes, 1):
            print(f"\n{'-' * 60}\n[{i}/{len(notes)}] {note.chapter_id} (Note id={note.id})")
            t0 = time.time()
            new_nodes, reused, new_edges, skipped = process_note(
                db, note, dry_run=args.dry_run,
            )
            elapsed = time.time() - t0
            print(
                f"  → {new_nodes} new nodes, {reused} reused, "
                f"{new_edges} new edges, {skipped} skipped  ({elapsed:.1f}s)"
            )
            total_new_nodes += new_nodes
            total_reused += reused
            total_new_edges += new_edges
            total_skipped += skipped

        total_elapsed = time.time() - t_start

        # ---- 汇总 ----
        print(f"\n\n{'=' * 60}\nSummary\n{'=' * 60}")
        print(f"  total time: {total_elapsed:.1f}s")
        print(f"  new KGNodes:   {total_new_nodes}")
        print(f"  reused nodes:  {total_reused}  (跨章合并节省的)")
        print(f"  new KGEdges:   {total_new_edges}")
        print(f"  skipped edges: {total_skipped}  (含 dup, 自环, 缺端点)")

        # 数据健康:按 type 分布
        if not args.dry_run:
            print(f"\n  KGNode by type:")
            for t in ["concept", "method", "example"]:
                c = (
                    db.query(KGNode)
                    .filter(KGNode.document_id == args.document_id, KGNode.type == t)
                    .count()
                )
                print(f"    {t:8s}: {c}")
            print(f"\n  KGEdge by type:")
            for t in ["requires", "related_to", "contrasts_with", "example_of"]:
                c = (
                    db.query(KGEdge)
                    .join(KGNode, KGEdge.source_node_id == KGNode.id)
                    .filter(KGNode.document_id == args.document_id, KGEdge.type == t)
                    .count()
                )
                print(f"    {t:18s}: {c}")
    finally:
        db.close()

    print(f"\n[ok] build_kg done")


if __name__ == "__main__":
    main()
