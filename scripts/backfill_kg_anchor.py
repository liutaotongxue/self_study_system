"""Phase 3-1-3 + 3-2 (A1 + A2):回填 kg_node.note_anchor_slug + cross_note_*。

对每个 KGNode,找它的 anchor:

匹配优先级(同 Note 内):
  1. heading 命中(label 出现在 ## 标题里)—— 这一节就是讲它的,最权威
  2. content 命中(label 出现在 section 正文)—— 这一节提到它,次选
  3. embedding 命中(--use-embedding-fallback):label 跟 heading 语义相似度
     ≥ threshold,救字面 mismatch(如 ε vs epsilon)

A2 跨 Note(home Note 全失败时):
  4. cross_embedding(--use-cross-note-fallback):在同 document 其他 Note 里跑
     embedding,找最佳 heading;写 cross_note_id + cross_note_slug

A2 label-length-aware threshold:
  - ≤2 词 label:0.85(短词在 embedding 空间噪音大,严格才能不假阳性)
  - ≥3 词 label:0.7 (home note Pass 3) / 0.75 (cross note Pass 4,搜索空间大,稍严)

跑法:
  python scripts/backfill_kg_anchor.py --document-id 2
  python scripts/backfill_kg_anchor.py --document-id 2 --use-embedding-fallback --use-cross-note-fallback
  python scripts/backfill_kg_anchor.py --dry-run

成本:Pass 1/2 $0;Pass 3/4 本地 sentence-transformers 模型,首跑下载 ~80MB,后续 0
"""
import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sla.db import SessionLocal  # noqa: E402
from sla.harness.kg import parse_markdown_sections, slugify_headings  # noqa: E402
from sla.models.domain import Note  # noqa: E402
from sla.models.kg import KGNode  # noqa: E402


@dataclass
class AnchorMatch:
    """单个 KGNode 的 anchor 解析结果。"""
    match_type: str = "miss"
    note_anchor_slug: str | None = None
    cross_note_id: int | None = None
    cross_note_slug: str | None = None
    debug: dict | None = None   # heading, score, target_chapter 等供 inspection


def label_aware_threshold(label: str, *, long_threshold: float) -> float:
    """短 label (≤2 语义 token) 强制 0.85,长 label 用 caller 给的 base。

    理由:embedding 在 1-2 词通用术语(`target` / `state`)上噪音大,需要更高 sim
    才可信;3+ 词 label 更 specific,base threshold 已经足够。

    "语义 token" 同时按 whitespace + hyphen split —— `exploration-exploitation` 是 2 个
    语义 token,不是 1 个 generic 名词,不该被算成短 label。
    """
    n_tokens = len(re.split(r"[-\s]+", label.strip()))
    return 0.85 if n_tokens <= 2 else long_threshold


def link_node_to_anchor(
    node: KGNode,
    db,
    embedding_model=None,
    cross_note_fallback: bool = False,
) -> AnchorMatch:
    """对单个 KGNode 找 anchor。Pass 1/2 字面,Pass 3 同 Note embedding,
    Pass 4 跨 Note embedding(--use-cross-note-fallback)。
    """
    if node.note_ref_id is None:
        return AnchorMatch(match_type="no_note")
    home_note = db.get(Note, node.note_ref_id)
    if home_note is None:
        return AnchorMatch(match_type="no_note")
    sections = parse_markdown_sections(home_note.content_md)
    if not sections:
        return AnchorMatch(match_type="no_sections")

    headings = [h for h, _ in sections]
    slugs = slugify_headings(headings)   # A2-1: collision-aware
    pat = re.compile(re.escape(node.label), re.IGNORECASE)

    # Pass 1:home note heading 命中
    for i, (h, _) in enumerate(sections):
        if pat.search(h):
            return AnchorMatch(match_type="heading", note_anchor_slug=slugs[i])

    # Pass 2:home note content 命中
    for i, (h, c) in enumerate(sections):
        if pat.search(c):
            return AnchorMatch(match_type="content", note_anchor_slug=slugs[i])

    if embedding_model is None:
        return AnchorMatch(match_type="miss")

    from sentence_transformers import util

    # Pass 3:home note embedding
    threshold_home = label_aware_threshold(node.label, long_threshold=0.7)
    emb_label = embedding_model.encode(node.label, convert_to_tensor=True)
    emb_home = embedding_model.encode(headings, convert_to_tensor=True)
    sims_home = util.cos_sim(emb_label, emb_home)[0]
    best_home_idx = int(sims_home.argmax())
    best_home_score = float(sims_home[best_home_idx])

    if best_home_score >= threshold_home:
        return AnchorMatch(
            match_type="embedding",
            note_anchor_slug=slugs[best_home_idx],
            debug={
                "heading": headings[best_home_idx],
                "score": best_home_score,
                "threshold": threshold_home,
            },
        )

    if not cross_note_fallback:
        return AnchorMatch(
            match_type="miss",
            debug={
                "heading": headings[best_home_idx],
                "score": best_home_score,
                "threshold": threshold_home,
            },
        )

    # Pass 4:跨 Note embedding(同 document 其他 Note,搜全部 heading)
    threshold_cross = label_aware_threshold(node.label, long_threshold=0.75)
    other_notes = (
        db.query(Note)
        .filter(Note.document_id == node.document_id, Note.id != node.note_ref_id)
        .order_by(Note.id)
        .all()
    )
    best_overall = {"score": -1.0, "note_id": None, "heading": None, "slug": None}
    for other in other_notes:
        secs = parse_markdown_sections(other.content_md)
        if not secs:
            continue
        oh = [h for h, _ in secs]
        os_ = slugify_headings(oh)
        emb_oh = embedding_model.encode(oh, convert_to_tensor=True)
        sims_o = util.cos_sim(emb_label, emb_oh)[0]
        idx = int(sims_o.argmax())
        sc = float(sims_o[idx])
        if sc > best_overall["score"]:
            best_overall = {
                "score": sc, "note_id": other.id,
                "heading": oh[idx], "slug": os_[idx],
            }

    if best_overall["note_id"] is not None and best_overall["score"] >= threshold_cross:
        return AnchorMatch(
            match_type="cross_embedding",
            cross_note_id=best_overall["note_id"],
            cross_note_slug=best_overall["slug"],
            debug={
                "heading": best_overall["heading"],
                "score": best_overall["score"],
                "threshold": threshold_cross,
                "target_note_id": best_overall["note_id"],
            },
        )

    # 跨 Note 也没过 threshold → 真 miss
    return AnchorMatch(
        match_type="miss",
        debug={
            "home_best_heading": headings[best_home_idx],
            "home_best_score": best_home_score,
            "cross_best_heading": best_overall["heading"],
            "cross_best_score": best_overall["score"],
            "cross_target_note_id": best_overall["note_id"],
        },
    )


def main():
    parser = argparse.ArgumentParser(prog="backfill_kg_anchor")
    parser.add_argument("--document-id", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-misses", action="store_true")
    parser.add_argument(
        "--use-embedding-fallback", action="store_true",
        help="Pass 3 同 Note embedding fallback",
    )
    parser.add_argument(
        "--use-cross-note-fallback", action="store_true",
        help="A2 Pass 4 跨 Note embedding fallback(同 document 其他 Note)",
    )
    parser.add_argument(
        "--embedding-model", type=str, default="all-MiniLM-L6-v2",
    )
    args = parser.parse_args()

    # --use-cross-note-fallback 必须先开 --use-embedding-fallback
    if args.use_cross_note_fallback and not args.use_embedding_fallback:
        print("warning: --use-cross-note-fallback 自动启用 --use-embedding-fallback")
        args.use_embedding_fallback = True

    embedding_model = None
    if args.use_embedding_fallback:
        from sentence_transformers import SentenceTransformer
        print(f"loading embedding model: {args.embedding_model} ...")
        embedding_model = SentenceTransformer(args.embedding_model)
        print(f"  Pass 3 threshold:label-aware (≤2 词 → 0.85,≥3 词 → 0.7)")
        if args.use_cross_note_fallback:
            print(f"  Pass 4 threshold:label-aware (≤2 词 → 0.85,≥3 词 → 0.75)")

    db = SessionLocal()
    try:
        # O4 gate(detect-only):note_ref 错配早停,防 §29 静默重演。
        # document_id=None 由 classify 内部封口(全库逐 doc 校验),非静默 no-op。
        from sla.harness.kg import NoteRefIntegrityError, validate_note_refs
        try:
            validate_note_refs(db, args.document_id)
        except NoteRefIntegrityError as e:
            print(f"FAIL note_ref integrity:\n{e}", file=sys.stderr)
            sys.exit(1)

        q = db.query(KGNode)
        if args.document_id is not None:
            q = q.filter(KGNode.document_id == args.document_id)
        nodes = q.order_by(KGNode.id).all()
        print(f"=== Processing {len(nodes)} KGNodes ===")
        if args.dry_run:
            print("(dry-run: 不写库)")

        stats: Counter[str] = Counter()
        embedding_hits: list[tuple[int, str, str, str, float]] = []
        cross_hits: list[tuple[int, str, str, int, str, float]] = []  # (id,lab,chap,tgt_note,head,sc)
        misses: list[tuple[int, str, str, dict | None]] = []
        updated = 0

        for n in nodes:
            r = link_node_to_anchor(
                n, db,
                embedding_model=embedding_model,
                cross_note_fallback=args.use_cross_note_fallback,
            )
            stats[r.match_type] += 1

            if r.match_type == "miss":
                misses.append((n.id, n.label, n.chapter_id or "?", r.debug))
            elif r.match_type == "embedding" and r.debug:
                embedding_hits.append(
                    (n.id, n.label, n.chapter_id or "?", r.debug["heading"], r.debug["score"])
                )
            elif r.match_type == "cross_embedding" and r.debug:
                cross_hits.append((
                    n.id, n.label, n.chapter_id or "?",
                    r.debug["target_note_id"], r.debug["heading"], r.debug["score"],
                ))

            if not args.dry_run:
                n.note_anchor_slug = r.note_anchor_slug
                n.cross_note_id = r.cross_note_id
                n.cross_note_slug = r.cross_note_slug
                updated += 1

        if not args.dry_run:
            db.commit()
            print(f"\n  wrote anchor fields to {updated} nodes")

        total = len(nodes)
        print(f"\n=== Match Distribution ===")
        for key in ("heading", "content", "embedding", "cross_embedding", "miss", "no_note", "no_sections"):
            cnt = stats.get(key, 0)
            pct = (100 * cnt / total) if total else 0
            bar = "█" * int(pct / 2)
            print(f"  {key:16s} {cnt:4d}  {pct:5.1f}%  {bar}")

        hit_rate = (
            stats.get("heading", 0) + stats.get("content", 0)
            + stats.get("embedding", 0) + stats.get("cross_embedding", 0)
        ) / total if total else 0
        print(f"\n  Total hit rate: {hit_rate * 100:.1f}%")

        if embedding_hits:
            print(f"\n=== Pass 3 Embedding Hits ({len(embedding_hits)}, sorted by score) ===")
            for nid, label, chap, heading, score in sorted(embedding_hits, key=lambda x: -x[4]):
                print(f"  node_id={nid:3d} chap={chap:8s} sim={score:.3f}  {label!r}")
                print(f"           → heading={heading!r}")

        if cross_hits:
            print(f"\n=== Pass 4 Cross-Note Hits ({len(cross_hits)}, sorted by score) ===")
            for nid, label, chap, tgt_note, heading, score in sorted(cross_hits, key=lambda x: -x[5]):
                print(f"  node_id={nid:3d} chap={chap:8s} → note_id={tgt_note}  sim={score:.3f}  {label!r}")
                print(f"           → heading={heading!r}")

        if misses and (args.show_misses or len(misses) <= 15):
            print(f"\n=== Misses ({len(misses)}) ===")
            for nid, label, chap, debug in misses:
                line = f"  node_id={nid:3d}  chapter={chap:8s}  label={label!r}"
                if debug:
                    if "cross_best_heading" in debug:
                        line += (
                            f"\n      home best: {debug['home_best_heading']!r} "
                            f"sim={debug['home_best_score']:.3f}"
                        )
                        line += (
                            f"\n      cross best: {debug['cross_best_heading']!r} "
                            f"sim={debug['cross_best_score']:.3f} "
                            f"(note_id={debug.get('cross_target_note_id')})"
                        )
                    elif "heading" in debug:
                        line += (
                            f"  [best heading={debug['heading']!r} "
                            f"sim={debug['score']:.3f}]"
                        )
                print(line)
    finally:
        db.close()

    print(f"\n[ok] backfill done")


if __name__ == "__main__":
    main()
