"""KG export(Phase 2-W3-6)—— 原生 schema 导出,无 lossy 适配。

**重要原则**:不向 math_graph(或任何 downstream consumer)做格式妥协。
我们输出 3 node types(concept/method/example)+ 4 edge types
(requires/related_to/contrasts_with/example_of)的原生 schema。
如果 downstream viewer 要消费,**它们改 loader**;我们的格式是 canonical。

输出结构:
    {
      "document_id": int,
      "document_title": str | None,
      "chapters": list[str],         # 排序后的 chapter_id 列表
      "nodes": [ ... ],              # 见 NodePayload
      "edges": [ ... ],              # 见 EdgePayload (引用 source/target external_id)
      "stats": { ... },              # 统计聚合,给 viewer 减少前端计算
    }
"""
from collections import Counter

from sla.db import SessionLocal
from sla.models.domain import Document
from sla.models.kg import KGEdge, KGNode


def to_native_json(document_id: int) -> dict:
    """把指定 document 的 KG 导出为 JSON-serializable dict。

    Edge 通过 `source_external_id` / `target_external_id` 引用 node,**不用内部 DB id**
    —— 让消费者按业务标识符工作,不被我们 DB schema 绑死。
    """
    db = SessionLocal()
    try:
        document = db.get(Document, document_id)
        document_title = document.title if document is not None else None

        nodes = (
            db.query(KGNode)
            .filter(KGNode.document_id == document_id)
            .order_by(KGNode.id)
            .all()
        )

        if not nodes:
            return {
                "document_id": document_id,
                "document_title": document_title,
                "chapters": [],
                "nodes": [],
                "edges": [],
                "stats": {
                    "total_nodes": 0,
                    "total_edges": 0,
                    "nodes_by_type": {},
                    "edges_by_type": {},
                },
            }

        node_ids = {n.id for n in nodes}
        id_to_ext = {n.id: n.external_id for n in nodes}

        # 边:两端节点都属于这个 document(理论上 KGEdge 不跨 document,这里 defensive)
        edges = (
            db.query(KGEdge)
            .filter(KGEdge.source_node_id.in_(node_ids))
            .filter(KGEdge.target_node_id.in_(node_ids))
            .order_by(KGEdge.id)
            .all()
        )

        chapters = sorted({n.chapter_id for n in nodes if n.chapter_id})

        nodes_payload = [
            {
                "external_id": n.external_id,
                "type": n.type,                            # concept | method | example
                "label": n.label,
                "description": n.description,
                "chapter_id": n.chapter_id,
                "note_ref_id": n.note_ref_id,
                "note_anchor_slug": n.note_anchor_slug,    # Phase 3-1:Note 内 ## section slug
                "cross_note_id": n.cross_note_id,          # Phase 3-2 A2:跨 Note anchor 目标
                "cross_note_slug": n.cross_note_slug,
                "tags": n.tags,
                "created_at": n.created_at.isoformat() if n.created_at else None,
            }
            for n in nodes
        ]

        edges_payload = [
            {
                "id": e.id,                                # stable DB id,前端 vis-network edge id 用
                "source_external_id": id_to_ext[e.source_node_id],
                "target_external_id": id_to_ext[e.target_node_id],
                "type": e.type,    # requires | related_to | contrasts_with | example_of
                "directed": e.directed,
                "notes": e.notes,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in edges
        ]

        stats = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "nodes_by_type": dict(Counter(n.type for n in nodes)),
            "edges_by_type": dict(Counter(e.type for e in edges)),
        }

        return {
            "document_id": document_id,
            "document_title": document_title,
            "chapters": chapters,
            "nodes": nodes_payload,
            "edges": edges_payload,
            "stats": stats,
        }
    finally:
        db.close()
