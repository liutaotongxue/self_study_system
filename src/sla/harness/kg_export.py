"""KG export (Phase 2-W3-6) -- native schema export, no lossy adaptation.

**Important principle**: we do not make format compromises for math_graph (or any downstream consumer).
We emit the native schema of 3 node types (concept/method/example) + 4 edge types
(requires/related_to/contrasts_with/example_of).
If a downstream viewer wants to consume it, **they update their loader**; our format is canonical.

Output structure:
    {
      "document_id": int,
      "document_title": str | None,
      "chapters": list[str],         # sorted list of chapter_id
      "nodes": [ ... ],              # see NodePayload
      "edges": [ ... ],              # see EdgePayload (references source/target external_id)
      "stats": { ... },              # aggregate stats so the viewer does less front-end computation
    }
"""
from collections import Counter

from sla.db import SessionLocal
from sla.models.domain import Document
from sla.models.kg import KGEdge, KGNode


def to_native_json(document_id: int) -> dict:
    """Export the specified document's KG as a JSON-serializable dict.

    Edges reference nodes via `source_external_id` / `target_external_id`, **not internal DB ids**
    -- letting consumers work in business identifiers without being bound to our DB schema.
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

        # Edges: both endpoints must belong to this document (KGEdge should not cross documents, defensive here)
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
                "note_anchor_slug": n.note_anchor_slug,    # Phase 3-1: ## section slug within the Note
                "cross_note_id": n.cross_note_id,          # Phase 3-2 A2: cross-Note anchor target
                "cross_note_slug": n.cross_note_slug,
                "tags": n.tags,
                "created_at": n.created_at.isoformat() if n.created_at else None,
            }
            for n in nodes
        ]

        edges_payload = [
            {
                "id": e.id,                                # stable DB id; used as the front-end vis-network edge id
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
