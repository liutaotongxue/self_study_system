"""KG (Knowledge Graph) ORM models.

One-to-one mapping with the 0002_kg_schema.py migration:
  - KGNode: concept / method / example nodes; external_id UNIQUE.
  - KGEdge: requires / related_to / contrasts_with / example_of;
            composite UNIQUE (source, target, type) prevents duplicate
            relations.

Design choices:
  - No backref to Document / Note (avoids circular imports between
    domain.py and kg.py).
  - Self-referential relation (KGNode <-> KGEdge <-> KGNode):
    outgoing_edges / incoming_edges use string foreign_keys to resolve
    the forward reference.
"""
from datetime import datetime

from sqlalchemy import Boolean, ForeignKey, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sla.db import Base


class KGNode(Base):
    """A concept node in the KG."""

    __tablename__ = "kg_node"

    id: Mapped[int] = mapped_column(primary_key=True)
    # external_id format: '<doc_id>_<chap_id>_<slug>', generated server-side
    # from the LLM-provided label. Uniqueness ensures that the same concept
    # appearing in multiple chapters merges into a single row.
    external_id: Mapped[str] = mapped_column(String(255), unique=True)
    type: Mapped[str] = mapped_column(String(20))  # concept | method | example
    label: Mapped[str] = mapped_column(String(500))
    description: Mapped[str | None] = mapped_column(Text, default=None)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id"))
    chapter_id: Mapped[str | None] = mapped_column(String(100), default=None)
    # The Note that first extracted this concept. Cross-chapter merges
    # keep the first value; description can still be accumulated/updated.
    note_ref_id: Mapped[int | None] = mapped_column(
        ForeignKey("note.id"), default=None,
    )
    tags: Mapped[list | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    # In-Note ## heading slug for click-to-jump from the viewer.
    # Computed and backfilled by scripts/backfill_kg_anchor.py.
    note_anchor_slug: Mapped[str | None] = mapped_column(String(200), default=None)
    # Cross-Note anchor: when the home Note has no dedicated section for
    # this concept, jump to another Note's heading instead.
    # Jump priority: note_anchor_slug > cross_note_* > start of Note.
    cross_note_id: Mapped[int | None] = mapped_column(
        ForeignKey("note.id"), default=None,
    )
    cross_note_slug: Mapped[str | None] = mapped_column(String(200), default=None)

    # Self-referential: edges originating from this node / arriving at it.
    # foreign_keys uses string forward refs since KGEdge isn't defined yet.
    outgoing_edges: Mapped[list["KGEdge"]] = relationship(
        back_populates="source_node",
        foreign_keys="KGEdge.source_node_id",
    )
    incoming_edges: Mapped[list["KGEdge"]] = relationship(
        back_populates="target_node",
        foreign_keys="KGEdge.target_node_id",
    )


class KGEdge(Base):
    """A relation edge in the KG."""

    __tablename__ = "kg_edge"
    __table_args__ = (
        UniqueConstraint(
            "source_node_id", "target_node_id", "type",
            name="uq_kg_edge_source_target_type",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source_node_id: Mapped[int] = mapped_column(ForeignKey("kg_node.id"))
    target_node_id: Mapped[int] = mapped_column(ForeignKey("kg_node.id"))
    # requires | related_to | contrasts_with | example_of
    type: Mapped[str] = mapped_column(String(30))
    # True for requires/example_of (directed), False for related_to/contrasts_with (undirected).
    directed: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    # Endpoints. foreign_keys disambiguates which FK to use (KGEdge has two FKs into kg_node).
    source_node: Mapped["KGNode"] = relationship(
        back_populates="outgoing_edges",
        foreign_keys=[source_node_id],
    )
    target_node: Mapped["KGNode"] = relationship(
        back_populates="incoming_edges",
        foreign_keys=[target_node_id],
    )
