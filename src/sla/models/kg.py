"""KG (Knowledge Graph) ORM 模型。Phase 2-W3-3.

跟 0002_kg_schema.py migration 一一对应:
  - KGNode:concept / method / example 节点,external_id UNIQUE
  - KGEdge:requires / related_to / contrasts_with / example_of,
            复合 UNIQUE (source, target, type) 防重复关系

设计选择:
  - 不加 backref 到 Document / Note(避免 domain.py 跟 kg.py 互相 import)
  - 自引用关系(KGNode ⇄ KGEdge ⇄ KGNode):outgoing_edges / incoming_edges
    用字符串 foreign_keys 解 forward ref
"""
from datetime import datetime

from sqlalchemy import Boolean, ForeignKey, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sla.db import Base


class KGNode(Base):
    """KG 概念节点。"""

    __tablename__ = "kg_node"

    id: Mapped[int] = mapped_column(primary_key=True)
    # external_id 格式: '<doc_id>_<chap_id>_<slug>',服务端根据 LLM 给的 label 生成
    # 唯一性保证跨章同名 concept 合并到一行
    external_id: Mapped[str] = mapped_column(String(255), unique=True)
    type: Mapped[str] = mapped_column(String(20))  # concept | method | example
    label: Mapped[str] = mapped_column(String(500))
    description: Mapped[str | None] = mapped_column(Text, default=None)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id"))
    chapter_id: Mapped[str | None] = mapped_column(String(100), default=None)
    # 首次抽出该 concept 的 Note;跨章合并时保留首次值,description 可累积更新
    note_ref_id: Mapped[int | None] = mapped_column(
        ForeignKey("note.id"), default=None,
    )
    tags: Mapped[list | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    # Phase 3-1:Note 内 ## heading slug,viewer 点节点跳转用;
    # 后台跑 scripts/backfill_kg_anchor.py 计算并回填
    note_anchor_slug: Mapped[str | None] = mapped_column(String(200), default=None)
    # Phase 3-2 A2:跨 Note anchor —— home Note 没专门 section 时,跳到别 Note 的某 ##
    # 跳转优先级:note_anchor_slug > cross_note_* > Note 起点
    cross_note_id: Mapped[int | None] = mapped_column(
        ForeignKey("note.id"), default=None,
    )
    cross_note_slug: Mapped[str | None] = mapped_column(String(200), default=None)

    # 自引用:本节点作为 source 出发的边 / 作为 target 到达的边
    # foreign_keys 用字符串 forward ref 避免 KGEdge 未定义时报错
    outgoing_edges: Mapped[list["KGEdge"]] = relationship(
        back_populates="source_node",
        foreign_keys="KGEdge.source_node_id",
    )
    incoming_edges: Mapped[list["KGEdge"]] = relationship(
        back_populates="target_node",
        foreign_keys="KGEdge.target_node_id",
    )


class KGEdge(Base):
    """KG 关系边。"""

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
    # True for requires/example_of(有向),False for related_to/contrasts_with(无向)
    directed: Mapped[bool] = mapped_column(Boolean, default=True)
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    # 两端节点。foreign_keys 指明用哪个 FK(KGEdge 有两个 FK 到 kg_node)
    source_node: Mapped["KGNode"] = relationship(
        back_populates="outgoing_edges",
        foreign_keys=[source_node_id],
    )
    target_node: Mapped["KGNode"] = relationship(
        back_populates="incoming_edges",
        foreign_keys=[target_node_id],
    )
