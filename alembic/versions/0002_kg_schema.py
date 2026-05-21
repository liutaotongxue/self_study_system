"""kg schema: kg_node + kg_edge tables (Phase 2 W3-2)

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-14

新增两张表:
  - kg_node:KG 概念节点(concept / method / example 三种 type)
  - kg_edge:KG 关系边(requires / related_to / contrasts_with / example_of)

设计决策(详见 RETROSPECTIVE.md §15):
  - kg_node 是一等公民,独立表(不沾 artifact)
  - external_id 唯一,格式 '<document_id>_<chapter_id>_<slug>',服务端生成
  - kg_edge 复合 UNIQUE (source, target, type) 防 LLM 重复抽
  - Agent 主 schema,math_graph 通过导出层 lossy 适配
"""
from alembic import op
import sqlalchemy as sa


revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---------- kg_node ----------
    op.create_table(
        "kg_node",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "external_id", sa.String(255), nullable=False, unique=True,
            comment="格式 '<doc_id>_<chap_id>_<slug>',服务端生成,跨章同名合并",
        ),
        sa.Column(
            "type", sa.String(20), nullable=False,
            comment="concept | method | example",
        ),
        sa.Column("label", sa.String(500), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column(
            "document_id", sa.Integer,
            sa.ForeignKey("document.id"), nullable=False,
            comment="所属 document(KG 是 document-scoped)",
        ),
        sa.Column(
            "chapter_id", sa.String(100),
            comment="首次出现的章节(跨章合并时保留首次值)",
        ),
        sa.Column(
            "note_ref_id", sa.Integer,
            sa.ForeignKey("note.id"),
            comment="抽自哪个 Note(首次抽取的 Note;跨章合并时保留首次值)",
        ),
        sa.Column("tags", sa.JSON),
        sa.Column(
            "created_at", sa.DateTime,
            server_default=sa.func.current_timestamp(), nullable=False,
        ),
    )
    op.create_index("ix_kg_node_document", "kg_node", ["document_id"])
    op.create_index("ix_kg_node_chapter", "kg_node", ["chapter_id"])
    op.create_index("ix_kg_node_type", "kg_node", ["type"])

    # ---------- kg_edge ----------
    op.create_table(
        "kg_edge",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "source_node_id", sa.Integer,
            sa.ForeignKey("kg_node.id"), nullable=False,
        ),
        sa.Column(
            "target_node_id", sa.Integer,
            sa.ForeignKey("kg_node.id"), nullable=False,
        ),
        sa.Column(
            "type", sa.String(30), nullable=False,
            comment="requires | related_to | contrasts_with | example_of",
        ),
        sa.Column(
            "directed", sa.Boolean, nullable=False, server_default=sa.true(),
            comment="True for requires/example_of, False for related_to/contrasts_with",
        ),
        sa.Column("notes", sa.Text),
        sa.Column(
            "created_at", sa.DateTime,
            server_default=sa.func.current_timestamp(), nullable=False,
        ),
        # 防重复关系:同 source → target 同 type 只一条
        sa.UniqueConstraint(
            "source_node_id", "target_node_id", "type",
            name="uq_kg_edge_source_target_type",
        ),
    )
    op.create_index("ix_kg_edge_source", "kg_edge", ["source_node_id"])
    op.create_index("ix_kg_edge_target", "kg_edge", ["target_node_id"])


def downgrade() -> None:
    # 先删 kg_edge(FK 到 kg_node),再删 kg_node
    for tbl in ["kg_edge", "kg_node"]:
        op.drop_table(tbl)
