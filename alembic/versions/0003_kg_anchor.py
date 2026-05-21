"""kg_node: add note_anchor_slug for paragraph-level navigation (Phase 3-1)

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-15

新增 kg_node.note_anchor_slug 字段(VARCHAR(200), nullable),
用于解决 3A 自用中 6 次撞墙的 "想点节点看出处但 Note 内没锚点" 痛点。

设计:
  - 锚到 Note 内的 markdown ## heading slug,而不是 chunk
  - 因为 Note 是 LLM 归一化产物,label 字面 match 率天然高;
    chunk 是 raw PDF,有 ligature / 断行 / 旧术语形式,match 率低
  - 见 docs/self_use_log.md "1 周后诊断" 段
"""
from alembic import op
import sqlalchemy as sa


revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "kg_node",
        sa.Column(
            "note_anchor_slug",
            sa.String(200),
            nullable=True,
            comment="Markdown heading slug,viewer 点节点后跳到 Note 内对应 section",
        ),
    )


def downgrade() -> None:
    op.drop_column("kg_node", "note_anchor_slug")
