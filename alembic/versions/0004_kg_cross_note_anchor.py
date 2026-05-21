"""kg_node: add cross-Note anchor for cross-chapter概念引用 (Phase 3-2 A2)

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-15

新增两个 nullable 字段:
  - cross_note_id (FK note.id):跳转目标 Note(跟 note_ref_id 可不同)
  - cross_note_slug (VARCHAR(200)):该目标 Note 内的 ## heading slug

用途:某 KG 节点的 home Note(note_ref_id)里没有专门 section 讲它,
     但同 document 其他 Note 里有 —— 这个字段对 viewer 提供跨 Note 跳转锚。
     A1 后剩 5 个 miss 节点的痛点。

设计原则:
  - 不动 KG 拓扑(不合并节点),只丰富 anchor 路由能力
  - viewer 跳转优先级:note_anchor_slug > cross_note_slug > Note 起点
  - 概念真合并(canonical_id)留给未来 phase,需要复习系统信号支撑
"""
from alembic import op
import sqlalchemy as sa


revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # SQLite batch_alter_table 要求 FK 有 name
    with op.batch_alter_table("kg_node") as batch_op:
        batch_op.add_column(
            sa.Column(
                "cross_note_id",
                sa.Integer(),
                sa.ForeignKey("note.id", name="fk_kg_node_cross_note_id_note"),
                nullable=True,
                comment="跨 Note anchor 目标 note.id(home note 无对应 section 时用)",
            ),
        )
        batch_op.add_column(
            sa.Column(
                "cross_note_slug",
                sa.String(200),
                nullable=True,
                comment="cross_note_id 那个 Note 内的 ## heading slug",
            ),
        )


def downgrade() -> None:
    with op.batch_alter_table("kg_node") as batch_op:
        batch_op.drop_column("cross_note_slug")
        batch_op.drop_column("cross_note_id")
