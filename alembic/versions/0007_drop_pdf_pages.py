"""drop pdf_page_start/pdf_page_end/book_page_end from document_structure

human-anchored pivot:结构=一发 LLM 从 TOC 抽,只需 book_page_start;
pdf 物理页/区间(原 1b agent + 1c gate 产物)随 1b-1e 退役一并去。
终列 8。链 head 0006,不碰 0001-0006。

batch 必须(SQLite 无裸 DROP COLUMN;0001-0006 全 create_table 无先例)。
copy_from 给【显式表】→ 全程不反射:frozen_at 的 server_default(0006 的
1a rider,非-ORM 写者鲁棒锚 memory:soundness-claims-enumerate-all-writers)
显式带上,堵掉 batch-reflection 静默丢之面
(memory:sqlite-batch-preserve-server-default)。

Revision ID: 0007
Revises: 0006
Create Date: 2026-05-20
"""
from alembic import op
import sqlalchemy as sa


revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def _table(meta, *, with_pdf_cols: bool) -> sa.Table:
    """document_structure 的显式权威态(batch copy_from 用,杜绝反射)。
    with_pdf_cols=True → 0006 的 11 列态;False → 0007 的 8 列态。
    两态都显式带 frozen_at server_default。"""
    cols = [
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("document_id", sa.Integer(),
                  sa.ForeignKey("document.id",
                                name="fk_document_structure_document"),
                  nullable=False),
        sa.Column("section_id", sa.String(100), nullable=False),
        sa.Column("chapter_id", sa.String(100), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("book_page_start", sa.Integer(), nullable=False),
    ]
    if with_pdf_cols:
        cols += [
            sa.Column("book_page_end", sa.Integer(), nullable=False),
            sa.Column("pdf_page_start", sa.Integer(), nullable=False),
            sa.Column("pdf_page_end", sa.Integer(), nullable=False),
        ]
    cols += [
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("frozen_at", sa.DateTime,
                  server_default=sa.func.current_timestamp(),
                  nullable=False),                      # ← 1a rider,显式
    ]
    return sa.Table(
        "document_structure", meta, *cols,
        sa.UniqueConstraint("document_id", "chapter_id",
                            name="uq_doc_structure_doc_chapter"),
    )


def upgrade() -> None:
    with op.batch_alter_table(
        "document_structure", schema=None,
        copy_from=_table(sa.MetaData(), with_pdf_cols=True),   # 显式源,不反射
    ) as b:
        b.drop_column("pdf_page_end")
        b.drop_column("pdf_page_start")
        b.drop_column("book_page_end")


def downgrade() -> None:
    # 回 11 列。pdf 页回滚后无语义 → 占位 server_default="0"(表非空时
    # NOT NULL 才能 roundtrip;已认这条)。frozen_at 仍走显式 copy_from。
    with op.batch_alter_table(
        "document_structure", schema=None,
        copy_from=_table(sa.MetaData(), with_pdf_cols=False),  # 8 列显式源
    ) as b:
        b.add_column(sa.Column("book_page_end", sa.Integer(),
                               server_default="0", nullable=False))
        b.add_column(sa.Column("pdf_page_start", sa.Integer(),
                               server_default="0", nullable=False))
        b.add_column(sa.Column("pdf_page_end", sa.Integer(),
                               server_default="0", nullable=False))
