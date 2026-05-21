"""document_structure: 冻结的权威章节结构(Phase 1a)

Revision ID: 0006
Revises: 0005
Create Date: 2026-05-19

7 字段契约结构经 agent-外确定性 gate 校验后冻结于此。单一写者 = 冻结步;
读者 = detect_sections 新分支。(document_id, chapter_id) 唯一 = 幂等锚
(与 Chunk 去重键同族),重 ingest 复用、不静默重 spawn。
链 head 0005,不碰 0001-0005(memory:check-migration-before-schema)。
frozen_at server_default 镜像 0001/0002 全库 created_at 房规(0005 无之系异类),
对非-ORM 写者鲁棒(memory:soundness-claims-enumerate-all-writers)。
"""
from alembic import op
import sqlalchemy as sa


revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document_structure",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("document_id", sa.Integer(),
                  sa.ForeignKey("document.id", name="fk_document_structure_document"),
                  nullable=False),
        sa.Column("section_id", sa.String(100), nullable=False),   # 原 TOC '1.3'
        sa.Column("chapter_id", sa.String(100), nullable=False),    # DB 键 'ch1.3'
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("book_page_start", sa.Integer(), nullable=False),
        sa.Column("book_page_end", sa.Integer(), nullable=False),
        sa.Column("pdf_page_start", sa.Integer(), nullable=False),
        sa.Column("pdf_page_end", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(20), nullable=False),         # llm | heuristic
        sa.Column("frozen_at", sa.DateTime,
                  server_default=sa.func.current_timestamp(), nullable=False),
        sa.UniqueConstraint("document_id", "chapter_id",
                            name="uq_doc_structure_doc_chapter"),
    )


def downgrade() -> None:
    op.drop_table("document_structure")
