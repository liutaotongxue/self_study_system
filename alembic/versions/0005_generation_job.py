"""generation_job: P3b UI-triggered chapter generate/rebuild jobs

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-17

P3b:网页触发"生成/重建"长 LLM 任务,subprocess + job-status + 轮询。
reconcile(pid+timeout 双信号)清死锁;mode=generate(study_book→build_kg→backfill)
/ rebuild(build_kg→backfill,无 study_book)。
"""
from alembic import op
import sqlalchemy as sa


revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "generation_job",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("document_id", sa.Integer(),
                  sa.ForeignKey("document.id", name="fk_generation_job_document"),
                  nullable=False),
        sa.Column("chapter_id", sa.String(200), nullable=False),
        sa.Column("mode", sa.String(16), nullable=False),       # generate | rebuild
        sa.Column("status", sa.String(16), nullable=False,
                  server_default="queued"),                     # queued|running|done|failed
        sa.Column("step", sa.String(16), nullable=True),        # study_book|build_kg|backfill
        sa.Column("pid", sa.Integer(), nullable=True),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("generation_job")
