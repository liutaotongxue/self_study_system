"""initial schema: runtime + domain tables

Revision ID: 0001
Revises:
Create Date: 2026-05-08

包含两类表:
- 应用层(domain):domain / document / chunk / note / question
- Harness 运行时(runtime):task / run / step / tool_call / tool_result / artifact / eval_result
"""
from alembic import op
import sqlalchemy as sa


revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---------- 应用层 ----------

    op.create_table(
        "domain",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.current_timestamp(), nullable=False),
    )

    op.create_table(
        "document",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("domain_id", sa.Integer, sa.ForeignKey("domain.id"), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("file_path", sa.String(1000)),
        sa.Column("total_pages", sa.Integer),
        sa.Column("parsed_outline", sa.JSON),
        sa.Column("status", sa.String(50), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.current_timestamp(), nullable=False),
    )

    op.create_table(
        "chunk",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("document_id", sa.Integer, sa.ForeignKey("document.id"), nullable=False),
        sa.Column("chapter_id", sa.String(100)),
        sa.Column("page_start", sa.Integer),
        sa.Column("page_end", sa.Integer),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", sa.LargeBinary),  # Phase 1C 起填
        sa.Column("created_at", sa.DateTime, server_default=sa.func.current_timestamp(), nullable=False),
    )
    op.create_index("ix_chunk_document_chapter", "chunk", ["document_id", "chapter_id"])

    op.create_table(
        "note",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("document_id", sa.Integer, sa.ForeignKey("document.id"), nullable=False),
        sa.Column("chapter_id", sa.String(100)),
        sa.Column("content_md", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.current_timestamp(), nullable=False),
    )

    op.create_table(
        "question",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("document_id", sa.Integer, sa.ForeignKey("document.id"), nullable=False),
        sa.Column("chapter_id", sa.String(100)),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("answer", sa.Text),
        sa.Column("difficulty", sa.Integer),
        sa.Column("tags", sa.JSON),
        sa.Column("source", sa.String(20), nullable=False, server_default="agent"),  # agent/user
        sa.Column("status", sa.String(20), nullable=False, server_default="unattempted"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.current_timestamp(), nullable=False),
    )

    # ---------- Harness 运行时 ----------

    op.create_table(
        "task",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("document_id", sa.Integer, sa.ForeignKey("document.id")),
        sa.Column("kind", sa.String(50), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("system_prompt", sa.Text, nullable=False),
        sa.Column("user_prompt", sa.Text, nullable=False),
        sa.Column("policy", sa.JSON, nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.current_timestamp(), nullable=False),
    )

    op.create_table(
        "run",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("task_id", sa.Integer, sa.ForeignKey("task.id"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),  # running/completed/failed/policy_halted
        sa.Column("started_at", sa.DateTime),
        sa.Column("ended_at", sa.DateTime),
        sa.Column("error", sa.Text),
    )
    op.create_index("ix_run_task", "run", ["task_id"])

    op.create_table(
        "step",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("run_id", sa.Integer, sa.ForeignKey("run.id"), nullable=False),
        sa.Column("idx", sa.Integer, nullable=False),
        sa.Column("model_input", sa.JSON),
        sa.Column("model_output", sa.JSON),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.current_timestamp(), nullable=False),
        sa.UniqueConstraint("run_id", "idx", name="uq_step_run_idx"),
    )

    op.create_table(
        "tool_call",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("step_id", sa.Integer, sa.ForeignKey("step.id"), nullable=False),
        sa.Column("anthropic_tool_use_id", sa.String(100)),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("input", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.current_timestamp(), nullable=False),
    )

    op.create_table(
        "tool_result",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("tool_call_id", sa.Integer, sa.ForeignKey("tool_call.id"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),  # ok/error/denied
        sa.Column("content", sa.JSON),
        sa.Column("reason", sa.Text),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.current_timestamp(), nullable=False),
    )

    op.create_table(
        "artifact",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("run_id", sa.Integer, sa.ForeignKey("run.id"), nullable=False),
        sa.Column("kind", sa.String(50), nullable=False),  # note/question/kg_node(后续)
        sa.Column("ref_table", sa.String(50), nullable=False),
        sa.Column("ref_id", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.current_timestamp(), nullable=False),
    )
    op.create_index("ix_artifact_run", "artifact", ["run_id"])

    op.create_table(
        "eval_result",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("run_id", sa.Integer, sa.ForeignKey("run.id"), nullable=False),
        sa.Column("rule_passed", sa.Boolean, nullable=False),
        sa.Column("rule_details", sa.JSON),
        sa.Column("llm_score", sa.Integer),
        sa.Column("llm_rationale", sa.Text),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.current_timestamp(), nullable=False),
    )


def downgrade() -> None:
    for tbl in [
        "eval_result", "artifact", "tool_result", "tool_call", "step", "run", "task",
        "question", "note", "chunk", "document", "domain",
    ]:
        op.drop_table(tbl)
