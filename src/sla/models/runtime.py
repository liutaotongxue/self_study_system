"""Harness runtime ORM models."""
import os
import sys
from datetime import datetime

from sqlalchemy import JSON, Boolean, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sla.db import Base


class Task(Base):
    __tablename__ = "task"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int | None] = mapped_column(ForeignKey("document.id"), default=None)
    kind: Mapped[str] = mapped_column(String(50))
    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[str | None] = mapped_column(Text, default=None)
    system_prompt: Mapped[str] = mapped_column(Text)
    user_prompt: Mapped[str] = mapped_column(Text)
    policy: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    runs: Mapped[list["Run"]] = relationship(back_populates="task")


class Run(Base):
    __tablename__ = "run"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("task.id"))
    status: Mapped[str] = mapped_column(String(20))
    started_at: Mapped[datetime | None] = mapped_column(default=None)
    ended_at: Mapped[datetime | None] = mapped_column(default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)

    task: Mapped["Task"] = relationship(back_populates="runs")
    steps: Mapped[list["Step"]] = relationship(back_populates="run", order_by="Step.idx")
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="run")


class Step(Base):
    __tablename__ = "step"
    __table_args__ = (UniqueConstraint("run_id", "idx", name="uq_step_run_idx"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run.id"))
    idx: Mapped[int] = mapped_column()
    model_input: Mapped[dict | list | None] = mapped_column(JSON, default=None)
    model_output: Mapped[dict | list | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    run: Mapped["Run"] = relationship(back_populates="steps")
    tool_calls: Mapped[list["ToolCall"]] = relationship(back_populates="step")


class ToolCall(Base):
    __tablename__ = "tool_call"

    id: Mapped[int] = mapped_column(primary_key=True)
    step_id: Mapped[int] = mapped_column(ForeignKey("step.id"))
    anthropic_tool_use_id: Mapped[str | None] = mapped_column(String(100), default=None)
    name: Mapped[str] = mapped_column(String(100))
    input: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    step: Mapped["Step"] = relationship(back_populates="tool_calls")
    result: Mapped["ToolResult | None"] = relationship(back_populates="tool_call", uselist=False)


class ToolResult(Base):
    __tablename__ = "tool_result"

    id: Mapped[int] = mapped_column(primary_key=True)
    tool_call_id: Mapped[int] = mapped_column(ForeignKey("tool_call.id"))
    status: Mapped[str] = mapped_column(String(20))  # ok/error/denied
    content: Mapped[dict | list | str | None] = mapped_column(JSON, default=None)
    reason: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    tool_call: Mapped["ToolCall"] = relationship(back_populates="result")


class Artifact(Base):
    __tablename__ = "artifact"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run.id"))
    kind: Mapped[str] = mapped_column(String(50))  # note/question/kg_node
    ref_table: Mapped[str] = mapped_column(String(50))
    ref_id: Mapped[int] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    run: Mapped["Run"] = relationship(back_populates="artifacts")


class EvalResult(Base):
    __tablename__ = "eval_result"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("run.id"))
    rule_passed: Mapped[bool] = mapped_column(Boolean)
    rule_details: Mapped[dict | None] = mapped_column(JSON, default=None)
    llm_score: Mapped[int | None] = mapped_column(default=None)
    llm_rationale: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class GenerationJob(Base):
    """A UI-triggered per-chapter generation/rebuild job.

    State machine + reconcile loop (uses pid + timeout as two independent
    signals).
    """

    __tablename__ = "generation_job"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id"))
    chapter_id: Mapped[str] = mapped_column(String(200))
    mode: Mapped[str] = mapped_column(String(16))                       # generate | rebuild
    status: Mapped[str] = mapped_column(String(16), default="queued")   # queued|running|done|failed
    step: Mapped[str | None] = mapped_column(String(16), default=None)  # study_book|build_kg|backfill
    pid: Mapped[int | None] = mapped_column(default=None)               # runner OS pid (POSIX liveness)
    detail: Mapped[str | None] = mapped_column(Text, default=None)      # Failure / gate-stop message
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(default=None)


# Deadlock floor, not an SLA: pick a value that no legal generation run will
# hit. If reconcile marks failed without killing the process and a false
# timeout fires on a live job, the lock would release -> concurrent
# generations could double-spend and accumulate Notes (breaking the three
# semantic states). False-kill-of-live-job is much worse than slow-recover-
# of-dead-job. The real generation cost was never measured (study_book was
# skipped in those experiments), so we pick a generous high value rather
# than estimating an SLA — see no-fabricated-numbers principle.
GENERATION_MAX_MINUTES = 90


def reconcile_generation_jobs(db):
    """Lazy on-read reconciler (no background scheduler — minimal single-user).

    Two INDEPENDENT signals, each sufficient on its own to declare a job dead:
      - pid death (POSIX only): os.kill(pid, 0) raises ProcessLookupError.
      - timeout: running for longer than MAX (also our pid-reuse backstop;
        the only signal available on Windows).

    Flagged failed WITHOUT killing the process (killing mid-LLM-write risks
    data corruption; the floor is set high specifically to avoid false-kill).

    Windows guard: os.kill(pid, 0) on Windows is implemented as
    TerminateProcess (it actually kills the process!), not a probe — so we
    skip the pid check there and rely on timeout only. The 90-minute floor
    still clears stuck locks cross-platform. Windows-native probe is
    backlogged (O9).
    """
    now = datetime.utcnow()
    changed = False
    for j in db.query(GenerationJob).filter(GenerationJob.status == "running").all():
        dead, why = False, ""
        if j.pid is not None and sys.platform != "win32":   # POSIX-only liveness
            try:
                os.kill(j.pid, 0)
            except ProcessLookupError:
                dead, why = True, f"runner pid {j.pid} 不存活"
            except PermissionError:
                pass
        if not dead and (now - j.created_at).total_seconds() > GENERATION_MAX_MINUTES * 60:
            dead, why = True, f"running 超 {GENERATION_MAX_MINUTES}min,presumed dead"
        if dead:
            j.status, j.detail, j.ended_at = "failed", f"reconcile: {why}", now
            changed = True
    if changed:
        db.commit()
