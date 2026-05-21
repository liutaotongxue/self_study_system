"""Harness 运行时 ORM 模型。"""
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
    """P3b:UI 触发的章节 生成/重建 任务。状态机 + reconcile(pid+timeout 双信号)。"""

    __tablename__ = "generation_job"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id"))
    chapter_id: Mapped[str] = mapped_column(String(200))
    mode: Mapped[str] = mapped_column(String(16))                       # generate | rebuild
    status: Mapped[str] = mapped_column(String(16), default="queued")   # queued|running|done|failed
    step: Mapped[str | None] = mapped_column(String(16), default=None)  # study_book|build_kg|backfill
    pid: Mapped[int | None] = mapped_column(default=None)               # runner OS pid(POSIX liveness)
    detail: Mapped[str | None] = mapped_column(Text, default=None)      # 失败/gate-stop 原文
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(default=None)


# deadlock-floor,非 SLA:设到【合法 generate 撞不到】的高位 —— reconcile 标 failed
# 不杀进程,若 false-timeout 活 job→锁释放→并发双 generate→双花+study_book累加
# (击穿三态)。依据:false-kill-live-job ≫ 坏于 慢恢复-dead-job;真 generate 耗时
# 测法故意跳 study_book 未实测,故取高位余量而非估测 SLA(no-fabricated-numbers)。
GENERATION_MAX_MINUTES = 90


def reconcile_generation_jobs(db):
    """惰性 on-read(无后台调度,单用户最小化)。两【独立】信号,各自能单独判死:
      - pid 死(仅 POSIX):os.kill(pid,0) ProcessLookupError
      - 超时:running 超 MAX(独立兜底 pid-reuse;且为 Windows 唯一信号)
    标 failed【不杀进程】(杀 mid-LLM-write 有腐败风险;floor 设高位规避 false-kill)。
    Windows guard:os.kill(pid,0) 在 Win = TerminateProcess(杀!)非探测 → 跳过,
    降级 timeout-only(90min floor 跨平台仍清死锁)。Windows-native probe = O9 named-defer。"""
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
