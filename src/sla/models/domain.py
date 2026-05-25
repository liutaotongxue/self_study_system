"""Application-level ORM models."""
from datetime import datetime

from sqlalchemy import JSON, ForeignKey, LargeBinary, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sla.db import Base


class Domain(Base):
    __tablename__ = "domain"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    documents: Mapped[list["Document"]] = relationship(back_populates="domain")


class Document(Base):
    __tablename__ = "document"

    id: Mapped[int] = mapped_column(primary_key=True)
    domain_id: Mapped[int] = mapped_column(ForeignKey("domain.id"))
    title: Mapped[str] = mapped_column(String(500))
    file_path: Mapped[str | None] = mapped_column(String(1000), default=None)
    total_pages: Mapped[int | None] = mapped_column(default=None)
    parsed_outline: Mapped[dict | None] = mapped_column(JSON, default=None)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    domain: Mapped["Domain"] = relationship(back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document")


class Chunk(Base):
    __tablename__ = "chunk"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id"))
    chapter_id: Mapped[str | None] = mapped_column(String(100), default=None)
    page_start: Mapped[int | None] = mapped_column(default=None)
    page_end: Mapped[int | None] = mapped_column(default=None)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, default=None)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    document: Mapped["Document"] = relationship(back_populates="chunks")


class Note(Base):
    __tablename__ = "note"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id"))
    chapter_id: Mapped[str | None] = mapped_column(String(100), default=None)
    content_md: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class Question(Base):
    __tablename__ = "question"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id"))
    chapter_id: Mapped[str | None] = mapped_column(String(100), default=None)
    content: Mapped[str] = mapped_column(Text)
    answer: Mapped[str | None] = mapped_column(Text, default=None)
    difficulty: Mapped[int | None] = mapped_column(default=None)
    tags: Mapped[list | None] = mapped_column(JSON, default=None)
    source: Mapped[str] = mapped_column(String(20), default="agent")  # agent/user
    status: Mapped[str] = mapped_column(String(20), default="unattempted")
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class DocumentStructure(Base):
    """Full-book chapter structure (human-anchored pivot).

    Writer = S2 "mark TOC pages -> one-shot LLM extraction" (single call,
        not an agent).
    Readers = S3 status page (this table is the spine that LEFT JOINs
        Chunk/Note/KGNode) + S5 (full-book outline injected into the
        learning agent context).
    UNIQUE (document_id, chapter_id) is the idempotency anchor — same
    dedup family as Chunk — so "re-extract skip / --force overwrite"
    is enforced at the DB layer, not just by code convention.
    chapter_id has the form 'ch1' / 'ch1.3' and must match
    Chunk.chapter_id exactly (S3 LEFT JOIN spine; any drift makes the
    status page silently empty).
    """
    __tablename__ = "document_structure"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id"))
    section_id: Mapped[str] = mapped_column(String(100))   # Original TOC numbering, e.g. '1.3'
    chapter_id: Mapped[str] = mapped_column(String(100))    # DB key, e.g. 'ch1.3'
    title: Mapped[str] = mapped_column(String(500))
    book_page_start: Mapped[int] = mapped_column()
    source: Mapped[str] = mapped_column(String(20))         # llm | heuristic
    frozen_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("document_id", "chapter_id",
                         name="uq_doc_structure_doc_chapter"),
    )
