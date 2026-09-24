"""
The data model, in two flavours:

1. SQLAlchemy ORM models  -> the database tables (Document, Chunk).
2. Pydantic schemas        -> the JSON shapes the API accepts and returns.

Keeping them separate is deliberate: the DB models are how data is stored; the
schemas are the API contract. They can evolve independently.
"""
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Text, DateTime, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pydantic import BaseModel

from app.config import settings
from app.db import Base


# ------------------------------------------------------------------ ORM tables

class Document(Base):
    __tablename__ = "document"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    # ingestion status: "pending" -> "processing" -> "ready" (or "failed").
    # Set when the upload is accepted; updated by the background ingest task.
    status: Mapped[str] = mapped_column(Text, default="pending")
    # where the raw uploaded file is stored, so the background worker can read it
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    # one document has many chunks; deleting the document deletes its chunks
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunk"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer)     # order within the document
    text: Mapped[str] = mapped_column(Text)
    # the embedding vector — this is what similarity search runs against
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embed_dim))

    document: Mapped["Document"] = relationship(back_populates="chunks")


# ------------------------------------------------------------------ API schemas

class DocumentOut(BaseModel):
    id: int
    filename: str
    status: str
    chunk_count: int


class ChatRequest(BaseModel):
    question: str


class Citation(BaseModel):
    document: str          # filename the passage came from
    chunk_index: int
    text: str              # the passage itself


class AgentTrace(BaseModel):
    """What the agent actually did to reach its answer.

    The agent already tracked all of this internally; exposing it is what makes
    the "agentic" part observable instead of a claim. A caller can see that a
    weak first retrieval was graded, the query rewritten, and the search run
    again -- which is the whole difference between this and plain RAG.
    """

    attempts: int                  # how many retrievals it took
    rewritten: bool                # did it give up on the original wording?
    final_query: str               # what it actually searched for in the end
    best_score: float              # similarity of the strongest chunk found
    chunks_found: int              # how many passed the relevance gate


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refused: bool          # True when we declined because nothing relevant was found
    trace: AgentTrace | None = None   # optional so tests can build a bare response
