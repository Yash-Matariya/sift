"""
Ingestion: turn an uploaded document into stored, embedded chunks.

The flow is: read the file's text -> split it into overlapping chunks ->
embed each chunk into a vector -> store the chunks (with vectors) in Postgres.
Retrieval later searches those vectors.

Ingestion runs in a background Celery worker (see app/tasks.py), not in the
upload request, so uploads return immediately. `ingest_into` is written to be
idempotent — it clears any existing chunks first — so a retried task can't
create duplicate chunks.
"""
import io

from pypdf import PdfReader
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Document, Chunk
from app.storage import read_upload
from app.llm import embed


def _read_text(filename: str, data: bytes) -> str:
    """Extract plain text from an uploaded file (PDF or text)."""
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


def _chunk(text: str) -> list[str]:
    """Split text into overlapping pieces so context isn't cut mid-thought.

    Overlap means neighbouring chunks share some text, so an answer that
    straddles a boundary is still fully present in at least one chunk.

    Both edges of a window are moved to a space, so no chunk starts or ends in
    the middle of a word. The first version cut at exact character counts, and
    a passage shown as a source began "eclined the card" (from "declined").
    Only a single word longer than the whole window, which has no space to move
    to, is still cut.
    """
    size, overlap = settings.chunk_size, settings.chunk_overlap
    text = " ".join(text.split())         # runs of spaces and newlines -> one space
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        # Don't end inside a word: back up to the last space in the window.
        if end < len(text) and text[end] != " ":
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunks.append(text[start:end])
        if end == len(text):
            break
        # Step back by the overlap, then forward to the start of the next word.
        start = max(end - overlap, start + 1)
        if text[start - 1] != " ":
            space = text.find(" ", start, end)
            if space != -1:
                start = space + 1
    return chunks


class DocumentGone(ValueError):
    """The document was deleted before it could be ingested. Not worth retrying."""


def ingest_into(session: Session, document_id: int) -> None:
    """Chunk, embed, and store the document's file. Idempotent and retry-safe.

    Reads the document's saved file, clears any chunks from a previous run,
    then stores fresh chunks and marks the document ready. On any failure the
    status is set to "failed" and the error re-raised so the task can record it.
    """
    doc = session.get(Document, document_id)
    if doc is None:
        raise DocumentGone(f"document {document_id} not found")

    doc.status = "processing"
    session.commit()

    try:
        data = read_upload(doc.source_path)
        text = _read_text(doc.filename, data)

        # Idempotency: remove any chunks from a previous (partial) run so a
        # retry produces the same clean result instead of duplicates.
        session.execute(delete(Chunk).where(Chunk.document_id == doc.id))

        for index, piece in enumerate(_chunk(text)):
            vector = embed(settings.embed_document_prefix + piece)   # one embedding per chunk (via Ollama)
            session.add(Chunk(document_id=doc.id, chunk_index=index,
                              text=piece, embedding=vector))

        doc.status = "ready"
        session.commit()                      # chunks + status committed together
    except Exception:
        session.rollback()
        # mark failed in its own transaction so the status is visible
        doc = session.get(Document, document_id)
        if doc is not None:
            doc.status = "failed"
            session.commit()
        raise


def chunk_count(session: Session, document_id: int) -> int:
    """How many chunks a document currently has."""
    from sqlalchemy import func
    return session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
    ) or 0
