"""
Ingestion: turn an uploaded document into stored, embedded chunks.

The flow is: read the file's text -> split it into overlapping chunks ->
embed each chunk into a vector -> store the chunks (with vectors) in Postgres.
Retrieval later searches those vectors.
"""
import io

from pypdf import PdfReader
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Document, Chunk
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
    """
    size, overlap = settings.chunk_size, settings.chunk_overlap
    text = text.strip()
    chunks, i = [], 0
    while i < len(text):
        piece = text[i:i + size].strip()
        if piece:
            chunks.append(piece)
        i += size - overlap
    return chunks


def ingest_document(session: Session, filename: str, data: bytes) -> Document:
    text = _read_text(filename, data)
    doc = Document(filename=filename)
    session.add(doc)
    session.flush()                       # assigns doc.id before we reference it
    for index, piece in enumerate(_chunk(text)):
        vector = embed(piece)             # one embedding per chunk (via Ollama)
        session.add(Chunk(document_id=doc.id, chunk_index=index,
                          text=piece, embedding=vector))
    session.commit()
    session.refresh(doc)
    return doc
