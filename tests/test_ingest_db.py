"""
Integration tests that need real Postgres + pgvector (skipped otherwise).

These exercise the ingestion pipeline against a real database: that ingestion
stores chunks and marks the document ready, and — the important one — that
re-running ingestion is idempotent and doesn't create duplicate chunks.

The embedding model is mocked (a fixed valid vector), so no Ollama is needed;
we're testing the database logic, not the model.
"""
from app import ingest
from app.config import settings
from app.models import Document


def _fixed_vector(text: str):
    return [0.0] * settings.embed_dim     # a valid 768-dim vector


def test_ingest_stores_chunks_and_marks_ready(db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "embed", _fixed_vector)
    monkeypatch.setattr(settings, "uploads_dir", str(tmp_path))

    # a document whose file has enough text to produce several chunks
    path = tmp_path / "doc.txt"
    path.write_text("sentence. " * 300)
    doc = Document(filename="doc.txt", status="pending", source_path=str(path))
    db_session.add(doc)
    db_session.commit()

    ingest.ingest_into(db_session, doc.id)

    db_session.refresh(doc)
    assert doc.status == "ready"
    assert ingest.chunk_count(db_session, doc.id) > 1


def test_reingest_is_idempotent(db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "embed", _fixed_vector)
    monkeypatch.setattr(settings, "uploads_dir", str(tmp_path))

    path = tmp_path / "doc.txt"
    path.write_text("sentence. " * 300)
    doc = Document(filename="doc.txt", status="pending", source_path=str(path))
    db_session.add(doc)
    db_session.commit()

    ingest.ingest_into(db_session, doc.id)
    first = ingest.chunk_count(db_session, doc.id)

    ingest.ingest_into(db_session, doc.id)          # run it again
    second = ingest.chunk_count(db_session, doc.id)

    assert first == second                          # no duplicate chunks
