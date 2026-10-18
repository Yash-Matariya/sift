"""Re-embed every stored document with the current settings.

    .venv/bin/python -m app.reindex

Needed after changing the embedding model or its prefixes (embed_query_prefix,
embed_document_prefix): stored vectors made the old way are not comparable with
questions embedded the new way. Runs ingestion in this process, one document at a
time; ingestion is idempotent, so a document's old chunks are replaced, not added to.
"""
from sqlalchemy import select

from app.db import SessionLocal
from app.ingest import ingest_into
from app.models import Document


def main() -> None:
    session = SessionLocal()
    try:
        ids = session.scalars(select(Document.id).order_by(Document.id)).all()
        for doc_id in ids:
            doc = session.get(Document, doc_id)
            print(f"  re-embedding {doc_id}: {doc.filename} ...", end="", flush=True)
            try:
                ingest_into(session, doc_id)
                print(" done")
            except Exception as exc:          # one bad file shouldn't stop the rest
                print(f" failed: {exc}")
    finally:
        session.close()


if __name__ == "__main__":
    main()
