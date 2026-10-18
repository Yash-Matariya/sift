"""
Celery tasks: the slow work that runs in a background worker, off the request path.

The only task is document ingestion. The upload endpoint saves the file, creates
a `pending` document row, and calls `ingest_document_task.delay(doc.id)`; a worker
picks it up and runs the chunk/embed/store pipeline, updating the document status.

The task is idempotent (ingest_into clears existing chunks first) and retries on
failure with backoff, so a transient error — like the embedding model being
briefly unavailable — recovers on its own without creating duplicate chunks.
"""
from app.celery_app import celery
from app.db import SessionLocal
from app.ingest import DocumentGone, ingest_into


@celery.task(bind=True, max_retries=3, default_retry_delay=5)
def ingest_document_task(self, document_id: int):
    """Ingest a document by id. One session per task run."""
    session = SessionLocal()
    try:
        ingest_into(session, document_id)
    except DocumentGone:
        return              # deleted before the worker got to it: nothing to do, nothing to retry
    except Exception as exc:
        # Retry transient failures (e.g. embedding service down) with backoff.
        # ingest_into already marked the document "failed"; a retry will re-run
        # it cleanly because it's idempotent.
        raise self.retry(exc=exc)
    finally:
        session.close()
