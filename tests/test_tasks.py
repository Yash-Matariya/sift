"""
Test the Celery ingestion task in eager mode.

Eager mode runs the task in-process with no broker, so we can verify the task
wiring — that queuing it actually runs the ingestion — without Redis or a worker.
We mock the actual ingestion so no database or model is needed here.
"""
from app import tasks
from app.celery_app import celery


def test_task_runs_ingestion_eagerly(monkeypatch):
    celery.conf.task_always_eager = True
    celery.conf.task_eager_propagates = True

    called = {}

    def fake_ingest_into(session, document_id):
        called["doc_id"] = document_id

    # a dummy session factory so the task doesn't touch a real database
    class DummySession:
        def close(self):
            pass

    monkeypatch.setattr(tasks, "ingest_into", fake_ingest_into)
    monkeypatch.setattr(tasks, "SessionLocal", lambda: DummySession())

    tasks.ingest_document_task.delay(42)     # eager -> runs now, in-process
    assert called["doc_id"] == 42
