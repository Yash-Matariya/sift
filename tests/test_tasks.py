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



class _DummySession:
    def close(self):
        pass


def _eager_with_retries(monkeypatch):
    """Eager mode that actually retries.

    With task_eager_propagates on (the setting the other tests use), Celery
    re-raises the Retry the moment a task asks for one, so a retry is never
    attempted. Turned off, apply() catches the Retry and runs the task again
    in-process, ignoring the countdown -- which is what these tests observe.
    """
    monkeypatch.setattr(celery.conf, "task_always_eager", True)
    monkeypatch.setattr(celery.conf, "task_eager_propagates", False)
    monkeypatch.setattr(tasks, "SessionLocal", lambda: _DummySession())


def test_a_transient_failure_is_retried_and_recovers(monkeypatch):
    """The embedding service is down for two attempts, then back: the task succeeds."""
    _eager_with_retries(monkeypatch)
    attempts = []

    def flaky(session, document_id):
        attempts.append(document_id)
        if len(attempts) < 3:
            raise ConnectionError("model service unavailable")

    monkeypatch.setattr(tasks, "ingest_into", flaky)
    result = tasks.ingest_document_task.delay(7)
    assert attempts == [7, 7, 7]                  # two failures, then success on the third try
    assert result.state == "SUCCESS"


def test_a_permanent_failure_gives_up_after_three_retries(monkeypatch):
    _eager_with_retries(monkeypatch)
    attempts = []

    def always_fails(session, document_id):
        attempts.append(document_id)
        raise ConnectionError("still down")

    monkeypatch.setattr(tasks, "ingest_into", always_fails)
    result = tasks.ingest_document_task.delay(9)
    assert len(attempts) == 4                     # the first try plus max_retries=3
    assert result.state == "FAILURE"
    assert isinstance(result.result, ConnectionError)
