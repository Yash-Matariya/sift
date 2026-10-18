"""
The HTTP layer, end to end through FastAPI's TestClient.

main.py is thin on purpose, but thin is not the same as correct: status codes,
the upload's 202-then-poll contract, validation of an empty question, and the
404 for a document that does not exist are all promises a client relies on.
These run against the real test database (skipped without one); the model is
never called -- the agent and the embedder are replaced.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app import ingest, main, tasks
from app.celery_app import celery
from app.config import settings
from app.db import get_session
from sqlalchemy import exc as sa_exc, func, select

from app.models import AgentTrace, ChatResponse, Chunk, Document


@pytest.fixture
def client(db_session, tmp_path, monkeypatch):
    """A client whose requests use the test database and a temporary upload folder."""
    monkeypatch.setattr(settings, "uploads_dir", str(tmp_path))

    def _session():
        yield db_session

    main.app.dependency_overrides[get_session] = _session
    # No `with`: the lifespan (init_db on the real database) must not run in tests.
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


@pytest.fixture
def queued(monkeypatch):
    """Record queued ingestion tasks instead of running them."""
    ids = []
    monkeypatch.setattr(tasks.ingest_document_task, "delay", lambda doc_id: ids.append(doc_id))
    return ids


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_returns_202_pending_and_queues_ingestion(client, queued, tmp_path):
    r = client.post("/documents", files={"file": ("notes.txt", b"hello world", "text/plain")})
    assert r.status_code == 202                       # accepted, not done: the client polls
    body = r.json()
    assert body["status"] == "pending"
    assert body["chunk_count"] == 0
    assert queued == [body["id"]]                     # the slow work was handed to the worker
    saved = list(tmp_path.iterdir())
    assert len(saved) == 1 and saved[0].read_bytes() == b"hello world"


def test_status_of_a_missing_document_is_404(client):
    r = client.get("/documents/999999")
    assert r.status_code == 404
    assert r.json()["detail"] == "document not found"


def test_list_documents(client, queued):
    client.post("/documents", files={"file": ("a.txt", b"one", "text/plain")})
    client.post("/documents", files={"file": ("b.txt", b"two", "text/plain")})
    names = sorted(d["filename"] for d in client.get("/documents").json())
    assert names == ["a.txt", "b.txt"]


def test_list_stays_in_upload_order_after_a_status_change(client, queued, db_session):
    ids = [client.post("/documents", files={"file": (n, b"x", "text/plain")}).json()["id"]
           for n in ("a.txt", "b.txt", "c.txt")]
    # Postgres writes an UPDATE as a new row version further along the table,
    # so without an ORDER BY the first document comes back last -- the page's
    # list reshuffled every time a document finished processing.
    db_session.get(Document, ids[0]).status = "ready"
    db_session.commit()
    assert [d["id"] for d in client.get("/documents").json()] == ids


def test_empty_question_is_400(client):
    r = client.post("/chat", json={"question": "   "})
    assert r.status_code == 400


def test_missing_question_field_is_422(client):
    r = client.post("/chat", json={})
    assert r.status_code == 422                       # Pydantic rejects the body before our code runs


def test_chat_returns_the_agents_answer_with_its_trace(client, monkeypatch):
    def fake_answer(session, question):
        return ChatResponse(
            answer=f"echo: {question}", citations=[], refused=False,
            trace=AgentTrace(attempts=2, rewritten=True, final_query="better query",
                             best_score=0.61, chunks_found=3),
        )

    monkeypatch.setattr(main, "answer_question", fake_answer)
    body = client.post("/chat", json={"question": "what is sift?"}).json()
    assert body["answer"] == "echo: what is sift?"
    assert body["trace"]["attempts"] == 2
    assert body["trace"]["rewritten"] is True


def test_upload_then_ingest_then_ready(client, db_session, monkeypatch):
    """The whole flow: upload, the worker ingests, the status endpoint reports ready."""
    celery.conf.task_always_eager = True
    celery.conf.task_eager_propagates = True
    monkeypatch.setattr(ingest, "embed", lambda text: [0.0] * settings.embed_dim)
    # The task opens its own session; point it at the test database.
    monkeypatch.setattr(tasks, "SessionLocal", sessionmaker(bind=db_session.get_bind()))

    doc = client.post("/documents", files={"file": ("long.txt", b"sentence. " * 300, "text/plain")}).json()
    status = client.get(f"/documents/{doc['id']}").json()
    assert status["status"] == "ready"
    assert status["chunk_count"] > 1


def test_the_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


# ---- when a dependency is down, and fixing what it left behind -------------

def _queue_down(doc_id):
    raise ConnectionError("Error 61 connecting to localhost:6379. Connection refused.")


def test_upload_while_the_queue_is_down_is_503_and_the_document_is_failed(client, monkeypatch):
    # Before the fix: the row was committed as "pending", .delay() raised, the client got a 500,
    # and the document stayed pending for ever with nothing coming to pick it up.
    monkeypatch.setattr(tasks.ingest_document_task, "delay", _queue_down)
    r = client.post("/documents", files={"file": ("a.txt", b"x", "text/plain")})
    assert r.status_code == 503
    assert "Redis" in r.json()["detail"]
    [doc] = client.get("/documents").json()
    assert doc["status"] == "failed"                  # visibly broken, and retryable


def test_retry_queues_a_failed_document_again(client, queued, db_session):
    doc = client.post("/documents", files={"file": ("a.txt", b"x", "text/plain")}).json()
    db_session.get(Document, doc["id"]).status = "failed"
    db_session.commit()
    r = client.post(f"/documents/{doc['id']}/retry")
    assert r.status_code == 202
    assert r.json()["status"] == "pending"
    assert queued == [doc["id"], doc["id"]]           # queued at upload, and again by the retry


def test_retry_while_the_queue_is_still_down_is_503(client, queued, db_session, monkeypatch):
    doc = client.post("/documents", files={"file": ("a.txt", b"x", "text/plain")}).json()
    monkeypatch.setattr(tasks.ingest_document_task, "delay", _queue_down)
    assert client.post(f"/documents/{doc['id']}/retry").status_code == 503
    assert client.get(f"/documents/{doc['id']}").json()["status"] == "failed"


def test_retry_a_ready_document_is_409(client, queued, db_session):
    doc = client.post("/documents", files={"file": ("a.txt", b"x", "text/plain")}).json()
    db_session.get(Document, doc["id"]).status = "ready"
    db_session.commit()
    assert client.post(f"/documents/{doc['id']}/retry").status_code == 409


def test_delete_removes_the_document_its_chunks_and_its_file(client, db_session, monkeypatch, tmp_path):
    celery.conf.task_always_eager = True
    celery.conf.task_eager_propagates = True
    monkeypatch.setattr(ingest, "embed", lambda text: [0.0] * settings.embed_dim)
    monkeypatch.setattr(tasks, "SessionLocal", sessionmaker(bind=db_session.get_bind()))
    doc = client.post("/documents", files={"file": ("long.txt", b"sentence. " * 300, "text/plain")}).json()
    assert client.get(f"/documents/{doc['id']}").json()["chunk_count"] > 1

    assert client.delete(f"/documents/{doc['id']}").status_code == 204
    assert client.get(f"/documents/{doc['id']}").status_code == 404
    left = db_session.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == doc["id"]))
    assert left == 0                                  # the relationship's cascade took the chunks
    assert list(tmp_path.iterdir()) == []             # and the stored file is gone


def test_delete_a_missing_document_is_404(client):
    assert client.delete("/documents/999999").status_code == 404


def test_the_worker_skips_a_deleted_document_instead_of_retrying(db_session, monkeypatch):
    monkeypatch.setattr(tasks, "SessionLocal", sessionmaker(bind=db_session.get_bind()))
    assert tasks.ingest_document_task.apply(args=[999999]).successful()


def test_database_down_is_503_not_500(client, monkeypatch):
    def down(session, question):
        raise sa_exc.OperationalError("SELECT 1", {}, Exception("connection refused"))

    monkeypatch.setattr(main, "answer_question", down)
    r = client.post("/chat", json={"question": "anything"})
    assert r.status_code == 503
    assert "database" in r.json()["detail"]
