"""
The FastAPI application: upload documents, check their status, and ask questions.

This is the thin HTTP layer — each endpoint validates input, calls the right
service, and returns a typed response. The real work lives in the ingest,
retrieval, rag, and agent modules.

Uploads are handled asynchronously: the endpoint saves the file, creates a
`pending` document, and queues a background Celery task to chunk and embed it,
returning immediately (202). The client polls the status endpoint until the
document is `ready`.
"""
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, UploadFile, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import select
from sqlalchemy import exc as sa_exc
from sqlalchemy.orm import Session

from app.db import get_session, init_db
from app.models import Document, DocumentOut, ChatRequest, ChatResponse
from app.ingest import chunk_count
from app.storage import save_upload, delete_upload
from app.tasks import ingest_document_task
from app.agent import answer_question


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()          # enable pgvector + create tables on startup
    yield


app = FastAPI(
    title="Sift",
    description="An agentic RAG document assistant",
    lifespan=lifespan,
)


@app.exception_handler(sa_exc.OperationalError)
async def database_down(request: Request, exc: sa_exc.OperationalError):
    """Postgres isn't reachable: say so plainly instead of a bare 500 with a stack trace."""
    return JSONResponse(status_code=503, content={
        "detail": "The database isn't reachable. Start Postgres (docker compose up -d db) and try again."})


QUEUE_DOWN = ("The file is saved, but the job queue (Redis) isn't reachable, so it wasn't sent to the "
              "worker. Start Redis (docker compose up -d redis), then press Retry.")


def _queue(session: Session, doc: Document) -> None:
    """Hand the document to the worker.

    If Redis is down, `.delay()` raises. Without this, the row was already committed as "pending",
    the client got a 500, and the document stayed pending for ever with nothing coming to pick it
    up. Now it is marked "failed" and the client gets a 503 that says what to do.
    """
    try:
        ingest_document_task.delay(doc.id)
    except Exception:
        doc.status = "failed"
        session.commit()
        raise HTTPException(status_code=503, detail=QUEUE_DOWN)


def _to_out(session: Session, doc: Document) -> DocumentOut:
    return DocumentOut(
        id=doc.id, filename=doc.filename, status=doc.status,
        chunk_count=chunk_count(session, doc.id),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentOut, status_code=202)
async def upload_document(file: UploadFile, session: Session = Depends(get_session)):
    """Accept a document, save it, and queue ingestion in the background.

    Returns immediately with status "pending"; a worker chunks and embeds it.
    """
    data = await file.read()

    doc = Document(filename=file.filename, status="pending")
    session.add(doc)
    session.flush()                        # assigns doc.id
    doc.source_path = save_upload(doc.id, file.filename, data)
    session.commit()

    _queue(session, doc)                   # queue the slow work; return now
    return _to_out(session, doc)


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(session: Session = Depends(get_session)):
    docs = session.scalars(select(Document).order_by(Document.id)).all()
    return [_to_out(session, d) for d in docs]


@app.get("/documents/{doc_id}", response_model=DocumentOut)
def document_status(doc_id: int, session: Session = Depends(get_session)):
    """Check a document's ingestion status (pending / processing / ready / failed)."""
    doc = session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    return _to_out(session, doc)


@app.post("/documents/{doc_id}/retry", response_model=DocumentOut, status_code=202)
def retry_document(doc_id: int, session: Session = Depends(get_session)):
    """Queue ingestion again for a document that is "failed", or "pending" with nothing picking it up.

    Safe to repeat: ingestion clears a document's old chunks before storing new ones.
    """
    doc = session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    if doc.status in ("ready", "processing"):
        raise HTTPException(status_code=409, detail=f"document is already {doc.status}")
    if not doc.source_path or not Path(doc.source_path).exists():
        raise HTTPException(status_code=409,
                            detail="the uploaded file is no longer stored; delete this and upload it again")
    doc.status = "pending"
    session.commit()
    _queue(session, doc)
    return _to_out(session, doc)


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: int, session: Session = Depends(get_session)):
    """Delete a document, its chunks (the ORM relationship cascades) and its stored file."""
    doc = session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    path = doc.source_path
    session.delete(doc)
    session.commit()
    delete_upload(path)
    return Response(status_code=204)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, session: Session = Depends(get_session)):
    """Ask a question; get an answer grounded in the documents, with citations."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    return answer_question(session, req.question)


# ---------------------------------------------------------------------------
# The UI.
#
# A single static page, served by the API that powers it. No build step and no
# npm: this is a Python project, and a front end you cannot open without a
# toolchain is a worse demo, not a better one.
#
# Declared last so it can never shadow an API route, and by explicit file path
# rather than a mounted directory, so nothing else in the package is exposed.
# ---------------------------------------------------------------------------

_UI = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def ui():
    """The chat interface. /docs still has the interactive API."""
    return FileResponse(_UI)
