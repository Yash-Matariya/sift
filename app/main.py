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

from fastapi import FastAPI, UploadFile, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_session, init_db
from app.models import Document, DocumentOut, ChatRequest, ChatResponse
from app.ingest import chunk_count
from app.storage import save_upload
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

    ingest_document_task.delay(doc.id)     # queue the slow work; return now
    return _to_out(session, doc)


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(session: Session = Depends(get_session)):
    docs = session.scalars(select(Document)).all()
    return [_to_out(session, d) for d in docs]


@app.get("/documents/{doc_id}", response_model=DocumentOut)
def document_status(doc_id: int, session: Session = Depends(get_session)):
    """Check a document's ingestion status (pending / processing / ready / failed)."""
    doc = session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    return _to_out(session, doc)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, session: Session = Depends(get_session)):
    """Ask a question; get an answer grounded in the documents, with citations."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    return answer_question(session, req.question)
