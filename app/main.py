"""
The FastAPI application: upload documents, list them, and ask questions.

This is the thin HTTP layer — each endpoint validates input, calls the right
service (ingest / rag), and returns a typed response. All the real work lives
in the ingest, retrieval, and rag modules.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.db import get_session, init_db
from app.models import Document, Chunk, DocumentOut, ChatRequest, ChatResponse
from app.ingest import ingest_document
from app.rag import answer_question


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()          # enable pgvector + create tables on startup
    yield


app = FastAPI(
    title="Sift",
    description="An agentic RAG document assistant",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentOut)
async def upload_document(file: UploadFile, session: Session = Depends(get_session)):
    """Upload a document; it's chunked, embedded, and stored."""
    data = await file.read()
    doc = ingest_document(session, file.filename, data)
    count = session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == doc.id)
    )
    return DocumentOut(id=doc.id, filename=doc.filename, chunk_count=count)


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(session: Session = Depends(get_session)):
    docs = session.scalars(select(Document)).all()
    return [
        DocumentOut(id=d.id, filename=d.filename, chunk_count=len(d.chunks))
        for d in docs
    ]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, session: Session = Depends(get_session)):
    """Ask a question; get an answer grounded in the documents, with citations."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    return answer_question(session, req.question)
