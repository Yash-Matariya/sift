"""
The RAG flow: retrieve relevant chunks, then answer *from that context* with
citations — or refuse when nothing relevant was found.

The refusal is the important part: it's what stops the model inventing an answer
when the documents don't contain one.
"""
from sqlalchemy.orm import Session

from app.retrieval import retrieve
from app.llm import generate
from app.models import ChatResponse, Citation

PROMPT = """Answer the question using ONLY the context below. \
If the context does not contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


def answer_question(session: Session, question: str) -> ChatResponse:
    hits = retrieve(session, question)

    # Nothing relevant -> refuse rather than let the model make something up.
    if not hits:
        return ChatResponse(
            answer="I don't have anything relevant in the documents to answer that.",
            citations=[],
            refused=True,
        )

    # Build the context from the retrieved chunks, numbered so answers can cite them.
    context = "\n\n".join(f"[{i + 1}] {chunk.text}" for i, (chunk, _) in enumerate(hits))
    answer = generate(PROMPT.format(context=context, question=question))

    citations = [
        Citation(document=chunk.document.filename,
                 chunk_index=chunk.chunk_index,
                 text=chunk.text)
        for chunk, _ in hits
    ]
    return ChatResponse(answer=answer, citations=citations, refused=False)
