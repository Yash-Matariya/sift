"""
The building blocks for turning retrieved chunks into a grounded answer.

`generate_answer` builds a prompt that grounds the model in the retrieved context
and instructs it to answer only from that context, then returns the answer with
citations. `refusal` is the "I don't know" response. The agent (app/agent.py)
decides *when* to call these; this module is *how* an answer is produced.
"""
from app.llm import generate
from app.models import ChatResponse, Citation

PROMPT = """Answer the question using ONLY the context below. \
If the context does not contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(hits, question: str) -> ChatResponse:
    """Build the grounded prompt from the retrieved chunks and answer, with citations."""
    # Number the chunks so the context is clear and citations line up with sources.
    context = "\n\n".join(
        f"[{i + 1}] {chunk.text}" for i, (chunk, _) in enumerate(hits)
    )
    answer = generate(PROMPT.format(context=context, question=question))

    citations = [
        Citation(document=chunk.document.filename,
                 chunk_index=chunk.chunk_index,
                 text=chunk.text)
        for chunk, _ in hits
    ]
    return ChatResponse(answer=answer, citations=citations, refused=False)


def refusal() -> ChatResponse:
    """The response when nothing relevant was found — refuse, don't invent."""
    return ChatResponse(
        answer="I don't have anything relevant in the documents to answer that.",
        citations=[],
        refused=True,
    )
