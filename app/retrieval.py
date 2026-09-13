"""
Retrieval: find the chunks most relevant to a question.

Embed the question, then ask Postgres (via pgvector) for the nearest chunk
vectors by cosine distance. We drop weak matches, which is what lets the
assistant *refuse* instead of answering from irrelevant text.
"""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Chunk
from app.llm import embed


def retrieve(session: Session, question: str, k: int | None = None):
    """Return [(chunk, similarity), ...] for the top-k most similar chunks."""
    k = k or settings.top_k
    query_vec = embed(question)

    # cosine_distance: 0 = identical direction, 2 = opposite. similarity = 1 - distance.
    distance = Chunk.embedding.cosine_distance(query_vec).label("distance")
    rows = session.execute(
        select(Chunk, distance).order_by(distance).limit(k)
    ).all()

    hits = []
    for chunk, dist in rows:
        similarity = 1.0 - dist
        if similarity >= settings.min_score:   # weak matches are dropped
            hits.append((chunk, similarity))
    return hits
