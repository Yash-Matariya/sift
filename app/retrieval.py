"""
Retrieval: find the chunks most relevant to a question.

Two modes (set by settings.retrieval_mode):

- "vector"  — semantic search only: embed the question and find the nearest chunk
              vectors by cosine distance in pgvector.
- "hybrid"  — semantic search fused with keyword search (Postgres full-text), so
              exact terms a pure embedding might miss (codes, names) are caught.
              The two ranked lists are merged with Reciprocal Rank Fusion (RRF).

Either way we keep a relevance gate: a chunk must be a strong semantic match
(similarity >= min_score) OR a literal keyword match to count. If nothing passes,
retrieve returns nothing, which is what lets the assistant *refuse* rather than
answer from irrelevant text.
"""
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Chunk
from app.llm import embed


def rrf_fuse(ranked_lists: list[list[int]], k: int, rrf_k: int | None = None) -> list[int]:
    """Reciprocal Rank Fusion: merge several ranked lists of ids into one.

    Each id scores sum(1 / (rrf_k + rank)) across the lists it appears in
    (rank is 0-based). Ids ranked highly in either list rise to the top, so an
    item found by *both* semantic and keyword search is favoured. Pure function
    (no DB), so it's easy to test.
    """
    rrf_k = settings.rrf_k if rrf_k is None else rrf_k
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores, key=lambda i: scores[i], reverse=True)
    return ordered[:k]


def _vector_search(session: Session, question: str, limit: int):
    """Return [(chunk, similarity)] for the nearest chunks by cosine distance."""
    query_vec = embed(question)
    # cosine_distance: 0 = identical direction, 2 = opposite. similarity = 1 - distance.
    distance = Chunk.embedding.cosine_distance(query_vec).label("distance")
    rows = session.execute(
        select(Chunk, distance).order_by(distance).limit(limit)
    ).all()
    return [(chunk, 1.0 - dist) for chunk, dist in rows]


def _keyword_search(session: Session, question: str, limit: int):
    """Return [chunk, ...] matching the question by Postgres full-text search."""
    tsv = func.to_tsvector("english", Chunk.text)
    tsq = func.plainto_tsquery("english", question)
    rank = func.ts_rank(tsv, tsq).label("rank")
    rows = session.execute(
        select(Chunk, rank).where(tsv.op("@@")(tsq)).order_by(rank.desc()).limit(limit)
    ).all()
    return [chunk for chunk, _ in rows]


def retrieve(session: Session, question: str, k: int | None = None):
    """Return [(chunk, score)] for the top-k relevant chunks, or [] if none qualify."""
    k = k or settings.top_k
    wide = max(k, 20)   # retrieve wide, then filter/fuse down to k

    vec = _vector_search(session, question, wide)
    similarity = {chunk.id: sim for chunk, sim in vec}
    chunks_by_id = {chunk.id: chunk for chunk, _ in vec}

    if settings.retrieval_mode != "hybrid":
        # vector-only: keep strong semantic matches, best first
        hits = [(chunk, sim) for chunk, sim in vec if sim >= settings.min_score]
        return hits[:k]

    # hybrid: add keyword matches and fuse the rankings
    kw = _keyword_search(session, question, wide)
    keyword_ids = {chunk.id for chunk in kw}
    for chunk in kw:
        chunks_by_id.setdefault(chunk.id, chunk)

    vec_ranked = [chunk.id for chunk, _ in vec]
    kw_ranked = [chunk.id for chunk in kw]
    fused_ids = rrf_fuse([vec_ranked, kw_ranked], k=wide)

    # Relevance gate: a chunk counts if it's a strong semantic match OR a literal
    # keyword match. Score is the semantic similarity where known, else min_score
    # for a keyword-only hit. Nothing passing -> [] -> the assistant refuses.
    results = []
    for cid in fused_ids:
        sim = similarity.get(cid)
        if sim is not None and sim >= settings.min_score:
            results.append((chunks_by_id[cid], sim))
        elif cid in keyword_ids:
            results.append((chunks_by_id[cid], settings.min_score))
    return results[:k]
