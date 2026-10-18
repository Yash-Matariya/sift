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
    query_vec = embed(settings.embed_query_prefix + question)
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


def _same_text(text: str) -> str:
    """The key two passages share when they are word-for-word the same (spacing and case aside)."""
    return " ".join(text.split()).lower()


def _one_per_text(candidates):
    """Keep one copy of passages that are word-for-word identical.

    Uploading a file twice stores every passage twice. Without this, the copies take several
    of the top-k slots with the same information and push other documents out entirely: three
    copies of a handbook filled all four slots and an email holding the answer never reached
    the model. The kept passage remembers the others in `.duplicates`, so the answer can still
    cite every place the text appears.
    """
    kept, seen = [], {}
    for chunk, score in candidates:
        key = _same_text(chunk.text)
        if key in seen:
            seen[key].duplicates.append(chunk)
            continue
        chunk.duplicates = []
        seen[key] = chunk
        kept.append((chunk, score))
    return kept


def retrieve(session: Session, question: str, k: int | None = None, extra_queries=()):
    """Return [(chunk, score)] for the top-k relevant chunks, or [] if none qualify.

    `extra_queries` are more searches for the same question (multi-query retrieval): each is
    searched the same way, and every ranked list is fused together with RRF.
    """
    k = k or settings.top_k
    wide = max(k, 20)   # retrieve wide, then filter/fuse down to k
    queries = [question, *[q for q in extra_queries if q and q.strip() != question.strip()]]

    similarity: dict[int, float] = {}
    chunks_by_id: dict[int, Chunk] = {}
    ranked_lists: list[list[int]] = []
    keyword_ids: set[int] = set()
    for q in queries:
        vec = _vector_search(session, q, wide)
        for chunk, sim in vec:
            chunks_by_id.setdefault(chunk.id, chunk)
            similarity[chunk.id] = max(sim, similarity.get(chunk.id, -1.0))   # best over the searches
        ranked_lists.append([chunk.id for chunk, _ in vec])
        if settings.retrieval_mode == "hybrid":
            kw = _keyword_search(session, q, wide)
            for chunk in kw:
                chunks_by_id.setdefault(chunk.id, chunk)
                keyword_ids.add(chunk.id)
            ranked_lists.append([chunk.id for chunk in kw])

    if settings.retrieval_mode != "hybrid" and len(queries) == 1:
        # vector-only: keep strong semantic matches, best first
        ordered = ranked_lists[0]
    else:
        # hybrid, or several searches: fuse the rankings
        ordered = rrf_fuse(ranked_lists, k=wide * len(queries))

    # Relevance gate: a chunk counts if it's a strong semantic match OR a literal
    # keyword match. Score is the semantic similarity where known, else min_score
    # for a keyword-only hit. Nothing passing -> [] -> the assistant refuses.
    results = []
    for cid in ordered:
        sim = similarity.get(cid)
        if sim is not None and sim >= settings.min_score:
            results.append((chunks_by_id[cid], sim))
        elif cid in keyword_ids:
            results.append((chunks_by_id[cid], settings.min_score))
    return _one_per_text(results)[:k]
