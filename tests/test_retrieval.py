"""
Tests for the retrieval gate and hybrid fusion.

We mock the vector and keyword searches (so no database or model is needed) and
check the logic: vector mode drops weak matches; hybrid mode keeps strong
semantic matches AND literal keyword matches, and drops the rest.
"""
from app import retrieval
from app.config import settings


class FakeChunk:
    def __init__(self, cid):
        self.id = cid
        self.text = f"chunk {cid}"
        self.chunk_index = cid


def test_vector_mode_drops_weak_matches(monkeypatch):
    monkeypatch.setattr(settings, "retrieval_mode", "vector")
    monkeypatch.setattr(settings, "min_score", 0.35)
    c1, c2 = FakeChunk(1), FakeChunk(2)
    monkeypatch.setattr(retrieval, "_vector_search",
                        lambda s, q, limit: [(c1, 0.8), (c2, 0.2)])

    hits = retrieval.retrieve(session=None, question="q", k=5)
    ids = [c.id for c, _ in hits]
    assert ids == [1]              # c2 (0.2 < 0.35) dropped


def test_hybrid_keeps_strong_vector_and_keyword_only(monkeypatch):
    monkeypatch.setattr(settings, "retrieval_mode", "hybrid")
    monkeypatch.setattr(settings, "min_score", 0.35)
    c1, c2, c3 = FakeChunk(1), FakeChunk(2), FakeChunk(3)
    # c1 strong semantic, c2 weak semantic, c3 keyword-only (exact-term match)
    monkeypatch.setattr(retrieval, "_vector_search",
                        lambda s, q, limit: [(c1, 0.8), (c2, 0.2)])
    monkeypatch.setattr(retrieval, "_keyword_search",
                        lambda s, q, limit: [c3])

    hits = retrieval.retrieve(session=None, question="q", k=5)
    ids = {c.id for c, _ in hits}
    assert 1 in ids                # strong semantic kept
    assert 3 in ids                # keyword-only kept (catches exact terms)
    assert 2 not in ids            # weak, non-keyword dropped


def test_hybrid_returns_nothing_when_all_weak(monkeypatch):
    monkeypatch.setattr(settings, "retrieval_mode", "hybrid")
    monkeypatch.setattr(settings, "min_score", 0.35)
    c1 = FakeChunk(1)
    monkeypatch.setattr(retrieval, "_vector_search",
                        lambda s, q, limit: [(c1, 0.1)])
    monkeypatch.setattr(retrieval, "_keyword_search", lambda s, q, limit: [])

    hits = retrieval.retrieve(session=None, question="q", k=5)
    assert hits == []              # nothing qualifies -> the agent will refuse
