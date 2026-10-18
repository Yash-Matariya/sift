"""
Retrieval against a real Postgres with pgvector and full-text search.

test_retrieval.py checks the fusion and the gate with the searches faked. This
checks the searches themselves: that the cosine-distance query ranks by
similarity, that full-text search catches an exact code an embedding would
miss, and that hybrid mode lets it through the gate. The embedder is replaced
by hand-made vectors, so the test controls every similarity exactly.
"""
import pytest

from app import retrieval
from app.config import settings
from app.models import Chunk, Document


def _axis(i: int) -> list[float]:
    """A unit vector along axis i: cosine similarity is 1 with itself, 0 with any other axis."""
    v = [0.0] * settings.embed_dim
    v[i] = 1.0
    return v


@pytest.fixture
def corpus(db_session):
    doc = Document(filename="handbook.txt", status="ready")
    db_session.add(doc)
    db_session.flush()
    chunks = {
        "refunds": Chunk(document_id=doc.id, chunk_index=0, embedding=_axis(0),
                         text="Refunds are issued within 14 days of a return."),
        "shipping": Chunk(document_id=doc.id, chunk_index=1, embedding=_axis(1),
                          text="Standard shipping takes three to five working days."),
        "error": Chunk(document_id=doc.id, chunk_index=2, embedding=_axis(2),
                       text="ERR-4471 means the upload was rejected by the scanner."),
    }
    db_session.add_all(chunks.values())
    db_session.commit()
    return chunks


def test_vector_search_ranks_by_cosine_similarity(db_session, corpus, monkeypatch):
    monkeypatch.setattr(retrieval, "embed", lambda q: _axis(0))
    hits = retrieval._vector_search(db_session, "how do refunds work", limit=3)
    assert hits[0][0].id == corpus["refunds"].id
    assert hits[0][1] == pytest.approx(1.0)          # same direction
    assert hits[1][1] == pytest.approx(0.0)          # orthogonal


def test_keyword_search_finds_an_exact_code(db_session, corpus):
    hits = retrieval._keyword_search(db_session, "ERR-4471 means", limit=5)
    assert [c.id for c in hits] == [corpus["error"].id]


def test_hybrid_lets_an_exact_code_through_the_gate(db_session, corpus, monkeypatch):
    # The question's embedding points at "shipping" -- semantically wrong -- but the
    # exact code appears in the error chunk, and keyword search catches it.
    monkeypatch.setattr(settings, "retrieval_mode", "hybrid")
    monkeypatch.setattr(retrieval, "embed", lambda q: _axis(1))
    hits = retrieval.retrieve(db_session, "ERR-4471 means", k=4)
    ids = {c.id for c, _ in hits}
    assert corpus["error"].id in ids                 # keyword match survives the gate
    assert corpus["refunds"].id not in ids           # neither similar nor a keyword match


def test_vector_only_mode_misses_the_exact_code(db_session, corpus, monkeypatch):
    # The same question without full-text search: the reason hybrid is the default.
    monkeypatch.setattr(settings, "retrieval_mode", "vector")
    monkeypatch.setattr(retrieval, "embed", lambda q: _axis(1))
    hits = retrieval.retrieve(db_session, "ERR-4471 means", k=4)
    assert corpus["error"].id not in {c.id for c, _ in hits}


# ---- duplicates and multi-query ---------------------------------------------

@pytest.fixture
def with_copies(db_session, corpus):
    """The handbook uploaded twice more: the same refunds passage in three documents."""
    copies = []
    for n in (2, 3):
        doc = Document(filename=f"handbook_copy_{n}.txt", status="ready")
        db_session.add(doc)
        db_session.flush()
        copies.append(Chunk(document_id=doc.id, chunk_index=0, embedding=_axis(0),
                            text="Refunds are issued within 14 days of a return."))
    db_session.add_all(copies)
    db_session.commit()
    return copies


def test_identical_passages_take_one_slot_and_remember_their_sources(db_session, corpus, with_copies, monkeypatch):
    # Points mostly at refunds, a little at shipping. Without de-duplication the three refunds
    # copies take three of the slots; with it they take one, and shipping gets a slot.
    v = _axis(0)
    v[1] = 0.8
    monkeypatch.setattr(settings, "retrieval_mode", "vector")
    monkeypatch.setattr(retrieval, "embed", lambda q: v)
    hits = retrieval.retrieve(db_session, "refunds", k=2)
    texts = [c.text for c, _ in hits]
    assert len(set(texts)) == len(texts) == 2            # no passage twice
    refunds = next(c for c, _ in hits if c.text.startswith("Refunds"))
    assert len(refunds.duplicates) == 2                 # the two other copies, kept for citations


def test_extra_queries_find_what_the_question_alone_misses(db_session, corpus, monkeypatch):
    # The question's embedding points at refunds; the split-out search points at the error code.
    monkeypatch.setattr(settings, "retrieval_mode", "vector")
    monkeypatch.setattr(retrieval, "embed", lambda q: _axis(2) if "scanner" in q else _axis(0))
    alone = {c.id for c, _ in retrieval.retrieve(db_session, "refunds and uploads", k=4)}
    both = {c.id for c, _ in retrieval.retrieve(db_session, "refunds and uploads", k=4,
                                                 extra_queries=["upload scanner error"])}
    assert corpus["error"].id not in alone
    assert {corpus["refunds"].id, corpus["error"].id} <= both


def test_questions_and_passages_get_their_task_labels(db_session, corpus, monkeypatch):
    seen = []
    monkeypatch.setattr(retrieval, "embed", lambda q: seen.append(q) or _axis(0))
    retrieval._vector_search(db_session, "how do refunds work", limit=1)
    assert seen == [settings.embed_query_prefix + "how do refunds work"]
