"""
Tests for the LangGraph agentic loop.

We mock retrieval, generation, and query-rewriting, so the whole graph — the
retrieve/grade/rewrite/generate/refuse flow and its loop bound — runs with no
database and no model. Each test scripts what retrieval returns on each call and
asserts the path the agent takes.
"""
import pytest

from app import agent
from app.models import ChatResponse
from app.config import settings


def _install(monkeypatch, retrieve_returns):
    """Wire fakes into the agent. retrieve_returns is a list, one per retrieval."""
    calls = {"retrieve": 0, "rewrite": 0, "generate": 0}
    seq = iter(retrieve_returns)

    def fake_retrieve(session, query, k=None):
        calls["retrieve"] += 1
        return next(seq)

    def fake_rewrite(question):
        calls["rewrite"] += 1
        return "rewritten: " + question

    def fake_generate_answer(hits, question):
        calls["generate"] += 1
        return ChatResponse(answer="grounded answer", citations=[], refused=False)

    monkeypatch.setattr(agent, "retrieve", fake_retrieve)
    monkeypatch.setattr(agent, "rewrite_query", fake_rewrite)
    monkeypatch.setattr(agent, "generate_answer", fake_generate_answer)
    return calls


def _hit(score):
    return (object(), score)   # grade only looks at the score


def test_confident_hit_answers_without_rewrite(monkeypatch):
    calls = _install(monkeypatch, [[_hit(0.9)]])
    resp = agent.answer_question(session=None, question="q")
    assert resp.refused is False
    assert resp.answer == "grounded answer"
    assert calls["retrieve"] == 1
    assert calls["rewrite"] == 0
    assert calls["generate"] == 1


def test_weak_then_good_after_rewrite(monkeypatch):
    # first retrieval weak (below good_score), second (after rewrite) confident
    calls = _install(monkeypatch, [[_hit(0.4)], [_hit(0.9)]])
    resp = agent.answer_question(session=None, question="q")
    assert resp.refused is False
    assert calls["retrieve"] == 2
    assert calls["rewrite"] == 1
    assert calls["generate"] == 1


def test_nothing_relevant_refuses(monkeypatch):
    # every retrieval returns nothing -> after the bounded loop, refuse
    empties = [[] for _ in range(settings.max_retrieval_attempts + 2)]
    calls = _install(monkeypatch, empties)
    resp = agent.answer_question(session=None, question="q")
    assert resp.refused is True
    assert calls["generate"] == 0
    # bounded: retrieval runs at most max_retrieval_attempts times
    assert calls["retrieve"] == settings.max_retrieval_attempts


def test_loop_is_bounded_then_best_effort(monkeypatch):
    # always weak but non-empty -> loops to the bound, then answers best-effort
    weak = [[_hit(0.4)] for _ in range(settings.max_retrieval_attempts + 2)]
    calls = _install(monkeypatch, weak)
    resp = agent.answer_question(session=None, question="q")
    assert resp.refused is False                       # had *something* to answer with
    assert calls["retrieve"] == settings.max_retrieval_attempts
    assert calls["rewrite"] == settings.max_retrieval_attempts - 1
    assert calls["generate"] == 1


# ---------------------------------------------------------------------------
# The trace: the agent's own account of what it did.
#
# It is assembled from the graph's final state rather than tracked separately,
# so these tests are really checking that the trace cannot drift from reality.
# ---------------------------------------------------------------------------

def test_trace_reports_a_single_confident_retrieval(monkeypatch):
    _install(monkeypatch, [[_hit(0.9)]])
    resp = agent.answer_question(session=None, question="what is the policy?")

    assert resp.trace is not None
    assert resp.trace.attempts == 1
    assert resp.trace.rewritten is False          # the first query was good enough
    assert resp.trace.final_query == "what is the policy?"
    assert resp.trace.best_score == 0.9
    assert resp.trace.chunks_found == 1


def test_trace_records_the_rewrite(monkeypatch):
    # weak first, confident second -> the agent rewrote, and the trace says so
    _install(monkeypatch, [[_hit(0.4)], [_hit(0.9)]])
    resp = agent.answer_question(session=None, question="original wording")

    assert resp.trace.attempts == 2
    assert resp.trace.rewritten is True
    assert resp.trace.final_query != "original wording"


def test_trace_survives_a_refusal(monkeypatch):
    # A refusal is the most interesting case to explain, so it must carry a
    # trace too -- "it tried N times and found nothing" is the whole answer.
    empties = [[] for _ in range(settings.max_retrieval_attempts + 2)]
    _install(monkeypatch, empties)
    resp = agent.answer_question(session=None, question="unrelated question")

    assert resp.refused is True
    assert resp.trace is not None
    assert resp.trace.attempts == settings.max_retrieval_attempts
    assert resp.trace.chunks_found == 0
    assert resp.trace.best_score == 0.0
