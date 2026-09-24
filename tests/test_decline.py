"""The second refusal path: the model declining even though retrieval passed.

There are two ways Sift can end up not answering, and they are different:

1. **Nothing relevant was retrieved.** The similarity gate catches it and the
   agent refuses without ever calling the model. Covered in test_agent.py.
2. **Something was retrieved, but it did not answer the question.** The gate
   cannot catch this — the passages looked close enough. The model is the last
   line of defence, and the system prompt tells it to say it does not know.

This file covers the second one, which was a real bug: the model said "I don't
know" and the API reported `refused: false` with citations attached, so a
non-answer was presented as an answer sourced from those passages.
"""
from app import rag


class _FakeChain:
    """Stands in for the LCEL chain. The real one is a Pydantic model whose
    `invoke` cannot be monkeypatched, so the whole chain object is replaced."""

    def __init__(self, reply: str):
        self.reply = reply

    def invoke(self, _inputs):
        return self.reply


class _Chunk:
    def __init__(self):
        self.text = "some retrieved text"
        self.chunk_index = 0
        self.document = type("D", (), {"filename": "policy.txt"})()


def _hits():
    return [(_Chunk(), 0.5)]


def _answer_with(monkeypatch, reply: str):
    monkeypatch.setattr(rag, "ANSWER_CHAIN", _FakeChain(reply))
    return rag.generate_answer(_hits(), "a question")


def test_model_declining_is_reported_as_a_refusal(monkeypatch):
    r = _answer_with(monkeypatch, "I don't know.")

    assert r.refused is True, "a non-answer must not be reported as an answer"
    assert r.citations == [], "citations under 'I don't know' imply it came from them"


def test_a_real_answer_keeps_its_citations(monkeypatch):
    r = _answer_with(monkeypatch, "Employees may claim up to 300 GBP per year.")

    assert r.refused is False
    assert len(r.citations) == 1


def test_an_answer_that_merely_discusses_not_knowing_is_still_an_answer(monkeypatch):
    # The guard must not fire on a long, genuine answer that happens to contain
    # the phrase. This is why the check is limited to short replies.
    r = _answer_with(monkeypatch, (
        "The policy does not say what happens if you don't know your team's "
        "timezone, but it does require you to be reachable between 11:00 and "
        "16:00 in the timezone of your assigned team, so you would need to "
        "confirm that with your manager before arranging your schedule."
    ))

    assert r.refused is False
    assert len(r.citations) == 1


def test_detection_ignores_case_and_surrounding_whitespace(monkeypatch):
    assert _answer_with(monkeypatch, "\n  I Do Not Know.  \n").refused is True


def test_other_instructed_phrasings_are_caught(monkeypatch):
    for reply in ("The context does not contain that information.",
                  "I don't have enough information to answer."):
        assert _answer_with(monkeypatch, reply).refused is True, reply
