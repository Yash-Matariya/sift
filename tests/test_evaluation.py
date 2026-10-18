"""The evaluation metrics.

These are pure functions over collected results, so they need no database, no
embedding model and no LLM. What they protect is the honesty of the numbers --
in particular that a system which refuses everything cannot post a good score.
"""
from app.evaluation import CaseResult, EvalCase, recall_at_k, refusal_accuracy, score


def answerable(q="q", *, found, refused=False):
    case = EvalCase(question=q, answer_in=("the answer",))
    texts = ["... the answer is here ..."] if found else ["something unrelated"]
    return CaseResult(case=case, retrieved_texts=texts, refused=refused)


def unanswerable(q="q", *, refused):
    return CaseResult(case=EvalCase(question=q, expect_refusal=True),
                      retrieved_texts=[], refused=refused)


# ---- recall@k -------------------------------------------------------------

def test_recall_counts_only_answerable_questions():
    results = [
        answerable("a", found=True),
        answerable("b", found=False),
        unanswerable("c", refused=True),   # must not count either way
    ]
    assert recall_at_k(results) == 0.5


def test_recall_is_one_when_every_answer_is_found():
    assert recall_at_k([answerable("a", found=True),
                        answerable("b", found=True)]) == 1.0


def test_recall_of_an_empty_set_is_zero_not_a_crash():
    assert recall_at_k([]) == 0.0
    assert recall_at_k([unanswerable("only", refused=True)]) == 0.0


def test_a_hit_requires_every_expected_fragment():
    # Both fragments must appear, so a chunk that happens to contain one
    # common word does not score as a hit.
    case = EvalCase(question="q", answer_in=("300 GBP", "desk"))
    partial = CaseResult(case=case, retrieved_texts=["you may claim 300 GBP"])
    full = CaseResult(case=case, retrieved_texts=["claim 300 GBP towards a desk"])

    assert partial.hit() is False
    assert full.hit() is True


def test_a_hit_can_span_several_retrieved_chunks():
    # The answer may be split across chunk boundaries; that is still a hit,
    # because the model receives all of them together.
    case = EvalCase(question="q", answer_in=("300 GBP", "desk"))
    r = CaseResult(case=case, retrieved_texts=["... 300 GBP ...", "... a desk ..."])
    assert r.hit() is True


# ---- refusal --------------------------------------------------------------

def test_refusal_accuracy_counts_only_unanswerable_questions():
    results = [
        unanswerable("a", refused=True),
        unanswerable("b", refused=False),   # should have declined
        answerable("c", found=True),        # irrelevant to this metric
    ]
    assert refusal_accuracy(results) == 0.5


def test_refusing_everything_does_not_look_good():
    # The point of keeping the two metrics apart. A system that declines every
    # question scores perfectly on refusal and zero on recall, and the false
    # refusals name exactly what it broke.
    results = [
        answerable("a", found=True, refused=True),
        answerable("b", found=True, refused=True),
        unanswerable("c", refused=True),
    ]
    report = score(results)

    assert report.refusal_accuracy == 1.0
    assert report.recall_at_k == 1.0          # retrieval worked...
    assert len(report.false_refusals) == 2    # ...but it refused anyway
    assert "a" in report.false_refusals


# ---- the report -----------------------------------------------------------

def test_report_counts_and_lists_the_failures():
    results = [
        answerable("found it", found=True),
        answerable("missed it", found=False),
        unanswerable("declined", refused=True),
        unanswerable("should have declined", refused=False),
    ]
    report = score(results)

    assert report.answerable == 2
    assert report.unanswerable == 2
    assert report.recall_at_k == 0.5
    assert report.refusal_accuracy == 0.5
    assert report.missed == ["missed it"]
    assert report.wrongly_answered == ["should have declined"]


def test_summary_names_an_unanswerable_question_that_got_answered():
    # This line was missing: a run with refusal accuracy at 80% still ended
    # "no misses and no false refusals", and the one bad answer was easy to miss.
    report = score([answerable("a", found=True), unanswerable("Who is the CEO?", refused=False)])
    assert "Who is the CEO?" in report.summary()
    assert "no wrong answers" not in report.summary()


def test_summary_says_so_when_nothing_failed():
    report = score([answerable("a", found=True), unanswerable("b", refused=True)])
    assert "no misses, no false refusals, no wrong answers" in report.summary()
