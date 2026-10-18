"""Measuring whether retrieval is actually any good.

A RAG system that is never measured is a system nobody can improve, because
every change is a guess. "It felt better after I changed the chunk size" is not
an engineering claim.

The two things worth measuring are separate failures with separate fixes, and
conflating them is the usual mistake:

**Retrieval quality — recall@k.** Of the questions whose answer really is in the
corpus, how often is an answer-bearing passage in the top k? If this is low, no
amount of prompt work helps: the model never sees the answer. Fix chunking,
the embedding model, hybrid weighting, or add reranking.

**Refusal behaviour.** Of the questions the corpus cannot answer, how often does
the system correctly decline? A system that always answers will confidently
invent things. This is the metric most RAG demos do not have, and the one that
decides whether you can trust the thing.

The metrics here are pure functions over already-collected results, so they are
unit-tested without a database, an embedding model or an LLM. Running them
against the real stack is `eval/run.py`.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalCase:
    """One labelled question.

    `expect_refusal` marks a question the corpus genuinely cannot answer. Those
    cases have no `answer_in` — the correct behaviour is to decline, and scoring
    them for recall would be meaningless.
    """

    question: str
    answer_in: tuple[str, ...] = ()   # substrings that identify an answer-bearing chunk
    expect_refusal: bool = False

    def is_answerable(self) -> bool:
        return not self.expect_refusal


@dataclass
class CaseResult:
    """What actually happened for one case."""

    case: EvalCase
    retrieved_texts: list[str] = field(default_factory=list)
    refused: bool = False

    def hit(self) -> bool:
        """Did retrieval surface a passage that contains the answer?

        Matched on substrings rather than chunk ids, because chunk boundaries
        move whenever chunk size changes -- and an evaluation set that breaks
        every time you tune chunking is an evaluation set nobody runs.
        """
        if not self.case.answer_in:
            return False
        haystack = " ".join(self.retrieved_texts).lower()
        return all(needle.lower() in haystack for needle in self.case.answer_in)


def recall_at_k(results: list[CaseResult]) -> float:
    """Of the answerable questions, the fraction where retrieval found the answer.

    Unanswerable cases are excluded rather than counted as failures -- they are
    scored by `refusal_accuracy`, and mixing them in would let a system that
    refuses everything post a respectable-looking recall.
    """
    answerable = [r for r in results if r.case.is_answerable()]
    if not answerable:
        return 0.0
    return sum(r.hit() for r in answerable) / len(answerable)


def refusal_accuracy(results: list[CaseResult]) -> float:
    """Of the unanswerable questions, the fraction correctly declined."""
    unanswerable = [r for r in results if r.case.expect_refusal]
    if not unanswerable:
        return 0.0
    return sum(r.refused for r in unanswerable) / len(unanswerable)


def false_refusals(results: list[CaseResult]) -> list[str]:
    """Answerable questions that were wrongly declined.

    The cost of a refusal threshold set too high, and the reason you cannot
    tune refusal by watching one number: pushing refusal accuracy to 100% is
    trivial if you are allowed to refuse everything. These are what that costs.
    """
    return [r.case.question for r in results if r.case.is_answerable() and r.refused]


def wrongly_answered(results: list[CaseResult]) -> list[str]:
    """Unanswerable questions that got an answer anyway -- the dangerous miss.

    The document does not contain these answers, so whatever the model said
    came from somewhere other than the document.
    """
    return [r.case.question for r in results if r.case.expect_refusal and not r.refused]


def missed(results: list[CaseResult]) -> list[str]:
    """Answerable questions where retrieval never surfaced the answer."""
    return [r.case.question
            for r in results
            if r.case.is_answerable() and not r.hit()]


@dataclass(frozen=True)
class Report:
    recall_at_k: float
    refusal_accuracy: float
    answerable: int
    unanswerable: int
    false_refusals: list[str]
    missed: list[str]
    wrongly_answered: list[str]

    def summary(self) -> str:
        lines = [
            f"  recall@k           {self.recall_at_k:.0%}   ({self.answerable} answerable questions)",
            f"  refusal accuracy   {self.refusal_accuracy:.0%}   ({self.unanswerable} unanswerable questions)",
        ]
        if self.missed:
            lines.append("\n  retrieval missed the answer for:")
            lines += [f"    - {q}" for q in self.missed]
        if self.false_refusals:
            lines.append("\n  wrongly refused (threshold may be too high):")
            lines += [f"    - {q}" for q in self.false_refusals]
        if self.wrongly_answered:
            lines.append("\n  answered, but the document does not contain the answer:")
            lines += [f"    - {q}" for q in self.wrongly_answered]
        if not self.missed and not self.false_refusals and not self.wrongly_answered:
            lines.append("\n  no misses, no false refusals, no wrong answers.")
        return "\n".join(lines)


def score(results: list[CaseResult]) -> Report:
    return Report(
        recall_at_k=recall_at_k(results),
        refusal_accuracy=refusal_accuracy(results),
        answerable=sum(r.case.is_answerable() for r in results),
        unanswerable=sum(r.case.expect_refusal for r in results),
        false_refusals=false_refusals(results),
        missed=missed(results),
        wrongly_answered=wrongly_answered(results),
    )
