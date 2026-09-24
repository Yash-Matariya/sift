#!/usr/bin/env python
"""Run the evaluation set against the real stack and print the numbers.

    docker compose up -d db redis
    .venv/bin/python -m eval.run

Needs Postgres and Ollama, because measuring retrieval with a fake embedder
measures nothing. It ingests the corpus into a throwaway document, asks every
question, scores the results, and deletes what it created.

Two numbers come out:

  recall@k          of the answerable questions, how often retrieval surfaced
                    an answer-bearing passage. Low means the model never sees
                    the answer, and no prompt change will fix that.

  refusal accuracy  of the unanswerable questions, how often it correctly
                    declined. A system that always answers will invent things.

Read them together. Refusing everything scores 100% on the second and produces
a list of false refusals, which is why both are printed.

Exit code is non-zero if either number falls below the thresholds below, so
this can gate a release.
"""
import sys
import time

from app.agent import answer_question
from app.db import SessionLocal, init_db
from app.evaluation import CaseResult, score
from app.ingest import ingest_into
from app.models import Document
from app.retrieval import retrieve
from eval.dataset import CASES, CORPUS

# Below these, something has regressed badly enough to look at.
MIN_RECALL = 0.70
MIN_REFUSAL_ACCURACY = 0.50


def _ingest_corpus(session) -> Document:
    """Load the evaluation corpus as a document, reusing it if already there."""
    doc = Document(filename=CORPUS.split("/")[-1], source_path=CORPUS, status="pending")
    session.add(doc)
    session.commit()
    session.refresh(doc)

    print(f"  ingesting {CORPUS} ...", end="", flush=True)
    t0 = time.time()
    ingest_into(session, doc.id)
    print(f" done in {time.time() - t0:.1f}s")
    return doc


def main() -> int:
    init_db()
    session = SessionLocal()
    doc = None
    try:
        doc = _ingest_corpus(session)

        print(f"\n  running {len(CASES)} cases\n")
        results = []
        for case in CASES:
            # Retrieval is scored on the question as asked, so recall measures
            # the retriever rather than the agent's rewriting. The refusal
            # decision comes from the full agent, because that is the behaviour
            # a user actually gets.
            hits = retrieve(session, case.question)
            response = answer_question(session, case.question)

            results.append(CaseResult(
                case=case,
                retrieved_texts=[chunk.text for chunk, _ in hits],
                refused=response.refused,
            ))

            mark = "ok " if (
                (case.expect_refusal and response.refused)
                or (case.is_answerable() and results[-1].hit() and not response.refused)
            ) else "  ✗"
            print(f"    {mark} {case.question}")

        report = score(results)
        print("\n" + report.summary() + "\n")

        failed = False
        if report.recall_at_k < MIN_RECALL:
            print(f"  FAIL recall@k {report.recall_at_k:.0%} is below {MIN_RECALL:.0%}")
            failed = True
        if report.refusal_accuracy < MIN_REFUSAL_ACCURACY:
            print(f"  FAIL refusal accuracy {report.refusal_accuracy:.0%} "
                  f"is below {MIN_REFUSAL_ACCURACY:.0%}")
            failed = True
        return 1 if failed else 0

    finally:
        # Leave the database as it was found.
        if doc is not None:
            session.delete(doc)      # chunks cascade
            session.commit()
        session.close()


if __name__ == "__main__":
    sys.exit(main())
