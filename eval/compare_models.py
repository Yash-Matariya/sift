#!/usr/bin/env python
"""Run the evaluation set against several chat models and print a comparison.

    docker compose up -d db
    .venv/bin/python -m eval.compare_models llama3.1:8b qwen2.5:7b gemma3:4b

This is how the default chat model in app/config.py was chosen. "Pick a model"
is otherwise guesswork -- the eval set turns it into a measurement, and the
measurement is the answer to "why this model and not a bigger one".

What moves and what does not:

  recall@k          does NOT depend on the chat model. Retrieval is embeddings
                    plus full-text search over the same corpus every time, so
                    this number is a constant across the table. It is printed
                    anyway, because a column that does not move is itself
                    evidence: it shows which half of the system a model change
                    can and cannot affect.

  refusal accuracy  DOES depend on the chat model. Deciding that retrieved
                    passages do not answer the question, and saying so instead
                    of improvising, is the model's job. This is the column that
                    separates candidates.

  seconds           wall-clock for the whole run on this machine. A model that
                    scores the same and answers twice as fast is the better
                    choice, and on a laptop it is the difference between a demo
                    that lands and one that stalls.

The corpus is small and the question set is 20 cases, so treat the numbers as a
sanity check on a laptop, not a benchmark. That limitation is the honest thing
to say about it.
"""
import sys
import time

from app.agent import answer_question
from app.config import settings
from app.db import SessionLocal, init_db
from app.evaluation import CaseResult, score
from app.retrieval import retrieve
from eval.dataset import CASES
from eval.run import _ingest_corpus


def _run_one(session, model: str) -> tuple:
    """Score the whole case set against one chat model."""
    original = settings.chat_model
    settings.chat_model = model
    t0 = time.time()
    try:
        results = []
        for case in CASES:
            hits = retrieve(session, case.question)
            response = answer_question(session, case.question)
            results.append(CaseResult(
                case=case,
                retrieved_texts=[chunk.text for chunk, _ in hits],
                refused=response.refused,
            ))
        return score(results), time.time() - t0
    finally:
        settings.chat_model = original


def main(models: list[str]) -> int:
    if not models:
        print(__doc__)
        return 2

    missing = [m for m in models if not _installed(m)]
    if missing:
        print(f"  not installed: {', '.join(missing)}")
        print(f"  run: {' && '.join(f'ollama pull {m}' for m in missing)}")
        return 2

    init_db()
    session = SessionLocal()
    doc = None
    try:
        doc = _ingest_corpus(session)
        print()
        print(f"  {'model':<20} {'recall@k':>9} {'refusal':>9} {'seconds':>9}")
        print(f"  {'-' * 20} {'-' * 9} {'-' * 9} {'-' * 9}")

        rows = []
        for model in models:
            report, elapsed = _run_one(session, model)
            rows.append((model, report, elapsed))
            print(f"  {model:<20} {report.recall_at_k:>8.0%} "
                  f"{report.refusal_accuracy:>9.0%} {elapsed:>8.0f}s")

        best = max(rows, key=lambda r: (r[1].refusal_accuracy, -r[2]))
        print(f"\n  best on refusal accuracy, then speed: {best[0]}")
        print("  recall@k is identical across models because retrieval does "
              "not use the chat model.")
        print("  This ranking does NOT measure how well an answer reads, and "
              "with only "
              f"{sum(1 for c in CASES if c.expect_refusal)} refusal cases the "
              "refusal column moves in large steps.")
        print("  Read the answers before trusting the table.\n")
        return 0
    finally:
        if doc is not None:
            session.delete(doc)      # chunks cascade
            session.commit()
        session.close()


def _installed(model: str) -> bool:
    """Ask Ollama whether the model is pulled, so the run fails fast."""
    import httpx
    try:
        tags = httpx.get(f"{settings.ollama_url}/api/tags", timeout=10).json()
    except Exception:
        return False
    names = {m["name"] for m in tags.get("models", [])}
    return model in names or f"{model}:latest" in names


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
