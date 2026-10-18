#!/usr/bin/env python
"""A harder evaluation: duplicate uploads, a second document, and vague questions.

    DATABASE_URL=postgresql+psycopg://sift:sift@localhost:5435/sift_eval .venv/bin/python -m eval.hard

Run it against an empty database (the line above uses a separate one), because
anything else in the database competes for the same top-k slots.

The main set (eval/run.py) has one document and specific questions, so it
scored 100% recall and could not see the failure that prompted this file: the
handbook uploaded three times, a short email with the answer, and a question
like "company and my email?". Three identical copies of each handbook passage
filled every retrieval slot, the email never reached the model, and it
honestly said "I don't know".

Two numbers per case:
  reached   did the passages sent to the model contain every expected string?
  answered  did the final answer contain them?
"""
import sys

from app import agent
from app.agent import answer_question
from app.db import SessionLocal, init_db
from app.ingest import ingest_into
from app.models import Document
from app.retrieval import retrieve

HANDBOOK = "eval/corpus/handbook.txt"
EMAIL = "eval/corpus/shortlist_email.txt"

# (question, strings the passages must contain, strings the answer must contain)
CASES = [
    ("company and my email?", ("Nimbus", "arjun.rao.2003@gmail.com"), ("nimbus", "arjun.rao.2003@gmail.com")),
    ("what is my email address?", ("arjun.rao.2003@gmail.com",), ("arjun.rao.2003@gmail.com",)),
    ("who sent me the shortlisting mail?", ("Meera Iyer",), ("meera",)),
    ("by when do I have to fill the questionnaire?", ("9 October 2026",), ("9 october",)),
    ("which company shortlisted me and where is their office?", ("Nimbus", "Prestige Tech Park"), ("nimbus", "prestige")),
    ("internship email from the company?", ("Nimbus",), ("nimbus",)),
]


def _search(session, question):
    """The passages the agent's first retrieval would use (the new search() if present)."""
    search = getattr(agent, "search", None)
    return search(session, question) if search else retrieve(session, question)


def _add(session, path, name):
    doc = Document(filename=name, source_path=path, status="pending")
    session.add(doc)
    session.commit()
    ingest_into(session, doc.id)
    return doc


def main() -> int:
    init_db()
    session = SessionLocal()
    docs = []
    try:
        for i in range(3):                                   # the duplicate uploads
            docs.append(_add(session, HANDBOOK, f"handbook_copy_{i + 1}.txt"))
        docs.append(_add(session, EMAIL, "shortlist_email.txt"))

        reached = answered = 0
        for question, need, expect in CASES:
            hits = _search(session, question)
            text = " ".join(chunk.text for chunk, _ in hits).lower()
            r = all(s.lower() in text for s in need)
            reply = answer_question(session, question)
            a = (not reply.refused) and all(s in reply.answer.lower() for s in expect)
            reached += r
            answered += a
            print(f"  reached {'yes' if r else 'NO '}  answered {'yes' if a else 'NO '}  {question}")
        n = len(CASES)
        print(f"\n  reached the model  {reached}/{n}")
        print(f"  answered correctly {answered}/{n}")
        return 0
    finally:
        for doc in docs:
            session.delete(doc)
        session.commit()
        session.close()


if __name__ == "__main__":
    sys.exit(main())
