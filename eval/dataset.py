"""The labelled evaluation set.

Small and hand-written on purpose. Twenty questions you understand beats two
hundred you generated and never read, because the whole value is being able to
look at a failure and decide whether the system or the label is wrong.

**The corpus has to be big enough for the metric to mean anything.** The first
version of this set scored 100% recall against a two-chunk document while
`top_k` was 4 — every chunk was retrieved for every question, so the number
could not fail and measured nothing. The corpus is now an eleven-chunk handbook
covering fifteen distinct topics, so retrieving four of eleven is a real choice
and a bad retriever will miss.

Answers are identified by **substrings that must appear in the retrieved text**,
not by chunk ids. Chunk boundaries move every time you tune chunk size, and an
evaluation set that breaks whenever you tune the thing it measures is one
nobody runs twice.

The unanswerable questions are deliberately *plausible* — the sort of thing
someone would reasonably ask an employee handbook. Refusing an obviously
irrelevant question proves very little.
"""
from app.evaluation import EvalCase

CORPUS = "eval/corpus/handbook.txt"

CASES = [
    # --- answerable, spread across the document ---------------------------
    EvalCase("How much can I claim towards a desk or chair?", ("300",)),
    EvalCase("What are the core hours for remote workers?", ("11:00", "16:00")),
    EvalCase("Who is eligible to work remotely?", ("probation",)),
    EvalCase("How far from the office do I have to live to avoid coming in?", ("80",)),
    EvalCase("Can I work from a coffee shop wifi?", ("VPN",)),
    EvalCase("How long do I have to submit an expense claim?", ("sixty days",)),
    EvalCase("What is the mileage rate for using my own car?", ("45 pence",)),
    EvalCase("How much can I spend on meals when staying away overnight?", ("40 GBP",)),
    EvalCase("How many days of annual leave do I get?", ("twenty-eight",)),
    EvalCase("How many sick days can I self-certify?", ("seven",)),
    EvalCase("How long is parental leave for a primary caregiver?", ("twenty-six weeks",)),
    EvalCase("How much am I paid for being on call?", ("200 GBP",)),
    EvalCase("How quickly should code reviews happen?", ("one working day",)),
    EvalCase("When do I get my laptop as a new joiner?", ("first day",)),

    # Phrased nothing like the document, to exercise the rewrite path:
    # "commuting" and "paid back" do not appear; "reimbursed" does.
    EvalCase("Do I get paid back for commuting to the office?", ("not reimbursed",)),

    # --- unanswerable: plausible, but genuinely absent --------------------
    EvalCase("What is the company pension contribution rate?", expect_refusal=True),
    EvalCase("Who is the CEO of Acme Corp?", expect_refusal=True),
    EvalCase("What is the notice period for resignation?", expect_refusal=True),
    EvalCase("How often are salary reviews held?", expect_refusal=True),
    EvalCase("What is the dress code in the office?", expect_refusal=True),
]
