r"""
The agentic RAG loop, built with LangGraph.

Plain RAG retrieves once and answers. This agent can *decide*: it retrieves,
grades whether the chunks are good enough, and if they're weak it rewrites the
query and retrieves again — up to a bounded number of attempts — before either
answering (grounded, with citations) or refusing. It's a bounded
reason-act-observe loop: a state machine of nodes and edges over a shared state.

The graph:

    START -> retrieve -> [grade] --good/exhausted-with-hits--> generate -> END
                            |  \
                            |   \--nothing-at-all-and-exhausted--> refuse -> END
                            \--weak-and-attempts-left--> rewrite -> retrieve (loop)

The loop is bounded by settings.max_retrieval_attempts so it can never spin
forever. Retrieval, generation, and query-rewriting are module-level functions,
so tests replace them with fakes and exercise the whole graph without a database
or a model.
"""
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from sqlalchemy.orm import Session

from app.config import settings
from app.llm import generate
from app.retrieval import retrieve
from app.rag import generate_answer, refusal
from app.models import AgentTrace, ChatResponse


class AgentState(TypedDict, total=False):
    question: str          # the original question
    query: str             # the current (possibly rewritten) search query
    extra: list            # more searches the question was split into (multi-query)
    attempts: int          # how many times we've retrieved
    hits: list             # [(chunk, score)] from the last retrieval
    widened: bool          # searched wider after the model first declined
    response: ChatResponse # the final answer/refusal


# Multi-query retrieval. One embedding of "company and my email?" is a blur of two topics and
# matches neither well; searching "company" and "email address" separately, then merging, finds
# both. The prompt says it is not answering, for the same reason as REWRITE_PROMPT below.
EXPAND_PROMPT = """You help a document search engine. Split the user's question into the \
separate things it asks about, and write one short search query for each. If it asks about \
only one thing, write one query. You are NOT answering the question.

Reply with the queries only: one per line, at most 3, no numbering, no quotes, no explanation.

Question: {question}
Queries:"""


# The first version of this prompt just said "rewrite the question into a
# search query". llama3.1:8b then treated some questions as questions to
# answer, and refused: on 3 of 25 test questions it replied "I can't answer
# that. I'm happy to help with other questions though!" -- and that refusal
# became the search query. Saying plainly that it is not answering anything
# took that to 0 of 25.
REWRITE_PROMPT = """You turn a user's question into a search query for a document search \
engine. You are NOT answering the question, so it does not matter whether you \
know the answer. Keep the key terms, names and codes; drop filler words.

Reply with the search query only: one line, no quotes, no explanation.

Question: {question}
Search query:"""

# How a small model sounds when it refuses instead of doing the task.
_REFUSAL_OPENINGS = ("i can't", "i cannot", "i can not", "i'm sorry", "i am sorry",
                     "sorry", "i'm unable", "i am unable", "as an ai")
_REWRITE_MAX_WORDS = 20


def rewrite_query(question: str) -> str:
    """Use the model to rephrase the question for better retrieval.

    The reply goes straight into the search, so it is checked before it is
    trusted. A refusal, an empty reply or a paragraph of chat would retrieve
    nothing useful; any of those falls back to the original question, which is
    never worse than what the first attempt searched for.
    """
    reply = generate(REWRITE_PROMPT.format(question=question)).strip()
    query = reply.splitlines()[0].strip().strip("\"'") if reply else ""
    if (not query
            or len(query.split()) > _REWRITE_MAX_WORDS
            or query.lower().replace("\u2019", "'").startswith(_REFUSAL_OPENINGS)):
        return question
    return query


def expand_queries(question: str) -> list[str]:
    """Extra search queries covering the separate parts of a question ([] if off or unusable).

    The reply goes straight into the search, so each line is checked like rewrite_query's. If
    the model is unreachable, the plain search still runs: splitting is a bonus, never a
    requirement.
    """
    if not settings.multi_query:
        return []
    try:
        reply = generate(EXPAND_PROMPT.format(question=question))
    except Exception:
        return []
    out = []
    for line in reply.splitlines():
        q = line.strip().lstrip("-*•0123456789.) ").strip().strip("\"'")
        low = q.lower().replace("\u2019", "'")
        if (not q or len(q.split()) > _REWRITE_MAX_WORDS or low.startswith(_REFUSAL_OPENINGS)
                or low == question.strip().lower() or q in out):
            continue
        out.append(q)
    return out[:3]


def _search(session: Session, query: str, extra: list, k: int | None = None):
    """retrieve(), passing the extra searches only when there are some."""
    if extra:
        return retrieve(session, query, k=k, extra_queries=extra)
    return retrieve(session, query, k=k) if k else retrieve(session, query)


def search(session: Session, question: str):
    """The passages the agent's first retrieval uses: the question plus its split-out searches."""
    return _search(session, question, expand_queries(question))


def _best_score(hits) -> float:
    return max((score for _, score in hits), default=0.0)


def build_graph(session: Session):
    """Compile the agent graph with its nodes bound to a database session."""

    def retrieve_node(state: AgentState) -> AgentState:
        update = {"attempts": state.get("attempts", 0) + 1}
        extra = state.get("extra")
        if extra is None:                            # first retrieval: split the question once
            extra = expand_queries(state["question"])
            update["extra"] = extra
        if state["query"] != state["question"]:      # a rewritten query searches on its own
            extra = []
        update["hits"] = _search(session, state["query"], extra)
        return update

    def rewrite_node(state: AgentState) -> AgentState:
        return {"query": rewrite_query(state["question"])}

    def generate_node(state: AgentState) -> AgentState:
        response = generate_answer(state["hits"], state["question"])
        if response.refused and settings.widen_on_decline and state["hits"]:
            # Retrieval passed the gate but the model saw no answer: the answer may be just
            # outside the top-k. Look once more, wider, before giving up.
            extra = [] if state["query"] != state["question"] else state.get("extra", [])
            wider = _search(session, state["query"], extra, k=settings.wide_k)
            if len(wider) > len(state["hits"]):
                return {"response": generate_answer(wider, state["question"]),
                        "hits": wider, "widened": True}
        return {"response": response}

    def refuse_node(state: AgentState) -> AgentState:
        return {"response": refusal()}

    def grade(state: AgentState) -> str:
        """Decide what to do after a retrieval."""
        hits = state.get("hits", [])
        if _best_score(hits) >= settings.good_score:
            return "generate"                       # confident -> answer
        if state.get("attempts", 0) < settings.max_retrieval_attempts:
            return "rewrite"                        # weak -> try a better query
        if hits:
            return "generate"                       # out of tries but have something
        return "refuse"                             # nothing relevant -> refuse

    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("generate", generate_node)
    builder.add_node("refuse", refuse_node)

    builder.add_edge(START, "retrieve")
    builder.add_conditional_edges("retrieve", grade,
                                  {"generate": "generate", "rewrite": "rewrite",
                                   "refuse": "refuse"})
    builder.add_edge("rewrite", "retrieve")         # the loop
    builder.add_edge("generate", END)
    builder.add_edge("refuse", END)
    return builder.compile()


def answer_question(session: Session, question: str) -> ChatResponse:
    """Run the agentic RAG loop and return the answer (or refusal), with a trace.

    The trace is assembled from the final state rather than tracked separately,
    so it cannot drift from what the graph actually did.
    """
    graph = build_graph(session)
    final = graph.invoke({"question": question, "query": question, "attempts": 0})

    response = final["response"]
    hits = final.get("hits", [])
    final_query = final.get("query", question)

    response.trace = AgentTrace(
        attempts=final.get("attempts", 0),
        rewritten=final_query.strip() != question.strip(),
        final_query=final_query,
        best_score=round(_best_score(hits), 4),
        chunks_found=len(hits),
        searches=1 + len(final.get("extra") or []),
        widened=bool(final.get("widened")),
    )
    return response
