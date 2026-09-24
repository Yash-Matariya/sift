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
    attempts: int          # how many times we've retrieved
    hits: list             # [(chunk, score)] from the last retrieval
    response: ChatResponse # the final answer/refusal


REWRITE_PROMPT = """Rewrite the following question into a short search query that \
would retrieve the most relevant passages. Return only the rewritten query.

Question: {question}
Search query:"""


def rewrite_query(question: str) -> str:
    """Use the model to rephrase the question for better retrieval."""
    return generate(REWRITE_PROMPT.format(question=question)).strip()


def _best_score(hits) -> float:
    return max((score for _, score in hits), default=0.0)


def build_graph(session: Session):
    """Compile the agent graph with its nodes bound to a database session."""

    def retrieve_node(state: AgentState) -> AgentState:
        hits = retrieve(session, state["query"])
        return {"hits": hits, "attempts": state.get("attempts", 0) + 1}

    def rewrite_node(state: AgentState) -> AgentState:
        return {"query": rewrite_query(state["question"])}

    def generate_node(state: AgentState) -> AgentState:
        return {"response": generate_answer(state["hits"], state["question"])}

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
    )
    return response
