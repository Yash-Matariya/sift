"""
The building blocks for turning retrieved chunks into a grounded answer.

The answer path is composed with **LangChain (LCEL)**: a ChatPromptTemplate fills
in the retrieved context and the question, a Runnable calls the model, and a
StrOutputParser hands back a clean string — piped together as

    prompt | model | parser

Everything in that pipe is a Runnable, which is why it composes. The model step
wraps our own Ollama client (app/llm.py), so there's one place that talks to the
model and LangChain handles the prompt/parse plumbing around it.

`generate_answer` builds the grounded answer with citations; `refusal` is the
"I don't know" response. The agent (app/agent.py) decides *when* to call these;
this module is *how* an answer is produced.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from app.llm import generate
from app.models import ChatResponse, Citation

# The guardrail lives in the system message so it governs every answer:
# answer only from the retrieved context, and admit when it isn't there.
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the question using ONLY the context below. "
     "If the context does not contain the answer, say you don't know.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def _call_model(prompt_value) -> str:
    """The model step of the chain: render the prompt and ask Ollama."""
    return generate(prompt_value.to_string())


# prompt -> model -> clean string. Swapping the model means swapping this one step.
ANSWER_CHAIN = PROMPT | RunnableLambda(_call_model) | StrOutputParser()


# The model is the last line of defence, and the system prompt gives it an
# explicit instruction: "if the context does not contain the answer, say you
# don't know." When it honours that, the response IS a refusal and has to be
# reported as one -- otherwise `refused` is a lie and the caller shows a
# non-answer as though it were an answer, with citations attached to it.
#
# This is not fragile phrase-guessing at arbitrary prose. It checks for the one
# behaviour the prompt asked for, in one of two shapes:
#
#   - the reply OPENS with it: "I don't know. The context only covers remote
#     work, expenses and leave..." The explanation after it can be any length.
#     An earlier version only looked at replies of 200 characters or less, so
#     when the model explained its refusal, the refusal was reported as an
#     answer, with citations -- the eval caught it on "Who is the CEO?".
#   - a SHORT reply contains it anywhere ("Sorry, I don't know.").
#
# A long reply that merely mentions not knowing somewhere in the middle is an
# answer that discusses it, and is left alone.
_DECLINE_PHRASES = (
    "i don't know",
    "i do not know",
    "i don't have enough information",
    "i do not have enough information",
    "the context does not contain",
    "the context doesn't contain",
)
_DECLINE_MAX_CHARS = 200


def _model_declined(answer: str) -> bool:
    """Did the model follow the prompt's instruction to admit it doesn't know?"""
    stripped = answer.strip().lower().replace("\u2019", "'")   # curly apostrophe
    if stripped.startswith(_DECLINE_PHRASES):
        return True
    if len(stripped) > _DECLINE_MAX_CHARS:
        return False
    return any(phrase in stripped for phrase in _DECLINE_PHRASES)


def _sources(chunk) -> str:
    """Where a passage came from, including any word-for-word copies retrieval merged into it."""
    names = [c.document.filename for c in [chunk, *getattr(chunk, "duplicates", [])]]
    return ("source: " if len(names) == 1 else "sources: ") + ", ".join(names)


def generate_answer(hits, question: str) -> ChatResponse:
    """Build the grounded prompt from the retrieved chunks and answer, with citations."""
    # Number the chunks so the context is clear and citations line up with sources.
    # Each passage once; if the same text is in several documents, the model is told every
    # source, so it can say so in its answer.
    context = "\n\n".join(
        f"[{i + 1}] ({_sources(chunk)}) {chunk.text}" for i, (chunk, _) in enumerate(hits)
    )
    answer = ANSWER_CHAIN.invoke({"context": context, "question": question})

    # Retrieval passed the relevance gate but the passages still did not answer
    # the question. Report it as the refusal it is, and drop the citations --
    # sources under "I don't know" imply the answer came from them.
    if _model_declined(answer):
        return ChatResponse(answer=answer.strip(), citations=[], refused=True)

    citations = [
        Citation(document=c.document.filename, chunk_index=c.chunk_index, text=c.text)
        for chunk, _ in hits
        for c in [chunk, *getattr(chunk, "duplicates", [])]
    ]
    return ChatResponse(answer=answer, citations=citations, refused=False)


def refusal() -> ChatResponse:
    """The response when nothing relevant was found — refuse, don't invent."""
    return ChatResponse(
        answer="I don't have anything relevant in the documents to answer that.",
        citations=[],
        refused=True,
    )
