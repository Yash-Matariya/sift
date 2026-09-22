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


def generate_answer(hits, question: str) -> ChatResponse:
    """Build the grounded prompt from the retrieved chunks and answer, with citations."""
    # Number the chunks so the context is clear and citations line up with sources.
    context = "\n\n".join(
        f"[{i + 1}] {chunk.text}" for i, (chunk, _) in enumerate(hits)
    )
    answer = ANSWER_CHAIN.invoke({"context": context, "question": question})

    citations = [
        Citation(document=chunk.document.filename,
                 chunk_index=chunk.chunk_index,
                 text=chunk.text)
        for chunk, _ in hits
    ]
    return ChatResponse(answer=answer, citations=citations, refused=False)


def refusal() -> ChatResponse:
    """The response when nothing relevant was found — refuse, don't invent."""
    return ChatResponse(
        answer="I don't have anything relevant in the documents to answer that.",
        citations=[],
        refused=True,
    )
