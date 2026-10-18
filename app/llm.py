"""
The one place that talks to a language model.

Everything else in the app calls just two functions -- `embed()` and
`generate()` -- so the provider is a configuration choice rather than something
threaded through the codebase. Two backends, selected by `settings.llm_provider`:

  "ollama" (default)
      Models run locally. Free, no API key, and nothing leaves the machine --
      which matters for a *document* assistant, where the documents are the
      whole point. It also means anyone can clone this and run it.

  "openai"
      Any OpenAI-compatible endpoint: OpenAI itself, or a gateway such as
      OpenRouter, Groq or Together, or a self-hosted vLLM. Set LLM_PROVIDER=openai
      plus OPENAI_BASE_URL, OPENAI_API_KEY and the model names.

**Generation is freely swappable. Embedding is not**, and the asymmetry is the
interesting part. A generated answer is thrown away after the request, so
changing that model changes only quality. An embedding is *stored*: every vector
in the database was produced by one model, similarity between vectors from two
different models is meaningless, and the pgvector column has a fixed width. So
changing the embedding model means re-embedding every document and migrating the
column -- it is a data migration, not a config change. `embed()` therefore checks
the width it got back and fails loudly rather than silently poisoning the store.
"""
import httpx

from app.config import settings


class EmbeddingDimensionError(RuntimeError):
    """An embedding came back the wrong width for the configured column."""


# --------------------------------------------------------------------- ollama

def _ollama_embed(text: str) -> list[float]:
    resp = httpx.post(
        f"{settings.ollama_url}/api/embeddings",
        json={"model": settings.embed_model, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def _ollama_generate(prompt: str) -> str:
    resp = httpx.post(
        f"{settings.ollama_url}/api/generate",
        json={
            "model": settings.chat_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": settings.temperature},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# --------------------------------------------------------- openai-compatible

def _openai_headers() -> dict:
    return {"Authorization": f"Bearer {settings.openai_api_key}"}


def _openai_embed(text: str) -> list[float]:
    resp = httpx.post(
        f"{settings.openai_base_url}/embeddings",
        headers=_openai_headers(),
        json={"model": settings.openai_embed_model, "input": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def _openai_generate(prompt: str) -> str:
    resp = httpx.post(
        f"{settings.openai_base_url}/chat/completions",
        headers=_openai_headers(),
        json={
            "model": settings.openai_chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": settings.temperature,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# --------------------------------------------------------------- the two APIs

_EMBED = {"ollama": _ollama_embed, "openai": _openai_embed}
_GENERATE = {"ollama": _ollama_generate, "openai": _openai_generate}


def _backend(table: dict):
    try:
        return table[settings.llm_provider]
    except KeyError:
        raise ValueError(
            f"unknown llm_provider {settings.llm_provider!r}; "
            f"expected one of {sorted(table)}"
        ) from None


def embed(text: str) -> list[float]:
    """Turn a piece of text into a vector (a list of floats).

    Fails if the vector is not the width the `chunks.embedding` column was
    created with -- a mismatch means the configured model is not the one the
    stored vectors came from, and every distance computed against them would be
    nonsense.
    """
    vector = _backend(_EMBED)(text)
    if len(vector) != settings.embed_dim:
        raise EmbeddingDimensionError(
            f"embedding model returned {len(vector)} dimensions but the store "
            f"expects {settings.embed_dim}. Changing the embedding model means "
            f"re-embedding every document and altering the vector column."
        )
    return vector


def generate(prompt: str) -> str:
    """Ask the chat model for a single (non-streamed) answer."""
    return _backend(_GENERATE)(prompt)
