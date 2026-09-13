"""
The one place that talks to the language model.

We use Ollama, which runs models locally and exposes a simple HTTP API — free,
no API key. Two things we need: turn text into an embedding vector, and generate
an answer from a prompt. Keeping both behind this small module means the rest of
the app never knows which provider we use — swap Ollama for OpenAI here and
nothing else changes.
"""
import httpx

from app.config import settings


def embed(text: str) -> list[float]:
    """Turn a piece of text into a vector (a list of floats)."""
    resp = httpx.post(
        f"{settings.ollama_url}/api/embeddings",
        json={"model": settings.embed_model, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def generate(prompt: str) -> str:
    """Ask the chat model for a single (non-streamed) answer."""
    resp = httpx.post(
        f"{settings.ollama_url}/api/generate",
        json={"model": settings.chat_model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()
