"""
The provider seam in app/llm.py.

The rest of the app only calls `embed()` and `generate()`, so which provider is
behind them is a config choice. These tests cover that seam without a network:
the HTTP call is replaced, and what is asserted is that the right endpoint got
the right request shape and that the reply was unwrapped correctly.

The Ollama path is also exercised for real by eval/run.py, which needs a live
model. The OpenAI-compatible path is covered here only -- it has not been run
against a paid endpoint, and this file is the honest limit of that claim.

The dimension check is the test that matters most. Swapping the embedding model
is not a config change: the stored vectors came from one model, distances
between vectors from two different models are meaningless, and the pgvector
column has a fixed width. Failing loudly beats poisoning the store.
"""
import httpx
import pytest

from app import llm
from app.config import settings


class _Resp:
    """A stand-in for httpx.Response carrying just what llm.py reads."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture
def captured(monkeypatch):
    """Replace httpx.post and record what was sent."""
    calls = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return fake_post.reply

    fake_post.reply = _Resp({})
    monkeypatch.setattr(httpx, "post", fake_post)
    return calls, fake_post


def test_generate_uses_ollama_by_default(captured, monkeypatch):
    calls, post = captured
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    post.reply = _Resp({"response": "  hello  "})

    assert llm.generate("q") == "hello"          # stripped
    assert calls[0]["url"].endswith("/api/generate")
    assert calls[0]["json"]["stream"] is False
    assert calls[0]["json"]["options"]["temperature"] == settings.temperature


def test_generate_uses_openai_when_configured(captured, monkeypatch):
    calls, post = captured
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_base_url", "https://example.test/v1")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    post.reply = _Resp({"choices": [{"message": {"content": " hi "}}]})

    assert llm.generate("q") == "hi"
    assert calls[0]["url"] == "https://example.test/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert calls[0]["json"]["messages"] == [{"role": "user", "content": "q"}]
    assert calls[0]["json"]["temperature"] == settings.temperature


def test_temperature_is_zero_by_default():
    """Not a style preference. A grounded Q&A system should give the same answer
    to the same question, and the eval set cannot mean anything if the number
    moves run to run -- which it did, at Ollama's default of 0.8."""
    assert settings.temperature == 0.0


def test_embed_returns_the_vector_when_the_width_is_right(captured, monkeypatch):
    _, post = captured
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    post.reply = _Resp({"embedding": [0.0] * settings.embed_dim})

    assert len(llm.embed("text")) == settings.embed_dim


def test_embed_rejects_a_vector_of_the_wrong_width(captured, monkeypatch):
    """The failure this prevents is silent: a 1536-dim model against a 768-dim
    column stores nothing usable, and every later search returns nonsense."""
    _, post = captured
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    post.reply = _Resp({"embedding": [0.0] * 1536})

    with pytest.raises(llm.EmbeddingDimensionError) as err:
        llm.embed("text")
    assert "1536" in str(err.value) and str(settings.embed_dim) in str(err.value)


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "nope")
    with pytest.raises(ValueError, match="unknown llm_provider"):
        llm.generate("q")


def test_openai_embed_unwraps_the_vector(captured, monkeypatch):
    calls, fake_post = captured
    monkeypatch.setattr(settings, "llm_provider", "openai")
    fake_post.reply = _Resp({"data": [{"embedding": [0.5] * settings.embed_dim}]})
    assert llm.embed("hello") == [0.5] * settings.embed_dim
    assert calls[0]["url"].endswith("/embeddings")
    assert calls[0]["json"]["input"] == "hello"
