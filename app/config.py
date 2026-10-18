"""
Central configuration.

All settings come from environment variables (with sensible defaults for local
dev), read once into a single `settings` object the rest of the app imports.
This is the "config from the environment" rule — no secrets or hosts hardcoded
in the code.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # where the .env file is, and to ignore unknown keys
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- database ---
    database_url: str = "postgresql+psycopg://sift:sift@localhost:5435/sift"

    # --- which model provider app/llm.py talks to: "ollama" or "openai" ---
    # Local by default: free, no API key, and the documents never leave the
    # machine. "openai" is any OpenAI-compatible endpoint, for when answer
    # quality matters more than cost and privacy.
    llm_provider: str = "ollama"

    # --- Ollama (local LLM + embeddings; free, no API key) ---
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"     # produces 768-dim vectors
    embed_dim: int = 768                        # must match the model above
    chat_model: str = "llama3.1:8b"

    # How much the model is allowed to wander when picking the next token.
    # 0 for a grounded Q&A system: the answer should come from the retrieved
    # passages, not from creativity, and the same question should give the same
    # answer twice. Left at Ollama's default of 0.8, the eval scored 100% then
    # 80% then 100% on identical inputs -- a measurement that changes run to run
    # measures nothing.
    temperature: float = 0.0

    # --- OpenAI-compatible provider (only used when llm_provider="openai") ---
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    # Note the dimension: text-embedding-3-small is 1536, not 768, so switching
    # the embedding model also means altering the column and re-embedding.
    openai_embed_model: str = "text-embedding-3-small"

    # --- retrieval / RAG knobs ---
    chunk_size: int = 800          # characters per chunk
    chunk_overlap: int = 150       # characters shared between neighbouring chunks
    top_k: int = 4                 # how many chunks to retrieve per question
    # nomic-embed-text expects a task label in front of every text it embeds: search_query for a
    # question, search_document for a passage. Without them its similarities are noticeably worse.
    # Changing these means re-embedding what is stored: python -m app.reindex
    embed_query_prefix: str = "search_query: "
    embed_document_prefix: str = "search_document: "
    # Multi-query retrieval: ask the model to split a question into the separate things it asks
    # about, search for each, and merge the results ("company and my email?" is two searches).
    multi_query: bool = True
    # If the model says it doesn't know even though retrieval passed the gate, search once more,
    # wider (wide_k passages instead of top_k), before giving up.
    widen_on_decline: bool = True
    wide_k: int = 12
    min_score: float = 0.35        # below this similarity, we treat retrieval as "nothing relevant"

    # Retrieval mode: "vector" (semantic only) or "hybrid" (semantic + keyword, fused).
    retrieval_mode: str = "hybrid"
    rrf_k: int = 60                # Reciprocal Rank Fusion constant (standard default)

    # --- agentic loop (LangGraph) ---
    # If the best retrieved chunk is below this, the agent rewrites the query and
    # retries retrieval, up to max_retrieval_attempts, before answering or refusing.
    good_score: float = 0.55
    max_retrieval_attempts: int = 2

    # --- Celery (async ingestion) ---
    redis_url: str = "redis://localhost:6379/0"
    # Run tasks in-process instead of via a worker (used in tests; no broker needed).
    celery_eager: bool = False
    # Where uploaded files are stored so the worker can read them. In Docker this
    # is a shared volume; in production it would be object storage (e.g. S3).
    uploads_dir: str = "./uploads"


@lru_cache
def get_settings() -> Settings:
    # cached so the .env is read once, not on every import
    return Settings()


settings = get_settings()
