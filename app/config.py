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
    database_url: str = "postgresql+psycopg://sift:sift@localhost:5432/sift"

    # --- Ollama (local LLM + embeddings; free, no API key) ---
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"     # produces 768-dim vectors
    embed_dim: int = 768                        # must match the model above
    chat_model: str = "llama3.2"

    # --- retrieval / RAG knobs ---
    chunk_size: int = 800          # characters per chunk
    chunk_overlap: int = 150       # characters shared between neighbouring chunks
    top_k: int = 4                 # how many chunks to retrieve per question
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
