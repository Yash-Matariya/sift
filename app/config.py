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


@lru_cache
def get_settings() -> Settings:
    # cached so the .env is read once, not on every import
    return Settings()


settings = get_settings()
