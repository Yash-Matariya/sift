"""
Database setup: the SQLAlchemy engine, a session factory, and one-time schema
creation. Everything that touches Postgres goes through the session made here.
"""
from collections.abc import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker, DeclarativeBase

from app.config import settings


class Base(DeclarativeBase):
    """Base class all ORM models inherit from."""
    pass


# The engine is the connection pool to Postgres. One per process.
engine = create_engine(settings.database_url, pool_pre_ping=True)

# A factory that hands out sessions (one unit of work each).
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_session() -> Iterator[Session]:
    """FastAPI dependency: yields a session and always closes it."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Enable the pgvector extension and create the tables. Safe to run repeatedly."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    # import models so they're registered on Base before create_all
    from app import models  # noqa: F401
    Base.metadata.create_all(engine)
