"""
Shared test fixtures.

Most tests mock the LLM and don't touch a database. The `db_session` fixture is
for the few integration tests that need real Postgres with pgvector (vector
search can't run on SQLite). It connects to a test database if one is reachable
and otherwise skips those tests, so the suite runs anywhere.
"""
import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# A separate test database; override with TEST_DATABASE_URL.
TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://sift:sift@localhost:5432/sift_test",
)


def _postgres_available() -> bool:
    try:
        eng = create_engine(TEST_DB_URL)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        eng.dispose()
        return True
    except Exception:
        return False


@pytest.fixture
def db_session():
    if not _postgres_available():
        pytest.skip("no test Postgres available (set TEST_DATABASE_URL)")

    from app.db import Base
    from app import models  # noqa: F401  register models on Base

    engine = create_engine(TEST_DB_URL)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()
