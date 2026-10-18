"""
Shared test fixtures.

Most tests mock the LLM and don't touch a database. The `db_session` fixture is
for the few integration tests that need real Postgres with pgvector (vector
search can't run on SQLite). It skips those tests when no Postgres is reachable,
so the suite still runs on a machine with nothing started.

**The port matters.** The compose `db` publishes 5435, not 5432, because a
locally-installed Postgres commonly owns 5432. Defaulting this to 5432 meant the
connection reached the *native* server, which has no `sift_test` database, so
`_postgres_available()` returned False and the two ingestion tests skipped
silently on a machine where the project's own database was running fine. A test
that silently does not run is worse than no test: the idempotency guarantee it
exists to prove was never actually checked.

The test database is created on demand, so `docker compose up -d db` followed by
`pytest` is all it takes -- no manual createdb step to forget.
"""
import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# A separate database on the compose Postgres; override with TEST_DATABASE_URL.
TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://sift:sift@localhost:5435/sift_test",
)
# The same server, but the database that always exists -- you cannot issue
# CREATE DATABASE while connected to the database you are creating.
ADMIN_DB_URL = TEST_DB_URL.rsplit("/", 1)[0] + "/sift"
TEST_DB_NAME = TEST_DB_URL.rsplit("/", 1)[1]


def _ensure_test_database() -> bool:
    """Make sure the test database exists. False if the server is unreachable."""
    try:
        eng = create_engine(ADMIN_DB_URL, isolation_level="AUTOCOMMIT")
        with eng.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :n"),
                {"n": TEST_DB_NAME},
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{TEST_DB_NAME}"'))
        eng.dispose()
        return True
    except Exception:
        return False


@pytest.fixture
def db_session():
    if not _ensure_test_database():
        pytest.skip(
            f"no Postgres at {ADMIN_DB_URL} -- run `docker compose up -d db`, "
            "or set TEST_DATABASE_URL"
        )

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


@pytest.fixture(autouse=True)
def _one_model_call_at_a_time(monkeypatch):
    """Most tests check one behaviour with the model replaced. The extra model calls -- splitting
    a question into several searches, and the wider second search after a decline -- are off
    unless a test turns them on, so those tests never reach a real model."""
    from app.config import settings
    monkeypatch.setattr(settings, "multi_query", False)
    monkeypatch.setattr(settings, "widen_on_decline", False)
