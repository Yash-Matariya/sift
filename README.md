# Sift — an agentic RAG document assistant

> This is the **AI Document Assistant** project listed on my resume.

Upload your documents, ask questions in plain language, and get answers that **cite the exact passages they came from** — and that **say "I don't know" instead of making something up** when the documents don't contain the answer.

---

## What it is, in one plain paragraph

You give Sift a set of documents (PDF, text, markdown). It splits them into small overlapping pieces, turns each piece into a vector (an "embedding") with a local model, and stores them in Postgres with pgvector. When you ask a question, an agent retrieves the most relevant pieces, grades whether they're actually good enough, retries with a rewritten query if they're weak, and — only if it has relevant context — asks a local language model to write an answer using *those pieces*, with citations. If nothing relevant is found, it refuses instead of guessing. Ingesting a document runs in a background worker so uploads return immediately.

That is **Retrieval-Augmented Generation (RAG)** — and the exact thing this job's LangChain / LangGraph / RAG lines are about.

## Try it

```bash
docker compose up -d db redis
ollama pull nomic-embed-text && ollama pull llama3.2
.venv/bin/celery -A app.celery_app.celery worker --loglevel=info &
.venv/bin/uvicorn app.main:app
# open http://localhost:8000
```

Upload a document, wait for it to say **ready**, then ask something it covers —
and something it does not. The second one is the interesting case.

The UI is a single static page served by the API itself. No build step and no
npm: this is a Python project, and a front end you cannot open without a
toolchain would be a worse demo, not a better one.

## What it does (the features that are built)

- **Agentic RAG loop (LangGraph).** Not "search, then answer." A small state machine: *retrieve → grade whether the results are strong enough → if weak, rewrite the query and retrieve again (bounded) → answer with citations, or refuse.* That grade-and-maybe-retry-or-refuse loop is what makes it trustworthy.
- **Hybrid retrieval.** Fuses **vector search (pgvector, cosine)** with **keyword search (Postgres full-text)** using **Reciprocal Rank Fusion**, because semantic search misses exact terms (codes, names) and keyword search misses meaning. A similarity threshold gates relevance and drives refusal.
- **Citations + refusal, on two levels.** Every answer lists the source passages it used. It declines when nothing relevant is retrieved *and* when the passages were retrieved but still do not answer the question — in that second case the model itself says so, and the response is reported as a refusal with no citations attached. Sources under "I don't know" would imply the answer came from them.
- **A visible agent trace.** Every response carries what the agent actually did: how many times it retrieved, whether it gave up on your wording and rewrote the query, the best similarity it found, and how many passages passed the gate. That is what makes "agentic" observable instead of a claim.
- **Background ingestion (Celery).** Uploads return immediately (202 `pending`); a Celery worker chunks and embeds in the background and flips the document to `ready`. The task is **idempotent** (it clears existing chunks first) and **retries with backoff**, so a transient failure recovers without duplicating data.
- **Local + reproducible.** LLM and embeddings run locally via **Ollama** (free, no API key, data stays on the machine); the whole stack — Postgres/pgvector, Redis, Ollama, the API, and the worker — comes up with one `docker compose up`.

## Architecture

```
        upload a document                          ask a question
              │                                          │
              ▼                                          ▼
   POST /documents ──► Celery task ──► ready   POST /chat ──► LangGraph agent
     (save file,       (background)              (FastAPI)         │
      queue, 202)          │                                      ├─► retrieve  (hybrid: pgvector + full-text, RRF)
                    chunk → embed → store                         ├─► grade     (strong enough? else rewrite & retry, bounded)
                           │                                      ├─► generate  (LLM answers, with citations)
                           ▼                                      └─► refuse    (nothing relevant found)
                  Postgres + pgvector ◄───────────────────────────────┘
                  (documents, chunks, embeddings)     Redis = the Celery broker

   LLM + embeddings run locally via Ollama.
```

## Tech stack → mapped to the job description

| Job description asks for | Where Sift uses it |
|---|---|
| Python | the whole thing |
| FastAPI | the REST API (`app/main.py`) |
| PyTest | the test suite (`tests/`) |
| Celery | background document ingestion (`app/tasks.py`) |
| LangChain | the answer chain — prompt template, model, output parser composed with LCEL (`app/rag.py`) |
| LangGraph | the retrieve → grade → answer/refuse agent (`app/agent.py`) |
| RAG, embeddings, vector search | the core of the product (pgvector) |
| PostgreSQL | the database — documents, chunks, embeddings |
| REST APIs, async, concurrency | API design + async endpoints + Celery workers |
| Docker | docker-compose runs the whole stack |
| LLM prompting, structured output | the answer, grading, and query-rewrite prompts |

## Which code is where (so you can answer "walk me through it")

```
sift/
  docker-compose.yml     # postgres+pgvector, redis, ollama, api, worker — one command
  Dockerfile             # the image the api and worker both run from
  requirements.txt
  app/
    config.py            # all settings via pydantic-settings (models, chunking, thresholds)
    db.py                # SQLAlchemy engine (connection pool), session factory, init_db
    models.py            # ORM tables (Document, Chunk) + Pydantic API schemas
    storage.py           # save/read the raw uploaded file (so the worker can read it)
    llm.py               # the one place that calls Ollama: embed() and generate()
    ingest.py            # read -> chunk -> embed -> store; idempotent ingest_into()
    tasks.py             # the Celery task that runs ingestion in the background
    celery_app.py        # the Celery app (Redis broker)
    retrieval.py         # vector + keyword search, RRF fusion, the relevance gate
    rag.py               # LangChain (LCEL) answer chain: prompt | model | parser, + citations/refusal
    agent.py             # the LangGraph state machine (retrieve/grade/rewrite/generate/refuse)
    main.py              # FastAPI app and endpoints
  tests/
    test_rrf.py          # Reciprocal Rank Fusion (pure)
    test_chunk.py        # the chunker (pure)
    test_agent.py        # the LangGraph loop: confident / rewrite / refuse / bound
    test_retrieval.py    # the hybrid relevance gate
    test_tasks.py        # the Celery task runs ingestion (eager mode)
    test_ingest_db.py    # ingestion + idempotency against real Postgres (skips if none)
```

When they ask *"where does the answer get generated?"* → `app/rag.py`, `generate_answer`.
*"How do you stop it hallucinating?"* → the relevance gate in `app/retrieval.py` + the grade/refuse path in `app/agent.py` + the "answer only from context" prompt in `app/rag.py`.
*"Where's the vector search?"* → `app/retrieval.py`.

See **`CODE_TOUR.md`** for a guided, decision-by-decision read.

## Design decisions you can defend

- **Chunk size vs. accuracy.** Chunks too big → retrieval is vague; too small → context is lost. Sift uses ~800 chars with 150 overlap (in `config.py`), tunable.
- **Pure vector search misses exact terms.** A specific name or code may not embed distinctively → **hybrid** (vector + keyword, RRF) catches both meaning and literal matches.
- **The model answering from nothing.** Given weak retrieval it would invent → a **similarity threshold** gates relevance and, with the agent's grade/refuse path, it only answers when grounded.
- **Re-uploading duplicating data.** Re-ingesting a document could double its chunks → ingestion is **idempotent** (clears existing chunks first), so a retry produces the same clean result.
- **Big uploads freezing the API.** Embedding a large file would block requests → ingestion runs in a **Celery worker**; the upload returns immediately and the client polls status.

## How to run

```bash
docker compose up -d --build                              # postgres+pgvector, redis, ollama, api, worker
docker compose exec ollama ollama pull nomic-embed-text   # the embedding model (once)
docker compose exec ollama ollama pull llama3.2           # the chat model (once)
# open http://localhost:8000/docs for the interactive API
```

Then: POST a file to `/documents` (returns `202` with a `pending` status), poll
`GET /documents/{id}` until it's `ready`, then POST a question to `/chat` and get
an answer with citations (or a refusal).

## Tests

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```

The logic tests (RRF, chunking, the agent loop, the hybrid gate, the Celery task)
mock the model and need no services. The database-integration tests use real
Postgres with pgvector and **skip automatically** when it isn't available (set
`TEST_DATABASE_URL` to run them, e.g. against the compose `db`).

## Possible next steps (not built)

Honest roadmap, so the scope is clear: a reranking pass over retrieved chunks; a
`recall@k` / faithfulness evaluation harness on a labelled question set; streaming
answers token-by-token; per-user auth and document isolation; and Alembic
migrations in place of `create_all` for schema changes.

## The companion project

**Insight Extractor** — a small second AI service: send it free text, get back
clean structured JSON (sentiment, summary, topics, rating) from an LLM constrained
to a Pydantic schema, with validation and retries. Adds structured-output breadth
alongside Sift's retrieval depth.
