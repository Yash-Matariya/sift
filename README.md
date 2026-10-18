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
ollama pull nomic-embed-text && ollama pull llama3.1:8b
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
    test_chunk.py        # the chunker: sizes, overlap, never cuts a word (pure)
    test_agent.py        # the LangGraph loop: confident / rewrite / refuse / bound
    test_model_steps.py  # the query rewrite, and its guard against refusals
    test_decline.py      # the model saying "I don't know" is reported as a refusal
    test_retrieval.py    # the hybrid relevance gate
    test_retrieval_db.py # vector, keyword and hybrid search on real Postgres
    test_api.py          # every endpoint, upload -> ready -> ask, list order
    test_storage.py      # uploaded filenames cannot escape the uploads folder
    test_tasks.py        # the Celery task: runs, retries and recovers, gives up
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
docker compose exec ollama ollama pull llama3.1:8b        # the chat model (once)
# open http://localhost:8000/docs for the interactive API
```

Then: POST a file to `/documents` (returns `202` with a `pending` status), poll
`GET /documents/{id}` until it's `ready`, then POST a question to `/chat` and get
an answer with citations (or a refusal). A document that failed, or that sat `pending`
because the worker or Redis was down, can be queued again with
`POST /documents/{id}/retry`; `DELETE /documents/{id}` removes a document, its chunks
and its file. If Redis or Postgres is down, the API answers `503` with a message that
names it, rather than a bare `500`.

Retrieval splits a vague question into separate searches (multi-query) and merges
them, keeps one copy of passages that are word-for-word identical (so a file
uploaded twice can't crowd others out), and looks wider once before saying it
doesn't know. `eval/hard.py` measures exactly that situation, duplicates plus a
second document plus vague questions: 3 of 6 answered before these changes, 5 of 6
after, with the main set unchanged (recall@k 100%, refusal accuracy 80%). After
changing the embedding model or its prefixes, re-embed what is stored with
`.venv/bin/python -m app.reindex`.

## Tests

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```

The logic tests (RRF, chunking, the agent loop, the hybrid gate, the provider
seam, the Celery task) mock the model and need no services. The
database-integration tests use real Postgres with pgvector, because idempotency
is enforced by the database and an in-memory stand-in would not prove it. They
create their own `sift_test` database on the compose Postgres, so:

```bash
docker compose up -d db && .venv/bin/python -m pytest    # 83 passed
```

They still skip cleanly when no Postgres is reachable, so the suite runs
anywhere — but **that skip used to hide them**. The default `TEST_DATABASE_URL`
pointed at port 5432 while compose publishes 5435, so on a machine with a
locally-installed Postgres the connection reached the wrong server, found no
`sift_test` database, and skipped. Two tests reported as "skipped" in a run that
otherwise passed, and the idempotency guarantee they exist to prove was never
actually checked. A test that silently does not run is worse than no test.

## Measuring it

A RAG system nobody measures is one nobody can improve, because every change is
a guess. There is a labelled set of 20 questions over an 11-chunk handbook —
15 the document answers and 5 it does not — and two numbers:

```bash
docker compose up -d db
.venv/bin/python -m eval.run
```

```
  recall@k           100%   (15 answerable questions)
  refusal accuracy   100%   (5 unanswerable questions)
```

**recall@k** is of the answerable questions, how often retrieval surfaced a
passage containing the answer. If this is low the model never sees the answer
and no prompt change will help. **Refusal accuracy** is of the unanswerable
questions, how often it correctly declined.

They are reported together on purpose: refusing everything scores 100% on the
second, so the report also lists every *false* refusal.

**The numbers can fail, which is the point.** An earlier version of this set
scored 100% against a two-chunk document while `top_k` was 4 — every chunk was
retrieved for every question, so recall could not fail and measured nothing.
The corpus is now large enough that retrieval has to choose, and degrading it
on purpose moves the numbers:

| | recall@k |
|---|---|
| as shipped | 100% |
| `TOP_K=1` | 87% |
| `MIN_SCORE=0.9` (gate too strict) | **13%**, and the run fails |

That last row is the tradeoff made visible: you can push refusal accuracy up by
raising the threshold, and it costs you the answers.

The metric functions in `app/evaluation.py` are pure, so they are unit-tested
without a database, an embedding model or an LLM.

### Three bugs that only showed up in use

All 69 tests passed while these were live. They came from clicking through the
page and reading the eval's answers, not from the test suite.

- **The query rewrite searched for a refusal.** Asked to rewrite *"What is the
  CEO's favourite colour?"* as a search query, `llama3.1:8b` replied *"I can't
  answer that. I'm happy to help with other questions though!"* — and the agent
  searched for that sentence. The old prompt did this on 3 of 25 questions. The
  prompt now says plainly that the model is not answering anything (0 of 25),
  and `rewrite_query` falls back to the original question if the reply is
  empty, long, or opens like a refusal.
- **Chunks began mid-word.** The chunker cut at exact character counts, so a
  source passage began *"eclined the card"*. Both edges of a chunk now move to
  a space.
- **A long "I don't know" counted as an answer.** Refusals were only recognised
  in replies of 200 characters or less. After the chunker change the model
  explained its refusal at more length — *"I don't know. The provided context
  does not mention the CEO..."* — and the reply was reported as an answer with
  citations. The eval caught it: refusal accuracy fell to 80%. A reply that
  *opens* with "I don't know" is now a refusal at any length. The eval report
  also lists these cases by name now; before, that run still ended "no misses".

### Choosing the model

"Which model should this use?" is guesswork until the eval set turns it into a
measurement. `eval/compare_models.py` runs the whole case set against several
chat models and prints the comparison:

```bash
.venv/bin/python -m eval.compare_models llama3.1:8b qwen2.5:7b gemma3:4b
```

| model | size | recall@k | refusal accuracy | 20 cases |
|---|---|---|---|---|
| **llama3.1:8b** (default) | 4.9 GB | 100% | **100%** | 39s |
| qwen2.5:7b | 4.7 GB | 100% | 100% | 66s |
| gemma3:4b | 3.3 GB | 100% | 80% | 50s |

Measured on a MacBook M4 Pro (24 GB) at `TEMPERATURE=0`. Treat the seconds
column as indicative — it is wall-clock on a laptop with other things running.

**recall@k does not move, and that is the most informative part of the table.**
Retrieval is embeddings plus full-text search; it never calls the chat model. So
a model change cannot affect it, and the half of the system that decides whether
the answer is even *available* is the half that is not the model. Refusal
accuracy does move, because deciding that retrieved passages do not answer the
question is the model's job.

`llama3.1:8b` ties `qwen2.5:7b` on both columns and is the faster of the two,
but it would still be the default if it had only tied the smaller model: the
table does not score how well an answer *reads*, and on a real question
`gemma3:4b` produced *"**Both** must have twelve months of continuous service"* —
a pronoun with no antecedent, the kind of small incoherence a 4B model produces.
With only 5 refusal cases the refusal column also moves in 20-point steps, so an
80% is one case, not a verdict, and it has swapped between models from one
version of the code to the next. The numbers narrow the field; reading the
answers picks the winner.

`gemma3:4b` is the one to use on a machine with less memory: 3.3 GB rather than
4.9 GB, the same recall, and one refusal case short.

**This table is only meaningful because generation runs at temperature 0.** It
did not, at first: Ollama's default is 0.8, and the same eval scored 100%, then
80%, then 100% on identical inputs. A grounded Q&A system should give the same
answer to the same question anyway — the answer is supposed to come from the
retrieved passages, not from sampling — so `TEMPERATURE=0` is both the correct
product behaviour and the thing that makes the measurement mean anything. Three
consecutive runs now give the same numbers.

### Local by default, hosted by configuration

Everything goes through `app/llm.py`, so the provider is one environment
variable (`LLM_PROVIDER=ollama|openai`) rather than a change spread across the
codebase.

Local is the default deliberately: it is free, it needs no API key, anyone can
clone this and run it, and — for a *document* assistant — the documents never
leave the machine, which is the objection a real user would raise first. A
frontier model would write better prose and reason better across several
retrieved passages; that is the trade, and it is one config line.

**Generation is swappable; embedding is not.** A generated answer is discarded
after the request, so changing that model changes only quality. An embedding is
*stored*: every vector in the database came from one model, distances between
vectors from different models are meaningless, and the pgvector column has a
fixed width. Switching it is a data migration, not a config change — so
`embed()` checks the width it got back and raises rather than silently poisoning
the store.

## Possible next steps (not built)

Honest roadmap, so the scope is clear: a reranking pass over retrieved chunks;
faithfulness scoring (does the answer only claim what the context supports);
streaming answers token-by-token; per-user auth and document isolation; and
Alembic migrations in place of `create_all` for schema changes.

## The companion project

**Insight Extractor** — a small second AI service: send it free text, get back
clean structured JSON (sentiment, summary, topics, rating) from an LLM constrained
to a Pydantic schema, with validation and retries. Adds structured-output breadth
alongside Sift's retrieval depth.
