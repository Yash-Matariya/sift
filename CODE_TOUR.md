# Code Tour — Sift

Read this with the code open. It walks every file in the order the design makes sense, and calls out the decisions an interviewer will probe. By the end you can walk anyone through the whole system and defend each choice. Everything here is in the code — nothing is aspirational.

Read in this order: `config.py` → `models.py` → `db.py` → `llm.py` → `storage.py` → `ingest.py` → `tasks.py` → `celery_app.py` → `retrieval.py` → `rag.py` → `agent.py` → `main.py` → the tests.

---

## 1. `config.py` — everything tunable, in one place

All settings come from the environment via `pydantic-settings`: model names, `chunk_size`/`chunk_overlap`, `top_k`, `min_score` (the relevance threshold), `retrieval_mode` (`vector` or `hybrid`), `good_score` and `max_retrieval_attempts` (the agent loop), and the Redis URL. Nothing is hard-coded, so the same image runs anywhere by changing env vars.

**Probe — "how would you tune retrieval?"** → "Chunk size and overlap first, then `min_score` and the retrieval mode — all in config."

## 2. `models.py` — the data, two kinds

- **ORM tables:** `Document` (filename, `status`, `source_path`, and a `chunks` relationship with `cascade="all, delete-orphan"`) and `Chunk` (`document_id` foreign key, `chunk_index`, `text`, and the `embedding` as a pgvector `Vector(768)` column).
- **Pydantic schemas:** `DocumentOut`, `ChatRequest`, `ChatResponse`, `Citation` — the API contract, kept separate from the tables.
- **Probe — "why store the vector in the DB?"** → pgvector lets me search by cosine distance in SQL, so vectors and relational data live in one Postgres.

## 3. `db.py` — engine, session, startup

The engine is the connection pool (`pool_pre_ping=True`). `get_session` is the FastAPI dependency that yields a session and always closes it (short-lived session = returns the pooled connection quickly). `init_db` enables the pgvector extension and creates the tables at startup.

## 4. `llm.py` — the one place that calls the model

`embed(text)` and `generate(prompt)` call Ollama's HTTP API. Everything else depends on these two functions, so tests mock them and you'd swap to a hosted API by changing only this file.

## 5. `storage.py` — the uploaded file

`save_upload` writes the raw bytes to the uploads dir and returns the path; `read_upload` reads them back. **Why it exists:** the worker is a *separate process*, so it can't see the request's in-memory bytes — the file is persisted so the worker can read it. In Docker the api and worker share an `uploads` volume; in production this would be S3.

## 6. `ingest.py` — read → chunk → embed → store (idempotently)

- `_read_text` handles PDF (pypdf) and text. `_chunk` splits into overlapping pieces so an idea crossing a boundary survives.
- **`ingest_into(session, document_id)`** is the pipeline and the file to know: it sets the document `processing`, reads its file, **deletes any existing chunks** (idempotency — a retry can't duplicate), embeds each chunk, stores them, and commits chunks + `ready` status **together in one transaction**. On failure it rolls back, marks the document `failed`, and re-raises.
- **Probe — "what if ingestion runs twice?"** → "It's idempotent — I clear existing chunks first, so a retry produces the same clean result, no duplicates."

## 7. `tasks.py` — the background job

`ingest_document_task` is the Celery task: one session per run, calls `ingest_into`, and on failure `self.retry` with backoff (up to 3 times). Because `ingest_into` is idempotent, retrying is safe. **Probe — "why retry?"** → "Transient failures, like the embedding model being briefly down, recover on their own; and it's safe because the task is idempotent."

## 8. `celery_app.py` — the Celery app

Points Celery at Redis (broker + result backend). `task_always_eager` (from config) runs tasks in-process for tests — no Redis needed.

## 9. `retrieval.py` — hybrid search + the relevance gate

The most decision-dense file after the agent.
- **`rrf_fuse`** — pure Reciprocal Rank Fusion: merges ranked lists so an item found by *both* vector and keyword search rises to the top. Unit-tested.
- **`_vector_search`** — embed the question, order by pgvector cosine distance, return `(chunk, similarity)`.
- **`_keyword_search`** — Postgres full-text (`to_tsvector` / `plainto_tsquery`, ranked by `ts_rank`).
- **`retrieve`** — in `vector` mode, keep matches above `min_score`. In `hybrid` mode, fuse vector + keyword by RRF, then apply the **relevance gate**: a chunk counts if it's a strong semantic match OR a literal keyword match. Nothing passing → returns `[]` → the agent refuses.
- **Probe — "why hybrid?"** → "Semantic search misses exact terms, keyword misses meaning; RRF fuses them. And the threshold gate is what lets the system refuse instead of answering from junk."

## 10. `rag.py` — the grounded answer, composed with LangChain (LCEL)

The chain is `ANSWER_CHAIN = PROMPT | RunnableLambda(_call_model) | StrOutputParser()`. A `ChatPromptTemplate` keeps the guardrail in its **system** message and takes `{context}` and `{question}`; the model step wraps our own Ollama client, so there's still exactly one place that talks to the model; `StrOutputParser` returns a clean string. Every piece is a **Runnable**, which is why they pipe together — that's LCEL.

`generate_answer(hits, question)` numbers the chunks, invokes that chain, and returns the answer with **citations** (source filename + chunk index, from your own code — so they can't be hallucinated). `refusal()` is the "I don't know" response. **Probe — "how did you use LangChain?"** → "The answer path is an LCEL chain — a ChatPromptTemplate fills the retrieved context and question, a Runnable calls the model, and a StrOutputParser returns a clean string, piped together as `prompt | model | parser`. Everything's a Runnable so it composes, and swapping the model means swapping one step."

**Probe — "how are citations trustworthy?"** → "They come from my code knowing which chunks it sent, not from the model."

## 11. `agent.py` — the LangGraph agentic loop (the headline)

The file that makes it *agentic*, and the one to know cold.
- **State** (`AgentState`): question, current query, attempts, hits, response.
- **Nodes:** `retrieve` (search + count the attempt), `rewrite` (LLM rephrases the query for better retrieval), `generate` (grounded answer), `refuse`.
- **`grade`** is the conditional edge — the decision: if the best hit is confident (`>= good_score`) → generate; else if attempts remain → rewrite and loop; else if there's anything → answer best-effort; else → refuse.
- **The graph:** `START → retrieve → [grade] → generate / rewrite→retrieve (loop) / refuse → END`. The loop is **bounded** by `max_retrieval_attempts`, so it can never spin forever.
- **Probe — "what makes it agentic, and how is it safe?"** → "It decides — retrieve, grade, and re-retrieve with a rewritten query if the results are weak, bounded so it can't loop forever, then answer or refuse. It's a bounded reason-act-observe loop, a state machine over a shared state."

## 12. `main.py` — the thin HTTP layer

- `POST /documents` — save the file, create a `pending` document, queue `ingest_document_task.delay(id)`, return **202** immediately.
- `GET /documents/{id}` — poll the status (`pending`/`processing`/`ready`/`failed`).
- `POST /chat` — validate, then call `agent.answer_question`.
- Each endpoint gets a per-request session via `Depends(get_session)`. Logic lives in the modules; the endpoints just wire them to HTTP.
- **Probe — "why 202 on upload?"** → "Ingestion is queued to a worker, so the upload returns immediately with `pending`; the client polls for `ready`."

## 13. The tests — what's verified

`tests/` covers the logic without services (RRF, chunking, the full agent loop including its bound, the hybrid gate, the Celery task in eager mode) and, guarded to skip when there's no Postgres, ingestion + idempotency against real pgvector. Run: `.venv/bin/python -m pytest`.

---

## The 60-second walkthrough

> "Sift is an agentic RAG assistant. You upload a document; the endpoint saves it, marks it pending, and queues a Celery task that chunks, embeds with a local model, and stores it in Postgres with pgvector — idempotently, so a retry can't duplicate. Asking a question runs a LangGraph agent: it retrieves with hybrid search — vector plus keyword fused by RRF — grades whether the results are strong enough, rewrites the query and retries if they're weak (bounded so it can't loop forever), then answers using only the retrieved chunks with citations, or refuses if nothing's relevant. Everything runs locally via Ollama and comes up with one docker compose command. Each file has one job, so I can point to exactly where retrieval, generation, refusal, or the agent loop lives."
