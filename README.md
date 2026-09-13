# Sift — an agentic RAG document assistant

Upload your documents, ask questions in plain language, and get answers that **cite the exact passages they came from** — and that **say "I don't know" instead of making something up** when the documents don't contain the answer.

> This is the flagship AI project. It is being built from scratch this week, piece by piece, with every part explained. The whole point is that you understand every file and can walk an interviewer through it.

---

## What it is, in one plain paragraph

You give Sift a set of documents (PDFs, text, markdown). It splits them into small pieces, turns each piece into a vector (an "embedding"), and stores them in Postgres. When you ask a question, it finds the most relevant pieces, checks whether they actually answer the question, and — only if they do — asks a language model to write an answer using *those pieces*, with citations. If nothing relevant is found, it refuses instead of guessing. Heavy work (embedding a big upload) runs in the background so the API stays fast.

That is **Retrieval-Augmented Generation (RAG)** — the single most in-demand GenAI skill, and the exact thing this job's LangChain / LangGraph / RAG lines are about.

## Why it's impressive (and each part is explainable, not magic)

- **Agentic, not naive RAG.** Most tutorials do "search, then answer." Sift uses a **LangGraph agent** — a small state machine that *retrieves → grades whether the results are relevant → answers with citations, or refuses*. That "grade and maybe refuse" loop is what makes it trustworthy, and it's a current, senior-looking pattern.
- **Hybrid retrieval.** It fuses **vector search (pgvector)** with **keyword search (Postgres full-text)**, because each catches what the other misses. We measure this with a **recall@k** script, so you can say "the hybrid is better, and here are the numbers" — not "I assumed it was better."
- **Citations + refusal.** Every answer points at its sources; when unsure, it declines. This is the honesty an interviewer loves.
- **Real backend engineering.** FastAPI, typed with Pydantic, JWT auth, **Celery** background ingestion, PyTest, all in **Docker** with one command. It's a *system*, not a script.

## Architecture

```
        upload a document                          ask a question
              │                                          │
              ▼                                          ▼
   POST /documents  ─────►  Celery task ───►   POST /chat  ──►  LangGraph agent
        (FastAPI)          (background)          (FastAPI)          │
                                │                                   ├─► retrieve   (hybrid: pgvector + full-text)
                        chunk → embed → store                       ├─► grade      (are these actually relevant?)
                                │                                   ├─► answer     (LLM writes it, with citations)
                                ▼                                   └─► or refuse  (nothing relevant found)
                      Postgres + pgvector  ◄───────────────────────────┘
                      (documents, chunks, embeddings)

   LLM + embeddings run locally via Ollama (free, no API key) — same setup pattern as before.
```

## Tech stack → mapped to the job description

| Job description asks for | Where Sift uses it |
|---|---|
| Python | the whole thing |
| FastAPI | the REST API |
| PyTest | the test suite |
| Celery | background document ingestion |
| LangChain | models, prompts, embeddings, vector store glue |
| LangGraph | the retrieve → grade → answer/refuse agent |
| RAG, embeddings, vector search | the core of the product (pgvector) |
| PostgreSQL | the database (documents, chunks, users) |
| REST APIs, OOP, concurrency | API design, service classes, async + Celery |
| Docker | docker-compose runs the whole stack |
| LLM prompting, structured output | the answer + grading prompts |

One project, almost the entire JD.

## Which code is where (so you can answer "walk me through it")

```
sift/
  docker-compose.yml         # postgres+pgvector, redis, ollama, the api — one command to run it all
  requirements.txt
  app/
    main.py                  # creates the FastAPI app and wires everything together
    config.py                # settings (DB url, model names) via pydantic-settings
    api/
      routes_auth.py         # register / login, returns a JWT
      routes_docs.py         # upload / list / delete documents
      routes_chat.py         # ask a question — streams the answer back
    core/
      security.py            # JWT create/verify, password hashing
      db.py                  # SQLAlchemy engine + session
    models/
      db_models.py           # tables: users, documents, chunks (chunk holds text + embedding)
      schemas.py             # Pydantic request/response models
    ingest/
      chunker.py             # split a document into overlapping chunks
      embedder.py            # turn text into a vector (Ollama)
      tasks.py               # the Celery task: chunk -> embed -> store
    retrieval/
      hybrid.py              # run vector search + full-text search, fuse the results
    agent/
      graph.py               # the LangGraph state machine (the node wiring)
      nodes.py               # retrieve(), grade(), answer(), refuse()
      prompts.py             # the prompt templates
    eval/
      recall.py              # measure recall@k on a question set
  tests/
    test_api.py              # endpoints work, auth required
    test_retrieval.py        # hybrid finds the right chunk
    test_agent_refuses.py    # asks something not in the docs -> it refuses
```

When they say *"where does the answer get generated?"* → `app/agent/nodes.py`, the `answer()` node.
*"How do you stop it hallucinating?"* → the `grade()` node in the same file, plus the refuse path in `graph.py`.
*"Where's the vector search?"* → `app/retrieval/hybrid.py`. You'll know every one of these cold, because you build them.

## The real challenges we'll hit — and solve (your honest interview stories)

These are genuine problems this design runs into. We solve them during the build, and then they're *yours* to tell:

1. **Chunk size vs. accuracy.** Chunks too big → retrieval is vague; too small → context is lost. We pick a size + overlap and can explain the tradeoff.
2. **Pure vector search misses exact terms.** A question with a specific name/number retrieved the wrong thing → we added **hybrid** (vector + keyword) and measured the improvement with recall@k.
3. **The model answered from nothing.** Given junk retrieval it still made something up → we added the **grade node** and a **refusal path**, so it only answers when grounded.
4. **Re-uploading duplicated data.** Ingesting the same doc twice doubled the chunks → **idempotent upsert** keyed on `(document_id, chunk_index)`.
5. **Big uploads froze the API.** Embedding a large file blocked requests → moved ingestion to **Celery** so the API stays responsive.
6. **Users could see each other's documents.** → every query is scoped by `user_id` (structural isolation).

Each one is a two-minute story that shows real judgment.

## The other two projects on the resume

- **Insight Extractor** — a *small* second AI service (one main endpoint): send it free text, get back clean structured JSON (sentiment, category, entities) using an LLM constrained to a Pydantic schema. Reuses Sift's stack, so it's quick to build and easy to know fully. Adds breadth (structured output) to the resume.
- **Smart Attendance** — your real diploma project (SSIP 2023, 2nd prize). Stays as-is, short.

## How to run

```bash
docker compose up -d                                   # postgres+pgvector, ollama, the api
docker compose exec ollama ollama pull nomic-embed-text  # the embedding model (once)
docker compose exec ollama ollama pull llama3.2          # the chat model (once)
# open http://localhost:8000/docs  for the interactive API
```

Then, from the docs page (or curl): POST a file to `/documents`, then POST a
question to `/chat` and get an answer with citations (or a refusal).

## Build plan (7 days, alongside the study articles)

| Day | Build | Study alongside |
|---|---|---|
| 1 | Python essentials; project skeleton + FastAPI health + Docker up | Python basics articles |
| 2 | Auth (JWT), DB models, document upload endpoint | FastAPI + Pydantic articles |
| 3 | Chunking + embeddings (Ollama) + store; Celery ingestion | Embeddings + Celery articles |
| 4 | Hybrid retrieval (pgvector + full-text) + recall@k | RAG + pgvector articles |
| 5 | The LangGraph agent: retrieve → grade → answer/refuse + citations | LangChain + LangGraph articles |
| 6 | Streaming, tests (PyTest), Insight Extractor (small) | testing + LLM articles |
| 7 | Polish, README, finalize resume with the real challenges; mock Q&A | interview articles |

## Status

`[MVP]` — the core RAG loop is built and runnable: upload -> chunk -> embed (Ollama)
-> store (pgvector) -> retrieve -> answer with citations, or refuse when nothing
relevant is found. FastAPI + SQLAlchemy + Docker.

Next, layered on top: hybrid retrieval (vector + full-text), a LangGraph agent
(retrieve -> grade -> answer/refuse), background ingestion (Celery), streaming,
and tests.
