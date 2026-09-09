# go-rag-api

A retrieval-augmented generation (RAG) API in Go: POST a document and it is
chunked, embedded, and stored; POST a question and it is answered from the most
similar chunks. Stack: [chi](https://github.com/go-chi/chi) for HTTP, Ollama for
embeddings (`mxbai-embed-large`) and generation (any local chat model),
PostgreSQL + pgvector for vector search. The architecture is hexagonal: a
stdlib-only core surrounded by swappable adapters.

## Why hexagonal

The dependency rule: everything points inward. `internal/domain` and
`internal/application` hold the entities, errors, use cases, chunking, and
prompt building, and they import **zero third-party packages** — only the Go
standard library. That is not convention, it is enforced:
`internal/application/architecture_test.go` runs `go list -deps` over the core
packages and fails the build the moment any external dependency creeps in.

The payoff is testability. Every application test runs against hand-written
port fakes — no database, no Ollama, no network — and the inbound HTTP adapter
is tested the same way, against two small interfaces instead of concrete use
cases.

## Why the ports live in domain/ports

Interfaces belong to the consumer that needs them. `internal/domain/ports`
declares exactly what the core needs from the outside world — `Embedder`,
`Generator`, `VectorStore`, `DocumentLoader` — and the outbound adapters
implement those shapes, proving it with compile-time assertions
(`var _ ports.Embedder = (*Client)(nil)` and friends). The core never imports
an adapter, so dependencies point inward and adapters can be replaced without
touching the core.

The inbound side mirrors that philosophy: `internal/adapters/inbound/http`
defines its own narrow consumer interfaces for the use cases (an indexer and an
answerer, just the methods the handlers actually call) instead of importing
`*application.IndexDocument`. The only wiring happens in `cmd/api`, where
passing the concrete use cases to the handler constructor is itself the
compile-time check that both sides still fit.

## Why Ollama runs natively but Postgres runs in Docker

Ollama needs direct access to the GPU (or Apple Silicon) and keeps
multi-gigabyte model files on the host; containerizing it means GPU passthrough
friction and duplicated model downloads, so it runs natively and the API talks
to `localhost:11434`. Postgres with pgvector, on the other hand, is disposable
stateless-ish infrastructure: one `make compose-up` gives a throwaway database
whose entire schema lives in a single migration file.

## Trade-off: EMBED_DIMS fixed at 1024

The `chunks.embedding` column is `vector(1024)`, matching `mxbai-embed-large`.
The schema is therefore coupled to the embedding model: switching to a model
with a different dimension requires a new migration (drop and recreate the
column and its HNSW index), not a config change. That coupling is accepted
deliberately: embedding models change rarely, an explicit migration makes a
dimension change visible and reviewable instead of a silent runtime mismatch,
and pgvector indexes are dimension-bound anyway.

## Swapping Ollama for an external API

Ollama is just the current adapter choice. To use an external embedding/LLM
API, write one new package under `internal/adapters/outbound/` implementing
`ports.Embedder` and `ports.Generator`, then change the single
`ollama.NewClient(...)` call in `cmd/api/main.go`. The core, the HTTP layer,
and the schema do not move.

## Layout

```
cmd/api/                          composition root (main.go — the only wiring)
internal/config/                  environment-variable configuration
internal/domain/                  entities, errors, and ports (stdlib only)
internal/application/             use cases, chunking, prompt building
internal/adapters/inbound/http/   chi router, handlers, DTOs
internal/adapters/outbound/       ollama (embedder+generator), pgvector (store), loader (files)
migrations/                       SQL schema (vector(1024), HNSW index)
```

## Quickstart

The binary reads real environment variables only (stdlib `os.LookupEnv`; no
dotenv dependency). The Makefile exports `.env` when present; outside `make`,
export the variables in your shell yourself. Required: `DATABASE_URL`,
`OLLAMA_URL`, `EMBED_MODEL`, `LLM_MODEL`. Optional: `EMBED_DIMS` (default
`1024`), `HTTP_PORT` (default `8080`). See `.env.example`.

```bash
cp .env.example .env   # then review the values
make compose-up        # start postgres + pgvector
make migrate           # apply migrations/000001_init.up.sql
make run               # start the API on :8080
```

With Ollama running locally and the models pulled:

```bash
curl -s localhost:8080/health

curl -s -X POST localhost:8080/documents \
  -d '{"title":"Go proverbs","content":"Don'\''t communicate by sharing memory."}'

curl -s -X POST localhost:8080/documents -d '{"path":"./README.md"}'

curl -s -X POST localhost:8080/query \
  -d '{"text":"what are the go proverbs?","top_k":5}'
```

`top_k` defaults to 5 when omitted and is capped at 20.
