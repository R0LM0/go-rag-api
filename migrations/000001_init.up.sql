-- 000001_init.up.sql: initial schema for the RAG API.
-- vector(1024) matches the mxbai-embed-large embedding model and EMBED_DIMS
-- from .env.example; changing the model requires changing this dimension.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT   NOT NULL,
    content     TEXT   NOT NULL,
    position    INTEGER NOT NULL,
    embedding   vector(1024) NOT NULL
);

-- HNSW index for fast approximate nearest-neighbor search using cosine
-- distance (the <=> operator used by the application's similarity queries).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Plain index to look up all chunks of a document efficiently.
CREATE INDEX IF NOT EXISTS chunks_document_id_idx
    ON chunks (document_id);
