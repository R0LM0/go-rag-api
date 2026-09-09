// Package pgvector implements the VectorStore port on top of PostgreSQL with
// the pgvector extension, using pgx v5 and pgvector-go.
package pgvector

import (
	"context"
	"fmt"

	"github.com/R0LM0/go-rag-api/internal/domain"
	"github.com/R0LM0/go-rag-api/internal/domain/ports"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
	pgvectorpgx "github.com/pgvector/pgvector-go/pgx"
)

// Store persists chunks in a Postgres table with a vector column and answers
// cosine-distance similarity searches through an HNSW index.
type Store struct {
	pool *pgxpool.Pool
}

// Compile-time conformance assertion: Store satisfies the VectorStore port.
var _ ports.VectorStore = (*Store)(nil)

// NewStore opens a pgx connection pool for databaseURL and verifies the
// connection with a ping. The pgvector types are registered on every pooled
// connection so the vector column transparently maps to []float32.
func NewStore(ctx context.Context, databaseURL string) (*Store, error) {
	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		return nil, fmt.Errorf("pgvector: parse database URL: %w", err)
	}
	config.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		return pgvectorpgx.RegisterTypes(ctx, conn)
	}
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("pgvector: create connection pool: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("pgvector: ping database: %w", err)
	}
	return &Store{pool: pool}, nil
}

// SaveChunks batch-inserts all chunks in a single transaction. An empty slice
// is a no-op.
func (s *Store) SaveChunks(ctx context.Context, chunks []domain.Chunk) error {
	if len(chunks) == 0 {
		return nil
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("pgvector: begin transaction to save %d chunks: %w", len(chunks), err)
	}
	defer tx.Rollback(ctx) // no-op once committed

	batch := &pgx.Batch{}
	for _, chunk := range chunks {
		batch.Queue(
			"INSERT INTO chunks (id, document_id, content, position, embedding) VALUES ($1, $2, $3, $4, $5)",
			chunk.ID,
			chunk.DocumentID,
			chunk.Content,
			chunk.Position,
			pgvector.NewVector(chunk.Embedding),
		)
	}
	if err := tx.SendBatch(ctx, batch).Close(); err != nil {
		return fmt.Errorf("pgvector: insert %d chunks: %w", len(chunks), err)
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("pgvector: commit chunk inserts: %w", err)
	}
	return nil
}

// SearchSimilar returns up to topK chunks ordered by cosine distance
// (the <=> operator, backed by the HNSW index) from the given embedding.
//
// The embedding column is deliberately not selected: the caller already
// holds the query embedding, and returning 1024 floats per result row would
// be wasted bandwidth. Chunk.Embedding is therefore left nil.
func (s *Store) SearchSimilar(ctx context.Context, embedding []float32, topK int) ([]domain.Chunk, error) {
	rows, err := s.pool.Query(
		ctx,
		"SELECT id, document_id, content, position FROM chunks ORDER BY embedding <=> $1 LIMIT $2",
		pgvector.NewVector(embedding),
		topK,
	)
	if err != nil {
		return nil, fmt.Errorf("pgvector: similarity query: %w", err)
	}
	defer rows.Close()

	chunks := make([]domain.Chunk, 0)
	for rows.Next() {
		var chunk domain.Chunk
		if err := rows.Scan(&chunk.ID, &chunk.DocumentID, &chunk.Content, &chunk.Position); err != nil {
			return nil, fmt.Errorf("pgvector: scan similarity result row: %w", err)
		}
		chunks = append(chunks, chunk)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("pgvector: iterate similarity results: %w", err)
	}
	return chunks, nil
}

// Close releases every pooled connection.
func (s *Store) Close() {
	s.pool.Close()
}
