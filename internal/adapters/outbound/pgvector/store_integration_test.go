//go:build integration

package pgvector

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/R0LM0/go-rag-api/internal/domain"
	"github.com/jackc/pgx/v5"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/modules/postgres"
)

// oneHot returns a dim-sized vector with 1 at index hot and 0 elsewhere, so
// cosine distances between different one-hot vectors are maximally
// predictable (1 - dot product).
func oneHot(dim, hot int) []float32 {
	vec := make([]float32, dim)
	vec[hot] = 1
	return vec
}

// applyMigrations reads migrations/000001_init.up.sql relative to the
// repository root and executes it as a single script over the simple query
// protocol, which allows multiple statements in one round trip.
func applyMigrations(ctx context.Context, databaseURL string) error {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		return fmt.Errorf("pgvector: cannot locate the repository root from the test file")
	}
	repoRoot := filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(filepath.Dir(thisFile)))))
	migrationPath := filepath.Join(repoRoot, "migrations", "000001_init.up.sql")
	migrationSQL, err := os.ReadFile(migrationPath)
	if err != nil {
		return err
	}
	config, err := pgx.ParseConfig(databaseURL)
	if err != nil {
		return err
	}
	config.DefaultQueryExecMode = pgx.QueryExecModeSimpleProtocol
	conn, err := pgx.ConnectConfig(ctx, config)
	if err != nil {
		return err
	}
	defer conn.Close(ctx)
	if _, err := conn.Exec(ctx, string(migrationSQL)); err != nil {
		return err
	}
	return nil
}

// TestStoreIntegration exercises the Store against a real Postgres 17 with
// pgvector, started through testcontainers. It skips when Docker is not
// available so it degrades gracefully on machines without a Docker daemon.
func TestStoreIntegration(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	container, err := postgres.Run(
		ctx,
		"pgvector/pgvector:pg17",
		postgres.WithDatabase("rag"),
		postgres.WithUsername("rag"),
		postgres.WithPassword("rag"),
		// BasicWaitStrategies waits for the postgres init restart and for the
		// published port to actually serve connections (required on Windows).
		postgres.BasicWaitStrategies(),
	)
	if err != nil {
		t.Skipf("postgres container unavailable (is Docker running?): %v", err)
	}
	defer func() {
		_ = testcontainers.TerminateContainer(container)
	}()

	databaseURL, err := container.ConnectionString(ctx, "sslmode=disable")
	if err != nil {
		t.Fatalf("get connection string: %v", err)
	}
	if err := applyMigrations(ctx, databaseURL); err != nil {
		t.Fatalf("apply migrations: %v", err)
	}

	store, err := NewStore(ctx, databaseURL)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	defer store.Close()

	// Unknown content: the table is still empty, so the first search must
	// return an empty result set rather than an error.
	if got, err := store.SearchSimilar(ctx, oneHot(1024, 0), 3); err != nil {
		t.Fatalf("search empty store: %v", err)
	} else if len(got) != 0 {
		t.Fatalf("search on empty store returned %d chunks, want 0", len(got))
	}

	// Empty input is a no-op.
	if err := store.SaveChunks(ctx, nil); err != nil {
		t.Fatalf("SaveChunks(nil): %v", err)
	}

	chunks := []domain.Chunk{
		{ID: "chunk-a", DocumentID: "doc-1", Content: "alpha content", Position: 0, Embedding: oneHot(1024, 0)},
		{ID: "chunk-b", DocumentID: "doc-1", Content: "beta content", Position: 1, Embedding: oneHot(1024, 1)},
		{ID: "chunk-c", DocumentID: "doc-2", Content: "gamma content", Position: 0, Embedding: oneHot(1024, 2)},
	}
	if err := store.SaveChunks(ctx, chunks); err != nil {
		t.Fatalf("SaveChunks: %v", err)
	}

	// A query closest to chunk B must return chunk B first, with all row
	// fields round-tripped and no embedding loaded back.
	results, err := store.SearchSimilar(ctx, oneHot(1024, 1), 2)
	if err != nil {
		t.Fatalf("SearchSimilar: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("SearchSimilar topK=2 returned %d results, want 2", len(results))
	}
	first := results[0]
	if first.ID != "chunk-b" {
		t.Fatalf("closest result = %q, want chunk-b", first.ID)
	}
	if first.DocumentID != "doc-1" || first.Content != "beta content" || first.Position != 1 {
		t.Fatalf("closest result did not round-trip: %+v", first)
	}
	if first.Embedding != nil {
		t.Fatalf("SearchSimilar must not load the embedding column back, got %d floats", len(first.Embedding))
	}

	// topK is respected.
	limited, err := store.SearchSimilar(ctx, oneHot(1024, 1), 1)
	if err != nil {
		t.Fatalf("SearchSimilar topK=1: %v", err)
	}
	if len(limited) != 1 || limited[0].ID != "chunk-b" {
		t.Fatalf("SearchSimilar topK=1 returned %+v, want only chunk-b", limited)
	}
}
