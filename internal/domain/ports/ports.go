// Package ports defines the driven (outbound) interfaces the application
// core relies on. Adapters in later phases implement these interfaces; the
// core itself never imports adapter packages, so dependencies point inward.
package ports

import (
	"context"

	"github.com/R0LM0/go-rag-api/internal/domain"
)

// Embedder converts text into dense vector representations. Implementations
// are provided by adapters (for example an Ollama embeddings client).
type Embedder interface {
	// Embed returns the embedding vector for a single text.
	Embed(ctx context.Context, text string) ([]float32, error)
	// EmbedBatch returns one embedding per input text, preserving order.
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
}

// VectorStore persists chunks and answers similarity searches over their
// embeddings. Implementations are provided by adapters (for example a
// pgvector-backed repository).
type VectorStore interface {
	// SaveChunks stores chunks together with their embeddings.
	SaveChunks(ctx context.Context, chunks []domain.Chunk) error
	// SearchSimilar returns up to topK chunks whose embeddings are closest
	// to the given embedding.
	SearchSimilar(ctx context.Context, embedding []float32, topK int) ([]domain.Chunk, error)
}

// Generator produces natural-language text from a prompt. Implementations
// are provided by adapters (for example an LLM chat client).
type Generator interface {
	// Generate returns the generated text for the given prompt.
	Generate(ctx context.Context, prompt string) (string, error)
}

// DocumentLoader reads a document from an external source such as a file
// path. Implementations are provided by adapters.
type DocumentLoader interface {
	// Load reads and returns the document located at path.
	Load(ctx context.Context, path string) (domain.Document, error)
}
