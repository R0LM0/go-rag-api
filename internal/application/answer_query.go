package application

import (
	"context"
	"fmt"
	"strings"

	"github.com/R0LM0/go-rag-api/internal/domain"
	"github.com/R0LM0/go-rag-api/internal/domain/ports"
)

// noSourcesFallback is the answer returned when no chunk is similar enough
// to the question; the generator is intentionally not invoked in that case.
const noSourcesFallback = "I could not find any relevant information to answer your question."

// AnswerQuery is the application service that answers a question with
// retrieval-augmented generation: embed the question, retrieve similar
// chunks through the VectorStore port, and generate a grounded answer
// through the Generator port.
type AnswerQuery struct {
	embedder  ports.Embedder
	store     ports.VectorStore
	generator ports.Generator
}

// NewAnswerQuery returns an AnswerQuery service built from the given ports.
func NewAnswerQuery(embedder ports.Embedder, store ports.VectorStore, generator ports.Generator) *AnswerQuery {
	return &AnswerQuery{
		embedder:  embedder,
		store:     store,
		generator: generator,
	}
}

// Execute validates the query, embeds the question, retrieves the most
// similar chunks, and generates an answer grounded in them. When no chunk is
// retrieved it short-circuits and returns a fallback answer without calling
// the generator. All errors are wrapped so domain sentinels still match
// through errors.Is.
func (s *AnswerQuery) Execute(ctx context.Context, query domain.Query) (domain.Answer, error) {
	if strings.TrimSpace(query.Text) == "" {
		return domain.Answer{}, fmt.Errorf("executing query: %w", domain.ErrEmptyQueryText)
	}
	if query.TopK <= 0 {
		return domain.Answer{}, fmt.Errorf("executing query: %w", domain.ErrInvalidTopK)
	}

	embedding, err := s.embedder.Embed(ctx, query.Text)
	if err != nil {
		return domain.Answer{}, fmt.Errorf("executing query: embedding question: %w", err)
	}
	chunks, err := s.store.SearchSimilar(ctx, embedding, query.TopK)
	if err != nil {
		return domain.Answer{}, fmt.Errorf("executing query: searching similar chunks: %w", err)
	}
	if len(chunks) == 0 {
		return domain.Answer{Text: noSourcesFallback}, nil
	}

	text, err := s.generator.Generate(ctx, BuildPrompt(query.Text, chunks))
	if err != nil {
		return domain.Answer{}, fmt.Errorf("executing query: generating answer: %w", err)
	}
	return domain.Answer{Text: text, Sources: chunks}, nil
}
