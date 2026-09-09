package application

import (
	"context"
	"errors"

	"github.com/R0LM0/go-rag-api/internal/domain"
	"github.com/R0LM0/go-rag-api/internal/domain/ports"
)

// Shared sentinel errors returned by the fake ports in failure tests.
var (
	errEmbedderBoom  = errors.New("fake embedder failure")
	errStoreBoom     = errors.New("fake store failure")
	errGeneratorBoom = errors.New("fake generator failure")
	errLoaderBoom    = errors.New("fake loader failure")
)

// fakeEmbedder is a configurable ports.Embedder double that records calls.
type fakeEmbedder struct {
	embedFn      func(ctx context.Context, text string) ([]float32, error)
	embedBatchFn func(ctx context.Context, texts []string) ([][]float32, error)

	embedTexts     []string
	embedBatchText [][]string
}

// Embed records the call and delegates to embedFn when configured.
func (f *fakeEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	f.embedTexts = append(f.embedTexts, text)
	if f.embedFn != nil {
		return f.embedFn(ctx, text)
	}
	return []float32{float32(len(text))}, nil
}

// EmbedBatch records the call and delegates to embedBatchFn when configured.
func (f *fakeEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	f.embedBatchText = append(f.embedBatchText, texts)
	if f.embedBatchFn != nil {
		return f.embedBatchFn(ctx, texts)
	}
	embeddings := make([][]float32, len(texts))
	for i, text := range texts {
		embeddings[i] = []float32{float32(len(text))}
	}
	return embeddings, nil
}

// searchCall records one SearchSimilar invocation.
type searchCall struct {
	embedding []float32
	topK      int
}

// fakeVectorStore is a configurable ports.VectorStore double that records calls.
type fakeVectorStore struct {
	saveFn   func(ctx context.Context, chunks []domain.Chunk) error
	searchFn func(ctx context.Context, embedding []float32, topK int) ([]domain.Chunk, error)

	savedChunks [][]domain.Chunk
	searchCalls []searchCall
}

// SaveChunks records the call and delegates to saveFn when configured.
func (f *fakeVectorStore) SaveChunks(ctx context.Context, chunks []domain.Chunk) error {
	f.savedChunks = append(f.savedChunks, chunks)
	if f.saveFn != nil {
		return f.saveFn(ctx, chunks)
	}
	return nil
}

// SearchSimilar records the call and delegates to searchFn when configured.
func (f *fakeVectorStore) SearchSimilar(ctx context.Context, embedding []float32, topK int) ([]domain.Chunk, error) {
	f.searchCalls = append(f.searchCalls, searchCall{embedding: embedding, topK: topK})
	if f.searchFn != nil {
		return f.searchFn(ctx, embedding, topK)
	}
	return nil, nil
}

// fakeGenerator is a configurable ports.Generator double that records prompts.
type fakeGenerator struct {
	generateFn func(ctx context.Context, prompt string) (string, error)
	prompts    []string
}

// Generate records the prompt and delegates to generateFn when configured.
func (f *fakeGenerator) Generate(ctx context.Context, prompt string) (string, error) {
	f.prompts = append(f.prompts, prompt)
	if f.generateFn != nil {
		return f.generateFn(ctx, prompt)
	}
	return "generated answer", nil
}

// fakeLoader is a configurable ports.DocumentLoader double that records paths.
type fakeLoader struct {
	loadFn func(ctx context.Context, path string) (domain.Document, error)
	paths  []string
}

// Load records the path and delegates to loadFn when configured.
func (f *fakeLoader) Load(ctx context.Context, path string) (domain.Document, error) {
	f.paths = append(f.paths, path)
	if f.loadFn != nil {
		return f.loadFn(ctx, path)
	}
	return domain.Document{}, nil
}

// Compile-time checks that the fakes satisfy the ports.
var (
	_ ports.Embedder       = (*fakeEmbedder)(nil)
	_ ports.VectorStore    = (*fakeVectorStore)(nil)
	_ ports.Generator      = (*fakeGenerator)(nil)
	_ ports.DocumentLoader = (*fakeLoader)(nil)
)
