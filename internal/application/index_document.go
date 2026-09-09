package application

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/R0LM0/go-rag-api/internal/domain"
	"github.com/R0LM0/go-rag-api/internal/domain/ports"
)

// IndexDocument is the application service that ingests a document: it
// chunks the content, embeds the chunks through the Embedder port, and
// stores them through the VectorStore port.
type IndexDocument struct {
	loader   ports.DocumentLoader
	embedder ports.Embedder
	store    ports.VectorStore
	chunks   ChunkConfig
}

// NewIndexDocument returns an IndexDocument service built from the given
// ports and chunk configuration. A zero ChunkConfig falls back to
// DefaultChunkConfig inside the chunker.
func NewIndexDocument(loader ports.DocumentLoader, embedder ports.Embedder, store ports.VectorStore, cfg ChunkConfig) *IndexDocument {
	return &IndexDocument{
		loader:   loader,
		embedder: embedder,
		store:    store,
		chunks:   cfg,
	}
}

// IndexContent validates a title and content pair, creates a new document
// with a generated ID, and indexes it: chunk, embed, and store. All errors
// are wrapped with context so callers can still match the domain sentinels
// with errors.Is.
func (s *IndexDocument) IndexContent(ctx context.Context, title, content string) (domain.Document, error) {
	if err := validateDocumentParts(title, content); err != nil {
		return domain.Document{}, err
	}
	id, err := newID()
	if err != nil {
		return domain.Document{}, fmt.Errorf("indexing document %q: %w", title, err)
	}
	doc := domain.Document{
		ID:        id,
		Title:     title,
		Content:   content,
		CreatedAt: time.Now().UTC(),
	}
	return s.indexDocument(ctx, doc)
}

// IndexFile loads a document through the DocumentLoader port and indexes it
// with the same logic as IndexContent.
func (s *IndexDocument) IndexFile(ctx context.Context, path string) (domain.Document, error) {
	doc, err := s.loader.Load(ctx, path)
	if err != nil {
		return domain.Document{}, fmt.Errorf("loading document from %q: %w", path, err)
	}
	if err := validateDocumentParts(doc.Title, doc.Content); err != nil {
		return domain.Document{}, fmt.Errorf("document loaded from %q: %w", path, err)
	}
	if doc.ID == "" {
		id, err := newID()
		if err != nil {
			return domain.Document{}, fmt.Errorf("indexing document loaded from %q: %w", path, err)
		}
		doc.ID = id
	}
	if doc.CreatedAt.IsZero() {
		doc.CreatedAt = time.Now().UTC()
	}
	return s.indexDocument(ctx, doc)
}

// indexDocument chunks, embeds, and stores one already-validated document.
func (s *IndexDocument) indexDocument(ctx context.Context, doc domain.Document) (domain.Document, error) {
	chunks, err := SplitIntoChunks(doc.ID, doc.Content, s.chunks)
	if err != nil {
		return domain.Document{}, fmt.Errorf("indexing document %s: %w", doc.ID, err)
	}

	texts := make([]string, len(chunks))
	for i, chunk := range chunks {
		texts[i] = chunk.Content
	}
	embeddings, err := s.embedder.EmbedBatch(ctx, texts)
	if err != nil {
		return domain.Document{}, fmt.Errorf("indexing document %s: embedding %d chunks: %w", doc.ID, len(texts), err)
	}
	if len(embeddings) != len(chunks) {
		return domain.Document{}, fmt.Errorf("indexing document %s: embedder returned %d embeddings for %d chunks", doc.ID, len(embeddings), len(chunks))
	}
	for i := range chunks {
		chunks[i].Embedding = embeddings[i]
	}

	if err := s.store.SaveChunks(ctx, chunks); err != nil {
		return domain.Document{}, fmt.Errorf("indexing document %s: saving %d chunks: %w", doc.ID, len(chunks), err)
	}
	return doc, nil
}

// validateDocumentParts enforces the domain invariants on title and content.
func validateDocumentParts(title, content string) error {
	if strings.TrimSpace(title) == "" {
		return fmt.Errorf("invalid document: %w", domain.ErrEmptyDocumentTitle)
	}
	if strings.TrimSpace(content) == "" {
		return fmt.Errorf("invalid document: %w", domain.ErrEmptyDocumentContent)
	}
	return nil
}
