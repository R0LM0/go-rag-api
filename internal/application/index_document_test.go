package application

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/R0LM0/go-rag-api/internal/domain"
)

func TestIndexContentHappyPath(t *testing.T) {
	embedder := &fakeEmbedder{}
	store := &fakeVectorStore{}
	svc := NewIndexDocument(&fakeLoader{}, embedder, store, ChunkConfig{MaxTokens: 5, OverlapTokens: 0})

	// Six 8-character paragraphs with maxChars = 20 produce three chunks of
	// two paragraphs each, saved in order.
	content := strings.Join(
		[]string{"aaaaaaaa", "bbbbbbbb", "cccccccc", "dddddddd", "eeeeeeee", "ffffffff"},
		"\n\n",
	)

	doc, err := svc.IndexContent(context.Background(), "Test document", content)
	if err != nil {
		t.Fatalf("IndexContent() error = %v, want nil", err)
	}
	if doc.ID == "" {
		t.Fatal("doc.ID must not be empty")
	}
	if len(doc.ID) != 32 {
		t.Errorf("doc.ID length = %d, want 32 hex characters", len(doc.ID))
	}
	if got, want := doc.Title, "Test document"; got != want {
		t.Errorf("doc.Title = %q, want %q", got, want)
	}
	if doc.CreatedAt.IsZero() {
		t.Error("doc.CreatedAt must not be zero")
	}
	if loc := doc.CreatedAt.Location(); loc != time.UTC {
		t.Errorf("doc.CreatedAt location = %v, want UTC", loc)
	}

	if len(store.savedChunks) != 1 {
		t.Fatalf("SaveChunks called %d times, want 1", len(store.savedChunks))
	}
	saved := store.savedChunks[0]
	if len(saved) != 3 {
		t.Fatalf("saved %d chunks, want 3", len(saved))
	}
	wantContents := []string{
		"aaaaaaaa\n\nbbbbbbbb",
		"cccccccc\n\ndddddddd",
		"eeeeeeee\n\nffffffff",
	}
	for i, chunk := range saved {
		if chunk.ID == "" {
			t.Errorf("saved chunk %d ID must not be empty", i)
		}
		if chunk.DocumentID != doc.ID {
			t.Errorf("saved chunk %d DocumentID = %q, want %q", i, chunk.DocumentID, doc.ID)
		}
		if chunk.Position != i {
			t.Errorf("saved chunk %d Position = %d, want %d", i, chunk.Position, i)
		}
		if chunk.Content != wantContents[i] {
			t.Errorf("saved chunk %d Content = %q, want %q", i, chunk.Content, wantContents[i])
		}
		if len(chunk.Embedding) != 1 {
			t.Fatalf("saved chunk %d embedding length = %d, want 1", i, len(chunk.Embedding))
		}
		if want := float32(len(chunk.Content)); chunk.Embedding[0] != want {
			t.Errorf("saved chunk %d embedding = %v, want [%v]", i, chunk.Embedding, want)
		}
	}

	// The embedder must have received the chunk texts in order.
	if len(embedder.embedBatchText) != 1 {
		t.Fatalf("EmbedBatch called %d times, want 1", len(embedder.embedBatchText))
	}
	texts := embedder.embedBatchText[0]
	if len(texts) != len(saved) {
		t.Fatalf("EmbedBatch received %d texts, want %d", len(texts), len(saved))
	}
	for i, text := range texts {
		if text != saved[i].Content {
			t.Errorf("EmbedBatch text %d = %q, want %q", i, text, saved[i].Content)
		}
	}
}

func TestIndexContentValidation(t *testing.T) {
	embedder := &fakeEmbedder{}
	store := &fakeVectorStore{}
	svc := NewIndexDocument(&fakeLoader{}, embedder, store, DefaultChunkConfig())

	_, err := svc.IndexContent(context.Background(), "   ", "some content")
	if !errors.Is(err, domain.ErrEmptyDocumentTitle) {
		t.Errorf("IndexContent with empty title: err = %v, want ErrEmptyDocumentTitle", err)
	}
	_, err = svc.IndexContent(context.Background(), "Title", " \n\t ")
	if !errors.Is(err, domain.ErrEmptyDocumentContent) {
		t.Errorf("IndexContent with empty content: err = %v, want ErrEmptyDocumentContent", err)
	}
	if len(embedder.embedBatchText) != 0 {
		t.Errorf("EmbedBatch called %d times, want 0 for invalid documents", len(embedder.embedBatchText))
	}
	if len(store.savedChunks) != 0 {
		t.Errorf("SaveChunks called %d times, want 0 for invalid documents", len(store.savedChunks))
	}
}

func TestIndexContentWrapsPortErrors(t *testing.T) {
	t.Run("embedder error", func(t *testing.T) {
		embedder := &fakeEmbedder{
			embedBatchFn: func(context.Context, []string) ([][]float32, error) {
				return nil, errEmbedderBoom
			},
		}
		store := &fakeVectorStore{}
		svc := NewIndexDocument(&fakeLoader{}, embedder, store, DefaultChunkConfig())

		_, err := svc.IndexContent(context.Background(), "Title", "Some content")
		if !errors.Is(err, errEmbedderBoom) {
			t.Errorf("err = %v, want wrapping errEmbedderBoom", err)
		}
		if len(store.savedChunks) != 0 {
			t.Error("SaveChunks must not be called when embedding fails")
		}
	})

	t.Run("store error", func(t *testing.T) {
		store := &fakeVectorStore{
			saveFn: func(context.Context, []domain.Chunk) error {
				return errStoreBoom
			},
		}
		svc := NewIndexDocument(&fakeLoader{}, &fakeEmbedder{}, store, DefaultChunkConfig())

		_, err := svc.IndexContent(context.Background(), "Title", "Some content")
		if !errors.Is(err, errStoreBoom) {
			t.Errorf("err = %v, want wrapping errStoreBoom", err)
		}
	})

	t.Run("embedding count mismatch", func(t *testing.T) {
		embedder := &fakeEmbedder{
			embedBatchFn: func(context.Context, []string) ([][]float32, error) {
				return nil, nil
			},
		}
		store := &fakeVectorStore{}
		svc := NewIndexDocument(&fakeLoader{}, embedder, store, DefaultChunkConfig())

		_, err := svc.IndexContent(context.Background(), "Title", "Some content")
		if err == nil {
			t.Fatal("err = nil, want error for embedding count mismatch")
		}
		if !strings.Contains(err.Error(), "embeddings") {
			t.Errorf("err = %v, want it to describe the embedding count mismatch", err)
		}
		if len(store.savedChunks) != 0 {
			t.Error("SaveChunks must not be called when embedding counts mismatch")
		}
	})
}

func TestIndexFile(t *testing.T) {
	t.Run("happy path via loader", func(t *testing.T) {
		loader := &fakeLoader{
			loadFn: func(context.Context, string) (domain.Document, error) {
				return domain.Document{Title: "Loaded", Content: "loaded content"}, nil
			},
		}
		embedder := &fakeEmbedder{}
		store := &fakeVectorStore{}
		svc := NewIndexDocument(loader, embedder, store, DefaultChunkConfig())

		doc, err := svc.IndexFile(context.Background(), "docs/hello.txt")
		if err != nil {
			t.Fatalf("IndexFile() error = %v, want nil", err)
		}
		if len(loader.paths) != 1 || loader.paths[0] != "docs/hello.txt" {
			t.Errorf("loader paths = %v, want [docs/hello.txt]", loader.paths)
		}
		if doc.ID == "" {
			t.Error("doc.ID must be generated when the loader returns none")
		}
		if doc.CreatedAt.IsZero() {
			t.Error("doc.CreatedAt must be set when the loader returns none")
		}
		if got, want := doc.Title, "Loaded"; got != want {
			t.Errorf("doc.Title = %q, want %q", got, want)
		}
		if len(store.savedChunks) != 1 {
			t.Fatalf("SaveChunks called %d times, want 1", len(store.savedChunks))
		}
		saved := store.savedChunks[0]
		if len(saved) != 1 {
			t.Fatalf("saved %d chunks, want 1", len(saved))
		}
		if saved[0].DocumentID != doc.ID {
			t.Errorf("saved chunk DocumentID = %q, want %q", saved[0].DocumentID, doc.ID)
		}
	})

	t.Run("loader error is wrapped", func(t *testing.T) {
		loader := &fakeLoader{
			loadFn: func(context.Context, string) (domain.Document, error) {
				return domain.Document{}, errLoaderBoom
			},
		}
		svc := NewIndexDocument(loader, &fakeEmbedder{}, &fakeVectorStore{}, DefaultChunkConfig())

		_, err := svc.IndexFile(context.Background(), "missing.txt")
		if !errors.Is(err, errLoaderBoom) {
			t.Errorf("err = %v, want wrapping errLoaderBoom", err)
		}
	})

	t.Run("loaded document is validated", func(t *testing.T) {
		loader := &fakeLoader{
			loadFn: func(context.Context, string) (domain.Document, error) {
				return domain.Document{Title: "Empty", Content: "   "}, nil
			},
		}
		svc := NewIndexDocument(loader, &fakeEmbedder{}, &fakeVectorStore{}, DefaultChunkConfig())

		_, err := svc.IndexFile(context.Background(), "empty.txt")
		if !errors.Is(err, domain.ErrEmptyDocumentContent) {
			t.Errorf("err = %v, want ErrEmptyDocumentContent", err)
		}
	})
}
