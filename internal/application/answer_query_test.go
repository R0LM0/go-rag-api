package application

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/R0LM0/go-rag-api/internal/domain"
)

func TestAnswerQueryHappyPath(t *testing.T) {
	embedder := &fakeEmbedder{}
	generator := &fakeGenerator{}
	retrieved := []domain.Chunk{
		{ID: "c1", DocumentID: "d1", Content: "Paris is the capital of France.", Position: 0, Embedding: []float32{1}},
		{ID: "c2", DocumentID: "d1", Content: "The Seine crosses Paris.", Position: 1, Embedding: []float32{2}},
	}
	store := &fakeVectorStore{
		searchFn: func(context.Context, []float32, int) ([]domain.Chunk, error) {
			return retrieved, nil
		},
	}
	svc := NewAnswerQuery(embedder, store, generator)

	query := domain.Query{Text: "What is the capital of France?", TopK: 2}
	answer, err := svc.Execute(context.Background(), query)
	if err != nil {
		t.Fatalf("Execute() error = %v, want nil", err)
	}
	if answer.Text != "generated answer" {
		t.Errorf("answer.Text = %q, want %q", answer.Text, "generated answer")
	}
	if !reflect.DeepEqual(answer.Sources, retrieved) {
		t.Errorf("answer.Sources = %+v, want the retrieved chunks", answer.Sources)
	}

	// The question must be embedded exactly once.
	if len(embedder.embedTexts) != 1 || embedder.embedTexts[0] != query.Text {
		t.Errorf("embedder texts = %v, want [%q]", embedder.embedTexts, query.Text)
	}
	// The store must receive the embedder output and the requested TopK.
	if len(store.searchCalls) != 1 {
		t.Fatalf("SearchSimilar called %d times, want 1", len(store.searchCalls))
	}
	call := store.searchCalls[0]
	if call.topK != 2 {
		t.Errorf("SearchSimilar topK = %d, want 2", call.topK)
	}
	if want := float32(len(query.Text)); len(call.embedding) != 1 || call.embedding[0] != want {
		t.Errorf("SearchSimilar embedding = %v, want [%v]", call.embedding, want)
	}
	// The generator prompt must contain the question and the chunk contents.
	if len(generator.prompts) != 1 {
		t.Fatalf("Generate called %d times, want 1", len(generator.prompts))
	}
	prompt := generator.prompts[0]
	if !strings.Contains(prompt, query.Text) {
		t.Errorf("prompt does not contain the question: %q", prompt)
	}
	if !strings.Contains(prompt, retrieved[0].Content) {
		t.Errorf("prompt does not contain the first chunk content: %q", prompt)
	}
	if !strings.Contains(prompt, retrieved[1].Content) {
		t.Errorf("prompt does not contain the second chunk content: %q", prompt)
	}
}

func TestAnswerQueryValidation(t *testing.T) {
	embedder := &fakeEmbedder{}
	store := &fakeVectorStore{}
	generator := &fakeGenerator{}
	svc := NewAnswerQuery(embedder, store, generator)

	_, err := svc.Execute(context.Background(), domain.Query{Text: "   ", TopK: 3})
	if !errors.Is(err, domain.ErrEmptyQueryText) {
		t.Errorf("empty query text: err = %v, want ErrEmptyQueryText", err)
	}
	for _, topK := range []int{0, -1} {
		_, err := svc.Execute(context.Background(), domain.Query{Text: "a question", TopK: topK})
		if !errors.Is(err, domain.ErrInvalidTopK) {
			t.Errorf("topK = %d: err = %v, want ErrInvalidTopK", topK, err)
		}
	}
	if len(embedder.embedTexts) != 0 {
		t.Errorf("Embed called %d times, want 0 for invalid queries", len(embedder.embedTexts))
	}
	if len(generator.prompts) != 0 {
		t.Errorf("Generate called %d times, want 0 for invalid queries", len(generator.prompts))
	}
}

func TestAnswerQueryNoResultsSkipsGenerator(t *testing.T) {
	generator := &fakeGenerator{}
	store := &fakeVectorStore{
		searchFn: func(context.Context, []float32, int) ([]domain.Chunk, error) {
			return nil, nil
		},
	}
	svc := NewAnswerQuery(&fakeEmbedder{}, store, generator)

	answer, err := svc.Execute(context.Background(), domain.Query{Text: "a question", TopK: 3})
	if err != nil {
		t.Fatalf("Execute() error = %v, want nil", err)
	}
	if answer.Text != noSourcesFallback {
		t.Errorf("answer.Text = %q, want the fallback message", answer.Text)
	}
	if len(answer.Sources) != 0 {
		t.Errorf("answer.Sources = %+v, want empty", answer.Sources)
	}
	if len(generator.prompts) != 0 {
		t.Errorf("Generate called %d times, want 0 when no chunks are retrieved", len(generator.prompts))
	}
}

func TestAnswerQueryWrapsPortErrors(t *testing.T) {
	t.Run("embedder error", func(t *testing.T) {
		embedder := &fakeEmbedder{
			embedFn: func(context.Context, string) ([]float32, error) {
				return nil, errEmbedderBoom
			},
		}
		svc := NewAnswerQuery(embedder, &fakeVectorStore{}, &fakeGenerator{})

		_, err := svc.Execute(context.Background(), domain.Query{Text: "a question", TopK: 3})
		if !errors.Is(err, errEmbedderBoom) {
			t.Errorf("err = %v, want wrapping errEmbedderBoom", err)
		}
	})

	t.Run("store error", func(t *testing.T) {
		store := &fakeVectorStore{
			searchFn: func(context.Context, []float32, int) ([]domain.Chunk, error) {
				return nil, errStoreBoom
			},
		}
		svc := NewAnswerQuery(&fakeEmbedder{}, store, &fakeGenerator{})

		_, err := svc.Execute(context.Background(), domain.Query{Text: "a question", TopK: 3})
		if !errors.Is(err, errStoreBoom) {
			t.Errorf("err = %v, want wrapping errStoreBoom", err)
		}
	})

	t.Run("generator error", func(t *testing.T) {
		store := &fakeVectorStore{
			searchFn: func(context.Context, []float32, int) ([]domain.Chunk, error) {
				return []domain.Chunk{{ID: "c1", DocumentID: "d1", Content: "some context", Position: 0}}, nil
			},
		}
		generator := &fakeGenerator{
			generateFn: func(context.Context, string) (string, error) {
				return "", errGeneratorBoom
			},
		}
		svc := NewAnswerQuery(&fakeEmbedder{}, store, generator)

		_, err := svc.Execute(context.Background(), domain.Query{Text: "a question", TopK: 3})
		if !errors.Is(err, errGeneratorBoom) {
			t.Errorf("err = %v, want wrapping errGeneratorBoom", err)
		}
	})
}
