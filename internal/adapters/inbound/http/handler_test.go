package http

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/R0LM0/go-rag-api/internal/adapters/outbound/loader"
	"github.com/R0LM0/go-rag-api/internal/domain"
)

// stubIndexer is a hand-written fake of the documentIndexer interface that
// records what the handler passed and returns the configured result.
type stubIndexer struct {
	contentCalled  bool
	contentTitle   string
	contentContent string
	fileCalled     bool
	filePath       string
	doc            domain.Document
	err            error
}

func (s *stubIndexer) IndexContent(_ context.Context, title, content string) (domain.Document, error) {
	s.contentCalled = true
	s.contentTitle = title
	s.contentContent = content
	return s.doc, s.err
}

func (s *stubIndexer) IndexFile(_ context.Context, path string) (domain.Document, error) {
	s.fileCalled = true
	s.filePath = path
	return s.doc, s.err
}

// stubAnswerer is a hand-written fake of the queryAnswerer interface.
type stubAnswerer struct {
	called   bool
	gotQuery domain.Query
	answer   domain.Answer
	err      error
}

func (s *stubAnswerer) Execute(_ context.Context, query domain.Query) (domain.Answer, error) {
	s.called = true
	s.gotQuery = query
	return s.answer, s.err
}

// newTestHandler builds a Handler wired to the given stubs.
func newTestHandler(indexer *stubIndexer, answerer *stubAnswerer) *Handler {
	return NewHandler(indexer, answerer)
}

// serve calls the handler method directly against a recorder — no server, no
// database, no Ollama.
func serve(t *testing.T, h http.HandlerFunc, method, target, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, target, strings.NewReader(body))
	rec := httptest.NewRecorder()
	h(rec, req)
	return rec
}

// decodeJSON unmarshals the recorder body into v, failing the test on a
// malformed body.
func decodeJSON(t *testing.T, rec *httptest.ResponseRecorder, v any) {
	t.Helper()
	if err := json.Unmarshal(rec.Body.Bytes(), v); err != nil {
		t.Fatalf("decode response body %q: %v", rec.Body.String(), err)
	}
}

// requireStatus asserts a status code and returns the recorder for further
// assertions.
func requireStatus(t *testing.T, rec *httptest.ResponseRecorder, want int) *httptest.ResponseRecorder {
	t.Helper()
	if rec.Code != want {
		t.Fatalf("status = %d, want %d; body: %s", rec.Code, want, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}
	return rec
}

func TestPostDocument_InlineContent(t *testing.T) {
	created := time.Date(2025, 1, 2, 3, 4, 5, 0, time.UTC)
	indexer := &stubIndexer{doc: domain.Document{ID: "doc-1", Title: "Go proverbs", CreatedAt: created}}
	h := newTestHandler(indexer, &stubAnswerer{})

	rec := requireStatus(t, serve(t, h.PostDocument, http.MethodPost, "/documents",
		`{"title":"Go proverbs","content":"Don't communicate by sharing memory."}`),
		http.StatusCreated)

	if !indexer.contentCalled || indexer.fileCalled {
		t.Errorf("use case calls: content=%v file=%v, want content=true file=false", indexer.contentCalled, indexer.fileCalled)
	}
	if indexer.contentTitle != "Go proverbs" || indexer.contentContent != "Don't communicate by sharing memory." {
		t.Errorf("use case received title=%q content=%q", indexer.contentTitle, indexer.contentContent)
	}

	var got IndexDocumentResponse
	decodeJSON(t, rec, &got)
	if got.ID != "doc-1" || got.Title != "Go proverbs" || !got.CreatedAt.Equal(created) {
		t.Errorf("response = %+v, want id=doc-1 title=Go proverbs createdAt=%v", got, created)
	}
}

func TestPostDocument_FilePath(t *testing.T) {
	indexer := &stubIndexer{doc: domain.Document{ID: "doc-2", Title: "notes"}}
	h := newTestHandler(indexer, &stubAnswerer{})

	rec := requireStatus(t, serve(t, h.PostDocument, http.MethodPost, "/documents",
		`{"path":"./notes.md"}`),
		http.StatusCreated)

	if !indexer.fileCalled || indexer.contentCalled {
		t.Errorf("use case calls: content=%v file=%v, want content=false file=true", indexer.contentCalled, indexer.fileCalled)
	}
	if indexer.filePath != "./notes.md" {
		t.Errorf("use case received path=%q, want ./notes.md", indexer.filePath)
	}

	var got IndexDocumentResponse
	decodeJSON(t, rec, &got)
	if got.ID != "doc-2" || got.Title != "notes" {
		t.Errorf("response = %+v, want id=doc-2 title=notes", got)
	}
}

func TestPostDocument_BadJSON(t *testing.T) {
	indexer := &stubIndexer{}
	h := newTestHandler(indexer, &stubAnswerer{})

	rec := requireStatus(t, serve(t, h.PostDocument, http.MethodPost, "/documents", `{"title":`),
		http.StatusBadRequest)

	if indexer.contentCalled || indexer.fileCalled {
		t.Error("use case must not be called for a malformed body")
	}
	var got ErrorResponse
	decodeJSON(t, rec, &got)
	if got.Error == "" {
		t.Error("error message must not be empty")
	}
}

func TestPostDocument_Validation(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"title and content together with path", `{"title":"t","content":"c","path":"./f.md"}`},
		{"path together with content", `{"content":"c","path":"./f.md"}`},
		{"path together with title", `{"title":"t","path":"./f.md"}`},
		{"nothing at all", `{}`},
		{"title without content", `{"title":"t"}`},
		{"content without title", `{"content":"c"}`},
		{"blank title and content", `{"title":"  ","content":"  "}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			indexer := &stubIndexer{}
			h := newTestHandler(indexer, &stubAnswerer{})

			rec := requireStatus(t, serve(t, h.PostDocument, http.MethodPost, "/documents", tt.body),
				http.StatusBadRequest)

			if indexer.contentCalled || indexer.fileCalled {
				t.Error("use case must not be called on validation failure")
			}
			var got ErrorResponse
			decodeJSON(t, rec, &got)
			if got.Error == "" {
				t.Error("error message must not be empty")
			}
		})
	}
}

func TestPostDocument_SentinelErrors(t *testing.T) {
	tests := []struct {
		name    string
		err     error
		wantMsg string
	}{
		{
			name:    "wrapped empty content sentinel",
			err:     fmt.Errorf("indexing document x: %w", domain.ErrEmptyDocumentContent),
			wantMsg: domain.ErrEmptyDocumentContent.Error(),
		},
		{
			name:    "wrapped empty title sentinel",
			err:     fmt.Errorf("loading document from q: %w", domain.ErrEmptyDocumentTitle),
			wantMsg: domain.ErrEmptyDocumentTitle.Error(),
		},
		{
			name:    "wrapped unsupported file type sentinel",
			err:     fmt.Errorf("loader: ./f.exe: %w", loader.ErrUnsupportedFileType),
			wantMsg: loader.ErrUnsupportedFileType.Error(),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			indexer := &stubIndexer{err: tt.err}
			h := newTestHandler(indexer, &stubAnswerer{})

			rec := requireStatus(t, serve(t, h.PostDocument, http.MethodPost, "/documents",
				`{"title":"t","content":"c"}`),
				http.StatusBadRequest)

			var got ErrorResponse
			decodeJSON(t, rec, &got)
			if got.Error != tt.wantMsg {
				t.Errorf("error message = %q, want %q", got.Error, tt.wantMsg)
			}
		})
	}
}

func TestPostDocument_InternalError(t *testing.T) {
	indexer := &stubIndexer{err: errors.New("db exploded with secret details")}
	h := newTestHandler(indexer, &stubAnswerer{})

	rec := requireStatus(t, serve(t, h.PostDocument, http.MethodPost, "/documents",
		`{"title":"t","content":"c"}`),
		http.StatusInternalServerError)

	var got ErrorResponse
	decodeJSON(t, rec, &got)
	if got.Error != "internal server error" {
		t.Errorf("error message = %q, want generic %q that leaks no internals", got.Error, "internal server error")
	}
}

func TestPostQuery_HappyPath(t *testing.T) {
	answerer := &stubAnswerer{answer: domain.Answer{
		Text: "Don't communicate by sharing memory.",
		Sources: []domain.Chunk{
			{ID: "c1", DocumentID: "doc-1", Content: "chunk one", Position: 0},
			{ID: "c2", DocumentID: "doc-1", Content: "chunk two", Position: 1},
		},
	}}
	h := newTestHandler(&stubIndexer{}, answerer)

	rec := requireStatus(t, serve(t, h.PostQuery, http.MethodPost, "/query",
		`{"text":"what are the go proverbs?","top_k":7}`),
		http.StatusOK)

	if !answerer.called {
		t.Fatal("use case was not called")
	}
	if want := (domain.Query{Text: "what are the go proverbs?", TopK: 7}); answerer.gotQuery != want {
		t.Errorf("use case received query %+v, want %+v", answerer.gotQuery, want)
	}

	var got QueryResponse
	decodeJSON(t, rec, &got)
	if got.Answer != "Don't communicate by sharing memory." {
		t.Errorf("answer = %q", got.Answer)
	}
	if len(got.Sources) != 2 {
		t.Fatalf("sources = %+v, want 2 entries", got.Sources)
	}
	if s := got.Sources[0]; s.DocumentID != "doc-1" || s.Content != "chunk one" || s.Position != 0 {
		t.Errorf("sources[0] = %+v, want document_id=doc-1 content=chunk one position=0", s)
	}
}

func TestPostQuery_TopKZeroNormalizedToDefault(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"top_k absent", `{"text":"q"}`},
		{"top_k explicitly zero", `{"text":"q","top_k":0}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			answerer := &stubAnswerer{answer: domain.Answer{Text: "a"}}
			h := newTestHandler(&stubIndexer{}, answerer)

			requireStatus(t, serve(t, h.PostQuery, http.MethodPost, "/query", tt.body), http.StatusOK)

			if answerer.gotQuery.TopK != DefaultTopK {
				t.Errorf("use case received TopK=%d, want default %d", answerer.gotQuery.TopK, DefaultTopK)
			}
		})
	}
}

func TestPostQuery_Validation(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"missing text", `{"top_k":5}`},
		{"blank text", `{"text":"   ","top_k":5}`},
		{"negative top_k", `{"text":"q","top_k":-1}`},
		{"top_k above maximum", `{"text":"q","top_k":21}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			answerer := &stubAnswerer{}
			h := newTestHandler(&stubIndexer{}, answerer)

			rec := requireStatus(t, serve(t, h.PostQuery, http.MethodPost, "/query", tt.body),
				http.StatusBadRequest)

			if answerer.called {
				t.Error("use case must not be called on validation failure")
			}
			var got ErrorResponse
			decodeJSON(t, rec, &got)
			if got.Error == "" {
				t.Error("error message must not be empty")
			}
		})
	}
}

func TestPostQuery_BadJSON(t *testing.T) {
	h := newTestHandler(&stubIndexer{}, &stubAnswerer{})

	rec := requireStatus(t, serve(t, h.PostQuery, http.MethodPost, "/query", `not json`),
		http.StatusBadRequest)

	var got ErrorResponse
	decodeJSON(t, rec, &got)
	if got.Error == "" {
		t.Error("error message must not be empty")
	}
}

func TestPostQuery_SentinelError(t *testing.T) {
	answerer := &stubAnswerer{err: fmt.Errorf("executing query: %w", domain.ErrEmptyQueryText)}
	h := newTestHandler(&stubIndexer{}, answerer)

	rec := requireStatus(t, serve(t, h.PostQuery, http.MethodPost, "/query", `{"text":"q"}`),
		http.StatusBadRequest)

	var got ErrorResponse
	decodeJSON(t, rec, &got)
	if got.Error != domain.ErrEmptyQueryText.Error() {
		t.Errorf("error message = %q, want %q", got.Error, domain.ErrEmptyQueryText.Error())
	}
}

func TestPostQuery_InternalError(t *testing.T) {
	answerer := &stubAnswerer{err: errors.New("ollama connection refused")}
	h := newTestHandler(&stubIndexer{}, answerer)

	rec := requireStatus(t, serve(t, h.PostQuery, http.MethodPost, "/query", `{"text":"q"}`),
		http.StatusInternalServerError)

	var got ErrorResponse
	decodeJSON(t, rec, &got)
	if got.Error != "internal server error" {
		t.Errorf("error message = %q, want generic internal server error", got.Error)
	}
}

func TestGetHealth(t *testing.T) {
	h := newTestHandler(&stubIndexer{}, &stubAnswerer{})

	rec := requireStatus(t, serve(t, h.GetHealth, http.MethodGet, "/health", ""),
		http.StatusOK)

	if got := strings.TrimSpace(rec.Body.String()); got != `{"status":"ok"}` {
		t.Errorf("body = %q, want %q", got, `{"status":"ok"}`)
	}
}

func TestRouterRoutes(t *testing.T) {
	router := NewRouter(newTestHandler(
		&stubIndexer{doc: domain.Document{ID: "d"}},
		&stubAnswerer{answer: domain.Answer{Text: "a"}},
	))

	t.Run("health through the router", func(t *testing.T) {
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/health", nil))
		if rec.Code != http.StatusOK {
			t.Fatalf("GET /health status = %d, want 200", rec.Code)
		}
	})

	t.Run("query through the router", func(t *testing.T) {
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/query",
			strings.NewReader(`{"text":"q"}`)))
		if rec.Code != http.StatusOK {
			t.Fatalf("POST /query status = %d, want 200; body: %s", rec.Code, rec.Body.String())
		}
	})

	t.Run("unknown route falls through to 404", func(t *testing.T) {
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/nope", nil))
		if rec.Code != http.StatusNotFound {
			t.Fatalf("GET /nope status = %d, want 404", rec.Code)
		}
	})
}
