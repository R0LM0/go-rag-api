package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/R0LM0/go-rag-api/internal/domain/ports"
)

// Compile-time conformance assertions: Client satisfies both ports.
var (
	_ ports.Embedder  = (*Client)(nil)
	_ ports.Generator = (*Client)(nil)
)

// newTestClient builds a Client pointed at handler, mirroring the production
// constructor while keeping the tests free of a real Ollama server.
func newTestClient(t *testing.T, handler http.HandlerFunc) (*Client, *httptest.Server) {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	client, err := NewClient(server.URL, "embed-test-model", "llm-test-model", nil)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	return client, server
}

// embedPayload captures the request the client sends to /api/embed.
type embedPayload struct {
	Model string          `json:"model"`
	Input json.RawMessage `json:"input"`
}

func TestNewClientDefaultsAndValidation(t *testing.T) {
	client, err := NewClient("http://localhost:11434", "embed", "llm", nil)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	if client.httpClient.Timeout != defaultTimeout {
		t.Fatalf("default timeout = %v, want %v", client.httpClient.Timeout, defaultTimeout)
	}
	if _, err := NewClient("://not-a-url", "embed", "llm", nil); err == nil {
		t.Fatal("expected error for unparseable base URL, got nil")
	}
	if _, err := NewClient("ftp://localhost:11434", "embed", "llm", nil); err == nil {
		t.Fatal("expected error for non-http scheme, got nil")
	}
}

func TestEmbed(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/api/embed" {
			t.Errorf("path = %s, want /api/embed", r.URL.Path)
		}
		var payload embedPayload
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if payload.Model != "embed-test-model" {
			t.Errorf("model = %q, want embed-test-model", payload.Model)
		}
		var input string
		if err := json.Unmarshal(payload.Input, &input); err != nil {
			t.Errorf("input is not a string: %v", err)
		}
		if input != "hello world" {
			t.Errorf("input = %q, want %q", input, "hello world")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddings":[[0.1,0.2,0.3]]}`))
	})

	vec, err := client.Embed(context.Background(), "hello world")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vec) != 3 || vec[0] != 0.1 || vec[1] != 0.2 || vec[2] != 0.3 {
		t.Fatalf("Embed returned %v, want [0.1 0.2 0.3]", vec)
	}
}

func TestEmbedEmptyResponseError(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddings":[]}`))
	})
	if _, err := client.Embed(context.Background(), "hello"); err == nil {
		t.Fatal("expected error for empty embeddings response, got nil")
	}
}

func TestEmbedBatchOrderAndCount(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		var payload embedPayload
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode request: %v", err)
		}
		var inputs []string
		if err := json.Unmarshal(payload.Input, &inputs); err != nil {
			t.Errorf("input is not an array: %v", err)
		}
		if len(inputs) != 2 || inputs[0] != "first" || inputs[1] != "second" {
			t.Errorf("inputs = %v, want [first second]", inputs)
		}
		w.Header().Set("Content-Type", "application/json")
		// Distinct vectors prove that order is preserved end to end.
		_, _ = w.Write([]byte(`{"embeddings":[[5,5],[1,1]]}`))
	})

	vecs, err := client.EmbedBatch(context.Background(), []string{"first", "second"})
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	if len(vecs) != 2 {
		t.Fatalf("EmbedBatch returned %d vectors, want 2", len(vecs))
	}
	if len(vecs[0]) != 2 || vecs[0][0] != 5 || vecs[0][1] != 5 {
		t.Fatalf("vecs[0] = %v, want [5 5]", vecs[0])
	}
	if len(vecs[1]) != 2 || vecs[1][0] != 1 || vecs[1][1] != 1 {
		t.Fatalf("vecs[1] = %v, want [1 1]", vecs[1])
	}
}

func TestEmbedBatchEmptyInput(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		t.Error("server must not be called for an empty batch")
	})
	vecs, err := client.EmbedBatch(context.Background(), nil)
	if err != nil {
		t.Fatalf("EmbedBatch(nil): %v", err)
	}
	if len(vecs) != 0 {
		t.Fatalf("EmbedBatch(nil) returned %d vectors, want 0", len(vecs))
	}
}

func TestEmbedBatchCountMismatch(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddings":[[1,2]]}`))
	})
	_, err := client.EmbedBatch(context.Background(), []string{"first", "second"})
	if err == nil {
		t.Fatal("expected count mismatch error, got nil")
	}
	if !strings.Contains(err.Error(), "2 input texts") {
		t.Fatalf("error %q should mention the expected input count", err)
	}
}

func TestGenerate(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			t.Errorf("path = %s, want /api/generate", r.URL.Path)
		}
		var payload struct {
			Model  string `json:"model"`
			Prompt string `json:"prompt"`
			Stream bool   `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if payload.Model != "llm-test-model" {
			t.Errorf("model = %q, want llm-test-model", payload.Model)
		}
		if payload.Prompt != "what is RAG?" {
			t.Errorf("prompt = %q, want %q", payload.Prompt, "what is RAG?")
		}
		if payload.Stream {
			t.Error("stream must be false")
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"response":"retrieval augmented generation"}`))
	})

	answer, err := client.Generate(context.Background(), "what is RAG?")
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if answer != "retrieval augmented generation" {
		t.Fatalf("Generate = %q, want %q", answer, "retrieval augmented generation")
	}
}

func TestNon2xxContainsStatusAndSnippet(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":"model 'missing' not found"}`))
	})
	_, err := client.Embed(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error for 404 response, got nil")
	}
	if !strings.Contains(err.Error(), "404") {
		t.Fatalf("error %q should contain the status code", err)
	}
	if !strings.Contains(err.Error(), "model 'missing' not found") {
		t.Fatalf("error %q should contain a body snippet", err)
	}
}

func TestNon2xxSnippetTruncated(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(strings.Repeat("x", 1000)))
	})
	_, err := client.Embed(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error for 500 response, got nil")
	}
	if got := len(err.Error()); got > 500 {
		t.Fatalf("error message has %d chars, want it bounded well below 1000", got)
	}
}

func TestMalformedJSONResponse(t *testing.T) {
	client, _ := newTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddings": [`))
	})
	_, err := client.Embed(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error for malformed JSON, got nil")
	}
	if !strings.Contains(err.Error(), "decode") {
		t.Fatalf("error %q should mention decoding", err)
	}
}

func TestUnreachableHost(t *testing.T) {
	client, err := NewClient("http://127.0.0.1:1", "embed", "llm", &http.Client{Timeout: 2 * time.Second})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	if _, err := client.Embed(context.Background(), "hello"); err == nil {
		t.Fatal("expected error for unreachable host, got nil")
	} else if !strings.Contains(err.Error(), "POST") {
		t.Fatalf("error %q should mention the failed POST", err)
	}
	if _, err := client.Generate(context.Background(), "hello"); err == nil {
		t.Fatal("expected error for unreachable host on Generate, got nil")
	}
}
