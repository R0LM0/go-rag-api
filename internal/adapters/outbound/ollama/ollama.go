// Package ollama implements the Embedder and Generator ports against a
// locally running Ollama server over plain HTTP (no authentication).
package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/R0LM0/go-rag-api/internal/domain/ports"
)

// defaultTimeout is the HTTP timeout used when the caller does not provide
// an *http.Client. Local LLM generation is slow, so the default is generous.
const defaultTimeout = 120 * time.Second

// maxErrorSnippet caps the response body snippet included in error messages
// for non-2xx responses, keeping logs readable even for large payloads.
const maxErrorSnippet = 200

// Client talks to an Ollama server and implements ports.Embedder and
// ports.Generator for the embedding and generation models configured at
// construction time.
type Client struct {
	baseURL    *url.URL
	httpClient *http.Client
	embedModel string
	llmModel   string
}

// Compile-time conformance assertions: Client satisfies both ports.
var (
	_ ports.Embedder  = (*Client)(nil)
	_ ports.Generator = (*Client)(nil)
)

// NewClient creates a Client for the Ollama server at baseURL. baseURL must
// parse into a valid http or https URL. A nil httpClient is replaced by a
// default client with a 120s timeout, sized for slow local generation.
func NewClient(baseURL, embedModel, llmModel string, httpClient *http.Client) (*Client, error) {
	parsed, err := url.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("ollama: parse base URL %q: %w", baseURL, err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, fmt.Errorf("ollama: base URL %q must use scheme http or https", baseURL)
	}
	if httpClient == nil {
		httpClient = &http.Client{Timeout: defaultTimeout}
	}
	return &Client{
		baseURL:    parsed,
		httpClient: httpClient,
		embedModel: embedModel,
		llmModel:   llmModel,
	}, nil
}

// embedRequest is the payload for the POST /api/embed endpoint. Ollama
// accepts "input" either as a single string or as an array of strings.
type embedRequest struct {
	Model string `json:"model"`
	Input any    `json:"input"`
}

// embedResponse is the payload returned by POST /api/embed.
type embedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

// generateRequest is the payload for the POST /api/generate endpoint.
type generateRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

// generateResponse is the payload returned by POST /api/generate when
// streaming is disabled.
type generateResponse struct {
	Response string `json:"response"`
}

// Embed returns the embedding vector for a single text.
func (c *Client) Embed(ctx context.Context, text string) ([]float32, error) {
	var resp embedResponse
	if err := c.postJSON(ctx, "api/embed", &embedRequest{Model: c.embedModel, Input: text}, &resp); err != nil {
		return nil, err
	}
	if len(resp.Embeddings) == 0 {
		return nil, fmt.Errorf("ollama: model %s returned no embeddings", c.embedModel)
	}
	return resp.Embeddings[0], nil
}

// EmbedBatch returns one embedding per input text, preserving input order.
// An empty input returns an empty slice without contacting the server.
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	var resp embedResponse
	if err := c.postJSON(ctx, "api/embed", &embedRequest{Model: c.embedModel, Input: texts}, &resp); err != nil {
		return nil, err
	}
	if len(resp.Embeddings) != len(texts) {
		return nil, fmt.Errorf("ollama: model %s returned %d embeddings for %d input texts", c.embedModel, len(resp.Embeddings), len(texts))
	}
	return resp.Embeddings, nil
}

// Generate returns the generated text for a single prompt with streaming
// disabled.
func (c *Client) Generate(ctx context.Context, prompt string) (string, error) {
	var resp generateResponse
	if err := c.postJSON(ctx, "api/generate", &generateRequest{Model: c.llmModel, Prompt: prompt, Stream: false}, &resp); err != nil {
		return "", err
	}
	return resp.Response, nil
}

// postJSON marshals payload as JSON, POSTs it to the endpoint resolved from
// path against the configured base URL, and decodes a 2xx response body
// into out. Non-2xx statuses and malformed JSON become wrapped errors.
func (c *Client) postJSON(ctx context.Context, path string, payload, out any) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("ollama: marshal request for %s: %w", path, err)
	}
	endpoint := c.baseURL.JoinPath(path)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint.String(), bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("ollama: build request for %s: %w", endpoint, err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("ollama: POST %s: %w", endpoint, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("ollama: read response from %s: %w", endpoint, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return fmt.Errorf("ollama: POST %s returned status %d: %s", endpoint, resp.StatusCode, snippet(data))
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("ollama: decode response from %s: %w", endpoint, err)
	}
	return nil
}

// snippet returns a whitespace-trimmed, truncated view of a response body
// for inclusion in error messages.
func snippet(body []byte) string {
	s := strings.TrimSpace(string(body))
	if len(s) > maxErrorSnippet {
		return s[:maxErrorSnippet] + "..."
	}
	return s
}
