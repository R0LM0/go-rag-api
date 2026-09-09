package http

import (
	"fmt"
	"strings"
	"time"

	"github.com/R0LM0/go-rag-api/internal/domain"
)

// TopK bounds for query requests.
const (
	// DefaultTopK is the number of similar chunks retrieved when a query does
	// not specify top_k.
	DefaultTopK = 5
	// MaxTopK caps how many chunks a single query may retrieve.
	MaxTopK = 20
)

// IndexDocumentRequest is the payload of POST /documents. Exactly one of
// (title + content) or path must be present: the pair indexes inline content,
// path delegates loading to the DocumentLoader port.
type IndexDocumentRequest struct {
	Title   string `json:"title"`
	Content string `json:"content"`
	Path    string `json:"path"`
}

// Validate enforces that exactly one of (title + content) or path is present,
// and that both parts of the inline pair are present when it is chosen.
func (r IndexDocumentRequest) Validate() error {
	hasInline := strings.TrimSpace(r.Title) != "" || strings.TrimSpace(r.Content) != ""
	hasPath := strings.TrimSpace(r.Path) != ""

	switch {
	case hasInline && hasPath:
		return fmt.Errorf("provide either title and content, or path, not both")
	case !hasInline && !hasPath:
		return fmt.Errorf("provide either title and content, or path")
	case hasPath:
		return nil
	}
	if strings.TrimSpace(r.Title) == "" || strings.TrimSpace(r.Content) == "" {
		return fmt.Errorf("both title and content are required when indexing inline content")
	}
	return nil
}

// QueryRequest is the payload of POST /query.
type QueryRequest struct {
	Text string `json:"text"`
	TopK int    `json:"top_k"`
}

// Validate checks that text is present and top_k is within [0, MaxTopK]. As a
// documented convenience it also normalizes the request: top_k == 0 (or
// absent from the JSON) is rewritten to DefaultTopK, so the use case always
// receives a valid TopK.
func (r *QueryRequest) Validate() error {
	if strings.TrimSpace(r.Text) == "" {
		return fmt.Errorf("text must not be empty")
	}
	if r.TopK < 0 {
		return fmt.Errorf("top_k must not be negative, got %d", r.TopK)
	}
	if r.TopK > MaxTopK {
		return fmt.Errorf("top_k must be at most %d, got %d", MaxTopK, r.TopK)
	}
	if r.TopK == 0 {
		r.TopK = DefaultTopK
	}
	return nil
}

// toDomain converts a validated and normalized request into a domain Query.
func (r QueryRequest) toDomain() domain.Query {
	return domain.Query{Text: r.Text, TopK: r.TopK}
}

// IndexDocumentResponse is returned by POST /documents on success.
type IndexDocumentResponse struct {
	ID        string    `json:"id"`
	Title     string    `json:"title"`
	CreatedAt time.Time `json:"created_at"`
}

// SourceResponse is one retrieved chunk listed in a QueryResponse.
type SourceResponse struct {
	DocumentID string `json:"document_id"`
	Content    string `json:"content"`
	Position   int    `json:"position"`
}

// QueryResponse is returned by POST /query on success.
type QueryResponse struct {
	Answer  string           `json:"answer"`
	Sources []SourceResponse `json:"sources"`
}

// ErrorResponse is the error envelope used for every non-2xx response.
type ErrorResponse struct {
	Error string `json:"error"`
}

// newIndexDocumentResponse maps a domain document onto the response DTO.
func newIndexDocumentResponse(doc domain.Document) IndexDocumentResponse {
	return IndexDocumentResponse{
		ID:        doc.ID,
		Title:     doc.Title,
		CreatedAt: doc.CreatedAt,
	}
}

// newQueryResponse maps a domain answer onto the response DTO. Sources is
// always a non-nil slice so it encodes as [] rather than null.
func newQueryResponse(answer domain.Answer) QueryResponse {
	sources := make([]SourceResponse, 0, len(answer.Sources))
	for _, chunk := range answer.Sources {
		sources = append(sources, SourceResponse{
			DocumentID: chunk.DocumentID,
			Content:    chunk.Content,
			Position:   chunk.Position,
		})
	}
	return QueryResponse{Answer: answer.Text, Sources: sources}
}
