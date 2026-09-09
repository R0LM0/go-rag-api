// Package http implements the inbound (driving) adapter of the hexagonal RAG
// API: it exposes the application use cases over JSON/HTTP. Mirroring the
// ports philosophy of the core, this package defines its own narrow
// consumer-side interfaces for the use cases instead of importing concrete
// application types — the consumer owns the interface it needs.
package http

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"

	"github.com/R0LM0/go-rag-api/internal/adapters/outbound/loader"
	"github.com/R0LM0/go-rag-api/internal/domain"
)

// documentIndexer is the narrow use-case interface POST /documents needs.
type documentIndexer interface {
	IndexContent(ctx context.Context, title, content string) (domain.Document, error)
	IndexFile(ctx context.Context, path string) (domain.Document, error)
}

// queryAnswerer is the narrow use-case interface POST /query needs.
type queryAnswerer interface {
	Execute(ctx context.Context, query domain.Query) (domain.Answer, error)
}

// Handler serves the three HTTP endpoints of the RAG API.
type Handler struct {
	indexer  documentIndexer
	answerer queryAnswerer
}

// NewHandler builds a Handler. Handing the concrete use cases to this
// constructor is the compile-time assertion that they satisfy the narrow
// interfaces above; this package itself never imports the application package.
func NewHandler(indexer documentIndexer, answerer queryAnswerer) *Handler {
	return &Handler{indexer: indexer, answerer: answerer}
}

// PostDocument handles POST /documents: it indexes inline content
// (title + content) or a file (path) through the indexing use case.
func (h *Handler) PostDocument(w http.ResponseWriter, r *http.Request) {
	var req IndexDocumentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	if err := req.Validate(); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	var (
		doc domain.Document
		err error
	)
	if req.Path != "" {
		doc, err = h.indexer.IndexFile(r.Context(), req.Path)
	} else {
		doc, err = h.indexer.IndexContent(r.Context(), req.Title, req.Content)
	}
	if err != nil {
		writeIndexingError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, newIndexDocumentResponse(doc))
}

// writeIndexingError maps known sentinel errors to 400 responses carrying the
// sentinel message, and everything else to a generic 500 that leaks no
// internals. Note the deliberate boundary exception: the outbound loader
// sentinel ErrUnsupportedFileType is mapped here so clients receive an
// actionable 400 for bad file types instead of an opaque 500.
func writeIndexingError(w http.ResponseWriter, err error) {
	for _, sentinel := range []error{
		domain.ErrEmptyDocumentTitle,
		domain.ErrEmptyDocumentContent,
		domain.ErrNoChunksGenerated,
		loader.ErrUnsupportedFileType,
	} {
		if errors.Is(err, sentinel) {
			writeError(w, http.StatusBadRequest, sentinel.Error())
			return
		}
	}
	writeError(w, http.StatusInternalServerError, "internal server error")
}

// PostQuery handles POST /query: it answers a question with
// retrieval-augmented generation through the query use case.
func (h *Handler) PostQuery(w http.ResponseWriter, r *http.Request) {
	var req QueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error())
		return
	}
	if err := req.Validate(); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	answer, err := h.answerer.Execute(r.Context(), req.toDomain())
	if err != nil {
		writeQueryError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, newQueryResponse(answer))
}

// writeQueryError maps query sentinels to 400 responses carrying the sentinel
// message, and everything else to a generic 500.
func writeQueryError(w http.ResponseWriter, err error) {
	for _, sentinel := range []error{
		domain.ErrEmptyQueryText,
		domain.ErrInvalidTopK,
	} {
		if errors.Is(err, sentinel) {
			writeError(w, http.StatusBadRequest, sentinel.Error())
			return
		}
	}
	writeError(w, http.StatusInternalServerError, "internal server error")
}

// GetHealth handles GET /health with a static liveness response.
func (h *Handler) GetHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, struct {
		Status string `json:"status"`
	}{"ok"})
}

// writeJSON writes v as a JSON response body with the given status code.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		// The status line is already written; nothing more can be sent to
		// the client. Encoding these simple DTOs cannot realistically fail.
		_ = err
	}
}

// writeError writes msg inside the standard error envelope.
func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, ErrorResponse{Error: msg})
}
