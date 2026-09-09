// Package domain contains the core entities and errors of the RAG API.
// It depends only on the Go standard library and defines the vocabulary
// shared by the application core and the outer adapters.
package domain

import "time"

// Document is an ingested source text identified by its title and content.
type Document struct {
	// ID is the unique identifier of the document.
	ID string
	// Title is the human-readable name of the document.
	Title string
	// Content is the full raw text of the document.
	Content string
	// CreatedAt is the UTC timestamp recorded when the document was indexed.
	CreatedAt time.Time
}

// Chunk is a piece of a Document prepared for embedding and retrieval.
type Chunk struct {
	// ID is the unique identifier of the chunk.
	ID string
	// DocumentID references the Document the chunk was produced from.
	DocumentID string
	// Content is the chunk text used as embedding input and prompt context.
	Content string
	// Position is the zero-based order of the chunk within its document.
	Position int
	// Embedding is the dense vector representation of Content.
	Embedding []float32
}

// Query is a retrieval-augmented question.
type Query struct {
	// Text is the natural-language question.
	Text string
	// TopK is the maximum number of similar chunks to retrieve.
	TopK int
}

// Answer is the response produced for a Query.
type Answer struct {
	// Text is the generated natural-language answer.
	Text string
	// Sources lists the chunks used as context for the answer.
	Sources []Chunk
}
