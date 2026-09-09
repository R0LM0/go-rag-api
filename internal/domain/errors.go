package domain

import "errors"

// ErrEmptyDocumentTitle is returned when a document is created without a title.
var ErrEmptyDocumentTitle = errors.New("document title must not be empty")

// ErrEmptyDocumentContent is returned when a document is created without content.
var ErrEmptyDocumentContent = errors.New("document content must not be empty")

// ErrDocumentNotFound is returned when a referenced document does not exist.
var ErrDocumentNotFound = errors.New("document not found")

// ErrEmptyQueryText is returned when a query is executed without text.
var ErrEmptyQueryText = errors.New("query text must not be empty")

// ErrInvalidTopK is returned when a query requests a non-positive number of results.
var ErrInvalidTopK = errors.New("topK must be greater than zero")

// ErrNoChunksGenerated is returned when chunking produces no chunks from the given content.
var ErrNoChunksGenerated = errors.New("no chunks generated from content")
