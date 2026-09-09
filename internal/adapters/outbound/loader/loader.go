// Package loader implements the DocumentLoader port for local text files.
package loader

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/R0LM0/go-rag-api/internal/domain"
	"github.com/R0LM0/go-rag-api/internal/domain/ports"
)

// ErrUnsupportedFileType is returned, wrapped, when Load is called with a
// path whose extension is not supported (.txt or .md).
var ErrUnsupportedFileType = errors.New("unsupported file type")

// supportedExtensions lists the file types FileLoader accepts, matched
// case-insensitively.
var supportedExtensions = map[string]bool{
	".txt": true,
	".md":  true,
}

// FileLoader reads .txt and .md documents from the local filesystem. Its
// zero value is ready to use.
type FileLoader struct{}

// Compile-time conformance assertion: FileLoader satisfies the
// DocumentLoader port.
var _ ports.DocumentLoader = (*FileLoader)(nil)

// Load reads the file at path and returns it as a Document. The Title is the
// base filename without its extension. ID stays empty because the
// application core owns identifier generation, and CreatedAt stays zero
// because the core owns timestamping at index time.
//
// Boundary decision: an empty file loads successfully with empty Content.
// Validating content emptiness (domain.ErrEmptyDocumentContent) is the
// application core's job, not the loader's.
func (FileLoader) Load(ctx context.Context, path string) (domain.Document, error) {
	rawExt := filepath.Ext(path)
	ext := strings.ToLower(rawExt)
	if !supportedExtensions[ext] {
		return domain.Document{}, fmt.Errorf("loader: %s: %w", path, ErrUnsupportedFileType)
	}
	if err := ctx.Err(); err != nil {
		return domain.Document{}, fmt.Errorf("loader: load %s canceled before reading: %w", path, err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return domain.Document{}, fmt.Errorf("loader: read file %s: %w", path, err)
	}
	return domain.Document{
		Title:   strings.TrimSuffix(filepath.Base(path), rawExt),
		Content: string(data),
	}, nil
}
