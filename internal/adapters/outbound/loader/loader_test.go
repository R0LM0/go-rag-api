package loader

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/R0LM0/go-rag-api/internal/domain/ports"
)

// Compile-time conformance assertion: FileLoader satisfies the port.
var _ ports.DocumentLoader = (*FileLoader)(nil)

// writeFile creates a file inside dir with the given name and content.
func writeFile(t *testing.T, dir, name, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
	return path
}

func TestLoadTXT(t *testing.T) {
	path := writeFile(t, t.TempDir(), "notes.txt", "plain text body")

	doc, err := (FileLoader{}).Load(context.Background(), path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if doc.Title != "notes" {
		t.Errorf("Title = %q, want %q", doc.Title, "notes")
	}
	if doc.Content != "plain text body" {
		t.Errorf("Content = %q, want %q", doc.Content, "plain text body")
	}
	// ID and CreatedAt belong to the application core, not the loader.
	if doc.ID != "" {
		t.Errorf("ID = %q, want empty", doc.ID)
	}
	if !doc.CreatedAt.IsZero() {
		t.Errorf("CreatedAt = %v, want zero value", doc.CreatedAt)
	}
}

func TestLoadMD(t *testing.T) {
	path := writeFile(t, t.TempDir(), "design-doc.md", "# heading\n\nbody")

	doc, err := (FileLoader{}).Load(context.Background(), path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if doc.Title != "design-doc" {
		t.Errorf("Title = %q, want %q (base filename without extension)", doc.Title, "design-doc")
	}
	if doc.Content != "# heading\n\nbody" {
		t.Errorf("Content = %q, want the raw file body", doc.Content)
	}
}

func TestLoadUnsupportedExtension(t *testing.T) {
	path := writeFile(t, t.TempDir(), "archive.bin", "bytes")

	_, err := (FileLoader{}).Load(context.Background(), path)
	if !errors.Is(err, ErrUnsupportedFileType) {
		t.Fatalf("Load error = %v, want ErrUnsupportedFileType", err)
	}
	if !strings.Contains(err.Error(), path) {
		t.Errorf("error %v should contain the offending path %s", err, path)
	}
}

func TestLoadMissingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "does-not-exist.txt")

	_, err := (FileLoader{}).Load(context.Background(), path)
	if err == nil {
		t.Fatal("expected error for missing file, got nil")
	}
	if errors.Is(err, ErrUnsupportedFileType) {
		t.Fatalf("missing file must not report an unsupported type: %v", err)
	}
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("error = %v, want it to wrap os.ErrNotExist", err)
	}
}

func TestLoadEmptyFileStillLoads(t *testing.T) {
	// Boundary decision: the loader does not validate content emptiness.
	// The application core (domain.ErrEmptyDocumentContent) owns that rule,
	// so an empty file must still load successfully.
	path := writeFile(t, t.TempDir(), "empty.txt", "")

	doc, err := (FileLoader{}).Load(context.Background(), path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if doc.Content != "" {
		t.Errorf("Content = %q, want empty", doc.Content)
	}
	if doc.Title != "empty" {
		t.Errorf("Title = %q, want %q", doc.Title, "empty")
	}
}
