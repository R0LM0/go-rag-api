package application

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// modulePath is this module's import path prefix.
const modulePath = "github.com/R0LM0/go-rag-api"

// TestCorePackagesDependOnlyOnStdlib runs `go list -deps` over the domain and
// application packages and fails if any dependency resolves outside the Go
// standard library or this module (for example chi, pgx, pgvector, or an
// Ollama client). This keeps the hexagonal core free of third-party imports.
func TestCorePackagesDependOnlyOnStdlib(t *testing.T) {
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("cannot determine the test file location")
	}
	repoRoot := filepath.Dir(filepath.Dir(filepath.Dir(thisFile)))

	patterns := []string{
		"./internal/domain/...",
		"./internal/application/...",
	}
	banned := []string{"chi", "pgx", "pgvector", "ollama"}

	for _, pattern := range patterns {
		cmd := exec.Command("go", "list", "-deps", pattern)
		cmd.Dir = repoRoot
		out, err := cmd.Output()
		if err != nil {
			if exitErr, ok := err.(*exec.ExitError); ok {
				t.Fatalf("go list -deps %s failed: %v: %s", pattern, err, exitErr.Stderr)
			}
			t.Fatalf("go list -deps %s failed: %v", pattern, err)
		}

		for _, rawLine := range strings.Split(string(out), "\n") {
			line := strings.TrimSpace(rawLine)
			if line == "" {
				continue
			}
			if line == modulePath || strings.HasPrefix(line, modulePath+"/") {
				continue
			}
			for _, name := range banned {
				if strings.Contains(line, name) {
					t.Errorf("%s depends on banned package %q", pattern, line)
				}
			}
			// Standard library import paths have no dot in their first path
			// segment ("fmt", "internal/cpu", "vendor/golang.org/x/..."),
			// while external modules do ("github.com/...").
			firstSegment := line
			if idx := strings.Index(line, "/"); idx >= 0 {
				firstSegment = line[:idx]
			}
			if strings.Contains(firstSegment, ".") {
				t.Errorf("%s depends on non-stdlib package %q; the core must use only the standard library", pattern, line)
			}
		}
	}
}
