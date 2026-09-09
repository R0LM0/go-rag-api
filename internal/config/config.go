// Package config loads the service configuration from environment variables
// using only the standard library. A .env file is deliberately not read here:
// it is exported into the environment by the Makefile or by the user's shell.
package config

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

// Defaults applied when the corresponding variable is unset.
const (
	defaultEmbedDims = 1024
	defaultHTTPPort  = 8080
)

// Config holds every setting the composition root needs to wire the adapters.
type Config struct {
	// DatabaseURL is the PostgreSQL connection string (with pgvector).
	DatabaseURL string
	// OllamaURL is the base URL of the local Ollama server.
	OllamaURL string
	// EmbedModel is the Ollama model used for embeddings.
	EmbedModel string
	// LLMModel is the Ollama model used for answer generation.
	LLMModel string
	// EmbedDims is the embedding dimension; it must match the vector(1024)
	// column defined in migrations/000001_init.up.sql.
	EmbedDims int
	// HTTPPort is the port the API server listens on.
	HTTPPort int
}

// Load reads configuration from the environment. The four string variables
// are required and must be non-empty; the integer variables fall back to
// their defaults when unset. All missing or invalid variables are reported
// together in a single joined error so operators see the full picture in one
// run instead of fixing them one at a time.
func Load() (Config, error) {
	var errs []error

	cfg := Config{
		DatabaseURL: stringVar("DATABASE_URL", &errs),
		OllamaURL:   stringVar("OLLAMA_URL", &errs),
		EmbedModel:  stringVar("EMBED_MODEL", &errs),
		LLMModel:    stringVar("LLM_MODEL", &errs),
	}

	var err error
	if cfg.EmbedDims, err = intVar("EMBED_DIMS", defaultEmbedDims, 1, math.MaxInt32); err != nil {
		errs = append(errs, err)
	}
	if cfg.HTTPPort, err = intVar("HTTP_PORT", defaultHTTPPort, 1, 65535); err != nil {
		errs = append(errs, err)
	}

	if len(errs) > 0 {
		return Config{}, fmt.Errorf("invalid configuration: %w", errors.Join(errs...))
	}
	return cfg, nil
}

// stringVar returns the value of the required environment variable name,
// appending an error to errs when it is unset or blank.
func stringVar(name string, errs *[]error) string {
	value, ok := os.LookupEnv(name)
	if !ok || strings.TrimSpace(value) == "" {
		*errs = append(*errs, fmt.Errorf("%s must be set to a non-empty value", name))
		return ""
	}
	return value
}

// intVar parses name as an integer within [min, max] and returns fallback
// when the variable is unset. A variable that is set but not an integer (an
// empty string included) or outside the range is an error.
func intVar(name string, fallback, min, max int) (int, error) {
	raw, ok := os.LookupEnv(name)
	if !ok {
		return fallback, nil
	}
	value, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil {
		return 0, fmt.Errorf("%s must be an integer, got %q", name, raw)
	}
	if value < min || value > max {
		return 0, fmt.Errorf("%s must be between %d and %d, got %d", name, min, max, value)
	}
	return value, nil
}
