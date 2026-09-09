// Package application implements the use cases of the RAG API on top of the
// domain model and the ports defined in internal/domain/ports. It contains no
// adapter code: HTTP handlers, database access, and provider clients are
// added in later phases around this core.
package application

import (
	"fmt"
	"strings"

	"github.com/R0LM0/go-rag-api/internal/domain"
)

// charsPerToken is the denominator of the chars-per-token heuristic used to
// estimate token counts without a tokenizer dependency.
const charsPerToken = 4

// chunkSeparator joins the paragraph units accumulated inside a single chunk.
const chunkSeparator = "\n\n"

// ChunkConfig controls how documents are split into chunks.
type ChunkConfig struct {
	// MaxTokens is the maximum estimated token size of a chunk.
	MaxTokens int
	// OverlapTokens is the approximate number of tokens from the end of a
	// chunk repeated at the start of the next chunk.
	OverlapTokens int
}

// DefaultChunkConfig returns the default chunking configuration: at most 500
// estimated tokens per chunk with approximately 50 tokens of overlap between
// consecutive chunks.
func DefaultChunkConfig() ChunkConfig {
	return ChunkConfig{MaxTokens: 500, OverlapTokens: 50}
}

// estimateTokens approximates the token count of text as len(text)/4.
//
// This chars-per-token heuristic is a deliberate trade-off: it keeps the core
// free of tokenizer dependencies (standard library only) at the cost of being
// an approximation, most accurate for English prose and less accurate for
// languages or content with different character-to-token ratios.
func estimateTokens(text string) int {
	return len(text) / charsPerToken
}

// maxChars returns the character budget of a chunk: the maximum number of
// characters whose estimated token count still fits cfg.MaxTokens.
func maxChars(cfg ChunkConfig) int {
	return cfg.MaxTokens * charsPerToken
}

// normalizeChunkConfig replaces invalid field values with safe defaults so a
// zero-value or user-supplied config can never make the chunker panic,
// produce empty chunks, or loop forever.
func normalizeChunkConfig(cfg ChunkConfig) ChunkConfig {
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = DefaultChunkConfig().MaxTokens
	}
	if cfg.OverlapTokens < 0 {
		cfg.OverlapTokens = 0
	}
	if cfg.OverlapTokens >= cfg.MaxTokens {
		cfg.OverlapTokens = cfg.MaxTokens / 2
	}
	return cfg
}

// SplitIntoChunks splits content into ordered chunks for embedding.
//
// Content is split into paragraphs on blank lines (both \n and \r\n line
// endings are handled); paragraphs are accumulated until adding the next one
// would exceed cfg.MaxTokens estimated tokens; a single paragraph larger than
// the limit is hard-split at the character limit; and every chunk after the
// first starts with an overlap tail of approximately cfg.OverlapTokens
// estimated tokens taken from the end of the previous chunk (shrunk when
// needed so no chunk ever exceeds the limit). Positions start at zero and
// increase sequentially.
func SplitIntoChunks(docID string, content string, cfg ChunkConfig) ([]domain.Chunk, error) {
	cfg = normalizeChunkConfig(cfg)
	if strings.TrimSpace(content) == "" {
		return nil, fmt.Errorf("chunking document %q: %w", docID, domain.ErrNoChunksGenerated)
	}

	limit := maxChars(cfg)
	units := splitParagraphs(content, limit)

	var texts []string
	var current []string
	currentLen := 0

	flush := func() {
		if len(current) == 0 {
			return
		}
		texts = append(texts, strings.Join(current, chunkSeparator))
		current = current[:0]
		currentLen = 0
	}

	separatorLen := len(chunkSeparator)
	for _, unit := range units {
		separatorExtra := 0
		if len(current) > 0 {
			separatorExtra = separatorLen
		}
		if len(current) > 0 && currentLen+separatorExtra+len(unit) > limit {
			previous := strings.Join(current, chunkSeparator)
			flush()
			if tail := overlapTail(previous, unit, cfg.OverlapTokens*charsPerToken, limit); tail != "" {
				current = append(current, tail)
				currentLen = len(tail)
				separatorExtra = separatorLen
			} else {
				separatorExtra = 0
			}
		}
		current = append(current, unit)
		currentLen += separatorExtra + len(unit)
	}
	flush()

	if len(texts) == 0 {
		return nil, fmt.Errorf("chunking document %q: %w", docID, domain.ErrNoChunksGenerated)
	}

	chunks := make([]domain.Chunk, len(texts))
	for i, text := range texts {
		id, err := newID()
		if err != nil {
			return nil, fmt.Errorf("chunking document %q: chunk %d: %w", docID, i, err)
		}
		chunks[i] = domain.Chunk{
			ID:         id,
			DocumentID: docID,
			Content:    text,
			Position:   i,
		}
	}
	return chunks, nil
}

// splitParagraphs returns the content split into paragraph units: the text is
// first split into paragraphs on blank lines (whitespace-only lines), each
// paragraph is trimmed, and any paragraph longer than limit characters is
// hard-split into consecutive pieces of at most limit characters.
func splitParagraphs(content string, limit int) []string {
	normalized := strings.ReplaceAll(content, "\r\n", "\n")
	normalized = strings.ReplaceAll(normalized, "\r", "\n")

	var paragraphs []string
	var lines []string
	flushParagraph := func() {
		text := strings.TrimSpace(strings.Join(lines, "\n"))
		lines = lines[:0]
		if text != "" {
			paragraphs = append(paragraphs, text)
		}
	}
	for _, line := range strings.Split(normalized, "\n") {
		if strings.TrimSpace(line) == "" {
			flushParagraph()
			continue
		}
		lines = append(lines, line)
	}
	flushParagraph()

	units := make([]string, 0, len(paragraphs))
	for _, paragraph := range paragraphs {
		if len(paragraph) <= limit {
			units = append(units, paragraph)
			continue
		}
		for start := 0; start < len(paragraph); start += limit {
			end := start + limit
			if end > len(paragraph) {
				end = len(paragraph)
			}
			piece := strings.TrimSpace(paragraph[start:end])
			if piece != "" {
				units = append(units, piece)
			}
		}
	}
	return units
}

// overlapTail returns the string to prepend to the next chunk: the last
// overlapChars characters of previous, shrunk when necessary so the overlap
// plus the upcoming unit still fits within limit characters.
func overlapTail(previous string, next string, overlapChars int, limit int) string {
	if overlapChars <= 0 || previous == "" {
		return ""
	}
	budget := limit - len(next) - len(chunkSeparator)
	if budget <= 0 {
		return ""
	}
	n := overlapChars
	if n > budget {
		n = budget
	}
	if len(previous) <= n {
		return previous
	}
	return previous[len(previous)-n:]
}
