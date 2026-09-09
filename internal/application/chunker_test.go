package application

import (
	"errors"
	"strings"
	"testing"

	"github.com/R0LM0/go-rag-api/internal/domain"
)

func TestSplitIntoChunks(t *testing.T) {
	tests := []struct {
		name     string
		docID    string
		content  string
		cfg      ChunkConfig
		wantErr  error
		validate func(t *testing.T, chunks []domain.Chunk)
	}{
		{
			name:     "empty content returns ErrNoChunksGenerated",
			docID:    "doc-1",
			content:  "",
			cfg:      DefaultChunkConfig(),
			wantErr:  domain.ErrNoChunksGenerated,
			validate: nil,
		},
		{
			name:     "whitespace-only content returns ErrNoChunksGenerated",
			docID:    "doc-1",
			content:  " \n\t \n",
			cfg:      DefaultChunkConfig(),
			wantErr:  domain.ErrNoChunksGenerated,
			validate: nil,
		},
		{
			name:    "single short paragraph produces one chunk",
			docID:   "doc-1",
			content: "Hello world",
			cfg:     DefaultChunkConfig(),
			wantErr: nil,
			validate: func(t *testing.T, chunks []domain.Chunk) {
				if len(chunks) != 1 {
					t.Fatalf("chunks = %d, want 1", len(chunks))
				}
				if chunks[0].Content != "Hello world" {
					t.Errorf("content = %q, want %q", chunks[0].Content, "Hello world")
				}
				if chunks[0].Position != 0 {
					t.Errorf("position = %d, want 0", chunks[0].Position)
				}
				if chunks[0].DocumentID != "doc-1" {
					t.Errorf("document id = %q, want %q", chunks[0].DocumentID, "doc-1")
				}
				if chunks[0].ID == "" {
					t.Error("chunk ID must not be empty")
				}
			},
		},
		{
			name:    "multi paragraph under limit joins into one chunk",
			docID:   "doc-1",
			content: "first paragraph\n\nsecond paragraph",
			cfg:     DefaultChunkConfig(),
			wantErr: nil,
			validate: func(t *testing.T, chunks []domain.Chunk) {
				if len(chunks) != 1 {
					t.Fatalf("chunks = %d, want 1", len(chunks))
				}
				want := "first paragraph\n\nsecond paragraph"
				if chunks[0].Content != want {
					t.Errorf("content = %q, want %q", chunks[0].Content, want)
				}
			},
		},
		{
			name:    "windows line endings are normalized",
			docID:   "doc-1",
			content: "para one\r\n\r\npara two",
			cfg:     DefaultChunkConfig(),
			wantErr: nil,
			validate: func(t *testing.T, chunks []domain.Chunk) {
				if len(chunks) != 1 {
					t.Fatalf("chunks = %d, want 1", len(chunks))
				}
				if strings.Contains(chunks[0].Content, "\r") {
					t.Errorf("chunk still contains \\r: %q", chunks[0].Content)
				}
				want := "para one\n\npara two"
				if chunks[0].Content != want {
					t.Errorf("content = %q, want %q", chunks[0].Content, want)
				}
			},
		},
		{
			name:    "split respects max token limit",
			docID:   "doc-1",
			content: strings.Join([]string{"aaaaaaaa", "bbbbbbbb", "cccccccc", "dddddddd", "eeeeeeee", "ffffffff"}, "\n\n"),
			cfg:     ChunkConfig{MaxTokens: 5, OverlapTokens: 0},
			wantErr: nil,
			validate: func(t *testing.T, chunks []domain.Chunk) {
				if len(chunks) != 3 {
					t.Fatalf("chunks = %d, want 3", len(chunks))
				}
				wantContents := []string{
					"aaaaaaaa\n\nbbbbbbbb",
					"cccccccc\n\ndddddddd",
					"eeeeeeee\n\nffffffff",
				}
				for i, chunk := range chunks {
					if chunk.Content != wantContents[i] {
						t.Errorf("chunk %d content = %q, want %q", i, chunk.Content, wantContents[i])
					}
					if got := estimateTokens(chunk.Content); got > 5 {
						t.Errorf("chunk %d estimated tokens = %d, want <= 5", i, got)
					}
				}
			},
		},
		{
			name:    "overlap repeats tail of previous chunk",
			docID:   "doc-1",
			content: strings.Join([]string{strings.Repeat("A", 24), "BBBBBBBB"}, "\n\n"),
			cfg:     ChunkConfig{MaxTokens: 6, OverlapTokens: 2},
			wantErr: nil,
			validate: func(t *testing.T, chunks []domain.Chunk) {
				if len(chunks) != 2 {
					t.Fatalf("chunks = %d, want 2", len(chunks))
				}
				first := chunks[0].Content
				second := chunks[1].Content
				if first != strings.Repeat("A", 24) {
					t.Errorf("first chunk = %q, want 24 As", first)
				}
				// OverlapTokens(2) * charsPerToken(4) = 8 characters of tail.
				tail := strings.Repeat("A", 8)
				if !strings.HasPrefix(second, tail) {
					t.Errorf("second chunk %q must start with overlap tail %q", second, tail)
				}
				if !strings.Contains(second, "BBBBBBBB") {
					t.Errorf("second chunk %q must contain the new paragraph", second)
				}
			},
		},
		{
			name:    "oversized paragraph is hard split at char limit",
			docID:   "doc-1",
			content: strings.Repeat("x", 100),
			cfg:     ChunkConfig{MaxTokens: 5, OverlapTokens: 0},
			wantErr: nil,
			validate: func(t *testing.T, chunks []domain.Chunk) {
				if len(chunks) != 5 {
					t.Fatalf("chunks = %d, want 5", len(chunks))
				}
				for i, chunk := range chunks {
					if chunk.Content != strings.Repeat("x", 20) {
						t.Errorf("chunk %d content = %q, want 20 x characters", i, chunk.Content)
					}
				}
			},
		},
		{
			name:    "positions are sequential starting at zero",
			docID:   "doc-1",
			content: strings.Join([]string{"p1", "p2", "p3", "p4", "p5", "p6"}, "\n\n"),
			cfg:     ChunkConfig{MaxTokens: 1, OverlapTokens: 0},
			wantErr: nil,
			validate: func(t *testing.T, chunks []domain.Chunk) {
				if len(chunks) != 6 {
					t.Fatalf("chunks = %d, want 6", len(chunks))
				}
				seen := make(map[string]bool, len(chunks))
				for i, chunk := range chunks {
					if chunk.Position != i {
						t.Errorf("chunk %d position = %d, want %d", i, chunk.Position, i)
					}
					if chunk.ID == "" {
						t.Errorf("chunk %d ID must not be empty", i)
					}
					if seen[chunk.ID] {
						t.Errorf("chunk %d reuses ID %q", i, chunk.ID)
					}
					seen[chunk.ID] = true
				}
			},
		},
		{
			name:    "zero config falls back to defaults",
			docID:   "doc-1",
			content: "some short content",
			cfg:     ChunkConfig{},
			wantErr: nil,
			validate: func(t *testing.T, chunks []domain.Chunk) {
				if len(chunks) != 1 {
					t.Fatalf("chunks = %d, want 1", len(chunks))
				}
				if chunks[0].Content != "some short content" {
					t.Errorf("content = %q, want %q", chunks[0].Content, "some short content")
				}
			},
		},
		{
			name:    "negative overlap is treated as zero",
			docID:   "doc-1",
			content: "a\n\nb\nc",
			cfg:     ChunkConfig{MaxTokens: 2, OverlapTokens: -5},
			wantErr: nil,
			validate: func(t *testing.T, chunks []domain.Chunk) {
				if len(chunks) != 1 {
					t.Fatalf("chunks = %d, want 1", len(chunks))
				}
				want := "a\n\nb\nc"
				if chunks[0].Content != want {
					t.Errorf("content = %q, want %q", chunks[0].Content, want)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks, err := SplitIntoChunks(tt.docID, tt.content, tt.cfg)
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("SplitIntoChunks() error = %v, want %v", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("SplitIntoChunks() error = %v, want nil", err)
			}
			if tt.validate == nil {
				t.Fatal("test case has no validator")
			}
			tt.validate(t, chunks)
		})
	}
}
