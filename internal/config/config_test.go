package config

import (
	"strings"
	"testing"
)

// setRequired sets the four required string variables to valid values so each
// table row can start from a known-good baseline and clear only what it wants
// to exercise.
func setRequired(t *testing.T) {
	t.Helper()
	t.Setenv("DATABASE_URL", "postgres://user:pass@localhost:5432/rag")
	t.Setenv("OLLAMA_URL", "http://localhost:11434")
	t.Setenv("EMBED_MODEL", "mxbai-embed-large")
	t.Setenv("LLM_MODEL", "llama3.1")
}

func TestLoad(t *testing.T) {
	tests := []struct {
		name  string
		setup func(t *testing.T)
		want  Config
		// wantErr marks rows expected to fail; wantMsgs lists substrings that
		// must all appear in the joined error message.
		wantErr  bool
		wantMsgs []string
	}{
		{
			name: "all variables set",
			setup: func(t *testing.T) {
				setRequired(t)
				t.Setenv("EMBED_DIMS", "768")
				t.Setenv("HTTP_PORT", "9090")
			},
			want: Config{
				DatabaseURL: "postgres://user:pass@localhost:5432/rag",
				OllamaURL:   "http://localhost:11434",
				EmbedModel:  "mxbai-embed-large",
				LLMModel:    "llama3.1",
				EmbedDims:   768,
				HTTPPort:    9090,
			},
		},
		{
			name:  "integer defaults applied when unset",
			setup: setRequired,
			want: Config{
				DatabaseURL: "postgres://user:pass@localhost:5432/rag",
				OllamaURL:   "http://localhost:11434",
				EmbedModel:  "mxbai-embed-large",
				LLMModel:    "llama3.1",
				EmbedDims:   defaultEmbedDims,
				HTTPPort:    defaultHTTPPort,
			},
		},
		{
			name: "empty required variable counts as missing",
			setup: func(t *testing.T) {
				setRequired(t)
				t.Setenv("EMBED_MODEL", "")
			},
			wantErr:  true,
			wantMsgs: []string{"EMBED_MODEL"},
		},
		{
			name: "every missing variable reported together",
			setup: func(t *testing.T) {
				t.Setenv("DATABASE_URL", "")
				t.Setenv("OLLAMA_URL", "")
				t.Setenv("EMBED_MODEL", "")
				t.Setenv("LLM_MODEL", "")
			},
			wantErr: true,
			wantMsgs: []string{
				"DATABASE_URL",
				"OLLAMA_URL",
				"EMBED_MODEL",
				"LLM_MODEL",
			},
		},
		{
			name: "missing string and invalid integer reported together",
			setup: func(t *testing.T) {
				setRequired(t)
				t.Setenv("LLM_MODEL", "")
				t.Setenv("HTTP_PORT", "abc")
			},
			wantErr:  true,
			wantMsgs: []string{"LLM_MODEL", "HTTP_PORT"},
		},
		{
			name: "EMBED_DIMS not an integer",
			setup: func(t *testing.T) {
				setRequired(t)
				t.Setenv("EMBED_DIMS", "not-a-number")
			},
			wantErr:  true,
			wantMsgs: []string{"EMBED_DIMS"},
		},
		{
			name: "HTTP_PORT above range",
			setup: func(t *testing.T) {
				setRequired(t)
				t.Setenv("HTTP_PORT", "70000")
			},
			wantErr:  true,
			wantMsgs: []string{"HTTP_PORT"},
		},
		{
			name: "HTTP_PORT zero is invalid",
			setup: func(t *testing.T) {
				setRequired(t)
				t.Setenv("HTTP_PORT", "0")
			},
			wantErr:  true,
			wantMsgs: []string{"HTTP_PORT"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.setup(t)

			got, err := Load()
			if tt.wantErr {
				if err == nil {
					t.Fatalf("Load() = %+v, want error mentioning %v", got, tt.wantMsgs)
				}
				for _, msg := range tt.wantMsgs {
					if !strings.Contains(err.Error(), msg) {
						t.Errorf("Load() error %q does not mention %q", err.Error(), msg)
					}
				}
				return
			}
			if err != nil {
				t.Fatalf("Load() unexpected error: %v", err)
			}
			if got != tt.want {
				t.Errorf("Load() = %+v, want %+v", got, tt.want)
			}
		})
	}
}
