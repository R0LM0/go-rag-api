package application

import (
	"strings"
	"testing"

	"github.com/R0LM0/go-rag-api/internal/domain"
)

func TestBuildPromptContainsQuestionAndNumberedContext(t *testing.T) {
	chunks := []domain.Chunk{
		{ID: "c1", DocumentID: "d1", Content: "Paris is the capital of France.", Position: 0},
		{ID: "c2", DocumentID: "d1", Content: "The Seine flows through Paris.", Position: 1},
	}
	question := "What is the capital of France?"
	prompt := BuildPrompt(question, chunks)

	if !strings.Contains(prompt, question) {
		t.Errorf("prompt does not contain the question: %q", prompt)
	}
	if !strings.Contains(prompt, "[1] Paris is the capital of France.") {
		t.Errorf("prompt does not number the first context chunk: %q", prompt)
	}
	if !strings.Contains(prompt, "[2] The Seine flows through Paris.") {
		t.Errorf("prompt does not number the second context chunk: %q", prompt)
	}
}

func TestBuildPromptInstructsToAnswerOnlyFromContext(t *testing.T) {
	prompt := BuildPrompt("a question", []domain.Chunk{{Content: "context text"}})
	lower := strings.ToLower(prompt)
	if !strings.Contains(lower, "only") {
		t.Errorf("prompt must instruct to answer only from the context: %q", prompt)
	}
	if !strings.Contains(lower, "enough information") {
		t.Errorf("prompt must instruct to admit missing information: %q", prompt)
	}
}

func TestBuildPromptIsDeterministic(t *testing.T) {
	chunks := []domain.Chunk{{Content: "alpha"}, {Content: "beta"}}
	first := BuildPrompt("question", chunks)
	second := BuildPrompt("question", chunks)
	if first != second {
		t.Errorf("BuildPrompt is not deterministic:\nfirst:  %q\nsecond: %q", first, second)
	}
}
