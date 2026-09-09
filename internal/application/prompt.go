package application

import (
	"fmt"
	"strings"

	"github.com/R0LM0/go-rag-api/internal/domain"
)

// promptInstruction is the directive placed at the top of every RAG prompt.
const promptInstruction = `Answer the question using ONLY the context provided below.
If the context does not contain the information needed to answer, say that you do not have enough information. Do not use any outside knowledge.`

// BuildPrompt renders the retrieval-augmented generation prompt for a
// question and its retrieved context chunks. It is a pure function: it
// performs no I/O, touches no ports, and returns the same output for the
// same inputs.
func BuildPrompt(question string, chunks []domain.Chunk) string {
	var b strings.Builder
	b.WriteString(promptInstruction)
	b.WriteString("\n\nContext:\n")
	for i, chunk := range chunks {
		fmt.Fprintf(&b, "[%d] %s\n", i+1, chunk.Content)
	}
	b.WriteString("\nQuestion: ")
	b.WriteString(question)
	b.WriteString("\n")
	return b.String()
}
