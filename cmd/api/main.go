// Command api is the composition root of the RAG service: the only place
// where configuration, adapters, and use cases are wired together. Everything
// else depends inward; this file depends on everything.
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	inboundhttp "github.com/R0LM0/go-rag-api/internal/adapters/inbound/http"
	"github.com/R0LM0/go-rag-api/internal/adapters/outbound/loader"
	"github.com/R0LM0/go-rag-api/internal/adapters/outbound/ollama"
	"github.com/R0LM0/go-rag-api/internal/adapters/outbound/pgvector"
	"github.com/R0LM0/go-rag-api/internal/application"
	"github.com/R0LM0/go-rag-api/internal/config"
)

// shutdownTimeout bounds the graceful drain window after SIGINT/SIGTERM.
const shutdownTimeout = 10 * time.Second

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

// run wires the whole application and blocks until the server stops. It
// returns errors instead of exiting so main stays the only place with
// fatal-exit behavior.
func run() error {
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("load configuration: %w", err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	ollamaClient, err := ollama.NewClient(cfg.OllamaURL, cfg.EmbedModel, cfg.LLMModel, nil)
	if err != nil {
		return fmt.Errorf("create ollama client: %w", err)
	}

	store, err := pgvector.NewStore(ctx, cfg.DatabaseURL)
	if err != nil {
		return fmt.Errorf("connect to postgres: %w", err)
	}
	defer store.Close()

	indexUseCase := application.NewIndexDocument(
		loader.FileLoader{}, ollamaClient, store, application.DefaultChunkConfig(),
	)
	answerUseCase := application.NewAnswerQuery(ollamaClient, store, ollamaClient)

	// Handing the concrete use cases to NewHandler is the compile-time
	// assertion that they satisfy the inbound adapter's narrow consumer
	// interfaces: if their method sets ever drift from what the HTTP layer
	// needs, this construction stops compiling.
	handler := inboundhttp.NewHandler(indexUseCase, answerUseCase)

	server := &http.Server{
		Addr:              fmt.Sprintf(":%d", cfg.HTTPPort),
		Handler:           inboundhttp.NewRouter(handler),
		ReadHeaderTimeout: 10 * time.Second,
	}

	serverErr := make(chan error, 1)
	go func() {
		log.Printf("rag api listening on %s", server.Addr)
		serverErr <- server.ListenAndServe()
	}()

	select {
	case err := <-serverErr:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			return fmt.Errorf("http server: %w", err)
		}
		return nil
	case <-ctx.Done():
		log.Println("shutdown signal received, draining in-flight requests")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			return fmt.Errorf("graceful shutdown: %w", err)
		}
		log.Println("rag api stopped")
		return nil
	}
}
