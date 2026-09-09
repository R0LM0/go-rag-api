package http

import (
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
)

// serverTimeout bounds the total processing time of a single request, sized
// for slow local LLM generation.
const serverTimeout = 120 * time.Second

// NewRouter builds the chi router with the middleware stack and the three API
// routes.
func NewRouter(h *Handler) chi.Router {
	r := chi.NewRouter()
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)
	r.Use(middleware.Timeout(serverTimeout))
	r.Post("/documents", h.PostDocument)
	r.Post("/query", h.PostQuery)
	r.Get("/health", h.GetHealth)
	return r
}
