# Makefile for the go-rag-api service.
#
# When a .env file exists it is included and exported so every target (and the
# api binary) sees DATABASE_URL and friends as real environment variables. The
# Go binary itself never reads .env; configuration loading is stdlib-only.

ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: run test test-integration migrate lint compose-up compose-down

run:
	go run ./cmd/api

test:
	go test ./...

test-integration:
	go test -tags integration ./...

migrate:
	psql "$(DATABASE_URL)" -f migrations/000001_init.up.sql

lint:
	go vet ./...
	@files=$$(gofmt -l .); \
	if [ -n "$$files" ]; then \
		echo "gofmt: the following files need formatting:"; \
		echo "$$files"; \
		exit 1; \
	fi

compose-up:
	docker compose up -d

compose-down:
	docker compose down
