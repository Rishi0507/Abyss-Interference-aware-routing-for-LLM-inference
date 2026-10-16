BIN ?= abyss

.PHONY: build web test bench figures traces clean

build:
	go build -trimpath -o $(BIN) ./cmd/abyss

web:
	cd web && npm ci && npm run build

test:
	go test ./...

traces:
	sh scripts/fetch-traces.sh

bench: build
	BIN=./$(BIN) sh scripts/run-benchmarks.sh

figures:
	python scripts/plot.py

clean:
	rm -f $(BIN) $(BIN).exe
