#!/usr/bin/env sh
set -e
go test ./internal/policy -run XXX -bench BenchmarkAbyssDecision -benchtime 200x | tee results/microbench.txt
python scripts/parse-microbench.py results/microbench.txt results/microbench.json
