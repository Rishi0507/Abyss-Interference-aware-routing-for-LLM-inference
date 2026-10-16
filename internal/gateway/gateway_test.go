package gateway

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/emulator"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/engine"
)

func startEmulator(t *testing.T, ctx context.Context, name string) *httptest.Server {
	t.Helper()
	s, err := emulator.New(emulator.Config{Name: name, Model: "llama-3.1-8b", Hardware: "h100", TP: 1, Engine: engine.DefaultConfig(), Speed: 20}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatal(err)
	}
	go s.Run(ctx)
	srv := httptest.NewServer(s.Handler())
	t.Cleanup(srv.Close)
	return srv
}

func TestGatewayStreamsThroughAbyss(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	a := startEmulator(t, ctx, "a")
	b := startEmulator(t, ctx, "b")
	g, err := New(Config{
		Policy:   "abyss",
		Backends: []BackendConfig{{Name: "a", URL: a.URL}, {Name: "b", URL: b.URL}},
	}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatal(err)
	}
	g.Run(ctx)
	gw := httptest.NewServer(g.Handler())
	defer gw.Close()

	for i := 0; i < 6; i++ {
		body := `{"model":"m","stream":true,"max_tokens":12,"messages":[{"role":"user","content":"` + strings.Repeat("shared system prompt ", 40) + `question"}]}`
		resp, err := http.Post(gw.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status %d", resp.StatusCode)
		}
		if resp.Header.Get("X-Abyss-Backend") == "" {
			t.Fatal("missing backend header")
		}
		tokens, done := 0, false
		sc := bufio.NewScanner(resp.Body)
		for sc.Scan() {
			line := sc.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			payload := strings.TrimPrefix(line, "data: ")
			if payload == "[DONE]" {
				done = true
				continue
			}
			var c chunk
			if json.Unmarshal([]byte(payload), &c) == nil && len(c.Choices) > 0 && c.Choices[0].Delta.Content != "" {
				tokens++
			}
		}
		resp.Body.Close()
		if tokens != 12 || !done {
			t.Fatalf("request %d: %d tokens, done=%v", i, tokens, done)
		}
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		s := g.snapshot()
		if s.Totals.Completed == 6 {
			if s.Totals.Good == 0 {
				t.Fatal("expected some requests to meet the objectives")
			}
			if cpt := g.CharsPerToken(); cpt < 3.5 || cpt > 4.5 {
				t.Fatalf("learned %.2f characters per token, emulator uses 4", cpt)
			}
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("gateway recorded %d completions", g.snapshot().Totals.Completed)
}

func TestRejectsInvalidJSON(t *testing.T) {
	g, err := New(Config{Backends: []BackendConfig{{Name: "x", URL: "http://127.0.0.1:1"}}}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatal(err)
	}
	gw := httptest.NewServer(g.Handler())
	defer gw.Close()
	resp, err := http.Post(gw.URL+"/v1/chat/completions", "application/json", strings.NewReader("{not json"))
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status %d", resp.StatusCode)
	}
}

func TestParseGauges(t *testing.T) {
	text := "# HELP x\nvllm:num_requests_waiting{model_name=\"m\"} 3\nvllm:num_requests_running{model_name=\"m\"} 7\nvllm:kv_cache_usage_perc{model_name=\"m\"} 0.42\n"
	g := parseGauges(strings.NewReader(text))
	if g.Waiting != 3 || g.Running != 7 || g.KVUsage != 0.42 {
		t.Fatalf("%+v", g)
	}
}
