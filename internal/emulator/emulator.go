package emulator

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"math/rand/v2"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/engine"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/perf"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/prompt"
)

type Config struct {
	Name          string        `json:"name"`
	Model         string        `json:"model"`
	Hardware      string        `json:"hardware"`
	TP            int           `json:"tp"`
	Engine        engine.Config `json:"engine"`
	Speed         float64       `json:"speed"`
	Bias          float64       `json:"bias"`
	Jitter        float64       `json:"jitter"`
	BlockBytes    int           `json:"block_bytes"`
	CharsPerToken float64       `json:"chars_per_token"`
	DefaultOutput int           `json:"default_output"`
}

type stream struct {
	notify chan struct{}
	done   chan struct{}
	closed bool
}

type Server struct {
	cfg     Config
	eng     *engine.Engine
	hasher  prompt.Hasher
	mu      sync.Mutex
	wake    chan struct{}
	start   time.Time
	streams map[*engine.Request]*stream
	nextID  atomic.Int64
	rng     *rand.Rand
	log     *slog.Logger
	queries atomic.Int64
	hits    atomic.Int64
}

func New(cfg Config, log *slog.Logger) (*Server, error) {
	if cfg.Speed <= 0 {
		cfg.Speed = 1
	}
	if cfg.Bias <= 0 {
		cfg.Bias = 1
	}
	if cfg.BlockBytes <= 0 {
		cfg.BlockBytes = 256
	}
	if cfg.CharsPerToken <= 0 {
		cfg.CharsPerToken = 4
	}
	if cfg.DefaultOutput <= 0 {
		cfg.DefaultOutput = 200
	}
	if cfg.Engine.MaxNumSeqs == 0 {
		cfg.Engine = engine.DefaultConfig()
	}
	rep, err := perf.NewReplica(perf.Deployment{ModelKey: cfg.Model, HardwareKey: cfg.Hardware, TP: cfg.TP})
	if err != nil {
		return nil, err
	}
	h := prompt.Hasher{BlockBytes: cfg.BlockBytes, CharsPerToken: cfg.CharsPerToken}
	cfg.Engine.BlockTokens = h.BlockTokens()
	s := &Server{cfg: cfg, hasher: h, wake: make(chan struct{}, 1), start: time.Now(), streams: make(map[*engine.Request]*stream), rng: rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), 1)), log: log}
	s.eng = engine.New(cfg.Engine, rep)
	s.eng.Listener = s
	noise := rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), 2))
	s.eng.TimeScale = func(d float64) float64 {
		f := cfg.Bias
		if cfg.Jitter > 0 {
			f *= math.Exp(cfg.Jitter*noise.NormFloat64() - cfg.Jitter*cfg.Jitter/2)
		}
		return d * f
	}
	return s, nil
}

func (s *Server) simNow() float64 { return time.Since(s.start).Seconds() * s.cfg.Speed }

func (s *Server) wallAt(sim float64) time.Time {
	return s.start.Add(time.Duration(sim / s.cfg.Speed * float64(time.Second)))
}

func (s *Server) OnFirstToken(r *engine.Request, t float64) { s.signal(r) }

func (s *Server) OnFinish(r *engine.Request, t float64) {
	if st, ok := s.streams[r]; ok && !st.closed {
		st.closed = true
		close(st.done)
	}
}

func (s *Server) signal(r *engine.Request) {
	if st, ok := s.streams[r]; ok {
		select {
		case st.notify <- struct{}{}:
		default:
		}
	}
}

func (s *Server) Run(ctx context.Context) {
	for {
		s.mu.Lock()
		busy := s.eng.Busy()
		end := s.eng.StepEnd()
		s.mu.Unlock()
		if !busy {
			select {
			case <-ctx.Done():
				return
			case <-s.wake:
			}
			continue
		}
		if d := time.Until(s.wallAt(end)); d > 0 {
			timer := time.NewTimer(d)
			select {
			case <-ctx.Done():
				timer.Stop()
				return
			case <-timer.C:
			}
		}
		s.mu.Lock()
		s.eng.Step()
		for _, r := range s.eng.Running() {
			if r.Generated > 0 {
				s.signal(r)
			}
		}
		s.mu.Unlock()
	}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.complete)
	mux.HandleFunc("POST /v1/completions", s.complete)
	mux.HandleFunc("GET /v1/models", s.models)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("GET /metrics", s.metrics)
	return mux
}

var words = strings.Fields("the model routes tokens across replicas while the cache keeps shared prefixes warm and the scheduler batches prefill with decode under a fixed token budget so every request meets its latency objective")

func (s *Server) complete(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 64<<20))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	req, raw, err := prompt.Parse(body)
	if err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}
	chat := strings.HasSuffix(r.URL.Path, "chat/completions")
	input := s.hasher.Tokens(req.Text)
	out := req.MaxTokens
	if out <= 0 {
		out = int(math.Min(2048, math.Max(8, float64(s.cfg.DefaultOutput)*math.Exp(0.7*s.rng.NormFloat64()))))
	}
	id := s.nextID.Add(1)
	er := engine.NewRequest(id, s.simNow(), input, out, s.hasher.Hashes(req.Text))
	st := &stream{notify: make(chan struct{}, 1), done: make(chan struct{})}
	s.mu.Lock()
	if !s.eng.Fits(er) {
		s.mu.Unlock()
		http.Error(w, `{"error":{"message":"prompt exceeds KV capacity"}}`, http.StatusBadRequest)
		return
	}
	s.streams[er] = st
	s.eng.Add(er, s.simNow())
	s.mu.Unlock()
	s.queries.Add(int64(input))
	select {
	case s.wake <- struct{}{}:
	default:
	}
	defer func() {
		s.mu.Lock()
		if er.State != engine.Done {
			s.eng.Remove(er)
		}
		delete(s.streams, er)
		s.mu.Unlock()
	}()
	model := req.Model
	if model == "" {
		model = s.cfg.Model
	}
	rid := fmt.Sprintf("cmpl-%s-%d", s.cfg.Name, id)
	created := time.Now().Unix()
	includeUsage := true
	if so, ok := raw["stream_options"].(map[string]any); ok {
		if v, ok := so["include_usage"].(bool); ok {
			includeUsage = v
		}
	}
	if !req.Stream {
		select {
		case <-st.done:
		case <-r.Context().Done():
			return
		}
		s.mu.Lock()
		gen, hit := er.Generated, er.Hit
		s.mu.Unlock()
		s.hits.Add(int64(hit))
		text := s.text(0, gen)
		var choice map[string]any
		if chat {
			choice = map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": text}, "finish_reason": "length"}
		} else {
			choice = map[string]any{"index": 0, "text": text, "finish_reason": "length"}
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"id": rid, "object": objectName(chat, false), "created": created, "model": model, "choices": []any{choice}, "usage": usage(input, gen, hit)})
		return
	}
	fl, _ := w.(http.Flusher)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)
	sent := 0
	write := func(v any) bool {
		b, _ := json.Marshal(v)
		if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
			return false
		}
		if fl != nil {
			fl.Flush()
		}
		return true
	}
	emit := func(upto int) bool {
		for sent < upto {
			tok := s.text(sent, sent+1)
			var choice map[string]any
			if chat {
				delta := map[string]any{"content": tok}
				if sent == 0 {
					delta["role"] = "assistant"
				}
				choice = map[string]any{"index": 0, "delta": delta, "finish_reason": nil}
			} else {
				choice = map[string]any{"index": 0, "text": tok, "finish_reason": nil}
			}
			if !write(map[string]any{"id": rid, "object": objectName(chat, true), "created": created, "model": model, "choices": []any{choice}}) {
				return false
			}
			sent++
		}
		return true
	}
	for {
		select {
		case <-r.Context().Done():
			return
		case <-st.notify:
			s.mu.Lock()
			g := er.Generated
			s.mu.Unlock()
			if !emit(g) {
				return
			}
			continue
		case <-st.done:
		}
		break
	}
	s.mu.Lock()
	g, hit := er.Generated, er.Hit
	s.mu.Unlock()
	s.hits.Add(int64(hit))
	if !emit(g) {
		return
	}
	var final map[string]any
	if chat {
		final = map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": "length"}
	} else {
		final = map[string]any{"index": 0, "text": "", "finish_reason": "length"}
	}
	write(map[string]any{"id": rid, "object": objectName(chat, true), "created": created, "model": model, "choices": []any{final}})
	if includeUsage {
		write(map[string]any{"id": rid, "object": objectName(chat, true), "created": created, "model": model, "choices": []any{}, "usage": usage(input, g, hit)})
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	if fl != nil {
		fl.Flush()
	}
}

func usage(input, output, hit int) map[string]any {
	return map[string]any{"prompt_tokens": input, "completion_tokens": output, "total_tokens": input + output, "prompt_tokens_details": map[string]any{"cached_tokens": hit}}
}

func objectName(chat, chunk bool) string {
	switch {
	case chat && chunk:
		return "chat.completion.chunk"
	case chat:
		return "chat.completion"
	default:
		return "text_completion"
	}
}

func (s *Server) text(from, to int) string {
	var b strings.Builder
	for i := from; i < to; i++ {
		b.WriteByte(' ')
		b.WriteString(words[i%len(words)])
	}
	return b.String()
}

func (s *Server) models(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": []any{map[string]any{"id": s.cfg.Model, "object": "model", "owned_by": "abyss-emulator"}}})
}

func (s *Server) metrics(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	running, waiting, usage := len(s.eng.Running()), len(s.eng.Waiting()), s.eng.KVUsage()
	st := s.eng.Stats
	s.mu.Unlock()
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	m := fmt.Sprintf("model_name=%q", s.cfg.Model)
	fmt.Fprintf(w, "# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running{%s} %d\n", m, running)
	fmt.Fprintf(w, "# TYPE vllm:num_requests_waiting gauge\nvllm:num_requests_waiting{%s} %d\n", m, waiting)
	fmt.Fprintf(w, "# TYPE vllm:kv_cache_usage_perc gauge\nvllm:kv_cache_usage_perc{%s} %g\n", m, usage)
	fmt.Fprintf(w, "# TYPE vllm:prefix_cache_queries_total counter\nvllm:prefix_cache_queries_total{%s} %d\n", m, st.PromptTokens)
	fmt.Fprintf(w, "# TYPE vllm:prefix_cache_hits_total counter\nvllm:prefix_cache_hits_total{%s} %d\n", m, st.HitTokens)
	fmt.Fprintf(w, "# TYPE vllm:num_preemptions_total counter\nvllm:num_preemptions_total{%s} %d\n", m, st.Preemptions)
}
