package gateway

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/perf"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/policy"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/prompt"
)

type backend struct {
	cfg      BackendConfig
	failures int
	served   int64
}

type decisionLog struct {
	T        float64       `json:"t"`
	ID       int64         `json:"id"`
	Input    int           `json:"input"`
	Blocks   int           `json:"blocks"`
	Backend  string        `json:"backend"`
	Action   string        `json:"action"`
	PredTTFT float64       `json:"pred_ttft"`
	PredTPOT float64       `json:"pred_tpot"`
	Micros   float64       `json:"micros"`
	Costs    []policy.Cost `json:"costs,omitempty"`
}

type Gateway struct {
	cfg      Config
	mu       sync.Mutex
	fleet    *policy.Fleet
	pol      policy.Policy
	backends []*backend
	hasher   prompt.Hasher
	client   *http.Client
	probe    *http.Client
	start    time.Time
	nextID   atomic.Int64
	done     *ring
	recent   []decisionLog
	series   []seriesPoint
	totals   totals
	log      *slog.Logger
	subsMu   sync.Mutex
	subs     map[chan []byte]struct{}
	cpt      float64
	cptN     int
}

type totals struct {
	Requests  int64 `json:"requests"`
	Completed int64 `json:"completed"`
	Good      int64 `json:"good"`
	Rejected  int64 `json:"rejected"`
	Errors    int64 `json:"errors"`
	Deferred  int64 `json:"deferred"`
}

type seriesPoint struct {
	T          float64 `json:"t"`
	RPS        float64 `json:"rps"`
	Attainment float64 `json:"attainment"`
	TTFTp50    float64 `json:"ttft_p50"`
	TTFTp99    float64 `json:"ttft_p99"`
	TPOTp50    float64 `json:"tpot_p50"`
	Inflight   int     `json:"inflight"`
}

func New(cfg Config, log *slog.Logger) (*Gateway, error) {
	if err := cfg.normalize(); err != nil {
		return nil, err
	}
	g := &Gateway{
		cfg:    cfg,
		hasher: prompt.Hasher{BlockBytes: cfg.BlockBytes, CharsPerToken: cfg.CharsPerToken},
		start:  time.Now(),
		done:   newRing(8192),
		log:    log,
		subs:   make(map[chan []byte]struct{}),
		client: &http.Client{Transport: &http.Transport{MaxIdleConnsPerHost: 512, MaxConnsPerHost: 0, IdleConnTimeout: 90 * time.Second, DisableCompression: true}},
		probe:  &http.Client{Timeout: 1500 * time.Millisecond},
	}
	g.fleet = &policy.Fleet{BlockTokens: g.hasher.BlockTokens()}
	ec := cfg.Engine
	ec.BlockTokens = g.hasher.BlockTokens()
	for i, bc := range cfg.Backends {
		d := perf.Deployment{ModelKey: bc.Model, HardwareKey: bc.Hardware, TP: bc.TP}
		if bc.Eff != nil {
			d.Eff = *bc.Eff
		}
		rep, err := perf.NewReplica(d)
		if err != nil {
			return nil, fmt.Errorf("backend %s: %w", bc.Name, err)
		}
		g.fleet.Backends = append(g.fleet.Backends, &policy.Backend{Index: i, Name: bc.Name, Rep: rep, EngineCfg: ec, Healthy: true})
		g.backends = append(g.backends, &backend{cfg: bc})
	}
	if err := g.SetPolicy(cfg.Policy); err != nil {
		return nil, err
	}
	return g, nil
}

func (g *Gateway) now() float64 { return time.Since(g.start).Seconds() }

func (g *Gateway) SetPolicy(name string) error {
	p, err := policy.New(name, uint64(time.Now().UnixNano()))
	if err != nil {
		return err
	}
	if ex, ok := p.(interface{ SetExplain(bool) }); ok {
		ex.SetExplain(true)
	}
	if pa, ok := p.(interface{ SetParallel(bool) }); ok {
		pa.SetParallel(true)
	}
	g.mu.Lock()
	g.pol = p
	g.cfg.Policy = name
	g.mu.Unlock()
	g.log.Info("routing policy set", "policy", name)
	return nil
}

func (g *Gateway) PolicyName() string {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.cfg.Policy
}

func (g *Gateway) Run(ctx context.Context) {
	for i := range g.backends {
		go g.health(ctx, i)
		if g.backends[i].cfg.Metrics {
			go g.scrape(ctx, i)
		}
	}
	go g.ticker(ctx)
}

func (g *Gateway) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", g.proxy)
	mux.HandleFunc("POST /v1/completions", g.proxy)
	mux.HandleFunc("GET /v1/models", g.models)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte("ok")) })
	mux.HandleFunc("GET /metrics", g.metrics)
	mux.HandleFunc("GET /api/live/state", g.state)
	mux.HandleFunc("GET /api/live/stream", g.stream)
	mux.HandleFunc("POST /api/live/policy", g.policyHandler)
	return mux
}

func jsonError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": msg, "type": "abyss_gateway", "code": code}})
}

func (g *Gateway) proxy(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 64<<20))
	if err != nil {
		jsonError(w, http.StatusBadRequest, "unable to read request body")
		return
	}
	req, _, err := prompt.Parse(body)
	if err != nil {
		jsonError(w, http.StatusBadRequest, "request body is not valid JSON")
		return
	}
	input := g.estimateTokens(len(req.Text))
	pr := &policy.Request{
		ID:        g.nextID.Add(1),
		Arrival:   g.now(),
		Input:     input,
		MaxTokens: req.MaxTokens,
		Hashes:    g.hasher.Hashes(req.Text),
		TTFTSLO:   g.cfg.SLO.TTFT(input),
		TPOTSLO:   g.cfg.SLO.TPOT,
		Backend:   -1,
		FirstAt:   -1,
		DoneAt:    -1,
	}
	if v := r.Header.Get("X-Abyss-Weight"); v != "" {
		if w, err := strconv.ParseFloat(v, 64); err == nil && w > 0 && w <= 100 {
			pr.Weight = w
		}
	}
	idx, ok := g.route(r.Context(), pr)
	if !ok {
		w.Header().Set("Retry-After", "1")
		jsonError(w, http.StatusTooManyRequests, "no replica can serve this request within its latency objective")
		return
	}
	b := g.backends[idx]
	up, err := http.NewRequestWithContext(r.Context(), http.MethodPost, b.cfg.URL+r.URL.Path, bytes.NewReader(body))
	if err != nil {
		g.finish(pr, false)
		jsonError(w, http.StatusInternalServerError, err.Error())
		return
	}
	for _, h := range []string{"Content-Type", "Authorization", "Accept", "X-Request-Id"} {
		if v := r.Header.Get(h); v != "" {
			up.Header.Set(h, v)
		}
	}
	resp, err := g.client.Do(up)
	if err != nil {
		g.markFailure(idx)
		g.finish(pr, false)
		jsonError(w, http.StatusBadGateway, "upstream "+b.cfg.Name+" unavailable")
		return
	}
	defer resp.Body.Close()
	for k, vs := range resp.Header {
		if strings.EqualFold(k, "Content-Length") {
			continue
		}
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
	w.Header().Set("X-Abyss-Backend", b.cfg.Name)
	w.WriteHeader(resp.StatusCode)
	if resp.StatusCode != http.StatusOK {
		io.Copy(w, resp.Body)
		g.finish(pr, false)
		return
	}
	if req.Stream || strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		g.relayStream(w, resp.Body, pr, len(req.Text))
		return
	}
	data, err := io.ReadAll(resp.Body)
	w.Write(data)
	var u struct {
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	json.Unmarshal(data, &u)
	g.learnTokens(len(req.Text), u.Usage.PromptTokens)
	g.mu.Lock()
	t := g.now()
	pr.FirstAt, pr.LastAt = t, t
	pr.Generated = max(u.Usage.CompletionTokens, 1)
	g.pol.FirstToken(g.fleetAt(t), pr)
	g.mu.Unlock()
	g.finish(pr, err == nil)
}

type chunk struct {
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			Reasoning string `json:"reasoning_content"`
		} `json:"delta"`
		Text string `json:"text"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

func (g *Gateway) estimateTokens(bytes int) int {
	g.mu.Lock()
	cpt := g.cpt
	g.mu.Unlock()
	if cpt <= 0 {
		cpt = g.cfg.CharsPerToken
	}
	return max(int(float64(bytes)/cpt+0.5), 1)
}

func (g *Gateway) learnTokens(bytes, tokens int) {
	if tokens <= 0 || bytes <= 0 {
		return
	}
	ratio := math.Min(math.Max(float64(bytes)/float64(tokens), 1), 12)
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.cpt <= 0 {
		g.cpt = g.cfg.CharsPerToken
	}
	g.cptN++
	w := math.Max(1/float64(g.cptN), 0.02)
	g.cpt = (1-w)*g.cpt + w*ratio
}

func (g *Gateway) CharsPerToken() float64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.cpt <= 0 {
		return g.cfg.CharsPerToken
	}
	return g.cpt
}

func (g *Gateway) relayStream(w http.ResponseWriter, body io.Reader, pr *policy.Request, textLen int) {
	fl, _ := w.(http.Flusher)
	rd := bufio.NewReaderSize(body, 64<<10)
	ok := false
	for {
		line, err := rd.ReadSlice('\n')
		if len(line) > 0 {
			if _, werr := w.Write(line); werr != nil {
				break
			}
			if bytes.HasPrefix(line, []byte("data:")) {
				payload := bytes.TrimSpace(line[5:])
				if bytes.Equal(payload, []byte("[DONE]")) {
					ok = true
				} else {
					g.observeChunk(payload, pr, textLen)
				}
			}
			if len(bytes.TrimSpace(line)) == 0 && fl != nil {
				fl.Flush()
			}
		}
		if err != nil {
			if errors.Is(err, bufio.ErrBufferFull) {
				continue
			}
			if errors.Is(err, io.EOF) {
				ok = true
			}
			break
		}
	}
	if fl != nil {
		fl.Flush()
	}
	g.finish(pr, ok)
}

func (g *Gateway) observeChunk(payload []byte, pr *policy.Request, textLen int) {
	var c chunk
	if json.Unmarshal(payload, &c) != nil {
		return
	}
	if c.Usage != nil && c.Usage.PromptTokens > 0 {
		g.learnTokens(textLen, c.Usage.PromptTokens)
	}
	n := 0
	for _, ch := range c.Choices {
		if ch.Delta.Content != "" || ch.Delta.Reasoning != "" || ch.Text != "" {
			n++
		}
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	t := g.now()
	if c.Usage != nil && c.Usage.CompletionTokens > pr.Generated {
		pr.Generated = c.Usage.CompletionTokens
		pr.LastAt = t
	}
	if n == 0 {
		return
	}
	if pr.FirstAt < 0 {
		pr.FirstAt = t
		pr.Generated = n
		pr.LastAt = t
		g.pol.FirstToken(g.fleetAt(t), pr)
		return
	}
	pr.Generated += n
	pr.LastAt = t
}

func (g *Gateway) fleetAt(t float64) *policy.Fleet {
	g.fleet.Now = t
	return g.fleet
}

func (g *Gateway) route(ctx context.Context, pr *policy.Request) (int, bool) {
	deadline := pr.Arrival + g.cfg.MaxDefer
	for {
		g.mu.Lock()
		t := g.now()
		start := time.Now()
		d := g.pol.Route(g.fleetAt(t), pr)
		micros := float64(time.Since(start).Nanoseconds()) / 1e3
		g.logDecision(t, pr, d, micros)
		switch d.Action {
		case policy.Dispatch:
			b := g.fleet.Backends[d.Backend]
			pr.Backend = d.Backend
			pr.Dispatched = t
			b.Inflight = append(b.Inflight, pr)
			g.pol.Dispatched(g.fleet, pr)
			g.totals.Requests++
			g.backends[d.Backend].served++
			g.mu.Unlock()
			return d.Backend, true
		case policy.Reject:
			g.totals.Requests++
			g.totals.Rejected++
			g.done.add(outcome{at: t, backend: -1, input: pr.Input})
			g.mu.Unlock()
			return -1, false
		}
		g.totals.Deferred++
		g.mu.Unlock()
		if t > deadline {
			return -1, false
		}
		select {
		case <-ctx.Done():
			return -1, false
		case <-time.After(20 * time.Millisecond):
		}
	}
}

func (g *Gateway) logDecision(t float64, pr *policy.Request, d policy.Decision, micros float64) {
	name := ""
	if d.Action == policy.Dispatch && d.Backend >= 0 && d.Backend < len(g.backends) {
		name = g.backends[d.Backend].cfg.Name
	}
	entry := decisionLog{T: t, ID: pr.ID, Input: pr.Input, Blocks: len(pr.Hashes), Backend: name, Action: d.Action.String(), PredTTFT: pr.PredTTFT, PredTPOT: pr.PredTPOT, Micros: micros, Costs: d.Costs}
	g.recent = append(g.recent, entry)
	if len(g.recent) > 64 {
		g.recent = g.recent[len(g.recent)-64:]
	}
}

func (g *Gateway) finish(pr *policy.Request, ok bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if pr.Done {
		return
	}
	t := g.now()
	pr.Done = true
	pr.DoneAt = t
	if pr.Backend >= 0 {
		g.fleet.Backends[pr.Backend].Remove(pr)
		g.pol.Finished(g.fleetAt(t), pr)
	}
	o := outcome{at: t, backend: pr.Backend, input: pr.Input, output: pr.Generated, ok: ok && pr.FirstAt >= 0}
	if o.ok {
		o.ttft = pr.FirstAt - pr.Arrival
		if pr.Generated > 1 {
			o.tpot = (pr.DoneAt - pr.FirstAt) / float64(pr.Generated-1)
		}
		o.good = o.ttft <= pr.TTFTSLO && (pr.Generated < g.cfg.SLO.TPOTMinTokens || o.tpot <= pr.TPOTSLO)
		g.totals.Completed++
		if o.good {
			g.totals.Good++
		}
	} else {
		g.totals.Errors++
	}
	g.done.add(o)
}

func (g *Gateway) markFailure(i int) {
	g.mu.Lock()
	defer g.mu.Unlock()
	b := g.backends[i]
	b.failures++
	if b.failures >= 2 {
		g.fleet.Backends[i].Healthy = false
	}
}

func (g *Gateway) health(ctx context.Context, i int) {
	b := g.backends[i]
	tick := time.NewTicker(2 * time.Second)
	defer tick.Stop()
	for {
		resp, err := g.probe.Get(b.cfg.URL + g.cfg.HealthPath)
		healthy := err == nil && resp.StatusCode < 500
		if resp != nil {
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
		}
		g.mu.Lock()
		if healthy {
			b.failures = 0
			g.fleet.Backends[i].Healthy = true
		} else {
			b.failures++
			if b.failures >= 2 {
				g.fleet.Backends[i].Healthy = false
			}
		}
		g.mu.Unlock()
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
	}
}

func (g *Gateway) scrape(ctx context.Context, i int) {
	b := g.backends[i]
	tick := time.NewTicker(250 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
		resp, err := g.probe.Get(b.cfg.URL + "/metrics")
		if err != nil {
			continue
		}
		gauges := parseGauges(resp.Body)
		resp.Body.Close()
		g.mu.Lock()
		gauges.At = g.now()
		g.fleet.Backends[i].Gauges = gauges
		g.mu.Unlock()
	}
}

func parseGauges(r io.Reader) policy.Gauges {
	var gs policy.Gauges
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64<<10), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if strings.HasPrefix(line, "#") {
			continue
		}
		name, val, ok := splitMetric(line)
		if !ok {
			continue
		}
		switch name {
		case "vllm:num_requests_waiting", "sglang:num_queue_reqs":
			gs.Waiting += int(val)
		case "vllm:num_requests_running", "sglang:num_running_reqs":
			gs.Running += int(val)
		case "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "sglang:token_usage":
			gs.KVUsage = val
		}
	}
	return gs
}

func splitMetric(line string) (string, float64, bool) {
	sp := strings.LastIndexByte(line, ' ')
	if sp < 0 {
		return "", 0, false
	}
	name := line[:sp]
	if i := strings.IndexByte(name, '{'); i >= 0 {
		name = name[:i]
	}
	var v float64
	if _, err := fmt.Sscanf(line[sp+1:], "%g", &v); err != nil {
		return "", 0, false
	}
	return strings.TrimSpace(name), v, true
}

func (g *Gateway) models(w http.ResponseWriter, r *http.Request) {
	for i, b := range g.backends {
		g.mu.Lock()
		healthy := g.fleet.Backends[i].Healthy
		g.mu.Unlock()
		if !healthy {
			continue
		}
		resp, err := g.probe.Get(b.cfg.URL + "/v1/models")
		if err != nil {
			continue
		}
		defer resp.Body.Close()
		w.Header().Set("Content-Type", "application/json")
		io.Copy(w, resp.Body)
		return
	}
	jsonError(w, http.StatusServiceUnavailable, "no healthy backend")
}
