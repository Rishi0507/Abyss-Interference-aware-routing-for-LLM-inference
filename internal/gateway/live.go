package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/engine"
)

type twinView struct {
	Running      int     `json:"running"`
	Waiting      int     `json:"waiting"`
	KVUsage      float64 `json:"kv_usage"`
	CachedTokens int     `json:"cached_tokens"`
	Capacity     int     `json:"capacity_tokens"`
	HostBlocks   int     `json:"host_blocks"`
	Scale        float64 `json:"speed_scale"`
	PrefillScale float64 `json:"prefill_scale"`
}

type backendView struct {
	Name       string    `json:"name"`
	URL        string    `json:"url"`
	Label      string    `json:"label"`
	Model      string    `json:"model"`
	Healthy    bool      `json:"healthy"`
	Inflight   int       `json:"inflight"`
	Prefilling int       `json:"prefilling"`
	Decoding   int       `json:"decoding"`
	Served     int64     `json:"served"`
	Twin       *twinView `json:"twin,omitempty"`
	Waiting    int       `json:"scraped_waiting"`
	KVUsage    float64   `json:"scraped_kv_usage"`
	Window     window    `json:"window"`
}

type snapshot struct {
	Time      float64       `json:"time"`
	Policy    string        `json:"policy"`
	Policies  []string      `json:"policies"`
	SLO       any           `json:"slo"`
	Totals    totals        `json:"totals"`
	Window    window        `json:"window"`
	Backends  []backendView `json:"backends"`
	Decisions []decisionLog `json:"decisions"`
	Series    []seriesPoint `json:"series"`
}

type twinSource interface {
	Twin(i int) *engine.Engine
	TwinScale(i int) float64
	TwinPrefillScale(i int) float64
}

func (g *Gateway) snapshot() snapshot {
	g.mu.Lock()
	defer g.mu.Unlock()
	t := g.now()
	s := snapshot{Time: t, Policy: g.cfg.Policy, Policies: livePolicies, SLO: g.cfg.SLO, Totals: g.totals, Window: g.done.window(t, 30, -1)}
	ts, hasTwin := g.pol.(twinSource)
	for i, b := range g.fleet.Backends {
		v := backendView{Name: b.Name, URL: g.backends[i].cfg.URL, Label: b.Rep.Label, Model: b.Rep.Model.Name, Healthy: b.Healthy, Inflight: len(b.Inflight), Served: g.backends[i].served, Waiting: b.Gauges.Waiting, KVUsage: b.Gauges.KVUsage}
		for _, r := range b.Inflight {
			if r.FirstAt < 0 {
				v.Prefilling++
			} else {
				v.Decoding++
			}
		}
		if hasTwin {
			if e := ts.Twin(i); e != nil {
				v.Twin = &twinView{Running: len(e.Running()), Waiting: len(e.Waiting()), KVUsage: e.KVUsage(), CachedTokens: e.CachedTokens(), Capacity: e.CapacityTokens(), HostBlocks: e.HostBlocks(), Scale: ts.TwinScale(i), PrefillScale: ts.TwinPrefillScale(i)}
			}
		}
		v.Window = g.done.window(t, 30, i)
		s.Backends = append(s.Backends, v)
	}
	s.Decisions = append([]decisionLog(nil), g.recent...)
	s.Series = append([]seriesPoint(nil), g.series...)
	return s
}

var livePolicies = []string{"abyss", "abyss-selfish", "llm-d", "dynamo", "sglang", "sita", "least-request", "round-robin"}

func (g *Gateway) ticker(ctx context.Context) {
	tick := time.NewTicker(500 * time.Millisecond)
	defer tick.Stop()
	n := 0
	for {
		select {
		case <-ctx.Done():
			return
		case <-tick.C:
		}
		n++
		if n%2 == 0 {
			g.mu.Lock()
			t := g.now()
			w := g.done.window(t, 10, -1)
			inflight := 0
			for _, b := range g.fleet.Backends {
				inflight += len(b.Inflight)
			}
			g.series = append(g.series, seriesPoint{T: t, RPS: w.RPS, Attainment: w.Attainment, TTFTp50: w.TTFTp50, TTFTp99: w.TTFTp99, TPOTp50: w.TPOTp50, Inflight: inflight})
			if len(g.series) > 300 {
				g.series = g.series[len(g.series)-300:]
			}
			g.mu.Unlock()
		}
		g.subsMu.Lock()
		if len(g.subs) == 0 {
			g.subsMu.Unlock()
			continue
		}
		g.subsMu.Unlock()
		b, _ := json.Marshal(g.snapshot())
		g.subsMu.Lock()
		for ch := range g.subs {
			select {
			case ch <- b:
			default:
			}
		}
		g.subsMu.Unlock()
	}
}

func (g *Gateway) state(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(g.snapshot())
}

func (g *Gateway) stream(w http.ResponseWriter, r *http.Request) {
	fl, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	ch := make(chan []byte, 4)
	g.subsMu.Lock()
	g.subs[ch] = struct{}{}
	g.subsMu.Unlock()
	defer func() {
		g.subsMu.Lock()
		delete(g.subs, ch)
		g.subsMu.Unlock()
	}()
	b, _ := json.Marshal(g.snapshot())
	fmt.Fprintf(w, "data: %s\n\n", b)
	fl.Flush()
	for {
		select {
		case <-r.Context().Done():
			return
		case b := <-ch:
			if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
				return
			}
			fl.Flush()
		}
	}
}

func (g *Gateway) policyHandler(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Policy string `json:"policy"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Policy == "" {
		jsonError(w, http.StatusBadRequest, "expected {\"policy\": \"name\"}")
		return
	}
	if err := g.SetPolicy(body.Policy); err != nil {
		jsonError(w, http.StatusBadRequest, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"policy": body.Policy})
}

func (g *Gateway) metrics(w http.ResponseWriter, r *http.Request) {
	s := g.snapshot()
	var b strings.Builder
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	gauge := func(name, help string, v float64, labels string) {
		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s gauge\n%s%s %g\n", name, help, name, name, labels, v)
	}
	counter := func(name, help string, v float64) {
		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s counter\n%s %g\n", name, help, name, name, v)
	}
	counter("abyss_requests_total", "Requests routed by the gateway.", float64(s.Totals.Requests))
	counter("abyss_requests_completed_total", "Requests completed successfully.", float64(s.Totals.Completed))
	counter("abyss_requests_good_total", "Requests that met both latency objectives.", float64(s.Totals.Good))
	counter("abyss_requests_rejected_total", "Requests rejected by admission control.", float64(s.Totals.Rejected))
	counter("abyss_requests_failed_total", "Requests that failed upstream.", float64(s.Totals.Errors))
	gauge("abyss_slo_attainment_ratio", "Fraction of requests meeting both objectives over the last 30 seconds.", s.Window.Attainment, "")
	gauge("abyss_ttft_p99_seconds", "Time to first token, 99th percentile, last 30 seconds.", s.Window.TTFTp99, "")
	gauge("abyss_tpot_p99_seconds", "Time per output token, 99th percentile, last 30 seconds.", s.Window.TPOTp99, "")
	gauge("abyss_chars_per_token", "Prompt bytes per token learned from backend usage reports.", g.CharsPerToken(), "")
	for _, bv := range s.Backends {
		l := fmt.Sprintf("{backend=%q}", bv.Name)
		fmt.Fprintf(&b, "abyss_backend_inflight%s %d\n", l, bv.Inflight)
		h := 0
		if bv.Healthy {
			h = 1
		}
		fmt.Fprintf(&b, "abyss_backend_healthy%s %d\n", l, h)
		if bv.Twin != nil {
			fmt.Fprintf(&b, "abyss_twin_kv_usage_ratio%s %g\n", l, bv.Twin.KVUsage)
			fmt.Fprintf(&b, "abyss_twin_speed_scale%s %g\n", l, bv.Twin.Scale)
			fmt.Fprintf(&b, "abyss_twin_prefill_scale%s %g\n", l, bv.Twin.PrefillScale)
		}
	}
	w.Write([]byte(b.String()))
}
