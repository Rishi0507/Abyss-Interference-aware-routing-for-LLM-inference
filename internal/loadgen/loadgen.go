package loadgen

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"sync"
	"time"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/metrics"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/workload"
)

type Config struct {
	Target        string
	Model         string
	TokenScale    float64
	OutputScale   float64
	MaxOutput     int
	BlockBytes    int
	CharsPerToken float64
	SLO           workload.SLO
	Timeout       time.Duration
	Progress      func(done, total int)
}

type Result struct {
	Sent    float64 `json:"sent"`
	TTFT    float64 `json:"ttft"`
	TPOT    float64 `json:"tpot"`
	Tokens  int     `json:"tokens"`
	Status  int     `json:"status"`
	Backend string  `json:"backend"`
	Input   int     `json:"input_tokens"`
	Err     string  `json:"error,omitempty"`
	TTFTSLO float64 `json:"ttft_slo"`
}

type Report struct {
	Requests   int            `json:"requests"`
	OK         int            `json:"ok"`
	Rejected   int            `json:"rejected"`
	Failed     int            `json:"failed"`
	Good       int            `json:"good"`
	Attainment float64        `json:"attainment"`
	TTFT       metrics.Dist   `json:"ttft"`
	TPOT       metrics.Dist   `json:"tpot"`
	Duration   float64        `json:"duration_s"`
	OutputTPS  float64        `json:"output_tps"`
	Backends   map[string]int `json:"backends"`
	Results    []Result       `json:"-"`
}

func Run(ctx context.Context, cfg Config, w *workload.Workload) (*Report, error) {
	if cfg.TokenScale <= 0 {
		cfg.TokenScale = 1
	}
	if cfg.OutputScale <= 0 {
		cfg.OutputScale = 1
	}
	if cfg.BlockBytes <= 0 {
		cfg.BlockBytes = 256
	}
	if cfg.CharsPerToken <= 0 {
		cfg.CharsPerToken = 4
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 10 * time.Minute
	}
	gatewayBlockTokens := float64(cfg.BlockBytes) / cfg.CharsPerToken
	perBlock := math.Max(1, math.Round(float64(w.BlockTokens)*cfg.TokenScale/gatewayBlockTokens))
	blockBytes := int(perBlock) * cfg.BlockBytes
	client := &http.Client{Timeout: cfg.Timeout, Transport: &http.Transport{MaxIdleConnsPerHost: 1024, DisableCompression: true}}
	results := make([]Result, len(w.Items))
	start := time.Now()
	t0 := w.Items[0].Arrival
	var wg sync.WaitGroup
	var mu sync.Mutex
	done := 0
	for i := range w.Items {
		it := w.Items[i]
		at := start.Add(time.Duration((it.Arrival - t0) * float64(time.Second)))
		select {
		case <-ctx.Done():
			wg.Wait()
			return summarize(results[:i], time.Since(start).Seconds(), cfg), ctx.Err()
		case <-time.After(time.Until(at)):
		}
		wg.Add(1)
		go func(i int, it workload.Item) {
			defer wg.Done()
			text := synth(it, w.BlockTokens, blockBytes, cfg)
			out := max(1, int(math.Round(float64(it.Output)*cfg.OutputScale)))
			if cfg.MaxOutput > 0 && out > cfg.MaxOutput {
				out = cfg.MaxOutput
			}
			r := send(ctx, client, cfg, text, out)
			r.Sent = time.Since(start).Seconds()
			r.Input = int(float64(len(text)) / cfg.CharsPerToken)
			r.TTFTSLO = cfg.SLO.TTFT(r.Input)
			results[i] = r
			mu.Lock()
			done++
			if cfg.Progress != nil {
				cfg.Progress(done, len(w.Items))
			}
			mu.Unlock()
		}(i, it)
	}
	wg.Wait()
	return summarize(results, time.Since(start).Seconds(), cfg), nil
}

func synth(it workload.Item, traceBlock, blockBytes int, cfg Config) []byte {
	full := it.Input / traceBlock
	if full > len(it.Hashes) {
		full = len(it.Hashes)
	}
	var b bytes.Buffer
	for i := 0; i < full; i++ {
		fill(&b, it.Hashes[i], blockBytes)
	}
	tail := int(float64(it.Input-full*traceBlock) * cfg.TokenScale * cfg.CharsPerToken)
	if tail > 0 {
		fill(&b, uint64(it.ID)*0x9e3779b97f4a7c15+0x1234567, tail)
	}
	return b.Bytes()
}

const alphabet = "abcdefghijklmnopqrstuvwxyz"

func fill(b *bytes.Buffer, seed uint64, n int) {
	x := seed | 1
	for i := 0; i < n; i++ {
		x ^= x << 13
		x ^= x >> 7
		x ^= x << 17
		if i%6 == 5 {
			b.WriteByte(' ')
			continue
		}
		b.WriteByte(alphabet[x%26])
	}
}

func send(ctx context.Context, client *http.Client, cfg Config, text []byte, maxTokens int) Result {
	body, _ := json.Marshal(map[string]any{
		"model":          cfg.Model,
		"messages":       []any{map[string]any{"role": "user", "content": string(text)}},
		"max_tokens":     maxTokens,
		"stream":         true,
		"ignore_eos":     true,
		"stream_options": map[string]any{"include_usage": true},
	})
	var res Result
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.Target+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		res.Err = err.Error()
		return res
	}
	req.Header.Set("Content-Type", "application/json")
	begin := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		res.Err = err.Error()
		return res
	}
	defer resp.Body.Close()
	res.Status = resp.StatusCode
	res.Backend = resp.Header.Get("X-Abyss-Backend")
	if resp.StatusCode != http.StatusOK {
		io.Copy(io.Discard, resp.Body)
		return res
	}
	var first, last time.Time
	rd := bufio.NewReader(resp.Body)
	for {
		line, err := rd.ReadBytes('\n')
		if bytes.HasPrefix(line, []byte("data:")) {
			payload := bytes.TrimSpace(line[5:])
			if !bytes.Equal(payload, []byte("[DONE]")) {
				var c struct {
					Choices []struct {
						Delta struct {
							Content string `json:"content"`
						} `json:"delta"`
					} `json:"choices"`
				}
				if json.Unmarshal(payload, &c) == nil && len(c.Choices) > 0 && c.Choices[0].Delta.Content != "" {
					now := time.Now()
					if first.IsZero() {
						first = now
					}
					last = now
					res.Tokens++
				}
			}
		}
		if err != nil {
			break
		}
	}
	if first.IsZero() {
		res.Err = "no tokens received"
		return res
	}
	res.TTFT = first.Sub(begin).Seconds()
	if res.Tokens > 1 {
		res.TPOT = last.Sub(first).Seconds() / float64(res.Tokens-1)
	}
	return res
}

func summarize(rs []Result, dur float64, cfg Config) *Report {
	rep := &Report{Requests: len(rs), Duration: dur, Backends: map[string]int{}, Results: rs}
	var ttft, tpot []float64
	var out int
	for _, r := range rs {
		switch {
		case r.Status == http.StatusTooManyRequests:
			rep.Rejected++
			continue
		case r.Status != http.StatusOK || r.Err != "":
			rep.Failed++
			continue
		}
		rep.OK++
		rep.Backends[r.Backend]++
		ttft = append(ttft, r.TTFT)
		if r.Tokens > 1 {
			tpot = append(tpot, r.TPOT)
		}
		out += r.Tokens
		if r.TTFT <= r.TTFTSLO && cfg.SLO.TPOTMet(r.TPOT, r.Tokens) {
			rep.Good++
		}
	}
	sort.Float64s(ttft)
	rep.TTFT = metrics.NewDist(ttft)
	rep.TPOT = metrics.NewDist(tpot)
	if rep.Requests > 0 {
		rep.Attainment = float64(rep.Good) / float64(rep.Requests)
	}
	if dur > 0 {
		rep.OutputTPS = float64(out) / dur
	}
	return rep
}

func (r *Report) String() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "requests %d  ok %d  rejected %d  failed %d  duration %.1fs\n", r.Requests, r.OK, r.Rejected, r.Failed, r.Duration)
	fmt.Fprintf(&b, "SLO attainment %.1f%%  output %.0f tok/s\n", 100*r.Attainment, r.OutputTPS)
	fmt.Fprintf(&b, "TTFT p50 %.0fms  p90 %.0fms  p99 %.0fms\n", 1000*r.TTFT.P50, 1000*r.TTFT.P90, 1000*r.TTFT.P99)
	fmt.Fprintf(&b, "TPOT p50 %.1fms  p90 %.1fms  p99 %.1fms\n", 1000*r.TPOT.P50, 1000*r.TPOT.P90, 1000*r.TPOT.P99)
	var names []string
	for k := range r.Backends {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, k := range names {
		name := k
		if name == "" {
			name = "(direct)"
		}
		fmt.Fprintf(&b, "  %-14s %d\n", name, r.Backends[k])
	}
	return b.String()
}
