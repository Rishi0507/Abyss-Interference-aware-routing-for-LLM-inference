package calibrate

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"net/http"
	"sort"
	"sync"
	"time"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/perf"
)

type Sample struct {
	Kind    string  `json:"kind"`
	Tokens  int     `json:"prompt_tokens"`
	Batch   int     `json:"batch"`
	Context int     `json:"context"`
	Seconds float64 `json:"seconds"`
}

type Options struct {
	Target        string
	Model         string
	PrefillSizes  []int
	DecodeBatches []int
	Repeats       int
	CharsPerToken float64
	Progress      func(string)
}

type Result struct {
	CharsPerTok float64         `json:"chars_per_token"`
	Deployment  perf.Deployment `json:"deployment"`
	Efficiency  perf.Efficiency `json:"efficiency"`
	Network     float64         `json:"network_overhead_s"`
	Samples     []Sample        `json:"samples"`
	Rows        []Row           `json:"rows"`
	MedianErr   float64         `json:"median_abs_rel_error"`
	MaxErr      float64         `json:"max_abs_rel_error"`
	DefaultErr  float64         `json:"default_median_abs_rel_error"`
}

type Row struct {
	Sample    Sample  `json:"sample"`
	Predicted float64 `json:"predicted_s"`
	Default   float64 `json:"default_predicted_s"`
}

var client = &http.Client{Timeout: 5 * time.Minute}

func text(rng *rand.Rand, tokens int, cpt float64) string {
	n := int(float64(tokens) * cpt)
	var b bytes.Buffer
	for b.Len() < n {
		w := 2 + rng.IntN(7)
		for i := 0; i < w; i++ {
			b.WriteByte(byte('a' + rng.IntN(26)))
		}
		b.WriteByte(' ')
	}
	return b.String()[:n]
}

type streamResult struct {
	ttft   float64
	itl    float64
	prompt int
	tokens int
	err    error
}

func stream(ctx context.Context, opt Options, prompt string, maxTokens int) streamResult {
	body, _ := json.Marshal(map[string]any{
		"model":          opt.Model,
		"messages":       []any{map[string]any{"role": "user", "content": prompt}},
		"max_tokens":     maxTokens,
		"stream":         true,
		"ignore_eos":     true,
		"temperature":    0,
		"stream_options": map[string]any{"include_usage": true},
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, opt.Target+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return streamResult{err: err}
	}
	req.Header.Set("Content-Type", "application/json")
	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return streamResult{err: err}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return streamResult{err: fmt.Errorf("status %d", resp.StatusCode)}
	}
	var res streamResult
	var stamps []time.Time
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		line := sc.Bytes()
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}
		payload := line[6:]
		if bytes.Equal(payload, []byte("[DONE]")) {
			break
		}
		var c struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *struct {
				Prompt int `json:"prompt_tokens"`
			} `json:"usage"`
		}
		if json.Unmarshal(payload, &c) != nil {
			continue
		}
		if c.Usage != nil {
			res.prompt = c.Usage.Prompt
		}
		if len(c.Choices) > 0 && c.Choices[0].Delta.Content != "" {
			stamps = append(stamps, time.Now())
		}
	}
	if len(stamps) == 0 {
		return streamResult{err: fmt.Errorf("no tokens")}
	}
	res.tokens = len(stamps)
	res.ttft = stamps[0].Sub(start).Seconds()
	if len(stamps) > 20 {
		skip := len(stamps) / 10
		res.itl = stamps[len(stamps)-1].Sub(stamps[skip]).Seconds() / float64(len(stamps)-1-skip)
	}
	return res
}

func Measure(ctx context.Context, opt Options) ([]Sample, float64, error) {
	if opt.CharsPerToken <= 0 {
		opt.CharsPerToken = 4
	}
	if opt.Repeats <= 0 {
		opt.Repeats = 3
	}
	if len(opt.PrefillSizes) == 0 {
		opt.PrefillSizes = []int{256, 1024, 2048, 4096, 8192, 16384}
	}
	if len(opt.DecodeBatches) == 0 {
		opt.DecodeBatches = []int{1, 4, 16, 32}
	}
	rng := rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), 7))
	say := func(f string, a ...any) {
		if opt.Progress != nil {
			opt.Progress(fmt.Sprintf(f, a...))
		}
	}
	warm := stream(ctx, opt, text(rng, 64, opt.CharsPerToken), 4)
	if warm.err != nil {
		return nil, 0, fmt.Errorf("warm-up request failed: %w", warm.err)
	}
	probeText := text(rng, 1024, opt.CharsPerToken)
	if probe := stream(ctx, opt, probeText, 1); probe.err == nil && probe.prompt > 0 {
		opt.CharsPerToken = float64(len(probeText)) / float64(probe.prompt)
		say("server reports %.2f characters per token", opt.CharsPerToken)
	}
	var out []Sample
	for _, n := range opt.PrefillSizes {
		var ts []float64
		toks := n
		for k := 0; k < opt.Repeats; k++ {
			r := stream(ctx, opt, text(rng, n, opt.CharsPerToken), 1)
			if r.err != nil {
				return out, opt.CharsPerToken, fmt.Errorf("prefill %d: %w", n, r.err)
			}
			ts = append(ts, r.ttft)
			if r.prompt > 0 {
				toks = r.prompt
			}
		}
		out = append(out, Sample{Kind: "prefill", Tokens: toks, Batch: 1, Seconds: median(ts)})
		say("prefill %6d tokens  %8.1f ms", toks, 1000*median(ts))
	}
	for _, b := range opt.DecodeBatches {
		var itls []float64
		var mu sync.Mutex
		var wg sync.WaitGroup
		var firstErr error
		for i := 0; i < b; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				r := stream(ctx, opt, text(rand.New(rand.NewPCG(rng.Uint64(), uint64(i))), 512, opt.CharsPerToken), 160)
				mu.Lock()
				defer mu.Unlock()
				if r.err != nil {
					firstErr = r.err
					return
				}
				if r.itl > 0 {
					itls = append(itls, r.itl)
				}
			}()
		}
		wg.Wait()
		if firstErr != nil {
			return out, opt.CharsPerToken, fmt.Errorf("decode batch %d: %w", b, firstErr)
		}
		out = append(out, Sample{Kind: "decode", Tokens: 512, Batch: b, Context: 512 + 80, Seconds: median(itls)})
		say("decode batch %3d     %8.2f ms per token", b, 1000*median(itls))
	}
	return out, opt.CharsPerToken, nil
}

func median(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	return s[len(s)/2]
}

func predict(rep *perf.Replica, s Sample, network float64, chunk int) float64 {
	if s.Kind == "prefill" {
		return rep.PrefillTime(s.Tokens, 0, chunk) + network
	}
	return rep.DecodeStepTime(s.Batch, float64(s.Context))
}

func Fit(d perf.Deployment, samples []Sample, chunk int) (*Result, error) {
	if chunk <= 0 {
		chunk = 8192
	}
	if len(samples) < 3 {
		return nil, fmt.Errorf("need at least three samples, got %d", len(samples))
	}
	base := perf.DefaultEfficiency()
	build := func(x []float64) (*perf.Replica, float64, bool) {
		e := base
		e.GEMM, e.Attention, e.Bandwidth, e.StepOverhead = x[0], x[1], x[2], x[3]
		if e.GEMM < 0.02 || e.GEMM > 1 || e.Attention < 0.02 || e.Attention > 1 || e.Bandwidth < 0.05 || e.Bandwidth > 1 || e.StepOverhead < 0 || e.StepOverhead > 0.1 || x[4] < 0 || x[4] > 0.5 {
			return nil, 0, false
		}
		dd := d
		dd.Eff = e
		r, err := perf.NewReplica(dd)
		if err != nil {
			return nil, 0, false
		}
		return r, x[4], true
	}
	loss := func(x []float64) float64 {
		r, net, ok := build(x)
		if !ok {
			return 1e9
		}
		var l float64
		for _, s := range samples {
			if s.Seconds <= 0 {
				continue
			}
			e := math.Log(predict(r, s, net, chunk) / s.Seconds)
			l += e * e
		}
		return l
	}
	start := []float64{base.GEMM, base.Attention, base.Bandwidth, base.StepOverhead, 0.002}
	best := nelderMead(loss, start, 4000)
	r, net, ok := build(best)
	if !ok {
		return nil, fmt.Errorf("fit left the feasible region")
	}
	dd := d
	dd.Eff = r.Eff
	res := &Result{Deployment: dd, Efficiency: r.Eff, Network: net, Samples: samples}
	defRep, _ := perf.NewReplica(perf.Deployment{ModelKey: d.ModelKey, HardwareKey: d.HardwareKey, TP: d.TP, MemUtil: d.MemUtil})
	var errs, defErrs []float64
	for _, s := range samples {
		p := predict(r, s, net, chunk)
		pd := predict(defRep, s, 0, chunk)
		res.Rows = append(res.Rows, Row{Sample: s, Predicted: p, Default: pd})
		errs = append(errs, math.Abs(p-s.Seconds)/s.Seconds)
		defErrs = append(defErrs, math.Abs(pd-s.Seconds)/s.Seconds)
	}
	sort.Float64s(errs)
	res.MedianErr = median(errs)
	res.MaxErr = errs[len(errs)-1]
	res.DefaultErr = median(defErrs)
	return res, nil
}

func nelderMead(f func([]float64) float64, x0 []float64, iters int) []float64 {
	n := len(x0)
	pts := make([][]float64, n+1)
	vals := make([]float64, n+1)
	for i := range pts {
		p := append([]float64(nil), x0...)
		if i > 0 {
			step := p[i-1] * 0.25
			if step == 0 {
				step = 0.001
			}
			p[i-1] += step
		}
		pts[i] = p
		vals[i] = f(p)
	}
	for it := 0; it < iters; it++ {
		idx := make([]int, n+1)
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return vals[idx[a]] < vals[idx[b]] })
		sp := make([][]float64, n+1)
		sv := make([]float64, n+1)
		for i, k := range idx {
			sp[i], sv[i] = pts[k], vals[k]
		}
		pts, vals = sp, sv
		if math.Abs(vals[n]-vals[0]) < 1e-12 {
			break
		}
		c := make([]float64, n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				c[j] += pts[i][j] / float64(n)
			}
		}
		at := func(t float64) []float64 {
			p := make([]float64, n)
			for j := range p {
				p[j] = c[j] + t*(pts[n][j]-c[j])
			}
			return p
		}
		xr := at(-1)
		fr := f(xr)
		switch {
		case fr < vals[0]:
			xe := at(-2)
			if fe := f(xe); fe < fr {
				pts[n], vals[n] = xe, fe
			} else {
				pts[n], vals[n] = xr, fr
			}
		case fr < vals[n-1]:
			pts[n], vals[n] = xr, fr
		default:
			xc := at(0.5)
			if fc := f(xc); fc < vals[n] {
				pts[n], vals[n] = xc, fc
			} else {
				for i := 1; i <= n; i++ {
					for j := range pts[i] {
						pts[i][j] = pts[0][j] + 0.5*(pts[i][j]-pts[0][j])
					}
					vals[i] = f(pts[i])
				}
			}
		}
	}
	bi := 0
	for i := range vals {
		if vals[i] < vals[bi] {
			bi = i
		}
	}
	return pts[bi]
}
