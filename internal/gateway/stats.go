package gateway

import (
	"sort"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/metrics"
)

type outcome struct {
	at      float64
	backend int
	ttft    float64
	tpot    float64
	good    bool
	ok      bool
	input   int
	output  int
}

type window struct {
	Requests   int     `json:"requests"`
	RPS        float64 `json:"rps"`
	Goodput    float64 `json:"goodput"`
	Attainment float64 `json:"attainment"`
	TTFTp50    float64 `json:"ttft_p50"`
	TTFTp99    float64 `json:"ttft_p99"`
	TPOTp50    float64 `json:"tpot_p50"`
	TPOTp99    float64 `json:"tpot_p99"`
	OutputTPS  float64 `json:"output_tps"`
}

type ring struct {
	buf  []outcome
	next int
	full bool
}

func newRing(n int) *ring { return &ring{buf: make([]outcome, n)} }

func (r *ring) add(o outcome) {
	r.buf[r.next] = o
	r.next = (r.next + 1) % len(r.buf)
	if r.next == 0 {
		r.full = true
	}
}

func (r *ring) each(fn func(o *outcome)) {
	n := r.next
	if r.full {
		n = len(r.buf)
	}
	for i := 0; i < n; i++ {
		fn(&r.buf[i])
	}
}

func (r *ring) window(now, span float64, backend int) window {
	var w window
	var ttft, tpot []float64
	var good, out int
	r.each(func(o *outcome) {
		if now-o.at > span || (backend >= 0 && o.backend != backend) {
			return
		}
		w.Requests++
		if o.good {
			good++
		}
		if o.ok {
			ttft = append(ttft, o.ttft)
			if o.output > 1 {
				tpot = append(tpot, o.tpot)
			}
			out += o.output
		}
	})
	if w.Requests == 0 {
		return w
	}
	sort.Float64s(ttft)
	sort.Float64s(tpot)
	w.RPS = float64(w.Requests) / span
	w.Goodput = float64(good) / span
	w.Attainment = float64(good) / float64(w.Requests)
	w.TTFTp50 = metrics.Quantile(ttft, 0.5)
	w.TTFTp99 = metrics.Quantile(ttft, 0.99)
	w.TPOTp50 = metrics.Quantile(tpot, 0.5)
	w.TPOTp99 = metrics.Quantile(tpot, 0.99)
	w.OutputTPS = float64(out) / span
	return w
}
