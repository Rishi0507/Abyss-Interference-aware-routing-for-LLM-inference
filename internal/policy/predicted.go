package policy

import (
	"math"
	"math/rand/v2"
)

func init() {
	register("llm-d-predicted", func(seed uint64) Policy { return newPredicted(seed) })
}

type predicted struct {
	inner    *Abyss
	rng      *rand.Rand
	gate     float64
	explore  float64
	breakTTF float64
	wTTFT    float64
}

func newPredicted(seed uint64) *predicted {
	c := DefaultAbyss()
	c.Selfish = true
	c.Explain = true
	return &predicted{inner: NewAbyss("llm-d-predicted", c), rng: newRNG(seed), gate: 0.8, explore: 0.01, breakTTF: 5, wTTFT: 0.8}
}

func (p *predicted) Name() string { return "llm-d-predicted" }

func (p *predicted) Route(f *Fleet, r *Request) Decision {
	costs := p.inner.Predict(f, r)
	if len(costs) == 0 {
		return Decision{Action: Reject, Reason: "no healthy backend"}
	}
	match := func(c Cost) float64 {
		if len(r.Hashes) == 0 {
			return 0
		}
		b := f.Backends[c.Backend]
		if b.Match != nil {
			return float64(b.Match(r.Hashes)) / float64(len(r.Hashes))
		}
		return float64(c.HitBlocks) / float64(len(r.Hashes))
	}
	cands := costs
	if p.rng.Float64() >= p.explore {
		var sticky []Cost
		for _, c := range costs {
			if match(c) >= p.gate {
				sticky = append(sticky, c)
			}
		}
		if len(sticky) > 0 {
			bestAll, bestSticky := math.Inf(1), math.Inf(1)
			for _, c := range costs {
				bestAll = math.Min(bestAll, c.PredTTFT)
			}
			for _, c := range sticky {
				bestSticky = math.Min(bestSticky, c.PredTTFT)
			}
			if bestSticky-bestAll <= p.breakTTF {
				cands = sticky
			}
		}
	}
	headroom := func(c Cost) float64 {
		return p.wTTFT*(r.TTFTSLO-c.PredTTFT)/r.TTFTSLO + (1-p.wTTFT)*(r.TPOTSLO-c.PredTPOT)/r.TPOTSLO
	}
	pick, fit := -1, math.Inf(1)
	for i, c := range cands {
		if h := headroom(c); h > 0 && h < fit {
			pick, fit = i, h
		}
	}
	if pick < 0 {
		best := math.Inf(-1)
		for i, c := range cands {
			if h := headroom(c); h > best {
				pick, best = i, h
			}
		}
	}
	c := cands[pick]
	r.PredTTFT = f.Now - r.Arrival + c.PredTTFT
	r.PredTPOT = c.PredTPOT
	return Decision{Action: Dispatch, Backend: c.Backend}
}

func (p *predicted) Dispatched(f *Fleet, r *Request) { p.inner.Dispatched(f, r) }
func (p *predicted) FirstToken(f *Fleet, r *Request) { p.inner.FirstToken(f, r) }
func (p *predicted) Finished(f *Fleet, r *Request)   { p.inner.Finished(f, r) }
