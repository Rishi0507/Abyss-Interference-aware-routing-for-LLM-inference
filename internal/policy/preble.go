package policy

import "math"

func init() {
	register("preble", func(seed uint64) Policy { return newPreble(seed) })
}

type loadEntry struct {
	at   float64
	cost float64
}

type preble struct {
	hooks
	trees   *cacheAware
	history [][]loadEntry
	window  float64
	outMean float64
	outN    float64
}

func newPreble(seed uint64) *preble {
	return &preble{trees: newCacheAware(seed), window: 60, outMean: 256}
}

func (p *preble) Name() string { return "preble" }

func (p *preble) ensure(f *Fleet) {
	p.trees.ensure(f)
	for len(p.history) < len(f.Backends) {
		p.history = append(p.history, nil)
	}
}

func (p *preble) load(i int, now float64) float64 {
	h := p.history[i]
	k := 0
	for k < len(h) && now-h[k].at > p.window {
		k++
	}
	h = h[k:]
	p.history[i] = h
	var s float64
	for _, e := range h {
		s += e.cost
	}
	return s
}

func (p *preble) prefill(b *Backend, tokens int) float64 {
	if tokens <= 0 || b.Rep == nil {
		return 0
	}
	return b.Rep.PrefillTime(tokens, 0, max(b.EngineCfg.MaxBatchedTokens, 1))
}

func (p *preble) Route(f *Fleet, r *Request) Decision {
	p.ensure(f)
	bt := max(f.BlockTokens, 1)
	best, bestHit := -1, -1
	for _, b := range f.Backends {
		if !b.Healthy {
			continue
		}
		if m := p.trees.trees[b.Index].match(r.Hashes); m > bestHit {
			best, bestHit = b.Index, m
		}
	}
	if best < 0 {
		return Decision{Action: Reject, Reason: "no healthy backend"}
	}
	if hit := bestHit * bt; hit > r.Input-hit {
		return dispatch(best)
	}
	pick, cost := -1, math.Inf(1)
	for _, b := range f.Backends {
		if !b.Healthy {
			continue
		}
		miss := r.Input - p.trees.trees[b.Index].match(r.Hashes)*bt
		c := p.load(b.Index, f.Now) + p.prefill(b, miss)
		if c < cost {
			pick, cost = b.Index, c
		}
	}
	return dispatch(pick)
}

func (p *preble) Dispatched(f *Fleet, r *Request) {
	p.ensure(f)
	b := f.Backends[r.Backend]
	bt := max(f.BlockTokens, 1)
	miss := r.Input - p.trees.trees[b.Index].match(r.Hashes)*bt
	cost := p.prefill(b, miss)
	if b.Rep != nil {
		cost += p.outMean * b.Rep.DecodeStepTime(max(len(b.Inflight), 1), float64(r.Input)) / float64(max(len(b.Inflight), 1))
	}
	p.history[b.Index] = append(p.history[b.Index], loadEntry{f.Now, cost})
	p.trees.Dispatched(f, r)
}

func (p *preble) Finished(f *Fleet, r *Request) {
	if r.Generated > 0 {
		p.outN++
		w := math.Max(1/p.outN, 0.01)
		p.outMean = (1-w)*p.outMean + w*float64(r.Generated)
	}
}
