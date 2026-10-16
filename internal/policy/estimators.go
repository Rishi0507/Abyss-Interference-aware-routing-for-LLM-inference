package policy

import (
	"math"
	"sort"
)

type lengthModel struct {
	ring   []float64
	next   int
	n      int
	sorted []float64
	suffix []float64
	dirty  int
	prior  float64
}

func newLengthModel(prior float64) *lengthModel {
	return &lengthModel{ring: make([]float64, 4096), prior: prior}
}

func (l *lengthModel) observe(n int) {
	if n <= 0 {
		return
	}
	l.ring[l.next] = float64(n)
	l.next = (l.next + 1) % len(l.ring)
	if l.n < len(l.ring) {
		l.n++
	}
	l.dirty++
	if l.dirty >= 64 || l.n < 64 {
		l.rebuild()
	}
}

func (l *lengthModel) rebuild() {
	l.sorted = append(l.sorted[:0], l.ring[:l.n]...)
	sort.Float64s(l.sorted)
	l.suffix = l.suffix[:0]
	l.suffix = append(l.suffix, make([]float64, len(l.sorted)+1)...)
	for i := len(l.sorted) - 1; i >= 0; i-- {
		l.suffix[i] = l.suffix[i+1] + l.sorted[i]
	}
	l.dirty = 0
}

func (l *lengthModel) remaining(generated, limit int) int {
	var r float64
	if len(l.sorted) < 16 {
		r = math.Max(l.prior-float64(generated), l.prior/4)
	} else {
		g := float64(generated)
		i := sort.SearchFloat64s(l.sorted, g+0.5)
		if i >= len(l.sorted) {
			r = math.Max(16, g/2)
		} else {
			r = l.suffix[i]/float64(len(l.sorted)-i) - g
		}
	}
	if limit > 0 && float64(generated)+r > float64(limit) {
		r = float64(limit - generated)
	}
	if r < 1 {
		r = 1
	}
	return int(math.Round(r))
}

func (l *lengthModel) Mean() float64 {
	if len(l.sorted) == 0 {
		return l.prior
	}
	return l.suffix[0] / float64(len(l.sorted))
}

type reuseEntry struct {
	last float64
	gap  float64
	n    int32
}

type reuseModel struct {
	m         map[uint64]reuseEntry
	firstGap  float64
	seen      float64
	reused    float64
	lastPrune float64
	horizon   float64
}

func newReuseModel(horizon float64) *reuseModel {
	return &reuseModel{m: make(map[uint64]reuseEntry, 1<<14), horizon: horizon}
}

func (p *reuseModel) touch(hashes []uint64, now float64) {
	for _, h := range hashes {
		e, ok := p.m[h]
		if !ok {
			p.m[h] = reuseEntry{last: now, n: 1}
			p.seen++
			continue
		}
		g := now - e.last
		if e.n == 1 {
			p.reused++
			if p.firstGap == 0 {
				p.firstGap = g
			} else {
				p.firstGap = 0.98*p.firstGap + 0.02*g
			}
			e.gap = g
		} else {
			e.gap = 0.7*e.gap + 0.3*g
		}
		e.n++
		e.last = now
		p.m[h] = e
	}
	if now-p.lastPrune > p.horizon/4 {
		for h, e := range p.m {
			if now-e.last > p.horizon {
				delete(p.m, h)
			}
		}
		p.lastPrune = now
	}
}

func (p *reuseModel) prob(h uint64, now, residency float64) float64 {
	e, ok := p.m[h]
	if !ok {
		return 0
	}
	age := now - e.last
	if e.n == 1 {
		if p.seen == 0 {
			return 0
		}
		t := math.Max(p.firstGap, 1)
		return p.reused / p.seen * math.Exp(-age/(3*t)) * (1 - math.Exp(-residency/t))
	}
	g := math.Max(e.gap, 0.5)
	return math.Exp(-age/(3*g)) * (1 - math.Exp(-residency/g))
}
