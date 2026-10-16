package workload

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
)

type segment struct {
	id  uint64
	len int
}

type sequence []segment

func (s sequence) tokens() int {
	n := 0
	for _, g := range s {
		n += g.len
	}
	return n
}

func (s sequence) blockHashes(bt int) []uint64 {
	total := s.tokens()
	n := total / bt
	out := make([]uint64, 0, n)
	var parent uint64 = 0xcbf29ce484222325
	seg, off := 0, 0
	for b := 0; b < n; b++ {
		var fp uint64 = 14695981039346656037
		need := bt
		for need > 0 {
			g := s[seg]
			take := g.len - off
			if take > need {
				take = need
			}
			fp = mix(fp ^ (g.id + uint64(off)*0x9e3779b97f4a7c15 + uint64(take)<<48))
			off += take
			need -= take
			if off == g.len {
				seg++
				off = 0
			}
		}
		parent = chain(parent, fp)
		out = append(out, parent)
	}
	return out
}

type Spec struct {
	Kind     string  `json:"kind"`
	Rate     float64 `json:"rate"`
	Duration float64 `json:"duration"`
	Seed     uint64  `json:"seed"`
	Burst    float64 `json:"burstiness"`
}

const (
	ClassChat uint8 = iota
	ClassAgent
	ClassRAG
)

var classNames = []string{"chat", "agent", "rag"}

func Generate(sp Spec) (*Workload, error) {
	if sp.Rate <= 0 {
		sp.Rate = 4
	}
	if sp.Duration <= 0 {
		sp.Duration = 600
	}
	if sp.Seed == 0 {
		sp.Seed = 7
	}
	g := &gen{rng: rand.New(rand.NewPCG(sp.Seed, sp.Seed^0x5851f42d4c957f2d)), bt: 64, burst: sp.Burst}
	var items []Item
	switch sp.Kind {
	case "chat":
		items = g.chat(sp.Rate, sp.Duration)
	case "agent":
		items = g.agent(sp.Rate, sp.Duration)
	case "rag":
		items = g.rag(sp.Rate, sp.Duration)
	case "mixed":
		items = append(items, g.chat(sp.Rate*0.45, sp.Duration)...)
		items = append(items, g.agent(sp.Rate*0.35, sp.Duration)...)
		items = append(items, g.rag(sp.Rate*0.20, sp.Duration)...)
	default:
		return nil, fmt.Errorf("unknown workload kind %q (chat, agent, rag, mixed)", sp.Kind)
	}
	sort.SliceStable(items, func(i, j int) bool { return items[i].Arrival < items[j].Arrival })
	for i := range items {
		items[i].ID = int64(i)
	}
	return &Workload{Name: sp.Kind, BlockTokens: g.bt, Items: items, Classes: classNames}, nil
}

type gen struct {
	rng     *rand.Rand
	bt      int
	nextSeg uint64
	session int64
	burst   float64
}

func (g *gen) seg(n int) segment {
	g.nextSeg++
	return segment{id: mix(g.nextSeg*0x2545f4914f6cdd1d + 0x1234567), len: n}
}

func (g *gen) lognormal(median, sigma, lo, hi float64) int {
	return int(clamp(median*math.Exp(sigma*g.rng.NormFloat64()), lo, hi))
}

func (g *gen) zipf(n int, s float64) int {
	u := g.rng.Float64()
	var norm float64
	for k := 1; k <= n; k++ {
		norm += 1 / math.Pow(float64(k), s)
	}
	acc := 0.0
	for k := 1; k <= n; k++ {
		acc += 1 / math.Pow(float64(k), s) / norm
		if u <= acc {
			return k - 1
		}
	}
	return n - 1
}

func (g *gen) arrivals(rate, dur float64) []float64 {
	var out []float64
	t := 0.0
	cv := g.burst
	for {
		if cv > 1 {
			shape := 1 / (cv * cv)
			t += gammaSample(g.rng, shape) / (shape * rate)
		} else {
			t += g.rng.ExpFloat64() / rate
		}
		if t >= dur {
			return out
		}
		out = append(out, t)
	}
}

func gammaSample(r *rand.Rand, k float64) float64 {
	if k < 1 {
		return gammaSample(r, k+1) * math.Pow(r.Float64(), 1/k)
	}
	d := k - 1.0/3
	c := 1 / math.Sqrt(9*d)
	for {
		x := r.NormFloat64()
		v := 1 + c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := r.Float64()
		if math.Log(u) < 0.5*x*x+d-d*v+d*math.Log(v) {
			return d * v
		}
	}
}

func (g *gen) emit(items *[]Item, at float64, prompt sequence, output segment, class uint8) {
	full := append(append(sequence{}, prompt...), output)
	*items = append(*items, Item{Session: g.session, Arrival: at, Input: prompt.tokens(), Output: output.len, Hashes: full.blockHashes(g.bt), Class: class})
}

func (g *gen) chat(rate, dur float64) []Item {
	var items []Item
	const templates = 12
	sys := make([]segment, templates)
	for i := range sys {
		sys[i] = g.seg(600 + g.rng.IntN(1800))
	}
	meanTurns := 3.3
	for _, start := range g.arrivals(rate/meanTurns, dur) {
		g.session++
		prompt := sequence{sys[g.zipf(templates, 1.2)]}
		turns := 1
		for turns < 12 && g.rng.Float64() < 0.7 {
			turns++
		}
		t := start
		for k := 0; k < turns && t < dur; k++ {
			prompt = append(prompt, g.seg(g.lognormal(120, 1.0, 8, 4000)))
			out := g.seg(g.lognormal(220, 0.8, 8, 2048))
			g.emit(&items, t, prompt, out, ClassChat)
			prompt = append(prompt, out)
			t += float64(out.len)*0.03 + 5 + g.rng.ExpFloat64()*25
		}
	}
	return items
}

func (g *gen) agent(rate, dur float64) []Item {
	var items []Item
	const variants = 4
	sys := make([]segment, variants)
	for i := range sys {
		sys[i] = g.seg(8000 + g.rng.IntN(6000))
	}
	meanSteps := 18.0
	for _, start := range g.arrivals(rate/meanSteps, dur) {
		g.session++
		prompt := sequence{sys[g.zipf(variants, 1.0)], g.seg(g.lognormal(300, 0.8, 30, 4000))}
		steps := 6 + g.rng.IntN(25)
		t := start
		for k := 0; k < steps && t < dur && prompt.tokens() < 100000; k++ {
			out := g.seg(g.lognormal(180, 0.7, 16, 1500))
			g.emit(&items, t, prompt, out, ClassAgent)
			prompt = append(prompt, out, g.seg(g.lognormal(1500, 1.0, 50, 12000)))
			t += float64(out.len)*0.03 + 0.5 + g.rng.ExpFloat64()*3
		}
	}
	return items
}

func (g *gen) rag(rate, dur float64) []Item {
	var items []Item
	sys := []segment{g.seg(800), g.seg(700), g.seg(900)}
	const pool = 2000
	docs := make([]segment, pool)
	for i := range docs {
		docs[i] = g.seg(400 + g.rng.IntN(800))
	}
	for _, t := range g.arrivals(rate, dur) {
		g.session++
		prompt := sequence{sys[g.zipf(len(sys), 1.5)]}
		picked := map[int]bool{}
		for len(picked) < 4 {
			d := g.zipfFast(pool)
			if !picked[d] {
				picked[d] = true
				prompt = append(prompt, docs[d])
			}
		}
		prompt = append(prompt, g.seg(g.lognormal(60, 0.6, 8, 400)))
		g.emit(&items, t, prompt, g.seg(g.lognormal(180, 0.7, 16, 1500)), ClassRAG)
	}
	return items
}

func (g *gen) zipfFast(n int) int {
	z := rand.NewZipf(g.rng, 1.1, 1, uint64(n-1))
	return int(z.Uint64())
}
