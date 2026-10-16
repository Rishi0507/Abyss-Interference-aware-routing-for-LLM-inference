package policy

import (
	"math"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/engine"
)

type seen struct {
	at  float64
	gen int
}

type twin struct {
	eng             *engine.Engine
	reqs            map[int64]*engine.Request
	seen            map[int64]seen
	simFirst        map[int64]float64
	scale           float64
	pscale          float64
	sigTTFT         float64
	sigTPOT         float64
	inserted        map[uint64]float64
	evictAge        float64
	evictN          int
	blockCost       float64
	prefillPerToken float64
	prefillLoad     float64
	lastSync        float64
	obsTime         float64
	obsTok          float64
	simBusy         float64
	simPrefill      float64
	simSteps        int64
	lastPrefill     int64
	order           []*engine.Request
	base            *engine.Fork
	with            *engine.Fork
}

func newTwin(b *Backend, now float64) *twin {
	cfg := b.EngineCfg
	t := &twin{
		eng:      engine.New(cfg, b.Rep),
		reqs:     make(map[int64]*engine.Request),
		seen:     make(map[int64]seen),
		simFirst: make(map[int64]float64),
		scale:    1,
		pscale:   1,
		sigTTFT:  0.3,
		sigTPOT:  0.3,
		inserted: make(map[uint64]float64),
		lastSync: now,
		base:     engine.NewFork(),
		with:     engine.NewFork(),
	}
	t.eng.Listener = twinListener{t}
	t.eng.SplitScale = func(full, dec float64) (float64, float64) { return dec * t.scale, (full - dec) * t.pscale }
	t.eng.SetCacheHooks(func(h uint64) { t.inserted[h] = t.eng.Now() }, func(h uint64) {
		if at, ok := t.inserted[h]; ok {
			t.evictN++
			w := math.Max(0.02, 1/float64(t.evictN))
			t.evictAge = (1-w)*t.evictAge + w*(t.eng.Now()-at)
			delete(t.inserted, h)
		}
	})
	bt := t.eng.Cfg.BlockTokens
	chunk := t.eng.Cfg.MaxBatchedTokens
	t.blockCost = b.Rep.PrefillTime(bt, bt*8, chunk) - b.Rep.Eff.StepOverhead
	t.prefillPerToken = b.Rep.PrefillTime(chunk, chunk/2, chunk) / float64(chunk)
	return t
}

func (t *twin) residency() float64 {
	if t.evictN < 8 {
		return 600
	}
	return t.evictAge
}

func (t *twin) meanCtx(e *engine.Engine, fallback int) float64 {
	run := e.Running()
	if len(run) == 0 {
		return float64(fallback)
	}
	var s float64
	for _, x := range run {
		s += float64(x.KV)
	}
	return s / float64(len(run))
}

func (t *twin) sync(now float64) {
	if now <= t.lastSync {
		return
	}
	busy0 := t.eng.Stats.BusyTime
	pre0 := t.eng.Stats.PrefillBusy
	steps0 := t.eng.Stats.Steps
	t.eng.AdvanceTo(now)
	var obsDT, obsTok float64
	t.order = append(append(t.order[:0], t.eng.Running()...), t.eng.Waiting()...)
	for _, er := range t.order {
		pr := er.Ref.(*Request)
		id := pr.ID
		if pr.Done {
			continue
		}
		if pr.FirstAt >= 0 {
			if _, ok := t.seen[id]; !ok && pr.Dispatched >= 0 {
				sim, ok := t.simFirst[id]
				if !ok {
					sim = now + float64(er.Remaining())*t.prefillPerToken*t.pscale
				}
				t.observePrefill(pr.FirstAt-pr.Dispatched, sim-pr.Dispatched)
			}
			if er.State != engine.Active || er.Prefilling() {
				t.eng.ForceDecoding(er, pr.Generated)
			} else if pr.Generated > er.Generated {
				t.eng.SetGenerated(er, pr.Generated)
			}
			if prev, ok := t.seen[id]; ok && pr.Generated > prev.gen && pr.LastAt > prev.at {
				obsDT += pr.LastAt - prev.at
				obsTok += float64(pr.Generated - prev.gen)
			}
			t.seen[id] = seen{pr.LastAt, pr.Generated}
		} else if er.State == engine.Active && !er.Prefilling() {
			t.eng.HoldPrefill(er)
		}
	}
	dt := now - t.lastSync
	pf := t.eng.Stats.PrefillTokens - t.lastPrefill
	t.lastPrefill = t.eng.Stats.PrefillTokens
	load := float64(pf) * t.prefillPerToken * t.pscale / dt
	a := 1 - math.Exp(-dt/20)
	t.prefillLoad = (1-a)*t.prefillLoad + a*math.Min(load, 1)
	t.simBusy += t.eng.Stats.BusyTime - busy0
	t.simPrefill += t.eng.Stats.PrefillBusy - pre0
	t.simSteps += t.eng.Stats.Steps - steps0
	t.obsTime += obsDT
	t.obsTok += obsTok
	if t.obsTok >= 64 && t.simSteps >= 16 {
		observed := t.obsTime / t.obsTok
		dec := t.simBusy - t.simPrefill
		if observed > 0 && dec > 0.3*t.simBusy {
			r := (observed*float64(t.simSteps) - t.simPrefill) / dec
			r = math.Min(math.Max(r, 0.5), 2)
			t.scale = math.Min(math.Max(t.scale*math.Pow(r, 0.2), 0.33), 3)
		}
		t.obsTime, t.obsTok, t.simBusy, t.simPrefill, t.simSteps = 0, 0, 0, 0, 0
	}
	t.lastSync = now
}

type twinListener struct{ t *twin }

func (l twinListener) OnFirstToken(r *engine.Request, at float64) {
	if _, ok := l.t.simFirst[r.ID]; !ok {
		l.t.simFirst[r.ID] = at
	}
}

func (l twinListener) OnFinish(*engine.Request, float64) {}

func (t *twin) observePrefill(observed, simulated float64) {
	if observed <= 0 || simulated < 0.02 {
		return
	}
	e := math.Min(math.Max(math.Log(observed/simulated), -0.5), 0.5)
	t.pscale = math.Min(math.Max(t.pscale*math.Exp(0.025*e), 0.33), 3)
}
