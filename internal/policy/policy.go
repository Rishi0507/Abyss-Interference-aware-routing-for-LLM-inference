package policy

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/engine"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/perf"
)

type Request struct {
	ID        int64
	Session   int64
	Class     uint8
	Arrival   float64
	Input     int
	MaxTokens int
	Hashes    []uint64
	TTFTSLO   float64
	TPOTSLO   float64
	Weight    float64

	Backend    int
	Dispatched float64
	FirstAt    float64
	LastAt     float64
	Generated  int
	Done       bool
	DoneAt     float64
	Deferrals  int
	PredTTFT   float64
	PredTPOT   float64
	Oracle     int
	Ext        any
}

func (r *Request) Blocks() int { return len(r.Hashes) }

func (r *Request) weight() float64 {
	if r.Weight > 0 {
		return r.Weight
	}
	return 1
}

type Gauges struct {
	Waiting int
	Running int
	KVUsage float64
	At      float64
}

type Backend struct {
	Index     int
	Name      string
	Rep       *perf.Replica
	EngineCfg engine.Config
	Inflight  []*Request
	Gauges    Gauges
	Match     func(hashes []uint64) int
	Healthy   bool
}

func (b *Backend) Outstanding() int { return len(b.Inflight) }

func (b *Backend) Remove(r *Request) {
	for i, x := range b.Inflight {
		if x == r {
			copy(b.Inflight[i:], b.Inflight[i+1:])
			b.Inflight[len(b.Inflight)-1] = nil
			b.Inflight = b.Inflight[:len(b.Inflight)-1]
			return
		}
	}
}

type Fleet struct {
	Now         float64
	Backends    []*Backend
	BlockTokens int
}

func (f *Fleet) Healthy() []*Backend {
	out := make([]*Backend, 0, len(f.Backends))
	for _, b := range f.Backends {
		if b.Healthy {
			out = append(out, b)
		}
	}
	return out
}

type Action uint8

const (
	Dispatch Action = iota
	Defer
	Reject
)

func (a Action) String() string {
	return [...]string{"dispatch", "defer", "reject"}[a]
}

type Cost struct {
	Backend      int     `json:"backend"`
	Own          float64 `json:"own"`
	Interference float64 `json:"interference"`
	Cache        float64 `json:"cache"`
	Delay        float64 `json:"delay"`
	Total        float64 `json:"total"`
	PredTTFT     float64 `json:"pred_ttft"`
	PredTPOT     float64 `json:"pred_tpot"`
	HitBlocks    int     `json:"hit_blocks"`
	Evicted      int     `json:"evicted_blocks"`
}

type Decision struct {
	Action  Action
	Backend int
	Costs   []Cost
	Reason  string
}

type Policy interface {
	Name() string
	Route(f *Fleet, r *Request) Decision
	Dispatched(f *Fleet, r *Request)
	FirstToken(f *Fleet, r *Request)
	Finished(f *Fleet, r *Request)
}

type hooks struct{}

func (hooks) Dispatched(*Fleet, *Request) {}
func (hooks) FirstToken(*Fleet, *Request) {}
func (hooks) Finished(*Fleet, *Request)   {}

type Factory func(seed uint64) Policy

type ParamFactory func(seed uint64, args []float64) (Policy, error)

var (
	registry = map[string]Factory{}
	params   = map[string]ParamFactory{}
)

func register(name string, f Factory) { registry[name] = f }

func registerParams(name string, f ParamFactory) { params[name] = f }

func New(spec string, seed uint64) (Policy, error) {
	spec = strings.ToLower(strings.TrimSpace(spec))
	name, rest, hasArgs := strings.Cut(spec, ":")
	if hasArgs {
		pf, ok := params[name]
		if !ok {
			return nil, fmt.Errorf("policy %q takes no parameters", name)
		}
		var args []float64
		for _, s := range strings.Split(rest, ",") {
			v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
			if err != nil {
				return nil, fmt.Errorf("policy %q: bad parameter %q", name, s)
			}
			args = append(args, v)
		}
		p, err := pf(seed, args)
		if err != nil {
			return nil, fmt.Errorf("policy %q: %w", name, err)
		}
		if n, ok := p.(interface{ setName(string) }); ok {
			n.setName(spec)
		}
		return p, nil
	}
	f, ok := registry[name]
	if !ok {
		return nil, fmt.Errorf("unknown policy %q (known: %s)", name, strings.Join(Names(), ", "))
	}
	return f(seed), nil
}

func want(args []float64, n int) error {
	if len(args) != n {
		return fmt.Errorf("expected %d parameters, got %d", n, len(args))
	}
	return nil
}

type named struct{ label string }

func (n *named) setName(s string) { n.label = s }

func (n *named) nameOr(def string) string {
	if n.label != "" {
		return n.label
	}
	return def
}

func Names() []string {
	out := make([]string, 0, len(registry))
	for k := range registry {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
