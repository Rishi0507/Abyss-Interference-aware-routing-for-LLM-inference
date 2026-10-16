package experiment

import (
	"fmt"
	"sort"
	"strings"
)

type Family struct {
	Name    string   `json:"name"`
	Default string   `json:"default"`
	Grid    []string `json:"grid"`
}

func Families() []Family {
	var llmd, sgl []string
	for _, a := range []float64{0, 1, 2, 3, 5, 8} {
		for _, b := range []float64{0, 1, 2, 4} {
			for _, c := range []float64{0, 1, 2, 4} {
				if a+b+c == 0 {
					continue
				}
				llmd = append(llmd, fmt.Sprintf("llm-d:%g,%g,%g", a, b, c))
			}
		}
	}
	for _, t := range []float64{0.1, 0.3, 0.5, 0.7, 0.9} {
		for _, a := range []float64{2, 8, 32, 64} {
			for _, r := range []float64{1.05, 1.25, 1.5, 2} {
				sgl = append(sgl, fmt.Sprintf("sglang:%g,%g,%g", t, a, r))
			}
		}
	}
	var dyn, ch []string
	for _, w := range []float64{0, 0.25, 0.5, 1, 2, 4, 8} {
		dyn = append(dyn, fmt.Sprintf("dynamo:%g", w))
	}
	for _, l := range []float64{1.05, 1.1, 1.25, 1.5, 2, 3} {
		ch = append(ch, fmt.Sprintf("chwbl:%g", l))
	}
	var si []string
	for _, c := range []int{4, 8, 12, 16, 24, 32, 40, 48, 64, 80} {
		for _, n := range []int{1, 2, 3, 4, 5, 6} {
			si = append(si, fmt.Sprintf("sita:%d,%d", c*1024, n))
		}
	}
	return []Family{
		{Name: "llm-d", Default: "llm-d", Grid: llmd},
		{Name: "dynamo", Default: "dynamo", Grid: dyn},
		{Name: "sglang", Default: "sglang", Grid: sgl},
		{Name: "chwbl", Default: "chwbl", Grid: ch},
		{Name: "sita", Default: "sita", Grid: si},
	}
}

type TuneChoice struct {
	Family     string  `json:"family"`
	Best       string  `json:"best"`
	TuneScore  float64 `json:"tuning_attainment"`
	Candidates int     `json:"candidates"`
}

type TuneResult struct {
	Preset    string       `json:"preset"`
	TuneSeed  int          `json:"tuning_seed"`
	EvalSeeds []int        `json:"evaluation_seeds"`
	Scales    []float64    `json:"scales"`
	Choices   []TuneChoice `json:"choices"`
	Eval      *Sweep       `json:"evaluation"`
}

func Tune(p Preset, scales []float64, tuneSeed int, evalSeeds []int, parallel int, extra []string, progress func(stage string, done, total int)) (*TuneResult, error) {
	fams := Families()
	var grid []string
	for _, f := range fams {
		grid = append(grid, f.Grid...)
	}
	sw, err := RunSweep(p, Spec{Policies: grid, Scales: scales, SeedList: []int{tuneSeed}, Parallel: parallel}, func(d, t int, _ Run) {
		if progress != nil {
			progress("tuning", d, t)
		}
	})
	if err != nil {
		return nil, err
	}
	score := map[string]float64{}
	count := map[string]int{}
	for _, r := range sw.Runs {
		if r.Result != nil {
			score[r.Policy] += r.Result.Summary.Attainment
			count[r.Policy]++
		}
	}
	res := &TuneResult{Preset: p.Name, TuneSeed: tuneSeed, EvalSeeds: evalSeeds, Scales: scales}
	eval := append([]string(nil), extra...)
	for _, f := range fams {
		best, bestScore := "", -1.0
		for _, g := range f.Grid {
			if count[g] == 0 {
				continue
			}
			s := score[g] / float64(count[g])
			if s > bestScore {
				best, bestScore = g, s
			}
		}
		res.Choices = append(res.Choices, TuneChoice{Family: f.Name, Best: best, TuneScore: bestScore, Candidates: len(f.Grid)})
		eval = append(eval, f.Default)
		if best != "" && !strings.EqualFold(best, f.Default) {
			eval = append(eval, best)
		}
	}
	sort.SliceStable(res.Choices, func(i, j int) bool { return res.Choices[i].Family < res.Choices[j].Family })
	ev, err := RunSweep(p, Spec{Policies: dedupe(eval), Scales: scales, SeedList: evalSeeds, Parallel: parallel}, func(d, t int, _ Run) {
		if progress != nil {
			progress("evaluation", d, t)
		}
	})
	if err != nil {
		return nil, err
	}
	res.Eval = ev
	return res, nil
}

func dedupe(xs []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, x := range xs {
		if !seen[x] {
			seen[x] = true
			out = append(out, x)
		}
	}
	return out
}
