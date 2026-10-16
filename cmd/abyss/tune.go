package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/experiment"
)

func cmdTune(args []string) error {
	fs := flag.NewFlagSet("tune", flag.ExitOnError)
	ff := addFleetFlags(fs)
	scales := fs.String("scales", "0.75,1", "load levels used for tuning and evaluation")
	tuneSeed := fs.Int("tune-seed", 0, "trace window or generator seed used for tuning")
	evalSeeds := fs.String("eval-seeds", "1,2", "held-out windows or seeds used for evaluation")
	parallel := fs.Int("parallel", 2, "simulations to run at once")
	extra := fs.String("compare", "abyss,abyss-selfish,llm-d-predicted,preble,least-work", "policies evaluated alongside the tuned baselines")
	out := fs.String("out", "", "output JSON path (default results/tuned/<preset>.json)")
	fs.Parse(args)
	p, err := ff.resolve()
	if err != nil {
		return err
	}
	var sc []float64
	for _, s := range splitList(*scales) {
		var v float64
		if _, err := fmt.Sscanf(s, "%g", &v); err == nil {
			sc = append(sc, v)
		}
	}
	start := time.Now()
	last := time.Now()
	res, err := experiment.Tune(p, sc, *tuneSeed, ints0(*evalSeeds), *parallel, splitPolicies(*extra), func(stage string, d, t int) {
		if time.Since(last) > 3*time.Second || d == t {
			last = time.Now()
			fmt.Fprintf(os.Stderr, "%s %d/%d\n", stage, d, t)
		}
	})
	if err != nil {
		return err
	}
	fmt.Println("\nbest parameters on the tuning window:")
	for _, c := range res.Choices {
		fmt.Printf("  %-8s %-22s attainment %.1f%% (%d candidates)\n", c.Family, c.Best, 100*c.TuneScore, c.Candidates)
	}
	fmt.Println("\nheld-out evaluation:")
	printPoints(res.Eval.Points)
	if *out == "" {
		*out = filepath.Join("results", "tuned", p.Name+".json")
	}
	if err := writeJSON(*out, res); err != nil {
		return err
	}
	fmt.Printf("\nwrote %s in %s\n", *out, time.Since(start).Round(time.Second))
	return nil
}

func ints0(s string) []int {
	var out []int
	for _, x := range splitList(s) {
		var v int
		if _, err := fmt.Sscanf(x, "%d", &v); err == nil && v >= 0 {
			out = append(out, v)
		}
	}
	return out
}
