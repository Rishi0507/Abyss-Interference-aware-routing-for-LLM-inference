package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime/pprof"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/cluster"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/experiment"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/metrics"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/perf"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/policy"
)

const defaultPolicies = "round-robin,least-request,chwbl,sglang,llm-d,dynamo,abyss-selfish,abyss"

type fleetFlags struct {
	preset   *string
	traceDir *string
	trace    *string
	kind     *string
	rate     *float64
	duration *float64
	model    *string
	hw       *string
	tp       *int
	replicas *int
	ttftBase *float64
	ttftPerK *float64
	tpot     *float64
	seed     *uint64
	calib    *string
	longPf   *int
	chunk    *int
	biasMean *float64
	biasSpr  *float64
	gemm     *float64
	attn     *float64
	bw       *float64
	tpotMin  *int
}

func addFleetFlags(fs *flag.FlagSet) *fleetFlags {
	return &fleetFlags{
		preset:   fs.String("preset", "conversation-70b", "experiment preset (see 'abyss catalog')"),
		traceDir: fs.String("traces", "traces", "directory holding the Mooncake trace files"),
		trace:    fs.String("trace", "", "override: Mooncake-format JSONL trace path"),
		kind:     fs.String("workload", "", "override: synthetic workload kind (chat, agent, rag, mixed)"),
		rate:     fs.Float64("rate", 0, "override: synthetic request rate (req/s)"),
		duration: fs.Float64("duration", 0, "override: workload duration in seconds"),
		model:    fs.String("model", "", "override: model key"),
		hw:       fs.String("hardware", "", "override: accelerator key"),
		tp:       fs.Int("tp", 0, "override: tensor parallel degree"),
		replicas: fs.Int("replicas", 0, "override: replica count"),
		ttftBase: fs.Float64("slo-ttft", 0, "override: TTFT SLO base in seconds"),
		ttftPerK: fs.Float64("slo-ttft-per-k", -1, "override: TTFT SLO allowance per 1k prompt tokens"),
		tpot:     fs.Float64("slo-tpot", 0, "override: TPOT SLO in seconds"),
		seed:     fs.Uint64("seed", 0, "override: random seed"),
		calib:    fs.String("calibration", "", "apply a 'abyss calibrate' result to every replica"),
		longPf:   fs.Int("long-prefill", 0, "override: vLLM long_prefill_token_threshold"),
		chunk:    fs.Int("max-batched-tokens", 0, "override: scheduler token budget per step"),
		biasMean: fs.Float64("bias-mean", 0, "override: mean hidden speed bias of replicas (1 = as modeled)"),
		biasSpr:  fs.Float64("bias-spread", -1, "override: spread of hidden per-replica speed bias"),
		gemm:     fs.Float64("hidden-gemm", 0, "multiplier on the real replicas' matrix-multiply efficiency, unknown to the router (1 = as modeled)"),
		attn:     fs.Float64("hidden-attention", 0, "multiplier on the real replicas' attention efficiency, unknown to the router"),
		bw:       fs.Float64("hidden-bandwidth", 0, "multiplier on the real replicas' memory bandwidth efficiency, unknown to the router"),
		tpotMin:  fs.Int("slo-tpot-min-tokens", -1, "override: apply the TPOT objective only to responses with at least this many tokens"),
	}
}

func (f *fleetFlags) resolve() (experiment.Preset, error) {
	experiment.TraceDir = *f.traceDir
	p, err := experiment.Find(*f.preset)
	if err != nil {
		return p, err
	}
	if *f.trace != "" {
		p.Workload = experiment.WorkloadRef{Trace: *f.trace, Duration: p.Workload.Duration}
	}
	if *f.kind != "" {
		p.Workload = experiment.WorkloadRef{Kind: *f.kind, Rate: 4, Duration: 600, Seed: 1}
	}
	if *f.rate > 0 {
		p.Workload.Rate = *f.rate
	}
	if *f.duration > 0 {
		p.Workload.Duration = *f.duration
	}
	if *f.model != "" || *f.hw != "" || *f.tp > 0 || *f.replicas > 0 {
		spec := p.Cluster.Replicas[0]
		total := 0
		for _, r := range p.Cluster.Replicas {
			total += r.Count
		}
		spec.Count = total
		if *f.model != "" {
			spec.Model = *f.model
		}
		if *f.hw != "" {
			spec.Hardware = *f.hw
		}
		if *f.tp > 0 {
			spec.TP = *f.tp
		}
		if *f.replicas > 0 {
			spec.Count = *f.replicas
		}
		p.Cluster.Replicas = []cluster.ReplicaSpec{spec}
	}
	if *f.ttftBase > 0 {
		p.Cluster.SLO.TTFTBase = *f.ttftBase
	}
	if *f.ttftPerK >= 0 {
		p.Cluster.SLO.TTFTPerK = *f.ttftPerK
	}
	if *f.tpot > 0 {
		p.Cluster.SLO.TPOT = *f.tpot
	}
	if *f.seed > 0 {
		p.Cluster.Seed = *f.seed
	}
	if *f.calib != "" {
		eff, err := loadCalibration(*f.calib)
		if err != nil {
			return p, err
		}
		specs := append([]cluster.ReplicaSpec(nil), p.Cluster.Replicas...)
		for i := range specs {
			specs[i].Efficiency = eff
		}
		p.Cluster.Replicas = specs
	}
	if *f.longPf > 0 {
		p.Cluster.Engine.LongPrefill = *f.longPf
	}
	if *f.chunk > 0 {
		p.Cluster.Engine.MaxBatchedTokens = *f.chunk
	}
	if *f.biasMean > 0 {
		p.Cluster.Noise.BiasMean = *f.biasMean
	}
	if *f.biasSpr >= 0 {
		p.Cluster.Noise.BiasSpread = *f.biasSpr
	}
	if *f.gemm > 0 {
		p.Cluster.Noise.GEMM = *f.gemm
	}
	if *f.attn > 0 {
		p.Cluster.Noise.Attention = *f.attn
	}
	if *f.bw > 0 {
		p.Cluster.Noise.Bandwidth = *f.bw
	}
	if *f.tpotMin >= 0 {
		p.Cluster.SLO.TPOTMinTokens = *f.tpotMin
	}
	return p, nil
}

func cmdSim(args []string) error {
	fs := flag.NewFlagSet("sim", flag.ExitOnError)
	ff := addFleetFlags(fs)
	pols := fs.String("policies", defaultPolicies, "comma separated policies")
	scale := fs.Float64("scale", 1, "arrival rate multiplier")
	seeds := fs.Int("seeds", 1, "independent trace windows or generator seeds")
	out := fs.String("out", "", "write JSON results to this path")
	verbose := fs.Bool("v", false, "print per-replica statistics")
	profile := fs.String("cpuprofile", "", "write a CPU profile to this path")
	fs.Parse(args)
	if *profile != "" {
		f, err := os.Create(*profile)
		if err != nil {
			return err
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	p, err := ff.resolve()
	if err != nil {
		return err
	}
	names := splitPolicies(*pols)
	fmt.Printf("preset %s, scale %.2fx, policies %s\n", p.Name, *scale, strings.Join(names, ", "))
	start := time.Now()
	sw, err := experiment.RunSweep(p, experiment.Spec{Policies: names, Scales: []float64{*scale}, Seeds: *seeds}, nil)
	if err != nil {
		return err
	}
	ws := sw.Workload
	fmt.Printf("workload: %d requests over %.0fs (%.2f req/s at 1x), mean input %.0f, mean output %.0f, prefix reuse %.1f%%\n",
		ws.Requests, ws.Duration, ws.Rate, ws.MeanInput, ws.MeanOutput, 100*ws.ReuseRatio)
	fmt.Printf("fleet: %s\n\n", strings.Join(sw.Fleet, ", "))
	printTable(sw.Runs)
	if *verbose {
		for _, r := range sw.Runs {
			if r.Result == nil {
				continue
			}
			fmt.Printf("\n%s\n", r.Policy)
			tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
			fmt.Fprintln(tw, "  replica\trequests\tbusy\thit rate\tpreemptions\tevictions\tmean batch\tprefill share")
			var pf, dc float64
			for _, rs := range r.Result.Summary.Replicas {
				pf += float64(rs.Prefill)
				dc += float64(rs.Decode)
			}
			for i, rs := range r.Result.Summary.Replicas {
				fmt.Fprintf(tw, "  %d %s\t%d\t%.0f%%\t%.1f%%\t%d\t%d\t%.0f\t%.0f%% of prefill, %.0f%% of decode\n", i, rs.Name, rs.Requests, 100*rs.Busy, 100*rs.HitRate, rs.Preemptions, rs.Evictions, rs.MeanBatch, 100*float64(rs.Prefill)/pf, 100*float64(rs.Decode)/dc)
			}
			fmt.Fprintln(tw, "  attainment by prompt length:", bucketLine(r.Result.Summary.LengthAttainment))
			tw.Flush()
		}
	}
	fmt.Printf("\ncompleted in %s\n", time.Since(start).Round(time.Millisecond))
	if *out != "" {
		return writeJSON(*out, sw)
	}
	return nil
}

func printTable(runs []experiment.Run) {
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "policy\tscale\tSLO met\tTTFT ok\tTPOT ok\tgoodput\tTTFT p50\tTTFT p99\tTPOT p50\tTPOT p99\thit rate\tpreempt\trejected\ttimeouts\t$/M good tok\tdecision\t")
	for _, r := range runs {
		if r.Err != "" {
			fmt.Fprintf(tw, "%s\t%.2f\terror: %s\t\n", r.Policy, r.Scale, r.Err)
			continue
		}
		s := r.Result.Summary
		fmt.Fprintf(tw, "%s\t%.2f\t%.1f%%\t%.1f%%\t%.1f%%\t%.2f/s\t%s\t%s\t%s\t%s\t%.1f%%\t%d\t%d\t%d\t%.2f\t%s\t\n",
			r.Policy, r.Scale, 100*s.Attainment, 100*s.TTFTAttain, 100*s.TPOTAttain, s.Goodput, ms(s.TTFT.P50), ms(s.TTFT.P99), ms(s.TPOT.P50), ms(s.TPOT.P99),
			100*s.HitRate, s.Preemptions, s.Rejected, s.TimedOut, s.CostPerMGood, us(s.DecisionMicros.Mean))
	}
	tw.Flush()
}

func ms(s float64) string {
	if s >= 10 {
		return fmt.Sprintf("%.1fs", s)
	}
	return fmt.Sprintf("%.0fms", s*1000)
}

func us(v float64) string {
	if v >= 1000 {
		return fmt.Sprintf("%.1fms", v/1000)
	}
	return fmt.Sprintf("%.0fus", v)
}

func cmdBench(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	ff := addFleetFlags(fs)
	pols := fs.String("policies", defaultPolicies, "comma separated policies")
	scales := fs.String("scales", "", "comma separated rate multipliers (default: preset sweep)")
	out := fs.String("out", "", "output JSON path (default results/<preset>.json)")
	seeds := fs.Int("seeds", 3, "independent trace windows or generator seeds")
	parallel := fs.Int("parallel", 0, "simulations to run at once (default: number of CPUs)")
	timeline := fs.Bool("timeline", true, "record per-bucket timelines")
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
	names := splitPolicies(*pols)
	start := time.Now()
	sw, err := experiment.RunSweep(p, experiment.Spec{Policies: names, Scales: sc, Seeds: *seeds, Timeline: *timeline, Parallel: *parallel}, func(done, total int, r experiment.Run) {
		status := "ok"
		if r.Err != "" {
			status = r.Err
		} else if r.Result != nil {
			status = fmt.Sprintf("SLO met %.1f%% in %.1fs", 100*r.Result.Summary.Attainment, r.Result.Wall)
		}
		fmt.Fprintf(os.Stderr, "[%d/%d] %-22s %.2fx seed %d  %s\n", done, total, r.Policy, r.Scale, r.Seed, status)
	})
	if err != nil {
		return err
	}
	fmt.Println()
	printPoints(sw.Points)
	fmt.Printf("\ncompleted in %s\n", time.Since(start).Round(time.Second))
	if *out == "" {
		*out = filepath.Join("results", p.Name+".json")
	}
	if err := writeJSON(*out, sw); err != nil {
		return err
	}
	fmt.Println("wrote", *out)
	return nil
}

func cmdCatalog(args []string) error {
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "MODEL\tPARAMS\tLAYERS\tKV/TOKEN")
	for _, k := range perf.ModelKeys() {
		m, _ := perf.LookupModel(k)
		fmt.Fprintf(tw, "%s\t%.1fB\t%d\t%.0f KiB\n", k, m.TotalParams()/1e9, m.Layers, m.KVBytesPerToken()/1024)
	}
	fmt.Fprintln(tw, "\nACCELERATOR\tBF16 TFLOPS\tHBM TB/s\tMEMORY\t$/GPU-HR")
	for _, k := range perf.HardwareKeys() {
		h, _ := perf.LookupHardware(k)
		fmt.Fprintf(tw, "%s\t%.0f\t%.2f\t%.0f GB\t%.2f\n", k, h.FLOPS/1e12, h.MemBW/1e12, h.MemBytes/1e9, h.PricePerHour)
	}
	fmt.Fprintln(tw, "\nPRESET\tDESCRIPTION")
	for _, p := range experiment.Presets() {
		fmt.Fprintf(tw, "%s\t%s\n", p.Name, p.Description)
	}
	fmt.Fprintln(tw, "\nPOLICY")
	for _, n := range policy.Names() {
		fmt.Fprintf(tw, "%s\n", n)
	}
	return tw.Flush()
}

func cmdTrace(args []string) error {
	fs := flag.NewFlagSet("trace", flag.ExitOnError)
	ff := addFleetFlags(fs)
	fs.Parse(args)
	p, err := ff.resolve()
	if err != nil {
		return err
	}
	w, err := experiment.Load(p.Workload)
	if err != nil {
		return err
	}
	b, _ := json.MarshalIndent(w.Summarize(), "", "  ")
	fmt.Println(string(b))
	reps, cfgs, err := cluster.Build(p.Cluster, w.BlockTokens)
	if err != nil {
		return err
	}
	for i, r := range reps {
		fmt.Printf("replica %d: %s, KV capacity %d tokens, prefill 8k %.0fms, decode step b=32 ctx=8k %.1fms, chunk %d\n",
			i, r.Label, r.KVTokens, 1000*r.PrefillTime(8192, 0, cfgs[i].MaxBatchedTokens), 1000*r.DecodeStepTime(32, 8192), cfgs[i].MaxBatchedTokens)
	}
	return nil
}

func splitList(s string) []string {
	var out []string
	for _, x := range strings.Split(s, ",") {
		if x = strings.TrimSpace(x); x != "" {
			out = append(out, x)
		}
	}
	return out
}

func splitPolicies(s string) []string {
	var out []string
	for _, x := range splitList(s) {
		if _, err := strconv.ParseFloat(x, 64); err == nil && len(out) > 0 && strings.Contains(out[len(out)-1], ":") {
			out[len(out)-1] += "," + x
			continue
		}
		out = append(out, x)
	}
	seen := map[string]bool{}
	uniq := out[:0]
	for _, x := range out {
		if !seen[x] {
			seen[x] = true
			uniq = append(uniq, x)
		}
	}
	return uniq
}

func writeJSON(path string, v any) error {
	if dir := filepath.Dir(path); dir != "" {
		os.MkdirAll(dir, 0o755)
	}
	b, err := json.MarshalIndent(v, "", " ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func bucketLine(b []metrics.LengthBucket) string {
	var parts []string
	for _, x := range b {
		parts = append(parts, fmt.Sprintf("%s %.0f%% (n=%d)", x.Label, 100*x.Attainment, x.Requests))
	}
	return strings.Join(parts, ", ")
}

func printPoints(pts []experiment.Point) {
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "policy\tscale\treq/s\tSLO met\tsd\tgoodput\tTTFT p50\tTTFT p99\tTPOT p50\tTPOT p99\thit rate\ttimeouts\t$/M good tok\tdecision\t")
	for _, p := range pts {
		fmt.Fprintf(tw, "%s\t%.2f\t%.2f\t%.1f%%\t%.1f\t%.2f/s\t%s\t%s\t%s\t%s\t%.1f%%\t%.1f%%\t%.2f\t%s\t\n",
			p.Policy, p.Scale, p.Rate, 100*p.Attainment, 100*p.AttainSD, p.Goodput, ms(p.TTFTP50), ms(p.TTFTP99), ms(p.TPOTP50), ms(p.TPOTP99),
			100*p.HitRate, 100*p.TimedOut, p.CostPerM, us(p.DecisionUS))
	}
	tw.Flush()
}
