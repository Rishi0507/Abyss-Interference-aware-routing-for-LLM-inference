package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"text/tabwriter"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/calibrate"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/perf"
)

func cmdCalibrate(args []string) error {
	fs := flag.NewFlagSet("calibrate", flag.ExitOnError)
	target := fs.String("target", "http://localhost:8000", "OpenAI-compatible server to measure")
	model := fs.String("model", "llama-3.1-8b", "model key from 'abyss catalog'")
	hw := fs.String("hardware", "h100", "accelerator key from 'abyss catalog'")
	tp := fs.Int("tp", 1, "tensor parallel degree of the server")
	name := fs.String("model-name", "", "model field sent in requests (default: any)")
	chunk := fs.Int("max-batched-tokens", 8192, "the server's scheduler token budget")
	sizes := fs.String("prefill", "256,1024,2048,4096,8192,16384", "prompt lengths to measure")
	batches := fs.String("batches", "1,4,16,32", "decode batch sizes to measure")
	repeats := fs.Int("repeats", 3, "repetitions per prompt length")
	out := fs.String("out", "", "write the fitted calibration to this JSON file")
	fs.Parse(args)
	opt := calibrate.Options{Target: strings.TrimRight(*target, "/"), Model: *name, Repeats: *repeats,
		PrefillSizes: ints(*sizes), DecodeBatches: ints(*batches), Progress: func(s string) { fmt.Println(" ", s) }}
	if opt.Model == "" {
		opt.Model = *model
	}
	fmt.Printf("measuring %s\n", opt.Target)
	ctx, cancel := signalContext()
	defer cancel()
	samples, cpt, err := calibrate.Measure(ctx, opt)
	if err != nil {
		return err
	}
	res, err := calibrate.Fit(perf.Deployment{ModelKey: *model, HardwareKey: *hw, TP: *tp}, samples, *chunk)
	if err != nil {
		return err
	}
	res.CharsPerTok = cpt
	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tw, "\nmeasurement\tmeasured\tfitted model\tdefault model\t")
	for _, r := range res.Rows {
		label := fmt.Sprintf("prefill %d tokens", r.Sample.Tokens)
		if r.Sample.Kind == "decode" {
			label = fmt.Sprintf("decode batch %d", r.Sample.Batch)
		}
		fmt.Fprintf(tw, "%s\t%.2f ms\t%.2f ms\t%.2f ms\t\n", label, 1000*r.Sample.Seconds, 1000*r.Predicted, 1000*r.Default)
	}
	tw.Flush()
	e := res.Efficiency
	fmt.Printf("\nfitted: GEMM %.3f, attention %.3f, bandwidth %.3f, step overhead %.2f ms, network %.2f ms\n", e.GEMM, e.Attention, e.Bandwidth, 1000*e.StepOverhead, 1000*res.Network)
	fmt.Printf("median error %.1f%% (max %.1f%%), default model %.1f%%\n", 100*res.MedianErr, 100*res.MaxErr, 100*res.DefaultErr)
	if *out != "" {
		b, _ := json.MarshalIndent(res, "", "  ")
		if err := os.WriteFile(*out, b, 0o644); err != nil {
			return err
		}
		fmt.Println("wrote", *out)
	}
	return nil
}

func ints(s string) []int {
	var out []int
	for _, x := range splitList(s) {
		var v int
		if _, err := fmt.Sscanf(x, "%d", &v); err == nil && v > 0 {
			out = append(out, v)
		}
	}
	return out
}

func loadCalibration(path string) (*perf.Efficiency, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var r calibrate.Result
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &r.Efficiency, nil
}
