package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/emulator"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/engine"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/experiment"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/gateway"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/loadgen"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/server"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/ui"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/workload"
)

func logger() *slog.Logger {
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
}

func serve(ctx context.Context, addr string, h http.Handler, log *slog.Logger) error {
	srv := &http.Server{Addr: addr, Handler: h, ReadHeaderTimeout: 10 * time.Second}
	errc := make(chan error, 1)
	go func() { errc <- srv.ListenAndServe() }()
	select {
	case <-ctx.Done():
		shut, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return srv.Shutdown(shut)
	case err := <-errc:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	}
}

func signalContext() (context.Context, context.CancelFunc) {
	return signal.NotifyContext(context.Background(), os.Interrupt)
}

func cmdGateway(args []string) error {
	fs := flag.NewFlagSet("gateway", flag.ExitOnError)
	cfgPath := fs.String("config", "configs/gateway.json", "gateway configuration file")
	listen := fs.String("listen", "", "override listen address")
	pol := fs.String("policy", "", "override routing policy")
	results := fs.String("results", "results", "directory with benchmark results for the dashboard")
	fs.Parse(args)
	cfg, err := gateway.LoadConfig(*cfgPath)
	if err != nil {
		return err
	}
	if *listen != "" {
		cfg.Listen = *listen
	}
	if *pol != "" {
		cfg.Policy = *pol
	}
	log := logger()
	g, err := gateway.New(cfg, log)
	if err != nil {
		return err
	}
	ctx, cancel := signalContext()
	defer cancel()
	g.Run(ctx)
	h := server.New(*results, ui.FS(), g.Handler()).Handler()
	var names []string
	for _, b := range cfg.Backends {
		names = append(names, b.Name+"="+b.URL)
	}
	log.Info("gateway listening", "addr", cfg.Listen, "policy", cfg.Policy, "backends", strings.Join(names, ", "))
	return serve(ctx, cfg.Listen, h, log)
}

func cmdEmulate(args []string) error {
	fs := flag.NewFlagSet("emulate", flag.ExitOnError)
	listen := fs.String("listen", ":9001", "listen address")
	name := fs.String("name", "", "replica name")
	model := fs.String("model", "llama-3.1-8b", "model key")
	hw := fs.String("hardware", "h100", "accelerator key")
	tp := fs.Int("tp", 1, "tensor parallel degree")
	speed := fs.Float64("speed", 1, "simulated seconds per wall second")
	bias := fs.Float64("bias", 1, "multiplier applied to modeled step times")
	jitter := fs.Float64("jitter", 0.05, "lognormal step time jitter")
	chunk := fs.Int("max-batched-tokens", 8192, "scheduler token budget per step")
	kv := fs.Int("kv-tokens", 0, "override KV cache capacity in tokens")
	host := fs.Float64("host-gb", 0, "host DRAM KV offload capacity per GPU in GB")
	fs.Parse(args)
	ec := engine.DefaultConfig()
	ec.MaxBatchedTokens = *chunk
	ec.KVTokenOverride = *kv
	ec.HostGBPerGPU = *host
	if *name == "" {
		*name = "emu" + strings.TrimPrefix(*listen, ":")
	}
	log := logger()
	s, err := emulator.New(emulator.Config{Name: *name, Model: *model, Hardware: *hw, TP: *tp, Engine: ec, Speed: *speed, Bias: *bias, Jitter: *jitter}, log)
	if err != nil {
		return err
	}
	ctx, cancel := signalContext()
	defer cancel()
	go s.Run(ctx)
	log.Info("emulator listening", "addr", *listen, "model", *model, "hardware", *hw, "tp", *tp, "bias", *bias)
	return serve(ctx, *listen, s.Handler(), log)
}

func cmdLoadgen(args []string) error {
	fs := flag.NewFlagSet("loadgen", flag.ExitOnError)
	ff := addFleetFlags(fs)
	target := fs.String("target", "http://localhost:8080", "gateway or server base URL")
	scale := fs.Float64("scale", 1, "arrival rate multiplier")
	tokenScale := fs.Float64("token-scale", 0.125, "prompt length multiplier")
	cpt := fs.Float64("chars-per-token", 4, "characters of synthesized prompt per server token (see 'abyss calibrate')")
	blockBytes := fs.Int("block-bytes", 256, "the gateway's prefix block size in bytes")
	outScale := fs.Float64("output-scale", 0.25, "output length multiplier")
	maxOut := fs.Int("max-output", 512, "cap on generated tokens per request")
	modelName := fs.String("model-name", "abyss-demo", "model field sent in requests")
	out := fs.String("out", "", "write per-request results as JSON")
	fs.Parse(args)
	p, err := ff.resolve()
	if err != nil {
		return err
	}
	w, err := experiment.Load(p.Workload)
	if err != nil {
		return err
	}
	w = w.Scale(*scale)
	slo := p.Cluster.SLO
	slo = workload.SLO{TTFTBase: slo.TTFTBase, TTFTPerK: slo.TTFTPerK / *tokenScale, TPOT: slo.TPOT}
	fmt.Printf("replaying %d requests from %s against %s\n", len(w.Items), p.Name, *target)
	ctx, cancel := signalContext()
	defer cancel()
	last := time.Now()
	rep, err := loadgen.Run(ctx, loadgen.Config{Target: strings.TrimRight(*target, "/"), Model: *modelName, TokenScale: *tokenScale, OutputScale: *outScale, MaxOutput: *maxOut, SLO: slo, CharsPerToken: *cpt, BlockBytes: *blockBytes,
		Progress: func(done, total int) {
			if time.Since(last) > 2*time.Second || done == total {
				last = time.Now()
				fmt.Fprintf(os.Stderr, "\r%d/%d completed", done, total)
			}
		}}, w)
	fmt.Fprintln(os.Stderr)
	if rep != nil {
		fmt.Print(rep.String())
		if *out != "" {
			b, _ := json.MarshalIndent(map[string]any{"summary": rep, "requests": rep.Results}, "", " ")
			os.WriteFile(*out, b, 0o644)
		}
	}
	return err
}

func cmdStudio(args []string) error {
	fs := flag.NewFlagSet("studio", flag.ExitOnError)
	listen := fs.String("listen", ":8090", "listen address")
	results := fs.String("results", "results", "directory with benchmark results")
	traces := fs.String("traces", "traces", "directory with trace files")
	fs.Parse(args)
	experiment.TraceDir = *traces
	log := logger()
	ctx, cancel := signalContext()
	defer cancel()
	log.Info("studio listening", "addr", *listen, "url", "http://localhost"+*listen)
	return serve(ctx, *listen, server.New(*results, ui.FS(), nil).Handler(), log)
}
