package server

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/cluster"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/experiment"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/metrics"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/perf"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/policy"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/workload"
)

type Server struct {
	ResultsDir string
	UI         fs.FS
	Live       http.Handler
	sem        chan struct{}
}

func New(resultsDir string, ui fs.FS, live http.Handler) *Server {
	return &Server{ResultsDir: resultsDir, UI: ui, Live: live, sem: make(chan struct{}, 2)}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/catalog", s.catalog)
	mux.HandleFunc("POST /api/simulate", s.simulate)
	mux.HandleFunc("GET /api/results", s.results)
	mux.HandleFunc("GET /api/results/{name}", s.result)
	mux.HandleFunc("GET /api/research", s.research)
	if s.Live != nil {
		mux.Handle("/v1/", s.Live)
		mux.Handle("/api/live/", s.Live)
		mux.Handle("GET /metrics", s.Live)
		mux.Handle("GET /healthz", s.Live)
	} else {
		mux.HandleFunc("GET /api/live/state", func(w http.ResponseWriter, r *http.Request) {
			writeJSON(w, http.StatusOK, map[string]any{"attached": false})
		})
	}
	if s.UI != nil {
		files := http.FileServer(http.FS(s.UI))
		mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
			p := strings.TrimPrefix(r.URL.Path, "/")
			if p == "" {
				p = "index.html"
			}
			if _, err := fs.Stat(s.UI, p); err != nil {
				r.URL.Path = "/"
			}
			if strings.HasPrefix(p, "assets/") {
				w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
			}
			files.ServeHTTP(w, r)
		})
	}
	return mux
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

type modelInfo struct {
	Key        string  `json:"key"`
	Name       string  `json:"name"`
	Params     float64 `json:"params_b"`
	KVPerToken float64 `json:"kv_kib_per_token"`
}

type hwInfo struct {
	Key    string  `json:"key"`
	Name   string  `json:"name"`
	TFLOPS float64 `json:"tflops"`
	HBM    float64 `json:"hbm_tbs"`
	Memory float64 `json:"memory_gb"`
	Price  float64 `json:"price"`
}

func (s *Server) catalog(w http.ResponseWriter, r *http.Request) {
	var ms []modelInfo
	for _, k := range perf.ModelKeys() {
		m, _ := perf.LookupModel(k)
		ms = append(ms, modelInfo{k, m.Name, m.TotalParams() / 1e9, m.KVBytesPerToken() / 1024})
	}
	var hs []hwInfo
	for _, k := range perf.HardwareKeys() {
		h, _ := perf.LookupHardware(k)
		hs = append(hs, hwInfo{k, h.Name, h.FLOPS / 1e12, h.MemBW / 1e12, h.MemBytes / 1e9, h.PricePerHour})
	}
	writeJSON(w, http.StatusOK, map[string]any{"models": ms, "hardware": hs, "policies": policy.Names(), "presets": experiment.Presets()})
}

type simRequest struct {
	Preset   string   `json:"preset"`
	Policies []string `json:"policies"`
	Scale    float64  `json:"scale"`
	Duration float64  `json:"duration"`
	Seed     int      `json:"seed"`
}

type decisionSample struct {
	T       float64       `json:"t"`
	Input   int           `json:"input"`
	Hit     int           `json:"hit_blocks"`
	Chosen  int           `json:"chosen"`
	Selfish int           `json:"selfish"`
	Costs   []policy.Cost `json:"costs"`
	TTFTSLO float64       `json:"ttft_slo"`
}

type simRun struct {
	Policy   string           `json:"policy"`
	Summary  metrics.Summary  `json:"summary"`
	Timeline []cluster.Bucket `json:"timeline"`
	Error    string           `json:"error,omitempty"`
}

type simResponse struct {
	Preset    experiment.Preset `json:"preset"`
	Workload  workload.Summary  `json:"workload"`
	Fleet     []string          `json:"fleet"`
	Runs      []simRun          `json:"runs"`
	Decisions []decisionSample  `json:"decisions"`
	Overrides float64           `json:"override_ratio"`
	Wall      float64           `json:"wall_s"`
}

func (s *Server) simulate(w http.ResponseWriter, r *http.Request) {
	var req simRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request"})
		return
	}
	p, err := experiment.Find(req.Preset)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	if req.Scale <= 0 {
		req.Scale = 1
	}
	if req.Duration > 0 {
		p.Workload.Duration = min(req.Duration, 1800)
	} else {
		p.Workload.Duration = min(p.Workload.Duration, 600)
	}
	if len(req.Policies) == 0 {
		req.Policies = []string{"round-robin", "llm-d", "dynamo", "abyss-selfish", "abyss"}
	}
	if len(req.Policies) > 10 {
		req.Policies = req.Policies[:10]
	}
	s.sem <- struct{}{}
	defer func() { <-s.sem }()
	start := time.Now()
	wl, err := experiment.Load(p.Workload)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	scaled := wl.Scale(req.Scale)
	cfg := p.Cluster
	cfg.Seed += uint64(req.Seed)
	resp := simResponse{Preset: p, Workload: scaled.Summarize()}
	resp.Runs = make([]simRun, len(req.Policies))
	var wg sync.WaitGroup
	var mu sync.Mutex
	var overrides, total int
	for i, name := range req.Policies {
		wg.Add(1)
		go func(i int, name string) {
			defer wg.Done()
			run := simRun{Policy: name}
			pol, err := policy.New(name, cfg.Seed)
			if err != nil {
				run.Error = err.Error()
				resp.Runs[i] = run
				return
			}
			opt := cluster.Options{Timeline: true, Buckets: 36}
			if name == "abyss" {
				if ex, ok := pol.(interface{ SetExplain(bool) }); ok {
					ex.SetExplain(true)
				}
				every := max(len(scaled.Items)/48, 1)
				n := 0
				opt.Explain = func(t float64, pr *policy.Request, d policy.Decision) {
					if d.Action != policy.Dispatch || len(d.Costs) < 2 {
						return
					}
					selfish := d.Costs[0].Backend
					best := d.Costs[0]
					for _, c := range d.Costs[1:] {
						if c.Own+0.05*c.PredTTFT/pr.TTFTSLO < best.Own+0.05*best.PredTTFT/pr.TTFTSLO {
							best = c
							selfish = c.Backend
						}
					}
					mu.Lock()
					total++
					if selfish != d.Backend {
						overrides++
					}
					n++
					if n%every == 0 || (selfish != d.Backend && len(resp.Decisions) < 96 && n%3 == 0) {
						hit := 0
						for _, c := range d.Costs {
							if c.Backend == d.Backend {
								hit = c.HitBlocks
							}
						}
						resp.Decisions = append(resp.Decisions, decisionSample{T: t, Input: pr.Input, Hit: hit, Chosen: d.Backend, Selfish: selfish, Costs: d.Costs, TTFTSLO: pr.TTFTSLO})
					}
					mu.Unlock()
				}
			}
			res, err := cluster.Run(cfg, scaled, pol, opt)
			if err != nil {
				run.Error = err.Error()
			} else {
				run.Summary = res.Summary
				run.Timeline = res.Timeline
				mu.Lock()
				resp.Fleet = res.Fleet
				mu.Unlock()
			}
			resp.Runs[i] = run
		}(i, name)
	}
	wg.Wait()
	sort.Slice(resp.Decisions, func(i, j int) bool { return resp.Decisions[i].T < resp.Decisions[j].T })
	if total > 0 {
		resp.Overrides = float64(overrides) / float64(total)
	}
	resp.Wall = time.Since(start).Seconds()
	writeJSON(w, http.StatusOK, resp)
}

type resultEntry struct {
	Name     string             `json:"name"`
	Title    string             `json:"title"`
	Preset   experiment.Preset  `json:"preset"`
	Workload workload.Summary   `json:"workload"`
	Fleet    []string           `json:"fleet"`
	Seeds    int                `json:"seeds"`
	Points   []experiment.Point `json:"points"`
	Scale    float64            `json:"detail_scale"`
	Details  []detail           `json:"details"`
}

type detail struct {
	Policy     string                 `json:"policy"`
	Attainment float64                `json:"slo_attainment"`
	Replicas   []metrics.ReplicaStat  `json:"replicas"`
	Lengths    []metrics.LengthBucket `json:"length_attainment"`
}

func (s *Server) load(name string) (*resultEntry, error) {
	b, err := os.ReadFile(filepath.Join(s.ResultsDir, name+".json"))
	if err != nil {
		return nil, err
	}
	var sw experiment.Sweep
	if err := json.Unmarshal(b, &sw); err != nil {
		return nil, err
	}
	if len(sw.Points) == 0 || len(sw.Fleet) == 0 {
		return nil, fmt.Errorf("%s is not a benchmark sweep", name)
	}
	e := &resultEntry{Name: name, Title: sw.Preset.Title, Preset: sw.Preset, Workload: sw.Workload, Fleet: sw.Fleet, Seeds: sw.Seeds, Points: sw.Points}
	best := -1.0
	for _, r := range sw.Runs {
		if best < 0 || abs(r.Scale-1) < abs(best-1) {
			best = r.Scale
		}
	}
	e.Scale = best
	for _, r := range sw.Runs {
		if r.Scale == best && r.Seed == 0 && r.Result != nil {
			e.Details = append(e.Details, detail{Policy: r.Policy, Attainment: r.Result.Summary.Attainment, Replicas: r.Result.Summary.Replicas, Lengths: r.Result.Summary.LengthAttainment})
		}
	}
	return e, nil
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func (s *Server) results(w http.ResponseWriter, r *http.Request) {
	entries, _ := os.ReadDir(s.ResultsDir)
	var out []resultEntry
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		if res, err := s.load(strings.TrimSuffix(e.Name(), ".json")); err == nil {
			out = append(out, *res)
		}
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) result(w http.ResponseWriter, r *http.Request) {
	name := filepath.Base(r.PathValue("name"))
	res, err := s.load(strings.TrimSuffix(name, ".json"))
	if err != nil {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
		return
	}
	writeJSON(w, http.StatusOK, res)
}

type researchSet struct {
	Name    string                  `json:"name"`
	Title   string                  `json:"title"`
	Choices []experiment.TuneChoice `json:"choices,omitempty"`
	Points  []experiment.Point      `json:"points"`
	Fleet   []string                `json:"fleet,omitempty"`
}

func (s *Server) readDir(dir string, fn func(name string, b []byte)) {
	entries, _ := os.ReadDir(filepath.Join(s.ResultsDir, dir))
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(s.ResultsDir, dir, e.Name()))
		if err == nil {
			fn(strings.TrimSuffix(e.Name(), ".json"), b)
		}
	}
}

func (s *Server) research(w http.ResponseWriter, r *http.Request) {
	out := map[string]any{}
	var tuned, robust, sens, scaling []researchSet
	s.readDir("tuned", func(name string, b []byte) {
		var t experiment.TuneResult
		if json.Unmarshal(b, &t) == nil && t.Eval != nil {
			tuned = append(tuned, researchSet{Name: name, Title: t.Eval.Preset.Title, Choices: t.Choices, Points: t.Eval.Points, Fleet: t.Eval.Fleet})
		}
	})
	load := func(dir string, dst *[]researchSet) {
		s.readDir(dir, func(name string, b []byte) {
			var sw experiment.Sweep
			if json.Unmarshal(b, &sw) == nil {
				*dst = append(*dst, researchSet{Name: name, Title: sw.Preset.Title, Points: sw.Points, Fleet: sw.Fleet})
			}
		})
	}
	load("robustness", &robust)
	load("sensitivity", &sens)
	load("scaling", &scaling)
	var calib json.RawMessage
	s.readDir("calibration", func(name string, b []byte) {
		if calib == nil {
			calib = b
		}
	})
	var micro json.RawMessage
	if b, err := os.ReadFile(filepath.Join(s.ResultsDir, "microbench.json")); err == nil {
		micro = b
	}
	out["tuned"], out["robustness"], out["sensitivity"], out["scaling"] = tuned, robust, sens, scaling
	out["calibration"], out["microbench"] = calib, micro
	writeJSON(w, http.StatusOK, out)
}
