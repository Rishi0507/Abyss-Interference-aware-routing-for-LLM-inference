package gateway

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/engine"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/perf"
	"github.com/Rishi0507/Abyss-Interference-aware-routing-for-LLM-inference/internal/workload"
)

type BackendConfig struct {
	Name     string           `json:"name"`
	URL      string           `json:"url"`
	Model    string           `json:"model"`
	Hardware string           `json:"hardware"`
	TP       int              `json:"tp"`
	Metrics  bool             `json:"scrape_metrics"`
	Eff      *perf.Efficiency `json:"efficiency,omitempty"`
}

type Config struct {
	Listen        string          `json:"listen"`
	Policy        string          `json:"policy"`
	Backends      []BackendConfig `json:"backends"`
	SLO           workload.SLO    `json:"slo"`
	BlockBytes    int             `json:"block_bytes"`
	CharsPerToken float64         `json:"chars_per_token"`
	MaxDefer      float64         `json:"max_defer_s"`
	Engine        engine.Config   `json:"engine"`
	HealthPath    string          `json:"health_path"`
}

func LoadConfig(path string) (Config, error) {
	var c Config
	b, err := os.ReadFile(path)
	if err != nil {
		return c, err
	}
	if err := json.Unmarshal(b, &c); err != nil {
		return c, fmt.Errorf("%s: %w", path, err)
	}
	return c, c.normalize()
}

func (c *Config) normalize() error {
	if c.Listen == "" {
		c.Listen = ":8080"
	}
	if c.Policy == "" {
		c.Policy = "abyss"
	}
	if c.BlockBytes <= 0 {
		c.BlockBytes = 256
	}
	if c.CharsPerToken <= 0 {
		c.CharsPerToken = 4
	}
	if c.SLO.TPOT <= 0 {
		c.SLO = workload.SLO{TTFTBase: 0.5, TTFTPerK: 0.05, TPOT: 0.05}
	}
	if c.MaxDefer <= 0 {
		c.MaxDefer = 5
	}
	if c.HealthPath == "" {
		c.HealthPath = "/health"
	}
	if c.Engine.MaxNumSeqs == 0 {
		c.Engine = engine.DefaultConfig()
	}
	if len(c.Backends) == 0 {
		return fmt.Errorf("no backends configured")
	}
	for i := range c.Backends {
		b := &c.Backends[i]
		b.URL = strings.TrimRight(b.URL, "/")
		if b.Name == "" {
			b.Name = fmt.Sprintf("backend-%d", i)
		}
		if b.Model == "" {
			b.Model = "llama-3.1-8b"
		}
		if b.Hardware == "" {
			b.Hardware = "h100"
		}
		if b.TP <= 0 {
			b.TP = 1
		}
	}
	return nil
}
