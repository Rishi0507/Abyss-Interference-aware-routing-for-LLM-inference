package main

import (
	"fmt"
	"os"
	"strings"
)

var version = "0.1.0"

type command struct {
	name    string
	summary string
	run     func(args []string) error
}

func commands() []command {
	return []command{
		{"sim", "Run one simulation and print a summary", cmdSim},
		{"bench", "Sweep policies across load levels for a preset", cmdBench},
		{"tune", "Tune baseline parameters on one window and evaluate on held-out windows", cmdTune},
		{"gateway", "Run the routing gateway in front of OpenAI-compatible backends", cmdGateway},
		{"emulate", "Run an emulated inference server backed by the engine model", cmdEmulate},
		{"loadgen", "Replay a trace or synthetic workload against an HTTP endpoint", cmdLoadgen},
		{"calibrate", "Measure a real server and fit the performance model to it", cmdCalibrate},
		{"studio", "Serve the dashboard and the simulation API", cmdStudio},
		{"catalog", "List models, accelerators, presets and policies", cmdCatalog},
		{"trace", "Summarize a trace or synthetic workload", cmdTrace},
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, "abyss %s: externality-priced request routing for LLM inference fleets\n\nUsage:\n  abyss <command> [flags]\n\nCommands:\n", version)
	for _, c := range commands() {
		fmt.Fprintf(os.Stderr, "  %-9s %s\n", c.name, c.summary)
	}
	fmt.Fprintf(os.Stderr, "\nRun 'abyss <command> -h' for command flags.\n")
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	name := strings.ToLower(os.Args[1])
	if name == "-h" || name == "--help" || name == "help" {
		usage()
		return
	}
	if name == "version" || name == "--version" {
		fmt.Println(version)
		return
	}
	for _, c := range commands() {
		if c.name == name {
			if err := c.run(os.Args[2:]); err != nil {
				fmt.Fprintln(os.Stderr, "error:", err)
				os.Exit(1)
			}
			return
		}
	}
	fmt.Fprintf(os.Stderr, "unknown command %q\n\n", name)
	usage()
	os.Exit(2)
}
