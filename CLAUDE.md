# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chrysalis is an interdisciplinary research framework studying criticality and transformation dynamics across physics, neuroscience, and ethnography. It models the "Eight-Step Navigation Protocol" — a universal pattern observed in phase transitions across physical, psychological, and cultural domains.

## Commands

### Python Simulations
```bash
source .venv/bin/activate
python simulations/phase_transitions/ising_2d.py
python simulations/phase_transitions/potts_2d.py
python simulations/phase_transitions/percolation_2d.py
python simulations/neuroscience/critical_brain.py
```

### Tools
```bash
python src/tools/supernote_parser.py       # Parse markdown observation notes
python src/tools/universality_plotter.py    # Generate universality comparison dashboard
```

### Gemini MCP Server (gemini-mcp/)
```bash
cd gemini-mcp
npm run build        # tsc compile
npm test             # jest
npm run test:watch   # jest --watch
npm run lint         # eslint
npm run lint:fix     # eslint --fix
npm run dev          # tsx src/index.ts
npm run type-check   # tsc --noEmit
```

### Services
```bash
./launch_chrysalis.sh   # Starts Ollama server + Open WebUI
```

## Architecture
- **simulations/phase_transitions/**: Three classical models (Ising 2D, Potts 2D, Percolation 2D) demonstrating universality — different microscopic dynamics exhibiting identical critical behavior
- **simulations/neuroscience/**: Neural branching process model (critical brain hypothesis, σ_c = 1.0)
- **src/tools/**: Analysis utilities — `supernote_parser.py` organizes observations by Eight-Step Protocol sections, `universality_plotter.py` generates cross-system comparison dashboards
- **data/ethnographic/**: Structured ethnographic observations; **data/raw_notes/**: Raw markdown notes
- **research/**: Working notes, processed observations, references, and theoretical models
- **docs/**: Foundational theory, framework design, and methodology documentation
- **gemini-mcp/**: TypeScript MCP server for Google Gemini (v1.3.2), separate Node.js project with its own build system
- **prompts/templates/**: LLM prompt templates

### Key Dependencies
- **Python**: numpy, matplotlib, networkx, pandas (see `requirements.txt`)
- **Node.js** (gemini-mcp): @google/generative-ai, @modelcontextprotocol/sdk, winston, dotenv

### Simulation Pattern
All simulations follow a common structure with research-grade features:
- **Control parameter sweep** through a critical point with measurement of order parameter, susceptibility, and specific heat
- **Multi-seed ensemble averaging** (seeds: 42, 137, 256, 314, 999) with bootstrap error bars (n_bootstrap=200)
- **Finite-size scaling** across multiple lattice sizes with Binder cumulant analysis
- **Two output figures**: `*_results.png` (4-panel overview) and `*_scaling.png` (4-panel scaling analysis)

#### Model-specific features
- **Ising** (T_c ≈ 2.269): Wolff cluster algorithm, autocorrelation time comparison, exact exponents β/ν=1/8, γ/ν=7/4, ν=1
- **Potts** (T_c ≈ 0.995): Wolff cluster with FK bond probability, exact exponents β/ν=2/15, γ/ν=26/15, ν=5/6
- **Percolation** (p_c ≈ 0.5927): Union-Find with path compression, spanning probability, cluster size distribution with MLE power-law fit (τ=187/91), exact exponents β/ν=5/48, γ/ν=43/24, ν=4/3
- **Critical Brain** (σ_c = 1.0): Sparse set-based avalanche propagation, size/duration distributions with MLE fits, size-duration scaling (γ_sd=2), mean-field exponents P(s)~s^{-3/2}, P(d)~d^{-2}
