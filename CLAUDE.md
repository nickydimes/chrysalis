# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chrysalis is an interdisciplinary research framework studying criticality and transformation dynamics across physics, neuroscience, and ethnography. It models the "Eight-Step Navigation Protocol" — a universal pattern observed in phase transitions across physical, psychological, and cultural domains.

There are two copies: `chrysalis/` (main, fully developed) and `chrysalis-1/` (minimal fork with physics simulations only).

## Commands

### Python Simulations (chrysalis/)
```bash
source chrysalis/.venv/bin/activate
python chrysalis/simulations/phase_transitions/ising_2d.py
python chrysalis/simulations/phase_transitions/potts_2d.py
python chrysalis/simulations/phase_transitions/percolation_2d.py
python chrysalis/simulations/neuroscience/critical_brain.py
```

### Tools
```bash
python chrysalis/src/tools/supernote_parser.py       # Parse markdown observation notes
python chrysalis/src/tools/universality_plotter.py    # Generate universality comparison dashboard
```

### Gemini MCP Server (chrysalis/gemini-mcp/)
```bash
cd chrysalis/gemini-mcp
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
./chrysalis/launch_chrysalis.sh   # Starts Ollama server + Open WebUI
```

## Architecture

### chrysalis/ (main project)
- **simulations/phase_transitions/**: Three classical models (Ising 2D, Potts 2D, Percolation 2D) demonstrating universality — different microscopic dynamics exhibiting identical critical behavior
- **simulations/neuroscience/**: Neural branching process model (critical brain hypothesis, σ_c = 1.0)
- **src/tools/**: Analysis utilities — `supernote_parser.py` organizes observations by Eight-Step Protocol sections, `universality_plotter.py` generates cross-system comparison dashboards
- **data/ethnographic/**: Structured ethnographic observations; **data/raw_notes/**: Raw markdown notes
- **research/**: Working notes, processed observations, references, and theoretical models
- **docs/**: Foundational theory, framework design, and methodology documentation
- **gemini-mcp/**: TypeScript MCP server for Google Gemini (v1.3.2), separate Node.js project with its own build system
- **prompts/templates/**: LLM prompt templates

### Key Dependencies
- **Python**: numpy, scipy, matplotlib, networkx, pandas (see `requirements.txt`)
- **Node.js** (gemini-mcp): @google/generative-ai, @modelcontextprotocol/sdk, winston, dotenv

### Simulation Pattern
All phase transition simulations follow the same structure: sweep a control parameter through a critical point, measure order parameter/susceptibility/specific heat, and generate 4-panel visualization figures. Critical points: Ising T_c ≈ 2.269, Potts T_c ≈ 0.995, Percolation p_c ≈ 0.5927.
