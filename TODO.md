# Chrysalis Project TODO

Prioritized by value to the project. Items grouped into tiers.

---

## Tier 1: Critical / High Impact

These items fix broken things, fill core gaps, or unlock entire workflows.

### 1. Add real ethnographic data to `data/raw_notes/`
Currently only `test_observation.md` exists. The entire analysis pipeline
(supernote_parser → validate_notes → analyze_observations → synthesize_insights)
is built but has nothing real to process. Every downstream tool is blocked
on this. Adding even 5-10 real observation notes transforms the project from
a demo to an active research platform.

### 2. Write the Eight-Step Protocol reference document
`docs/framework/` has only a placeholder README. The protocol table in
`README.md` is the only definition. A proper reference document with
theoretical grounding, examples, and the relationship to criticality would
anchor the entire project. Every tool and analysis script references the
protocol but there is no canonical detailed description.

### 3. Create prompt templates in `prompts/templates/`
Directory exists but is completely empty. The analysis scripts
(`analyze_observations.py`, `synthesize_insights.py`) have prompts
hardcoded as inline strings. Extract these into reusable, versioned
templates that can be iterated on independently of code. Templates needed:
- `observation_analysis.txt` — single-note analysis prompt
- `step_synthesis.txt` — cross-note synthesis for a protocol step
- `pattern_recognition.txt` — cross-step pattern identification
- `simulation_interpretation.txt` — connecting simulation results to theory

### 4. Remove self-referencing dependency in `gemini-mcp/package.json`
Line 60: `"@houtini/gemini-mcp": "^1.4.5"` — the package lists itself as
its own dependency (at a higher version than itself: 1.4.5 > 1.3.2). This
is a copy-paste bug that pulls the npm-published version into node_modules,
shadowing local code unpredictably. Remove this line.

### 5. Clean up duplicate/stale observation markdown files
`research/observations/` contains `purification.md`, `stillness.md`,
`integration.md`, and `master_integration_log.md` — all with duplicated
content from the test observation. These appear to be artifacts from an
earlier version of the parser that wrote markdown (not JSON). Since the
current pipeline uses `research/processed_notes/` (JSON), these stale
markdown files are misleading. Either delete them or regenerate from real
data.

### 6. Add `.mcp.json` for Claude Code integration
No MCP configuration exists. The gemini-mcp server must be started manually
and the Python scripts connect via hardcoded `localhost:3000`. Adding a
`.mcp.json` would let Claude Code auto-discover and use the Gemini tools
directly, which is the intended workflow for the MCP architecture.

---

## Tier 2: High Value / Structural Improvements

These items improve reliability, developer experience, and project maturity.

### 7. Add Python package structure
No `__init__.py` files exist in `simulations/` or `src/`. No `pyproject.toml`
or `setup.py`. This means:
- Imports between modules are fragile (require `sys.path` hacks)
- Can't install the project as a package
- No way to pin the project's own version
Add `pyproject.toml`, `__init__.py` files, and a proper package name.

### 8. Add tests for Python tools
Zero test coverage for:
- `supernote_parser.py` — YAML parsing, protocol extraction, edge cases
- `validate_notes.py` — schema validation logic
- `universality_plotter.py` — image loading, figure generation
- Simulation modules — verify critical exponents within tolerance
The gemini-mcp has tests (9 passing) but the Python side has none.

### 9. Expand `simulations.ipynb` to cover all four simulations
Currently only runs and displays Ising. Should include Potts, Percolation,
and Critical Brain with their results and scaling figures. Also add
narrative markdown cells explaining what each simulation demonstrates
about universality and criticality.

### 10. Write simulation methodology document
`docs/methodology/README.md` is a placeholder. Write the actual methodology:
- Models used and their physical significance
- Parameter choices and known exact values
- Finite-size scaling procedure and Binder cumulant analysis
- Statistical methods (bootstrap errors, autocorrelation time)
- How simulation results connect to the Eight-Step Protocol

### 11. Add `gemini-mcp` tests for chat, deep-research, and list-models
Only `gemini-web` and `gemini-summarize-web` have tests (9 total). The
three most important tools — `gemini_chat`, `gemini_deep_research`, and
`gemini_list_models` — have zero test coverage.

### 12. Fix `gemini-mcp` ts-jest warning
Tests emit: `Using hybrid module kind (Node16/18/Next) is only supported
in "isolatedModules: true"`. Add `isolatedModules: true` to
`tsconfig.json` or create a separate `tsconfig.test.json` for jest.

### 13. Add `.env` and log files to parent `.gitignore`
The parent `.gitignore` doesn't exclude:
- `gemini-mcp-stderr.log` (currently tracked as untracked)
- `gemini-mcp/logs/` (runtime logs)
- `*.png` in simulation output directories (large binary files)
- `research/processed_notes/` (generated output)

---

## Tier 3: Valuable Enhancements

These items add new capabilities or polish existing ones.

### 14. Build end-to-end analysis pipeline script
Currently the workflow is manual: run parser → start server → run analysis
→ run synthesis. Create a single orchestrator script (`run_pipeline.py` or
extend `launch_chrysalis.sh`) that:
- Parses raw notes
- Validates output
- Starts gemini-mcp if not running
- Runs analysis on all notes
- Synthesizes insights per protocol step
- Generates a summary report

### 15. Write foundational theory documents
`docs/foundational/README.md` describes what should be there but contains
no actual content. Write or curate documents on:
- Theories of criticality and phase transitions
- The critical brain hypothesis
- Universality classes and why they matter
- Anthropological frameworks for transformation/liminality

### 16. Add a 3D/XY model simulation
The current simulations cover Ising (Z2), Potts (Z3), and Percolation.
Adding the XY model (continuous O(2) symmetry) would demonstrate BKT
(Berezinskii-Kosterlitz-Thouless) transitions — a qualitatively different
universality class with topological defects (vortices). This would
strengthen the universality argument significantly.

### 17. Create a cross-system scaling comparison figure
The universality plotter composites individual simulation PNGs. A more
powerful figure would overlay scaled data from all systems on a single
plot — e.g., all order parameters collapsed onto one universal curve,
demonstrating that Ising, Potts, Percolation, and branching processes
share the same scaling function (within their universality class).

### 18. Add data export from simulations
Simulations currently only produce PNG figures. Add CSV/JSON export of
raw measurement data (T, m, chi, C_v, Binder cumulant per L) so that:
- Notebooks can reload and replot without re-running simulations
- Data can be shared or used in other analysis tools
- Reproducibility is improved

### 19. Populate `data/ethnographic/` with structured data
The directory has only a `.gitkeep`. The README describes it as
"Structured ethnographic and historical accounts." Define a schema
(extending `processed_note.schema.json`) and populate with curated
cross-cultural accounts of transformation at critical boundaries.

### 20. Add CI/CD configuration
No GitHub Actions, no pre-commit hooks, no linting config for Python.
Add:
- GitHub Actions workflow: run Python syntax check, gemini-mcp tests
- Pre-commit hooks: black/ruff for Python, eslint for TypeScript
- Automated simulation verification (check exponents within tolerance)

---

## Tier 4: Nice to Have

### 21. Lower `engines.node` requirement in `gemini-mcp/package.json`
Currently requires `>=24.0.0` but the code works fine on Node 18+. This
blocks users and causes confusing errors. Lower to `>=18.0.0`.

### 22. Add interactive parameter exploration to notebooks
The `simulations.ipynb` just runs the full simulation. Add ipywidgets
sliders for temperature, lattice size, and number of sweeps so users
can interactively explore phase transitions.

### 23. Add a `research/references/` bibliography
The README lists this directory but it doesn't exist. Create a
`references.bib` or `references.md` with key papers on criticality,
universality, the critical brain hypothesis, and related anthropological
frameworks.

### 24. Add type hints to Python tools
None of the Python scripts use type hints. Adding them would improve
IDE support and catch bugs earlier, especially in the MCP client scripts
where the `session.call_tool()` return type is important.

### 25. Containerize the project with Docker
A `Dockerfile` and `docker-compose.yml` would allow one-command setup:
Python venv, Node.js, gemini-mcp server, and Ollama all in one
reproducible environment. Currently requires manual setup of multiple
runtimes.
