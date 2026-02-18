# Chrysalis Project TODO

Prioritized by value to the project.

---

## Phase 2: Theoretical & Structural Polish (In Progress)

### ✅ Completed
- **1. Write the Eight-Step Protocol reference document**: Detailed theoretical grounding in `docs/framework/eight_step_protocol.md`, integrated with Ising, Potts, Percolation, XY, and Neural findings.
- **2. Create prompt templates**: All core templates (extraction, synthesis, interpretation, meta-analysis) are now versioned in `prompts/templates/`.
- **4. Build end-to-end analysis pipeline**: `run_pipeline.py` is operational and optimized for local GPU (RTX 4090).
- **Dashboard Enhancement**: Added 3D Knowledge Graph visualization.

### 📍 Current Priorities

#### 1. Expand Test Coverage
Added initial tests for `robustness_analyzer` and `knowledge_graph`. Need to expand to:
- `supernote_parser.py`
- `validate_notes.py`
- Simulation modules (verify exponents within tolerance)
- Integration tests for `run_pipeline.py`

#### 2. Expand `simulations.ipynb`
Currently only runs and displays Ising. Should include Potts, Percolation, XY, and Critical Brain with their results and scaling figures.

#### 3. Write Simulation Methodology Document
`docs/methodology/README.md` is a placeholder. Write the actual methodology:
- Models used and their physical significance.
- Finite-size scaling procedure and Binder cumulant analysis.
- Statistical methods (bootstrap errors, autocorrelation time).

#### 4. Add CI/CD and Pre-commit Hooks
Add:
- GitHub Actions workflow: Python linting (ruff) and tests (pytest).
- Pre-commit hooks: black/ruff for Python.

#### 5. Containerize with Docker
Update `Dockerfile` and `docker-compose.yml` to support Python 3.12, Node 22, and the SQLite/ChromaDB stack.

#### 6. Finalize Manuscript
Synthesize the primary results into `research/manuscript.tex` and ensure all figures (scaling, universality, 3D graph) are included.

#### 7. Lower `engines.node` requirement
Lower `gemini-mcp/package.json` Node requirement to `>=18.0.0` if possible to avoid user friction.
