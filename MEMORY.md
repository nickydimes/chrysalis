# Chrysalis Project Memory 🦋

**Last Updated**: 2026-02-18
**Status**: High-Efficiency Automation Phase

## 🌀 Project DNA
- **Mission**: Study sociocultural and psychological transformation using the physics of criticality and neuroscience.
- **Core Framework**: The **Eight-Step Navigation Protocol** (Universal dynamical pattern of transition).
- **Architecture**: Python simulations (Ising, Potts, Percolation, XY, Critical Brain) + Gemini MCP Server + SQLite/RAG Knowledge Base + Typer-based CLI Orchestrator.

## 📍 Current State (at a Glance)
- **Database**: `chrysalis.db` contains metadata for experiments and 1000+ Knowledge Graph triples.
- **Automation**: `run_pipeline.py` provides automated end-to-end execution (Ingest -> Extract -> Robustness -> Meta-Analysis).
- **Optimization**: Tiered LLM strategy for RTX 4090:
    - **`command-r`**: Meta-analysis and multi-document synthesis (RAG optimized).
    - **`deepseek-r1:32b`**: Reasoning/Extraction/Manuscript (Logic optimized).
    - **`llama3.1:8b`**: RAG/Light tasks (Speed optimized).
- **Dashboard**: `src/tools/dashboard.py` features 2D and 3D Knowledge Graph visualization (using Plotly `Scatter3d`).
- **Protocol**: `docs/framework/eight_step_protocol.md` is consolidated with XY (BKT stiffness/vortex binding) and Neural-Socio (universality/branching) findings.

## 🦋 The Eight-Step Navigation Protocol (Canonical)
1. **Purification** | 2. **Containment** | 3. **Anchoring** | 4. **Dissolution**
5. **Liminality** | 6. **Encounter** | 7. **Integration** | 8. **Emergence**

## 💡 Key Findings to Date
- **Cross-Modal Convergence**: Ritualized "tipping points" in ethnographic records map quantitatively to **spin-flip transitions** in Ising models.
- **Network Percolation**: Cultural collapses mirror **percolation events** where giant components fragment.
- **BKT Transition (XY Model)**: Socio-cultural "stiffness" (helicity modulus) and vortex binding provide robust order parameters for topological transitions; integrated into the framework.
- **Neural-Socio Universality**: Social cascades and neural avalanches share the same universality class (Mean-Field Directed Percolation), with power-law exponents \(\tau_s \approx 1.5\) and \(\tau_d \approx 2.0\).

## 🛑 Fixed Issues
- **Multimodal**: PDF ingestion now uses `unstructured` for better local partitioning.
- **Dashboard**: Added interactive Knowledge Graph visualization with 3D support using Plotly.
- **Mapping**: Completed Neural-Socio mapping using a new social cascade generator.
- **Protocol**: Consolidated theoretical framework with XY and Neural-Socio findings.
- **Pipeline**: Created `run_pipeline.py` for automated project flow; fixed path resolution and interpreter issues.
- **CLI**: Improved robustness by handling failed tool imports and preventing `NameError`.
- **Validation**: Added unit tests for robustness analysis and knowledge graph extraction.

## ⏭ Priority Next Steps
1. **Manuscript Completion**: Finalize the LaTeX manuscript with the latest integrated results and figures.
2. **Expand Tests**: Increase coverage for the discovery loop and simulation modules.
3. **Interactive 3D**: Explore more performant 3D rendering if the graph grows beyond Plotly's comfortable limits.
4. **External Grounding**: Run the grounding orchestrator against a larger set of literature queries using the optimized models.
