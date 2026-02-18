# Chrysalis Project DNA

## 🦋 Core Mission
Studying sociocultural and psychological transformation through the lens of physics (criticality, phase transitions) and neuroscience (critical brain hypothesis).

## 🛠 Tech Stack
- **Python 3.12:** Core simulations (Ising, Potts, Percolation, XY, Critical Brain).
- **TypeScript:** Gemini MCP Server (`gemini-mcp/`).
- **Data:** SQLite (`data/chrysalis.db`), RAG via ChromaDB (`vector_db/`).
- **CLI:** Typer-based orchestrator (`chrysalis_cli.py`).

## 📏 Engineering Standards
- **Type Hinting:** Mandatory for all new Python tools.
- **Root Discovery:** Use `os.getenv("CHRYSALIS_PROJECT_ROOT")` or `pyproject.toml` discovery logic (see `llm_clients.py`).
- **Dependencies:** Use `google-genai` (modern) and `langchain-ollama`/`langchain-chroma`.
- **Linting:** `ruff` and `black` are enforced via pre-commit hooks.

## 🌀 Theoretical Foundation: The Eight-Step Protocol
1. Purification | 2. Containment | 3. Anchoring | 4. Dissolution
5. Liminality | 6. Encounter | 7. Integration | 8. Emergence

## 📂 Key Paths
- `simulations/`: Hot loops accelerated with `@njit`.
- `src/tools/`: LLM-integrated research utilities.
- `data/raw/`: Entry point for discovery loop.
- `prompts/templates/`: Versioned LLM instructions.
