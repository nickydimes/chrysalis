import typer
import subprocess
import sys
import os
from pathlib import Path
from typing import Optional, List, Callable

# Ensure project parent directory is in sys.path before importing chrysalis sub-modules
# so that 'import chrysalis.src...' works.
project_root: Path = Path(__file__).parent.absolute()
project_parent: str = str(project_root.parent)
if project_parent not in sys.path:
    sys.path.insert(0, project_parent)


# Import tool main functions for direct calling
def _unavailable_tool():
    print("Error: This tool is unavailable due to missing dependencies.")
    sys.exit(1)


ingest_main = simulate_main = query_exp_main = analyze_ethno_main = _unavailable_tool
suggest_params_main = interpret_results_main = generate_hypotheses_main = (
    _unavailable_tool
)
rag_chat_main = index_rag_main = meta_analysis_main = init_db_main = _unavailable_tool
sync_db_main = synthesize_sim_main = discovery_main = extract_graph_main = (
    _unavailable_tool
)
query_graph_main = robustness_main = generate_synthetic_main = _unavailable_tool
evaluate_stress_main = generate_queries_main = grounding_main = _unavailable_tool
sql_analytics_main = optimize_sim_main = generate_manuscript_main = _unavailable_tool

try:
    from chrysalis.src.tools.ingest_ethnographic_data import main as ingest_main
    from chrysalis.simulations.run_experiment import main as simulate_main
    from chrysalis.simulations.query_experiments import main as query_exp_main
    from chrysalis.src.analysis.ethnographic_data import main as analyze_ethno_main
    from chrysalis.src.tools.suggest_sim_params import main as suggest_params_main
    from chrysalis.src.tools.interpret_sim_results import main as interpret_results_main
    from chrysalis.src.tools.generate_hypotheses import main as generate_hypotheses_main
    from chrysalis.src.rag.chat_with_project import main as rag_chat_main
    from chrysalis.src.rag.index_project import main as index_rag_main
    from chrysalis.src.analysis.meta_analysis import main as meta_analysis_main
    from chrysalis.src.tools.init_db import main as init_db_main
    from chrysalis.src.tools.sync_json_to_db import main as sync_db_main
    from chrysalis.src.tools.synthesize_simulation import main as synthesize_sim_main
    from chrysalis.src.tools.discovery_orchestrator import main as discovery_main
    from chrysalis.src.tools.extract_knowledge_graph import main as extract_graph_main
    from chrysalis.src.analysis.query_knowledge_graph import main as query_graph_main
    from chrysalis.src.analysis.robustness_analyzer import main as robustness_main
    from chrysalis.src.tools.generate_synthetic_ethnography import (
        main as generate_synthetic_main,
    )
    from chrysalis.src.tools.evaluate_stress_test import main as evaluate_stress_main
    from chrysalis.src.tools.generate_search_queries import (
        main as generate_queries_main,
    )
    from chrysalis.src.tools.grounding_orchestrator import main as grounding_main
    from chrysalis.src.analysis.sql_analytics import main as sql_analytics_main
    from chrysalis.src.tools.optimize_sim_params import main as optimize_sim_main
    from chrysalis.src.tools.generate_manuscript import main as generate_manuscript_main
except ImportError as e:
    # Fallback if not installed or during bootstrap
    print(f"Warning: Some tools could not be imported directly: {e}")

app = typer.Typer(
    help="Chrysalis Project: Interdisciplinary Research Framework for Criticality."
)


def _invoke_tool(func: Callable[[], None], args_list: List[str]) -> None:
    """Helper to invoke a tool's main function by mocking sys.argv."""
    old_argv = sys.argv
    # The first element of sys.argv is usually the script name
    sys.argv = ["chrysalis_cli"] + args_list
    print(f"DEBUG: Invoking {func.__name__} with sys.argv: {sys.argv}")
    try:
        func()
    except SystemExit as e:
        if e.code != 0:
            print(f"DEBUG: {func.__name__} exited with code {e.code}")
            raise typer.Exit(code=e.code)
    finally:
        sys.argv = old_argv


@app.command()
def ingest(
    input_file: str = typer.Argument(
        ..., help="Path to the raw ethnographic file (txt, md, pdf, wav, mp3)."
    ),
    llm_client: str = typer.Option(
        "gemini_api", help="LLM client to use (gemini_api, ollama)."
    ),
    llm_model: Optional[str] = typer.Option(None, help="Specific LLM model name."),
    template_name: str = typer.Option(
        "ethnographic_extraction_template", help="Prompt template name."
    ),
    output_dir: str = typer.Option(
        "data/ethnographic/", help="Output directory for structured JSON."
    ),
) -> None:
    """Automate structured data ingestion from various formats using an LLM."""
    args: List[str] = [
        input_file,
        "--llm_client",
        llm_client,
        "--template_name",
        template_name,
        "--output_dir",
        output_dir,
    ]
    if llm_model:
        args.extend(["--llm_model", llm_model])
    _invoke_tool(ingest_main, args)


@app.command()
def simulate(
    config_file: str = typer.Argument(
        ..., help="Path to the JSON configuration file for the experiment."
    )
) -> None:
    """Run simulation experiments with tracking."""
    _invoke_tool(simulate_main, [config_file])


@app.command()
def query_exp(
    name: Optional[str] = typer.Option(None, help="Filter by experiment name."),
    status: Optional[str] = typer.Option(
        None, help="Filter by status (running, completed, failed)."
    ),
    id: Optional[str] = typer.Option(None, help="Filter by exact experiment ID."),
    show_runs: bool = typer.Option(
        False, "--show-runs", help="Show individual run details."
    ),
    show_env: bool = typer.Option(
        False, "--show-env", help="Show environment metadata."
    ),
) -> None:
    """Query and display tracked simulation experiments."""
    args: List[str] = []
    if name:
        args.extend(["--name", name])
    if status:
        args.extend(["--status", status])
    if id:
        args.extend(["--id", id])
    if show_runs:
        args.append("--show_runs")
    if show_env:
        args.append("--show_env")
    _invoke_tool(query_exp_main, args)


@app.command()
def analyze_ethnographic(
    data_dir: str = typer.Option(
        "data/ethnographic", help="Directory with structured JSON records."
    )
) -> None:
    """Run the ethnographic data analysis suite."""
    # Note: currently ethnographic_data.py main doesn't take many arguments in its __main__ block
    _invoke_tool(analyze_ethno_main, [])


@app.command()
def suggest_params(
    llm_client: str = typer.Option("gemini_api", help="LLM client to use."),
    llm_model: Optional[str] = typer.Option(None, help="Specific LLM model name."),
) -> None:
    """Use an LLM to suggest simulation parameters from ethnographic data."""
    args: List[str] = ["--llm_client", llm_client]
    if llm_model:
        args.extend(["--llm_model", llm_model])
    _invoke_tool(suggest_params_main, args)


@app.command()
def synthesize_sim(
    model_name: str = typer.Argument(
        ..., help="Name of the new model (e.g., rumor_spread)."
    ),
    description: str = typer.Argument(
        ..., help="Description of the social transition logic."
    ),
    llm_client: str = typer.Option("gemini_api", help="LLM client to use."),
    llm_model: Optional[str] = typer.Option(None, help="Specific LLM model name."),
) -> None:
    """Synthesize a new Python simulation model from a natural language description."""
    args: List[str] = [model_name, description, "--llm_client", llm_client]
    if llm_model:
        args.extend(["--llm_model", llm_model])
    _invoke_tool(synthesize_sim_main, args)


@app.command()
def interpret_results(
    experiment_dir: str = typer.Argument(
        ..., help="Path to the base experiment results directory."
    ),
    llm_client: str = typer.Option("gemini_api", help="LLM client to use."),
    llm_model: Optional[str] = typer.Option(None, help="Specific LLM model name."),
) -> None:
    """Use an LLM to interpret aggregated simulation results."""
    args: List[str] = [experiment_dir, "--llm_client", llm_client]
    if llm_model:
        args.extend(["--llm_model", llm_model])
    _invoke_tool(interpret_results_main, args)


@app.command()
def generate_hypotheses(
    llm_client: str = typer.Option("gemini_api", help="LLM client to use."),
    llm_model: Optional[str] = typer.Option(None, help="Specific LLM model name."),
) -> None:
    """Generate research hypotheses from ethnographic data using an LLM."""
    args: List[str] = ["--llm_client", llm_client]
    if llm_model:
        args.extend(["--llm_model", llm_model])
    _invoke_tool(generate_hypotheses_main, args)


@app.command()
def meta_analyze(
    query: str = typer.Option(
        "Synthesize the primary cross-modal findings between ethnographic records and simulation results.",
        help="Specific question for meta-analysis.",
    ),
    llm_model: str = typer.Option("command-r", help="Local LLM model."),
    template_name: str = typer.Option(
        "meta_analysis_template", help="Prompt template name."
    ),
) -> None:
    """Perform a high-level meta-analysis of project findings using RAG."""
    args: List[str] = [
        "--query",
        query,
        "--llm_model",
        llm_model,
        "--template_name",
        template_name,
    ]
    _invoke_tool(meta_analysis_main, args)


@app.command()
def generate_manuscript(
    llm_client: str = typer.Option("ollama", help="LLM client to use."),
    llm_model: str = typer.Option("deepseek-r1:32b", help="LLM model name."),
    output_file: str = typer.Option(
        "research/manuscript.tex", help="Output LaTeX file path."
    ),
) -> None:
    """Synthesize a complete research manuscript in LaTeX format."""
    _invoke_tool(
        generate_manuscript_main,
        [
            "--llm_client",
            llm_client,
            "--llm_model",
            llm_model,
            "--output_file",
            output_file,
        ],
    )


@app.command()
def gen_queries(
    hypotheses_file: str = typer.Option(
        "research/hypotheses.md", help="Path to hypotheses file."
    ),
    llm_client: str = typer.Option("ollama", help="LLM client to use."),
    llm_model: str = typer.Option("llama3.1:8b", help="LLM model name."),
) -> None:
    """Generate academic search queries from research hypotheses."""
    _invoke_tool(
        generate_queries_main,
        [
            "--hypotheses_file",
            hypotheses_file,
            "--llm_client",
            llm_client,
            "--llm_model",
            llm_model,
        ],
    )


@app.command()
def ground_hypotheses(
    hypotheses_file: str = typer.Option(
        "research/hypotheses.md", help="Path to hypotheses file."
    ),
    llm_model: str = typer.Option("llama3.1:8b", help="Local LLM model."),
) -> None:
    """Evaluate project hypotheses against indexed external literature."""
    _invoke_tool(
        grounding_main, ["--hypotheses_file", hypotheses_file, "--llm_model", llm_model]
    )


@app.command()
def rag_chat(
    query: str = typer.Argument(..., help="Question about the project."),
    llm_model: str = typer.Option("llama3.1:8b", help="Ollama LLM model."),
    template_name: str = typer.Option(
        "rag_default_template", help="RAG prompt template name."
    ),
) -> None:
    """Chat with the project knowledge base using RAG."""
    args: List[str] = [
        query,
        "--llm_model",
        llm_model,
        "--template_name",
        template_name,
    ]
    _invoke_tool(rag_chat_main, args)


@app.command()
def index_rag() -> None:
    """Index project code and docs for RAG."""
    _invoke_tool(index_rag_main, [])


@app.command()
def init_db(
    db_path: str = typer.Option(
        "data/chrysalis.db", help="Path to the SQLite database."
    )
) -> None:
    """Initialize the Chrysalis SQLite database."""
    _invoke_tool(init_db_main, ["--db_path", db_path])


@app.command()
def sync_db(
    db_path: str = typer.Option(
        "data/chrysalis.db", help="Path to the SQLite database."
    ),
    log_path: str = typer.Option(
        "simulations/experiment_log.json", help="Path to experiment log JSON."
    ),
    ethno_dir: str = typer.Option(
        "data/ethnographic", help="Directory with ethnographic JSONs."
    ),
) -> None:
    """Sync JSON logs and records to the SQLite database."""
    _invoke_tool(
        sync_db_main,
        ["--db_path", db_path, "--log_path", log_path, "--ethno_dir", ethno_dir],
    )


@app.command()
def docs() -> None:
    """Build project documentation."""
    docs_dir: Path = Path("chrysalis/docs")
    if not docs_dir.exists():
        typer.echo("Error: docs directory not found.")
        raise typer.Exit(code=1)

    try:
        subprocess.run(["make", "html"], cwd=str(docs_dir), check=True)
        typer.echo(f"Docs built successfully in {docs_dir}/build/html")
    except Exception as e:
        typer.echo(f"Error building docs: {e}")


@app.command()
def test() -> None:
    """Run the project test suite."""
    try:
        subprocess.run(["pytest"], check=True)
    except Exception as e:
        typer.echo(f"Tests failed: {e}")


@app.command()
def extract_graph(
    llm_client: str = typer.Option("ollama", help="LLM client to use."),
    llm_model: str = typer.Option("deepseek-r1:32b", help="LLM model name."),
) -> None:
    """Extract Knowledge Graph triples from project data using an LLM."""
    _invoke_tool(
        extract_graph_main, ["--llm_client", llm_client, "--llm_model", llm_model]
    )


@app.command()
def generate_synthetic(
    scenario: str = typer.Argument(
        ...,
        help="Scenario type (stuck_liminal, sudden_emergence, reversing_phase, overlapping_encounter, shadow_protocol).",
    ),
    llm_client: str = typer.Option("ollama", help="LLM client to use."),
    llm_model: str = typer.Option("llama3.1:8b", help="LLM model name."),
) -> None:
    """Generate a synthetic 'Black Swan' ethnographic account."""
    _invoke_tool(
        generate_synthetic_main,
        [scenario, "--llm_client", llm_client, "--llm_model", llm_model],
    )


@app.command()
def stress_test(
    synthetic_file: str = typer.Argument(
        ..., help="Path to the synthetic ethnographic account."
    ),
    llm_client: str = typer.Option("ollama", help="LLM client to use."),
    llm_model: str = typer.Option("llama3.1:8b", help="LLM model name."),
) -> None:
    """Evaluate framework integrity against a Black Swan scenario."""
    _invoke_tool(
        evaluate_stress_main,
        [synthetic_file, "--llm_client", llm_client, "--llm_model", llm_model],
    )


@app.command()
def analyze_robustness(
    experiment_dir: str = typer.Argument(
        ..., help="Path to the simulation experiment directory."
    ),
    threshold: float = typer.Option(
        0.8, help="Order parameter threshold for 'Success'."
    ),
) -> None:
    """Quantify the robustness of the Eight-Step Protocol via sensitivity analysis."""
    _invoke_tool(robustness_main, [experiment_dir, "--threshold", str(threshold)])


@app.command()
def optimize_sim(
    config_file: str = typer.Argument(
        ..., help="Path to the base simulation configuration file."
    ),
    target: str = typer.Option(
        "susceptibility", help="Metric to maximize (susceptibility, specific_heat)."
    ),
    iterations: int = typer.Option(2, help="Number of refinement iterations."),
) -> None:
    """Auto-optimize simulation parameters to find the critical 'sweet spot'."""
    _invoke_tool(
        optimize_sim_main,
        [config_file, "--target", target, "--iterations", str(iterations)],
    )


@app.command()
def analyze_sql(
    query: Optional[str] = typer.Option(None, help="Custom SQL query to run."),
    view: Optional[str] = typer.Option(
        None, help="Name of a pre-defined view to display."
    ),
    list_views: bool = typer.Option(
        False, "--list-views", help="List available analytics views."
    ),
    output_csv: Optional[str] = typer.Option(None, help="Save result to a CSV file."),
) -> None:
    """Run SQL-native cross-modal analytics on Chrysalis data."""
    args: List[str] = []
    if query:
        args.extend(["--query", query])
    if view:
        args.extend(["--view", view])
    if list_views:
        args.append("--list_views")
    if output_csv:
        args.extend(["--output_csv", output_csv])
    _invoke_tool(sql_analytics_main, args)


@app.command()
def query_graph(
    node: Optional[str] = typer.Option(None, help="Get neighbors of a specific node."),
    discover: bool = typer.Option(
        False, "--discover", help="Find hidden relationships (Event -> Model paths)."
    ),
) -> None:
    """Query and explore the Chrysalis Knowledge Graph."""
    args: List[str] = []
    if node:
        args.extend(["--node", node])
    if discover:
        args.append("--discover")
    _invoke_tool(query_graph_main, args)


@app.command()
def dashboard() -> None:
    """Launch the Chrysalis Phase Transition Dashboard."""
    script_path: Path = Path(__file__).parent / "src" / "tools" / "dashboard.py"
    if not script_path.exists():
        typer.echo(f"Error: Dashboard script not found at {script_path}")
        raise typer.Exit(code=1)

    try:
        typer.echo("Launching dashboard... Press Ctrl+C to stop.")
        # Ensure chrysalis is in PYTHONPATH for the streamlit process
        env = os.environ.copy()
        project_root_path: Path = Path(__file__).parent.absolute()
        project_root_str: str = str(project_root_path)
        env["PYTHONPATH"] = f"{project_root_str}:{env.get('PYTHONPATH', '')}"

        streamlit_bin = project_root_path / ".venv" / "bin" / "streamlit"
        subprocess.run(
            [str(streamlit_bin), "run", str(script_path)], env=env, check=True
        )
    except KeyboardInterrupt:
        typer.echo("\nDashboard stopped.")
    except Exception as e:
        typer.echo(f"Error launching dashboard: {e}")


@app.command()
def discovery_loop(
    llm_client: str = typer.Option("ollama", help="LLM client to use."),
    llm_model: Optional[str] = typer.Option(None, help="Specific LLM model name."),
) -> None:
    """Run the autonomous research discovery loop."""
    args: List[str] = ["--llm_client", llm_client]
    if llm_model:
        args.extend(["--llm_model", llm_model])
    _invoke_tool(discovery_main, args)


if __name__ == "__main__":
    app()
