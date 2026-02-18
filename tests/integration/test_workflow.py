import pytest
import os
import json
import shutil
from pathlib import Path
from unittest.mock import patch
from typer.testing import CliRunner
from chrysalis.chrysalis_cli import app

runner = CliRunner()


@pytest.fixture
def temp_project_dir(tmp_path):
    """Fixture to set up a temporary directory structure for testing."""
    data_raw = tmp_path / "data" / "raw"
    data_ethno = tmp_path / "chrysalis" / "data" / "ethnographic"
    sim_results = tmp_path / "simulations" / "results"
    prompts = tmp_path / "chrysalis" / "prompts" / "templates"
    schema = tmp_path / "chrysalis" / "schema"

    data_raw.mkdir(parents=True)
    data_ethno.mkdir(parents=True)
    sim_results.mkdir(parents=True)
    prompts.mkdir(parents=True)
    schema.mkdir(parents=True)

    # Copy necessary real files to temp dir
    real_root = Path(__file__).parent.parent.parent
    shutil.copy(real_root / "pyproject.toml", tmp_path / "chrysalis" / "pyproject.toml")
    shutil.copy(
        real_root / "schema" / "ethnographic_record.json",
        schema / "ethnographic_record.json",
    )
    shutil.copy(
        real_root / "prompts" / "templates" / "ethnographic_extraction_template.md",
        prompts / "ethnographic_extraction_template.md",
    )

    # Also need rag_default_template.md
    shutil.copy(
        real_root / "prompts" / "templates" / "rag_default_template.md",
        prompts / "rag_default_template.md",
    )

    yield tmp_path


def test_full_workflow_integration(temp_project_dir):
    """
    Integration test for the full workflow:
    Ingest -> (Hypothesis Generation - skipped for speed) -> Simulation.
    """
    project_root = temp_project_dir / "chrysalis"
    # 1. Setup Sample Data
    sample_text = "Village Riverbend faced a flood. Community anchored around traditions. New fishing emerged."
    input_file = temp_project_dir / "data" / "raw" / "test_village.txt"
    input_file.write_text(sample_text)

    # 2. Mock LLM Response for Ingestion
    mock_json_response = {
        "title": "Riverbend Flood",
        "source": "Test",
        "period_or_event": "2020",
        "geographic_location": "River",
        "summary": "Flood transition in Riverbend.",
        "critical_elements_observed": ["Structural dissolution"],
        "eight_step_protocol_relevance": {
            "Anchoring": "Traditional anchoring",
            "Emergence": "Fishing economy",
        },
        "tags": ["flood", "transition"],
    }

    # Patch both clients just in case
    with (
        patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "fake-key", "CHRYSALIS_PROJECT_ROOT": str(project_root)},
        ),
        patch(
            "chrysalis.src.llm_integration.llm_clients.GeminiAPIClient.generate_json",
            return_value=mock_json_response,
        ),
        patch(
            "chrysalis.src.llm_integration.llm_clients.OllamaClient.generate_json",
            return_value=mock_json_response,
        ),
        patch("chrysalis.src.llm_integration.llm_clients.genai.Client"),
        patch(
            "chrysalis.src.llm_integration.llm_clients._load_ethnographic_schema",
            return_value={},
        ),
    ):

        # 3. Run Ingest Command
        os.chdir(project_root)

        result = runner.invoke(
            app,
            [
                "ingest",
                str(input_file),
                "--llm-client",
                "gemini_api",
                "--output-dir",
                "data/ethnographic",
            ],
            catch_exceptions=False,
        )

        if result.exit_code != 0:
            print(result.output)

        assert result.exit_code == 0
        output_json = (
            temp_project_dir
            / "chrysalis"
            / "data"
            / "ethnographic"
            / "test_village.json"
        )
        assert output_json.exists()
        with open(output_json, "r") as f:
            saved_data = json.load(f)
        assert saved_data["title"] == "Riverbend Flood"

    # 4. Run a Small Simulation Experiment
    sim_config = {
        "experiment_name": "Test_Sim",
        "simulation_module": "ising_2d",
        "output_base_dir": str(
            temp_project_dir / "simulations" / "results" / "Test_Sim"
        ),
        "global_params": {"eq_sweeps": 1, "meas_sweeps": 1},
        "parameter_sweep": {"N": [4]},
        "num_seeds": 1,
        "seed_offset": 42,
    }
    config_file = temp_project_dir / "test_sim_config.json"
    config_file.write_text(json.dumps(sim_config))

    with patch(
        "chrysalis.simulations.run_experiment.EXPERIMENT_LOG_PATH",
        temp_project_dir / "simulations" / "experiment_log.json",
    ):
        result = runner.invoke(
            app, ["simulate", str(config_file)], catch_exceptions=False
        )

        if result.exit_code != 0:
            print(result.output)

        assert result.exit_code == 0
        experiment_log_path = temp_project_dir / "simulations" / "experiment_log.json"
        assert experiment_log_path.exists()
        with open(experiment_log_path, "r") as f:
            log_data = json.load(f)
        assert log_data[0]["name"] == "Test_Sim"
        assert log_data[0]["status"] == "completed"


if __name__ == "__main__":
    pass
