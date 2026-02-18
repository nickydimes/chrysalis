# Chrysalis SQL Schema (Initial Design)

This document outlines the proposed SQLite database schema for the Chrysalis project, intended to replace the JSON-based logging and record-keeping for improved scalability and querying.

## 1. Table: `experiments`
Stores metadata and global configurations for each high-level experiment.

| Column | Type | Description |
|---|---|---|
| `experiment_id` | TEXT (PK) | Unique UUID for the experiment. |
| `name` | TEXT | Name of the experiment. |
| `simulation_module` | TEXT | The simulation model used (e.g., ising_2d). |
| `config_file` | TEXT | Path to the original config file. |
| `output_base_dir` | TEXT | Base directory for results. |
| `start_time` | TEXT | ISO timestamp. |
| `end_time` | TEXT | ISO timestamp. |
| `status` | TEXT | 'running', 'completed', 'failed'. |
| `git_commit` | TEXT | Git hash at the time of experiment. |
| `python_version` | TEXT | Full Python version string. |
| `os_info` | TEXT | Operating system details. |
| `global_params_json` | TEXT | JSON string of all global parameters. |

## 2. Table: `simulation_runs`
Stores details for individual runs within an experiment.

| Column | Type | Description |
|---|---|---|
| `run_id` | TEXT (PK) | Unique UUID for the run. |
| `experiment_id` | TEXT (FK) | Reference to `experiments.experiment_id`. |
| `seed` | INTEGER | Random seed used. |
| `status` | TEXT | 'completed', 'failed'. |
| `start_time` | TEXT | ISO timestamp. |
| `end_time` | TEXT | ISO timestamp. |
| `output_path` | TEXT | Path to run-specific results directory. |
| `parameters_json` | TEXT | JSON string of all parameters for this run (global + sweep). |
| `error_message` | TEXT | Error message if failed. |

## 3. Table: `ethnographic_records`
Stores structured ethnographic data.

| Column | Type | Description |
|---|---|---|
| `record_id` | INTEGER (PK) | Auto-incrementing ID. |
| `title` | TEXT | Title of the account. |
| `source` | TEXT | Origin of the account. |
| `period_or_event` | TEXT | Historical period or event. |
| `geographic_location` | TEXT | Location. |
| `summary` | TEXT | Concise summary. |
| `source_file` | TEXT | Name of the original JSON file. |
| `critical_elements_json` | TEXT | JSON list of observed elements. |
| `protocol_relevance_json` | TEXT | JSON object of Eight-Step relevance. |
| `tags_json` | TEXT | JSON list of tags. |

---

## Migration Plan:
1.  **Develop `src/tools/init_db.py`:** Create the SQLite database and tables.
2.  **Develop `src/tools/sync_json_to_db.py`:** A script to read existing `experiment_log.json` and ethnographic JSON files and populate the database.
3.  **Update `run_experiment.py`:** Modify it to optionally log directly to the SQLite database.
4.  **Update `ethnographic_data.py`:** Add a function to load data from the database into a Pandas DataFrame.
