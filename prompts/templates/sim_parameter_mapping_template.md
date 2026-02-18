You are an expert in complexity science and phase transitions.
Your task is to analyze aggregated ethnographic data and suggest specific parameters for simulation models (Ising, Potts, Percolation) that could represent the observed social and cultural dynamics.

Aggregated Ethnographic Data:
{AGGREGATED_DATA}

Available Simulation Models:
1.  **Ising Model (ising_2d):** Models binary transitions. 
    - Expected Args: `--N` (Lattice size), `--eq_sweeps`, `--meas_sweeps`.
2.  **Potts Model (potts_2d, q=3):** Models multi-state transitions.
    - Expected Args: `--N` (Lattice size), `--eq_sweeps`, `--meas_sweeps`.
3.  **Percolation Model (percolation_2d):** Models structural connectivity.
    - Expected Args: `--L` (Lattice size), `--n_realizations`.

Instructions:
1.  Review the aggregated observations, especially the 'Critical Elements' and 'Protocol Phase Relevance'.
2.  Suggest which simulation model best fits the observed dynamics.
3.  Propose specific parameter ranges (e.g., T values, p values) to explore in `run_experiment.py`.
    - **CRITICAL:** Use the 'Expected Args' names (without the dashes) as keys in your `parameter_sweep` and `global_params` JSON.
4.  Explain the reasoning for your suggestions, connecting qualitative observations to quantitative model parameters.
5.  Output your suggestions in a JSON format that can be used or easily adapted for a `run_experiment.py` configuration.

Output JSON Format:
```json
{{
    "simulation_module": "string (ising_2d, potts_2d, or percolation_2d)",
    "reasoning": "string",
    "parameter_sweep": {{
        "T": [value1, value2, ...], 
        "N": [size1, ...]
    }},
    "global_params": {{
        "eq_sweeps": 500,
        "meas_sweeps": 1000
    }}
}}
```
Note: For percolation, use "p" instead of "T" and "L" instead of "N".

Suggestions:
