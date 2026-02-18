You are an expert computational physicist and research software engineer.
Your task is to synthesize a NEW Python simulation model for the Chrysalis project based on a natural language description of a social or cultural transition.

Transformation Description:
{TRANSFORMATION_DESCRIPTION}

Model Name: {MODEL_NAME}

Project Requirements & Best Practices:
1.  **Libraries:** Use `numpy`, `matplotlib`, and `numba` (`@njit`) for performance-critical loops.
    *   *Note:* Avoid using Python dynamic lists or non-jittable functions inside `@njit` blocks. Use pre-allocated NumPy arrays instead.
2.  **Structure:**
    *   Include a clear docstring describing the model's physics/logic.
    *   Implement the core simulation logic (e.g., initialization, update steps, measurement).
    *   Implement `calculate_protocol_metrics(results)` to map results to the Eight-Step Navigation Protocol.
        *   **CRITICAL:** This function MUST return a dictionary with exactly these keys: "Purification", "Containment", "Anchoring", "Dissolution", "Liminality", "Encounter", "Integration", "Emergence".
        *   The values should be lists (or 1D arrays) of numerical values representing the progression of that phase.
    *   Implement a `run(args=None)` function that:
        *   Uses `argparse` for configuration (include parameters relevant to the specific transition).
        *   Handles `--output_dir` and `--seed`.
        *   Performs the simulation and generates results.
        *   Saves plots (e.g., `results.png`).
        *   Saves a comprehensive `results.json` containing `params`, `main_results`, and `protocol_metrics`.
3.  **Style:** Follow the patterns in existing project files (e.g., `ising_2d.py`). Ensure the code is clean, well-commented, and standalone.
4.  **Output:** Provide the complete, functional Python code within a code block (```python ... ```). Do not include conversational text outside the block.

Specific Guidance for this Transition:
Translate the qualitative concepts in the description into quantitative mechanisms (e.g., 'trust' as a coupling constant, 'information flow' as a connectivity parameter, 'stress' as temperature).

Generated Code:
