import argparse
from pathlib import Path
from chrysalis.src.llm_integration.llm_clients import (
    GeminiAPIClient,
    OllamaClient,
    LLMClient,
)
from chrysalis.simulations.aggregate_results import (
    load_experiment_results,
    group_runs_by_params,
    find_critical_temperature,
)


def main():
    parser = argparse.ArgumentParser(
        description="Provide LLM-powered interpretation of simulation results."
    )
    parser.add_argument(
        "experiment_base_dir",
        type=str,
        help="Path to the base directory of an experiment.",
    )
    parser.add_argument(
        "--template_name",
        type=str,
        default="sim_interpretation_template",
        help="Name of the prompt template to use.",
    )
    parser.add_argument(
        "--llm_client",
        type=str,
        choices=["gemini_api", "ollama"],
        default="gemini_api",
        help="Choose which LLM client to use.",
    )
    parser.add_argument(
        "--llm_model", type=str, default=None, help="Specify the LLM model name."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File to save the generated interpretation. Defaults to 'interpretation.md' in report dir.",
    )

    args = parser.parse_args()

    # Find project root relative to this script
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent

    base_dir = Path(args.experiment_base_dir)
    if not base_dir.exists():
        print(f"Error: Experiment base directory '{base_dir}' not found.")
        exit(1)

    template_path = project_root / "prompts" / "templates" / f"{args.template_name}.md"
    if not template_path.exists():
        # Fallback
        template_path = Path("chrysalis/prompts/templates") / f"{args.template_name}.md"
        if not template_path.exists():
            print(f"Error: Template file '{template_path}' not found.")
            exit(1)

    try:
        # 1. Load and process simulation data
        all_runs_data = load_experiment_results(base_dir)
        if not all_runs_data:
            print("No simulation run data found to interpret.")
            exit(0)

        # Generate a summary for the prompt (similar to aggregate_results.py)
        all_param_keys = set()
        for run in all_runs_data:
            all_param_keys.update(run["params"].keys())

        grouping_keys = [
            k
            for k in all_param_keys
            if k
            not in [
                "run_seed",
                "seed",
                "L_values",
                "n_autocorr_sweeps",
                "T_values",
                "p_values",
            ]
        ]
        grouped_by_common_params = group_runs_by_params(all_runs_data, grouping_keys)

        summary_lines = [f"Simulation Experiment Summary: {base_dir.name}"]
        summary_lines.append(f"Total individual runs: {len(all_runs_data)}\n")

        for params_tuple, runs in grouped_by_common_params.items():
            params_dict = dict(zip(grouping_keys, params_tuple))
            summary_lines.append(f"Parameter Group: {params_dict}")
            summary_lines.append(f"  Number of runs: {len(runs)}")

            if runs and "main_results" in runs[0]:
                main_res = runs[0]["main_results"]
                if "T" in main_res:
                    summary_lines.append(
                        f"  Temperature range: {min(main_res['T'])} to {max(main_res['T'])}"
                    )
                elif "p" in main_res:
                    summary_lines.append(
                        f"  p range: {min(main_res['p'])} to {max(main_res['p'])}"
                    )

                # Aggregate key metrics across runs in this group
                T_values_group = []
                susceptibility_values_group = []
                for run in runs:
                    if (
                        "main_results" in run
                        and "T" in run["main_results"]
                        and "susceptibility" in run["main_results"]
                    ):
                        T_values_group.extend(run["main_results"]["T"])
                        susceptibility_values_group.extend(
                            run["main_results"]["susceptibility"]
                        )
                    elif (
                        "main_results" in run
                        and "p" in run["main_results"]
                        and "susceptibility" in run["main_results"]
                    ):
                        T_values_group.extend(run["main_results"]["p"])
                        susceptibility_values_group.extend(
                            run["main_results"]["susceptibility"]
                        )

                if T_values_group:
                    crit_val, peak_sus = find_critical_temperature(
                        T_values_group, susceptibility_values_group
                    )
                    summary_lines.append(f"  Estimated Critical Point: {crit_val:.3f}")
                    summary_lines.append(f"  Peak Susceptibility: {peak_sus:.3f}")
            summary_lines.append("")

        experiment_summary = "\n".join(summary_lines)

        # 2. Prepare the LLM prompt
        template_content = template_path.read_text(encoding="utf-8")
        llm_prompt = template_content.replace(
            "{EXPERIMENT_SUMMARY}", experiment_summary
        )

        # 3. Initialize LLM Client
        llm_client: LLMClient
        if args.llm_client == "gemini_api":
            llm_client = GeminiAPIClient(
                model_name=(
                    args.llm_model if args.llm_model else "models/gemini-pro-latest"
                )
            )
        elif args.llm_client == "ollama":
            llm_client = OllamaClient(
                model_name=(
                    args.llm_model if args.llm_model else "llama3.3:70b-instruct-q4_K_M"
                )
            )

        print(f"Using LLM client: {args.llm_client} (model: {llm_client.model_name})")
        print("Generating interpretation...")

        interpretation = llm_client.generate_text(llm_prompt)

        # 4. Save and output
        if args.output_file:
            output_file_path = Path(args.output_file)
        else:
            report_dir = project_root / "simulations" / "reports" / base_dir.name
            report_dir.mkdir(parents=True, exist_ok=True)
            output_file_path = report_dir / "interpretation.md"

        output_file_path.write_text(interpretation, encoding="utf-8")

        print("\n--- Generated Interpretation ---")
        print(interpretation)
        print(f"\nInterpretation saved to: {output_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    main()
