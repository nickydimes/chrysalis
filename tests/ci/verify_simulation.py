import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Verify simulation results.")
    parser.add_argument("results_file", type=str, help="Path to the results.json file.")
    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        exit(1)

    with open(results_path, "r") as f:
        results = json.load(f)

    # Example verification: check if susceptibility is within a reasonable range for this small simulation
    # This is a very loose check and should be refined with more rigorous statistical analysis
    # for a real-world use case.
    susceptibility = results["measurements"]["chi"]
    if not (0.1 < susceptibility < 10.0):
        print(
            f"Error: Susceptibility ({susceptibility}) is outside the expected range."
        )
        exit(1)

    print("Simulation results verified successfully.")


if __name__ == "__main__":
    main()
