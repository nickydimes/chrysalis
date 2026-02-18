import argparse
import importlib
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Run various phase transition simulations.")
    parser.add_argument("simulation_name", type=str,
                        help="Name of the simulation to run (e.g., ising_2d, percolation_2d, potts_2d).")
    parser.add_argument("--args", nargs=argparse.REMAINDER,
                        help="Additional arguments to pass to the simulation script.")

    args = parser.parse_args()
    simulation_name = args.simulation_name

    try:
        # Add the parent directory of 'simulations' to sys.path
        # so 'simulations.phase_transitions' can be imported
        simulations_path = os.path.abspath(os.path.dirname(__file__))
        project_root = os.path.abspath(os.path.join(simulations_path, '..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Dynamically import the simulation module
        # The module will be 'simulations.phase_transitions.<simulation_name>'
        module_path = f"simulations.phase_transitions.{simulation_name}"
        simulation_module = importlib.import_module(module_path)

        # Assuming each simulation module has a 'run' function
        if hasattr(simulation_module, 'run') and callable(simulation_module.run):
            print(f"Running simulation: {simulation_name}")
            # Pass additional arguments if any
            if args.args:
                simulation_module.run(args.args)
            else:
                simulation_module.run()
            print(f"Simulation {simulation_name} finished.")
        else:
            print(f"Error: Simulation module '{simulation_name}' does not have a 'run' function.")

    except ImportError:
        print(f"Error: Simulation '{simulation_name}' not found in simulations.phase_transitions.")
    except Exception as e:
        print(f"An error occurred while running simulation '{simulation_name}': {e}")

if __name__ == "__main__":
    main()
