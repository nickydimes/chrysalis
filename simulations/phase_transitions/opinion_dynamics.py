import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import json
import argparse
from pathlib import Path

def calculate_protocol_metrics(results):
    """Calculate metrics according to Eight-Step Navigation Protocol."""
    protocol_metrics = {
        'order_parameter': results['consensus'][-1],
        'energy': np.mean(results['stress']),
        'entropy': -np.sum(results['opinion_distribution'] * np.log(results['opinion_distribution'])),
        'activity_level': np.std(results['polarization'])
    }
    return protocol_metrics

class OpinionDynamics:
    def __init__(self, L=50, tolerance=0.1, stress=0.01, noise_strength=0.1, neighborhood_size=8, seed=None):
        self.L = L
        self.tolerance = tolerance
        self.stress = stress
        self.noise_strength = noise_strength
        self.neighborhood_size = neighborhood_size
        np.random.seed(seed)
        self.grid = None

    def initialize(self):
        """Initialize agents with random opinions between 0 and 1."""
        self.grid = np.random.rand(self.L, self.L)

    @staticmethod
    @njit
    def update_opinions(grid, tolerance, stress, noise_strength, neighborhood_size=8):
        L = grid.shape[0]
        new_grid = grid.copy()
        
        for i in range(L):
            for j in range(L):
                neighbors = []
                # Moore neighborhood (8 surrounding cells)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        x = (i + dx) % L
                        y = (j + dy) % L
                        diff = abs(grid[i,j] - grid[x,y])
                        if diff < tolerance:
                            neighbors.append(grid[x,y])
                if len(neighbors) > 0:
                    avg_neighbor = np.mean(neighbors)
                    new_grid[i,j] = (grid[i,j] + avg_neighbor) / 2.0
                
                # Add stress-induced noise
                noise = np.random.normal(0, noise_strength * stress)
                new_grid[i,j] += noise
                if new_grid[i,j] < 0:
                    new_grid[i,j] = 0
                elif new_grid[i,j] > 1:
                    new_grid[i,j] = 1
        
        return new_grid

    def run_simulation(self, steps=1000, save_interval=10):
        results = {
            'opinions': [],
            'consensus': [],
            'polarization': []
        }
        
        self.initialize()
        for step in range(steps):
            if step % save_interval == 0:
                consensus = np.std(self.grid)
                polarization = np.max(self.grid) - np.min(self.grid)
                results['opinions'].append(self.grid.copy())
                results['consensus'].append(consensus)
                results['polarization'].append(polarization)
            self.grid = self.update_opinions(
                self.grid, self.tolerance, self.stress,
                self.noise_strength
            )
        return results

def run(args=None):
    parser = argparse.ArgumentParser(description='Opinion Dynamics Simulation')
    parser.add_argument('--L', type=int, default=50, help='Lattice size')
    parser.add_argument('--tolerance', type=float, default=0.1, help='Tolerance threshold')
    parser.add_argument('--stress', type=float, default=0.01, help='Stress (temperature)')
    parser.add_argument('--noise_strength', type=float, default=0.1, help='Noise strength')
    parser.add_argument('--steps', type=int, default=1000, help='Number of simulation steps')
    parser.add_argument('--save_interval', type=int, default=10, help='Save interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    
    args = parser.parse_args(args)
    
    # Initialize model
    model = OpinionDynamics(
        L=args.L,
        tolerance=args.tolerance,
        stress=args.stress,
        noise_strength=args.noise_strength,
        seed=args.seed
    )
    
    # Run simulation
    results = model.run_simulation(steps=args.steps, save_interval=args.save_interval)
    
    # Calculate metrics
    protocol_metrics = calculate_protocol_metrics(results)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plots
    plt.figure(figsize=(8,6))
    plt.plot(results['consensus'])
    plt.xlabel('Time')
    plt.ylabel('Consensus (Standard Deviation)')
    plt.title(f'Opinion Dynamics: Tolerance={args.tolerance}, Stress={args.stress}')
    plt.savefig(output_dir / 'results.png', dpi=300)
    plt.close()
    
    # Save results
    results_to_save = {
        'params': vars(args),
        'main_results': {
            'consensus': results['consensus'],
            'polarization': results['polarization']
        },
        'protocol_metrics': protocol_metrics
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_to_save, f)

if __name__ == '__main__':
    run()