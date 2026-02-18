"""
Generate Large-Scale Social Data for Neural-Socio Mapping
========================================================
Generates a synthetic social network (Barabási–Albert) and simulates
information cascades (social avalanches) using a threshold-based or
branching-based process, analogous to the Critical Brain simulation.

Output:
    data/raw/social_cascades_large.json
"""

import networkx as nx
import numpy as np
import json
from pathlib import Path


def generate_social_cascades(n_nodes=1000, m=3, n_cascades=500, p_transmit=0.1):
    """
    Generate social cascades on a Barabási-Albert network.
    """
    print(f"Generating Scale-Free Social Network (N={n_nodes}, m={m})...")
    G = nx.barabasi_albert_graph(n_nodes, m)

    cascades = []

    print(f"Simulating {n_cascades} social cascades...")
    for i in range(n_cascades):
        # Pick a random seed node
        seed = np.random.choice(G.nodes())

        # Simple branching process on the network
        active = {seed}
        all_activated = {seed}
        duration = 0

        while active:
            duration += 1
            next_active = set()
            for node in active:
                neighbors = list(G.neighbors(node))
                for neighbor in neighbors:
                    if neighbor not in all_activated:
                        if np.random.random() < p_transmit:
                            next_active.add(neighbor)
                            all_activated.add(neighbor)
            active = next_active
            if not active:
                break

        cascades.append(
            {
                "id": i,
                "seed_node": int(seed),
                "size": len(all_activated),
                "duration": duration,
                "nodes": [int(n) for n in all_activated],
            }
        )

    return {
        "network_metadata": {
            "n_nodes": n_nodes,
            "m": m,
            "avg_degree": np.mean([d for n, d in G.degree()]),
        },
        "cascades": cascades,
    }


def main():
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    # We use a transmission probability near the reciprocal of the average degree
    # to be near a 'critical' point in the branching process on the network.
    data = generate_social_cascades(n_nodes=2000, m=2, n_cascades=1000, p_transmit=0.25)

    output_path = output_dir / "social_cascades_large.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Successfully generated {len(data['cascades'])} social cascades.")
    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    main()
