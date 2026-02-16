"""
Neural Branching Process — Critical Brain Hypothesis
=====================================================
Simulates a neural branching process on a 100×100 grid with periodic boundary
conditions. Each active neuron probabilistically activates its neighbors with
branching ratio σ. At the critical point (σ_c = 1.0), avalanches follow
power-law distributions — the hallmark of criticality in neural systems.

Usage:
    python simulations/neuroscience/critical_brain.py

Produces: simulations/neuroscience/critical_brain_results.png
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# --- Model ---

def run_avalanche(N, sigma, rng, max_activations):
    """
    Run a single avalanche on an N×N grid with branching ratio sigma.

    Each active neuron attempts to activate each of its 4 neighbors with
    probability sigma/2. Neurons that have already fired (in the footprint)
    cannot be reactivated. This maps to bond percolation on the square
    lattice (p_c = 0.5), placing the critical point at sigma_c = 1.0.

    Returns (size, duration, footprint) where footprint is the boolean
    array of all neurons that fired during the avalanche.
    """
    active = np.zeros((N, N), dtype=bool)
    footprint = np.zeros((N, N), dtype=bool)

    # Seed one random neuron
    i, j = rng.integers(N), rng.integers(N)
    active[i, j] = True
    footprint[i, j] = True

    size = 1
    duration = 0
    p = sigma / 2.0

    while np.any(active):
        duration += 1
        new_active = np.zeros((N, N), dtype=bool)

        # Each active neuron fires to each of 4 neighbors with prob sigma/2
        for shift, axis in [(1, 0), (-1, 0), (1, 1), (-1, 1)]:
            neighbors = np.roll(active, shift, axis=axis)
            fires = neighbors & (rng.random((N, N)) < p)
            new_active |= fires

        # Neurons can only fire once per avalanche
        new_active &= ~footprint

        # Update states
        active = new_active
        footprint |= active
        size += np.sum(active)

        if size >= max_activations:
            break

    return size, duration, footprint


# --- Simulation ---

def simulate(N=100, sigma_values=None, n_avalanches=500, seed=42):
    """
    Sweep branching ratio σ and collect avalanche statistics.

    Returns dict with arrays: sigma, order_param, susceptibility,
    mean_duration, and snapshot footprints at three regimes.
    """
    if sigma_values is None:
        sigma_values = np.concatenate([
            np.linspace(0.2, 0.8, 8, endpoint=False),
            np.linspace(0.8, 1.3, 14, endpoint=False),  # dense near σ_c
            np.linspace(1.3, 2.0, 6),
        ])

    rng = np.random.default_rng(seed)
    max_act = N * N

    order_param = np.zeros(len(sigma_values))
    susceptibility = np.zeros(len(sigma_values))
    mean_duration = np.zeros(len(sigma_values))

    # Snapshot targets
    sigma_c = 1.0
    snap_targets = [0.5, 1.0, 1.5]
    snapshots = {}

    for idx, sigma in enumerate(sigma_values):
        print(f"  σ = {sigma:.3f}  ({idx + 1}/{len(sigma_values)})")

        sizes = np.zeros(n_avalanches)
        durations = np.zeros(n_avalanches)

        for a in range(n_avalanches):
            s, d, fp = run_avalanche(N, sigma, rng, max_act)
            sizes[a] = s
            durations[a] = d

            # Capture the largest avalanche footprint near each target σ
            for s_target in snap_targets:
                if abs(sigma - s_target) < 0.08 and (
                    s_target not in snapshots or np.sum(fp) > np.sum(snapshots[s_target])
                ):
                    snapshots[s_target] = fp.copy()

        order_param[idx] = np.mean(sizes) / max_act
        susceptibility[idx] = np.var(sizes)
        mean_duration[idx] = np.mean(durations)

    return {
        "sigma": sigma_values,
        "order_param": order_param,
        "susceptibility": susceptibility,
        "mean_duration": mean_duration,
        "snapshots": snapshots,
        "snap_targets": snap_targets,
    }


# --- Visualization ---

def plot_results(results, output_path):
    """Generate 4-panel figure summarizing the neural critical transition."""
    sigma = results["sigma"]
    sigma_c = 1.0

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Neural Branching Process — Critical Brain Hypothesis",
                 fontsize=16, fontweight="bold")

    # Panel 1: Order parameter
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(sigma, results["order_param"], "o-", color="#2c7bb6", markersize=4)
    ax1.axvline(sigma_c, color="gray", linestyle="--", alpha=0.7,
                label=r"$\sigma_c$ = 1.0")
    ax1.set_xlabel(r"Branching ratio $\sigma$")
    ax1.set_ylabel(r"$\langle s \rangle / N^2$")
    ax1.set_title("Order Parameter (Mean Avalanche Size)")
    ax1.legend()

    # Panel 2: Susceptibility
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(sigma, results["susceptibility"], "o-", color="#d7191c", markersize=4)
    ax2.axvline(sigma_c, color="gray", linestyle="--", alpha=0.7,
                label=r"$\sigma_c$ = 1.0")
    ax2.set_xlabel(r"Branching ratio $\sigma$")
    ax2.set_ylabel(r"Var$(s)$")
    ax2.set_title("Susceptibility (Avalanche Size Variance)")
    ax2.legend()

    # Panel 3: Mean duration
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(sigma, results["mean_duration"], "o-", color="#fdae61", markersize=4)
    ax3.axvline(sigma_c, color="gray", linestyle="--", alpha=0.7,
                label=r"$\sigma_c$ = 1.0")
    ax3.set_xlabel(r"Branching ratio $\sigma$")
    ax3.set_ylabel(r"$\langle d \rangle$")
    ax3.set_title("Mean Avalanche Duration")
    ax3.legend()

    # Panel 4: Avalanche footprint snapshots
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_axis_off()
    ax4.set_title("Avalanche Footprints")

    snapshots = results["snapshots"]
    snap_targets = results["snap_targets"]
    labels = [r"Subcritical" + "\n" + r"($\sigma$=0.5)",
              r"Critical" + "\n" + r"($\sigma$=1.0)",
              r"Supercritical" + "\n" + r"($\sigma$=1.5)"]

    for i, (s_target, label) in enumerate(zip(snap_targets, labels)):
        if s_target in snapshots:
            inset = fig.add_axes([0.58 + i * 0.14, 0.12, 0.12, 0.25])
            inset.imshow(snapshots[s_target].astype(float), cmap="Purples",
                         vmin=0, vmax=1, interpolation="nearest")
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title(label, fontsize=9)

    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08,
                        hspace=0.35, wspace=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nResults saved to {output_path}")


# --- Main ---

if __name__ == "__main__":
    print("Neural Branching Process Simulation")
    print("=" * 40)

    output_dir = Path(__file__).parent
    output_path = output_dir / "critical_brain_results.png"

    results = simulate(N=100, n_avalanches=500)
    plot_results(results, output_path)
