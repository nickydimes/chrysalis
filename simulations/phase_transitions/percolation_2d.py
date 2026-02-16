"""
2D Site Percolation Phase Transition Simulation
================================================
Simulates 2D site percolation on a square lattice. Each site is occupied with
probability p; clusters form by nearest-neighbor connectivity. A giant spanning
cluster appears at p_c ≈ 0.5927. Unlike Ising/Potts, this model has no energy,
temperature, or Boltzmann weights — it is purely geometric/probabilistic — yet
exhibits the same qualitative critical behavior (universality).

Usage:
    python simulations/phase_transitions/percolation_2d.py

Produces: simulations/phase_transitions/percolation_results.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import label
from pathlib import Path


# --- Model ---

STRUCT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])  # 4-connectivity


def generate_lattice(L, p, rng):
    """Generate an LxL lattice where each site is occupied with probability p."""
    return (rng.random((L, L)) < p).astype(int)


def find_clusters(grid):
    """Label connected clusters using 4-connectivity. Returns labeled array and count."""
    labeled, num_clusters = label(grid, structure=STRUCT)
    return labeled, num_clusters


def largest_cluster_fraction(labeled, L):
    """Fraction of sites in the largest cluster (P_infinity)."""
    if labeled.max() == 0:
        return 0.0
    sizes = np.bincount(labeled.ravel())[1:]  # skip background (label 0)
    return sizes.max() / (L * L)


def mean_cluster_size(labeled):
    """Mean cluster size chi = sum(s^2 * n_s) / sum(s * n_s), excluding largest cluster."""
    if labeled.max() == 0:
        return 0.0
    sizes = np.bincount(labeled.ravel())[1:]  # skip background
    if len(sizes) <= 1:
        return 0.0
    largest_idx = sizes.argmax()
    remaining = np.delete(sizes, largest_idx)
    if remaining.sum() == 0:
        return 0.0
    return np.sum(remaining ** 2) / np.sum(remaining)


# --- Simulation ---

def simulate(L=100, p_values=None, n_realizations=50, seed=42):
    """
    Run percolation simulation across a range of occupation probabilities.

    Returns dict with arrays: p, order_param, susceptibility, variance,
    and snapshot lattices at low/critical/high p.
    """
    p_c = 0.5927

    if p_values is None:
        p_values = np.concatenate([
            np.linspace(0.1, 0.45, 8, endpoint=False),
            np.linspace(0.45, 0.75, 14, endpoint=False),  # dense near p_c
            np.linspace(0.75, 0.95, 6),
        ])

    rng = np.random.default_rng(seed)

    order_param = np.zeros(len(p_values))
    susceptibility = np.zeros(len(p_values))
    variance = np.zeros(len(p_values))

    # Probabilities at which to capture snapshots
    snap_targets = [0.3, p_c, 0.8]  # sparse, critical, dense
    snapshots = {}
    snapshot_labels = {}

    for idx, p in enumerate(p_values):
        print(f"  p = {p:.3f}  ({idx + 1}/{len(p_values)})")

        p_inf_samples = np.zeros(n_realizations)
        chi_samples = np.zeros(n_realizations)

        for r in range(n_realizations):
            grid = generate_lattice(L, p, rng)
            labeled, _ = find_clusters(grid)
            p_inf_samples[r] = largest_cluster_fraction(labeled, L)
            chi_samples[r] = mean_cluster_size(labeled)

        order_param[idx] = np.mean(p_inf_samples)
        susceptibility[idx] = np.mean(chi_samples)
        variance[idx] = np.var(p_inf_samples)

        # Capture snapshot if close to a target probability
        for p_snap in snap_targets:
            if abs(p - p_snap) < 0.02 and p_snap not in snapshots:
                grid = generate_lattice(L, p_snap, rng)
                labeled, _ = find_clusters(grid)
                snapshots[p_snap] = grid
                snapshot_labels[p_snap] = labeled

    return {
        "p": p_values,
        "order_param": order_param,
        "susceptibility": susceptibility,
        "variance": variance,
        "snapshots": snapshots,
        "snapshot_labels": snapshot_labels,
        "snap_targets": snap_targets,
        "p_c": p_c,
    }


# --- Visualization ---

def plot_results(results, output_path):
    """Generate 4-panel figure summarizing the percolation transition."""
    p = results["p"]
    p_c = results["p_c"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("2D Site Percolation — Phase Transition", fontsize=16, fontweight="bold")

    # Panel 1: Order Parameter (P_infinity)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(p, results["order_param"], "o-", color="#2c7bb6", markersize=4)
    ax1.axvline(p_c, color="gray", linestyle="--", alpha=0.7, label=f"$p_c$ = {p_c}")
    ax1.set_xlabel("Occupation Probability $p$")
    ax1.set_ylabel(r"$P_\infty$")
    ax1.set_title("Order Parameter (Largest Cluster Fraction)")
    ax1.legend()

    # Panel 2: Susceptibility (mean cluster size)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(p, results["susceptibility"], "o-", color="#d7191c", markersize=4)
    ax2.axvline(p_c, color="gray", linestyle="--", alpha=0.7, label=f"$p_c$ = {p_c}")
    ax2.set_xlabel("Occupation Probability $p$")
    ax2.set_ylabel(r"$\chi$ (mean cluster size)")
    ax2.set_title("Susceptibility Analog")
    ax2.legend()

    # Panel 3: Variance of P_infinity (specific heat analog)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(p, results["variance"], "o-", color="#fdae61", markersize=4)
    ax3.axvline(p_c, color="gray", linestyle="--", alpha=0.7, label=f"$p_c$ = {p_c}")
    ax3.set_xlabel("Occupation Probability $p$")
    ax3.set_ylabel(r"Var$(P_\infty)$")
    ax3.set_title("Fluctuations (Specific Heat Analog)")
    ax3.legend()

    # Panel 4: Lattice snapshots
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_axis_off()
    ax4.set_title("Lattice Snapshots")

    snapshots = results["snapshots"]
    snapshot_labels = results["snapshot_labels"]
    snap_targets = results["snap_targets"]
    labels = ["Sparse\n(p=0.3)", f"Critical\n(p≈{p_c})", "Dense\n(p=0.8)"]

    for i, (p_snap, lbl) in enumerate(zip(snap_targets, labels)):
        if p_snap in snapshots:
            inset = fig.add_axes([0.58 + i * 0.14, 0.12, 0.12, 0.25])
            grid = snapshots[p_snap]
            labeled = snapshot_labels[p_snap]

            # Color: empty=white, occupied=light blue, largest cluster=red
            display = np.zeros(grid.shape)  # 0 = empty
            display[grid == 1] = 1  # occupied
            if labeled.max() > 0:
                sizes = np.bincount(labeled.ravel())[1:]
                largest_id = sizes.argmax() + 1
                display[labeled == largest_id] = 2  # largest cluster

            snap_cmap = ListedColormap(["white", "#abd9e9", "#d7191c"])
            inset.imshow(display, cmap=snap_cmap, vmin=0, vmax=2,
                         interpolation="nearest")
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title(lbl, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nResults saved to {output_path}")


# --- Main ---

if __name__ == "__main__":
    print("2D Site Percolation Simulation")
    print("=" * 40)

    output_dir = Path(__file__).parent
    output_path = output_dir / "percolation_results.png"

    results = simulate(L=100, n_realizations=50)
    plot_results(results, output_path)
