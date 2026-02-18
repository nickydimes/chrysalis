"""
2D Site Percolation Phase Transition Simulation
================================================
Simulates 2D site percolation on a square lattice. Each site is occupied with
probability p; clusters form by nearest-neighbor connectivity. A giant spanning
cluster appears at p_c ≈ 0.5927. Unlike Ising/Potts, this model has no energy,
temperature, or Boltzmann weights — it is purely geometric/probabilistic — yet
exhibits the same qualitative critical behavior (universality).

Features:
- Union-Find data structure with path compression and union-by-rank
- Multi-seed ensemble averaging with bootstrap error bars
- Spanning probability detection
- Finite-size scaling with cluster size distributions
- Power-law fitting via MLE at p_c

Usage:
    python simulations/phase_transitions/percolation_2d.py

Produces:
    simulations/phase_transitions/percolation_results.png
    simulations/phase_transitions/percolation_scaling.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numba import njit
from pathlib import Path
import json
import argparse


# --- Statistical Infrastructure ---

def bootstrap_error(samples, n_bootstrap=200):
    """Return (mean, stderr) via bootstrap resampling."""
    rng = np.random.default_rng(0)
    n = len(samples)
    means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[b] = np.mean(samples[idx])
    return np.mean(samples), np.std(means)


def powerlaw_mle(data, s_min=5):
    """MLE power-law exponent: alpha = 1 + n / sum(ln(x_i / (s_min - 0.5)))."""
    filtered = data[data >= s_min]
    if len(filtered) < 5:
        return np.nan
    n = len(filtered)
    alpha = 1 + n / np.sum(np.log(filtered / (s_min - 0.5)))
    return alpha


# --- Union-Find (numba-accelerated) ---

@njit
def _uf_find(parent, x):
    """Find root with path compression."""
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


@njit
def _uf_union(parent, rank, size, a, b):
    """Union by rank."""
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra == rb:
        return
    if rank[ra] < rank[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    size[ra] += size[rb]
    if rank[ra] == rank[rb]:
        rank[ra] += 1


@njit
def _find_clusters(grid):
    """Numba-accelerated cluster finding via Union-Find.
    Returns (labeled, sizes_array, n_clusters)."""
    L = grid.shape[0]
    n = L * L
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int32)
    size = np.ones(n, dtype=np.int32)

    # Build unions
    for i in range(L):
        for j in range(L):
            if grid[i, j] == 0:
                continue
            idx = i * L + j
            if j + 1 < L and grid[i, j + 1]:
                _uf_union(parent, rank, size, idx, i * L + j + 1)
            if i + 1 < L and grid[i + 1, j]:
                _uf_union(parent, rank, size, idx, (i + 1) * L + j)

    # Label clusters — assign sequential labels to roots
    labeled = np.zeros((L, L), dtype=np.int32)
    root_label = np.full(n, -1, dtype=np.int32)
    label_counter = 0
    # First pass: assign labels to roots
    for i in range(L):
        for j in range(L):
            if grid[i, j] == 0:
                continue
            root = _uf_find(parent, i * L + j)
            if root_label[root] < 0:
                root_label[root] = label_counter
                label_counter += 1
            labeled[i, j] = root_label[root] + 1  # 1-indexed

    # Collect sizes (one per cluster)
    sizes = np.empty(label_counter, dtype=np.int32)
    for idx in range(n):
        if root_label[idx] >= 0:
            sizes[root_label[idx]] = size[idx]

    return labeled, sizes, label_counter


# Python-facing wrapper (keeps existing API)
class UnionFind:
    """Union-Find with path compression and union-by-rank (Python fallback)."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# --- Model ---

def generate_lattice(L, p, rng):
    """Generate an LxL lattice where each site is occupied with probability p."""
    return (rng.random((L, L)) < p).astype(np.int8)


def find_clusters_uf(grid):
    """Label connected clusters using numba-accelerated Union-Find.
    Returns (labeled_array, cluster_sizes)."""
    labeled, sizes, n_clusters = _find_clusters(grid)
    return labeled, sizes if n_clusters > 0 else np.array([], dtype=np.int32)


def largest_cluster_fraction(labeled, L):
    """Fraction of sites in the largest cluster (P_infinity)."""
    if labeled.max() == 0:
        return 0.0
    sizes = np.bincount(labeled.ravel())[1:]
    return sizes.max() / (L * L)


def mean_cluster_size(labeled, L):
    """Mean cluster size chi = sum(s^2) / L^2, excluding largest cluster."""
    if labeled.max() == 0:
        return 0.0
    sizes = np.bincount(labeled.ravel())[1:]
    if len(sizes) <= 1:
        return 0.0
    largest_idx = sizes.argmax()
    remaining = np.delete(sizes, largest_idx)
    if len(remaining) == 0:
        return 0.0
    return np.sum(remaining ** 2) / (L * L)


def spanning_probability(labeled, L):
    """Check if any cluster spans from top to bottom or left to right."""
    if labeled.max() == 0:
        return False
    top_labels = set(labeled[0, :]) - {0}
    bottom_labels = set(labeled[L - 1, :]) - {0}
    if top_labels & bottom_labels:
        return True
    left_labels = set(labeled[:, 0]) - {0}
    right_labels = set(labeled[:, L - 1]) - {0}
    if left_labels & right_labels:
        return True
    return False


# --- Simulation ---

SEEDS = [42, 137, 256, 314, 999]


def _simulate_single(L, p_values, n_realizations, seed):
    """Run one percolation simulation with a given seed."""
    rng = np.random.default_rng(seed)

    p_c = 0.5927
    snap_targets = [0.3, p_c, 0.8]
    snapshots = {}
    snapshot_labels = {}

    order_param = np.zeros(len(p_values))
    susceptibility = np.zeros(len(p_values))
    variance = np.zeros(len(p_values))
    span_prob = np.zeros(len(p_values))
    cluster_sizes_at_pc = []

    for idx, p in enumerate(p_values):
        p_inf_samples = np.zeros(n_realizations)
        chi_samples = np.zeros(n_realizations)
        span_count = 0

        for r in range(n_realizations):
            grid = generate_lattice(L, p, rng)
            labeled, sizes = find_clusters_uf(grid)
            p_inf_samples[r] = largest_cluster_fraction(labeled, L)
            chi_samples[r] = mean_cluster_size(labeled, L)
            if spanning_probability(labeled, L):
                span_count += 1

            # Collect cluster sizes near p_c for distribution
            if abs(p - p_c) < 0.02 and len(sizes) > 0:
                # Exclude largest cluster
                if len(sizes) > 1:
                    largest_idx = sizes.argmax()
                    remaining = np.delete(sizes, largest_idx)
                    cluster_sizes_at_pc.extend(remaining.tolist())

        order_param[idx] = np.mean(p_inf_samples)
        susceptibility[idx] = np.mean(chi_samples)
        variance[idx] = np.var(p_inf_samples)
        span_prob[idx] = span_count / n_realizations

        for p_snap in snap_targets:
            if abs(p - p_snap) < 0.02 and p_snap not in snapshots:
                grid = generate_lattice(L, p_snap, rng)
                labeled, _ = find_clusters_uf(grid)
                snapshots[p_snap] = grid
                snapshot_labels[p_snap] = labeled

    return {
        "order_param": order_param,
        "susceptibility": susceptibility,
        "variance": variance,
        "span_prob": span_prob,
        "snapshots": snapshots,
        "snapshot_labels": snapshot_labels,
        "cluster_sizes_at_pc": np.array(cluster_sizes_at_pc) if cluster_sizes_at_pc else np.array([]),
    }


def simulate(L=100, p_values=None, n_realizations=200, seeds=None):
    """Run multi-seed ensemble percolation simulation with bootstrap errors."""
    p_c = 0.5927

    if p_values is None:
        p_values = np.concatenate([
            np.linspace(0.1, 0.45, 8, endpoint=False),
            np.linspace(0.45, 0.75, 14, endpoint=False),
            np.linspace(0.75, 0.95, 6),
        ])
    if seeds is None:
        seeds = SEEDS

    snap_targets = [0.3, p_c, 0.8]

    all_op = []
    all_sus = []
    all_var = []
    all_span = []
    all_cluster_sizes = []
    snapshots = {}
    snapshot_labels = {}

    for i, seed in enumerate(seeds):
        print(f"  Seed {seed} ({i + 1}/{len(seeds)}), L={L}")
        result = _simulate_single(L, p_values, n_realizations, seed)
        all_op.append(result["order_param"])
        all_sus.append(result["susceptibility"])
        all_var.append(result["variance"])
        all_span.append(result["span_prob"])
        if len(result["cluster_sizes_at_pc"]) > 0:
            all_cluster_sizes.extend(result["cluster_sizes_at_pc"].tolist())
        if not snapshots:
            snapshots = result["snapshots"]
            snapshot_labels = result["snapshot_labels"]

    all_op = np.array(all_op)
    all_sus = np.array(all_sus)
    all_var = np.array(all_var)
    all_span = np.array(all_span)

    n_p = len(p_values)
    op_mean = np.zeros(n_p)
    op_err = np.zeros(n_p)
    sus_mean = np.zeros(n_p)
    sus_err = np.zeros(n_p)
    var_mean = np.zeros(n_p)
    var_err = np.zeros(n_p)

    for t in range(n_p):
        op_mean[t], op_err[t] = bootstrap_error(all_op[:, t])
        sus_mean[t], sus_err[t] = bootstrap_error(all_sus[:, t])
        var_mean[t], var_err[t] = bootstrap_error(all_var[:, t])

    span_mean = np.mean(all_span, axis=0)

    return {
        "p": p_values,
        "order_param": op_mean,
        "op_err": op_err,
        "susceptibility": sus_mean,
        "sus_err": sus_err,
        "variance": var_mean,
        "var_err": var_err,
        "span_prob": span_mean,
        "snapshots": snapshots,
        "snapshot_labels": snapshot_labels,
        "snap_targets": snap_targets,
        "p_c": p_c,
        "L": L,
        "cluster_sizes_at_pc": np.array(all_cluster_sizes) if all_cluster_sizes else np.array([]),
    }


def simulate_scaling(L_values=None, p_values=None, n_realizations=200, seeds=None):
    """Run finite-size scaling: multiple L values."""
    if L_values is None:
        L_values = [50, 100, 200]
    if p_values is None:
        p_values = np.linspace(0.4, 0.8, 30)
    if seeds is None:
        seeds = SEEDS

    results_by_L = {}
    for L in L_values:
        print(f"\n--- L = {L} ---")
        results_by_L[L] = simulate(L=L, p_values=p_values,
                                    n_realizations=n_realizations, seeds=seeds)
    return results_by_L


# --- Visualization ---

def plot_results(results, output_path):
    """Generate 4-panel figure summarizing the percolation transition."""
    p = results["p"]
    p_c = results["p_c"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("2D Site Percolation — Phase Transition", fontsize=16, fontweight="bold")

    # Panel 1: Order Parameter (P_infinity)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.errorbar(p, results["order_param"], yerr=results["op_err"],
                 fmt="o-", color="#2c7bb6", markersize=4, capsize=2)
    ax1.axvline(p_c, color="gray", linestyle="--", alpha=0.7, label=f"$p_c$ = {p_c}")
    ax1.set_xlabel("Occupation Probability $p$")
    ax1.set_ylabel(r"$P_\infty$")
    ax1.set_title("Order Parameter (Largest Cluster Fraction)")
    ax1.legend()

    # Panel 2: Susceptibility (mean cluster size)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.errorbar(p, results["susceptibility"], yerr=results["sus_err"],
                 fmt="o-", color="#d7191c", markersize=4, capsize=2)
    ax2.axvline(p_c, color="gray", linestyle="--", alpha=0.7, label=f"$p_c$ = {p_c}")
    ax2.set_xlabel("Occupation Probability $p$")
    ax2.set_ylabel(r"$\chi$ (mean cluster size)")
    ax2.set_title("Susceptibility Analog")
    ax2.legend()

    # Panel 3: Variance of P_infinity (specific heat analog)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.errorbar(p, results["variance"], yerr=results["var_err"],
                 fmt="o-", color="#fdae61", markersize=4, capsize=2)
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

            display = np.zeros(grid.shape)
            display[grid == 1] = 1
            if labeled.max() > 0:
                sizes = np.bincount(labeled.ravel())[1:]
                largest_id = sizes.argmax() + 1
                display[labeled == largest_id] = 2

            snap_cmap = ListedColormap(["white", "#abd9e9", "#d7191c"])
            inset.imshow(display, cmap=snap_cmap, vmin=0, vmax=2,
                         interpolation="nearest")
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title(lbl, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nResults saved to {output_path}")


def plot_scaling(results_by_L, output_path):
    """Generate 4-panel finite-size scaling figure."""
    p_c = 0.5927
    # Known exact exponents for 2D percolation
    beta_nu = 5.0 / 48.0    # beta/nu
    gamma_nu = 43.0 / 24.0  # gamma/nu
    nu = 4.0 / 3.0
    tau_exact = 187.0 / 91.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("2D Site Percolation — Finite-Size Scaling", fontsize=16, fontweight="bold")

    colors = {50: "#2c7bb6", 100: "#fdae61", 200: "#d7191c"}

    # Panel 1: Spanning probability vs p
    ax1 = axes[0, 0]
    for L, res in results_by_L.items():
        ax1.plot(res["p"], res["span_prob"], "o-", color=colors.get(L, "black"),
                 markersize=3, label=f"L={L}")
    ax1.axvline(p_c, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Occupation Probability $p$")
    ax1.set_ylabel("Spanning Probability")
    ax1.set_title("Spanning Probability")
    ax1.legend()

    # Panel 2: P_inf scaling collapse
    ax2 = axes[0, 1]
    for L, res in results_by_L.items():
        p_arr = res["p"]
        P_inf = res["order_param"]
        x = (p_arr - p_c) * L ** (1.0 / nu)
        y = L ** beta_nu * P_inf
        ax2.plot(x, y, "o", color=colors.get(L, "black"), markersize=3,
                 label=f"L={L}")
    ax2.set_xlabel(r"$(p - p_c) \cdot L^{1/\nu}$")
    ax2.set_ylabel(r"$L^{\beta/\nu} \cdot P_\infty$")
    ax2.set_title(r"$P_\infty$ Scaling Collapse")
    ax2.legend()

    # Panel 3: Chi_max vs L log-log
    ax3 = axes[1, 0]
    L_arr = []
    chi_max_arr = []
    for L, res in results_by_L.items():
        L_arr.append(L)
        chi_max_arr.append(np.max(res["susceptibility"]))
    L_arr = np.array(L_arr, dtype=float)
    chi_max_arr = np.array(chi_max_arr)

    ax3.loglog(L_arr, chi_max_arr, "s-", color="#2c7bb6", markersize=8)
    if len(L_arr) >= 2:
        coeffs = np.polyfit(np.log(L_arr), np.log(chi_max_arr), 1)
        L_fit = np.linspace(L_arr.min(), L_arr.max(), 50)
        ax3.loglog(L_fit, np.exp(coeffs[1]) * L_fit ** coeffs[0], "--",
                   color="gray", alpha=0.7,
                   label=rf"slope = {coeffs[0]:.2f} (exact $\gamma/\nu$ = {gamma_nu:.2f})")
    ax3.set_xlabel("L")
    ax3.set_ylabel(r"$\chi_{\max}$")
    ax3.set_title(r"$\chi_{\max}$ vs $L$ (log-log)")
    ax3.legend()

    # Panel 4: Cluster size distribution at p_c with power-law fit
    ax4 = axes[1, 1]
    # Use the largest L available for best statistics
    largest_L = max(results_by_L.keys())
    cs = results_by_L[largest_L]["cluster_sizes_at_pc"]
    if len(cs) > 0:
        cs = cs[cs > 0]
        s_max = int(cs.max())
        bins = np.logspace(0, np.log10(max(s_max, 2)), 40)
        hist, bin_edges = np.histogram(cs, bins=bins, density=True)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        mask = hist > 0
        ax4.loglog(bin_centers[mask], hist[mask], "o", color="#2c7bb6", markersize=4)

        # MLE power-law fit
        tau_fit = powerlaw_mle(cs, s_min=5)
        if not np.isnan(tau_fit):
            s_fit = np.logspace(np.log10(5), np.log10(max(s_max, 10)), 50)
            p_fit = s_fit ** (-tau_fit)
            p_fit *= hist[mask][bin_centers[mask] >= 5][0] / (5 ** (-tau_fit)) if np.any(bin_centers[mask] >= 5) else 1
            ax4.loglog(s_fit, p_fit, "--", color="gray", alpha=0.7,
                       label=rf"$\tau$ = {tau_fit:.2f} (exact = {tau_exact:.2f})")
    ax4.set_xlabel("Cluster size $s$")
    ax4.set_ylabel("$P(s)$")
    ax4.set_title(f"Cluster Size Distribution at $p_c$ (L={largest_L})")
    ax4.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Scaling results saved to {output_path}")


def calculate_protocol_metrics(results):
    """
    Maps geometric percolation results to the Eight-Step Navigation Protocol.
    """
    p = np.atleast_1d(results["p"])
    p_inf = np.atleast_1d(results["order_param"])
    sus = np.atleast_1d(results["susceptibility"])
    var_p_inf = np.atleast_1d(results["variance"])
    span_prob = np.atleast_1d(results["span_prob"])
    
    # 1. Purification: filling the structure
    if np.max(p) > 0:
        purification = p / np.max(p)
    else:
        purification = p

    # 2. Containment: Order parameter relative to max
    if np.max(p_inf) > 0:
        containment = p_inf / np.max(p_inf)
    else:
        containment = p_inf

    # 3. Anchoring: Spanning probability (global connectivity)
    anchoring = span_prob

    # 4. Dissolution: Peak in p_inf fluctuations
    if np.max(var_p_inf) > 0:
        dissolution = var_p_inf / np.max(var_p_inf)
    else:
        dissolution = var_p_inf

    # 5. Liminality: Susceptibility peak
    if np.max(sus) > 0:
        liminality = sus / np.max(sus)
    else:
        liminality = sus

    # 6. Encounter: Correlation length proxy
    sq_sus = np.sqrt(sus)
    if np.max(sq_sus) > 0:
        encounter = sq_sus / np.max(sq_sus)
    else:
        encounter = sus

    # 7. Integration: Product of spanning probability and order parameter
    integration = span_prob * p_inf

    # 8. Emergence: Final spanning cluster strength
    emergence = p_inf

    metrics = {
        "Purification": purification,
        "Containment": containment,
        "Anchoring": anchoring,
        "Dissolution": dissolution,
        "Liminality": liminality,
        "Encounter": encounter,
        "Integration": integration,
        "Emergence": emergence
    }
    
    result_dict = {}
    for k, v in metrics.items():
        if hasattr(v, "tolist"):
            result_dict[k] = v.tolist()
        else:
            result_dict[k] = [float(v)]
            
    return result_dict


# --- Main ---

def run(args=None):
    """
    Runs the 2D Site Percolation Model simulation.
    Args:
        args: A list of arguments (e.g., from sys.argv[1:]). If None, uses default values or argparse.
    """
    parser = argparse.ArgumentParser(description="Run 2D Site Percolation Model simulation.")
    parser.add_argument("--L", type=int, default=100, help="Lattice size L")
    parser.add_argument("--p", type=float, default=None, help="Single probability to simulate")
    parser.add_argument("--n_realizations", type=int, default=200, help="Number of realizations per p-value")
    parser.add_argument("--L_values", type=int, nargs='*', default=[50, 100, 200],
                        help="Lattice sizes for finite-size scaling")
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).parent),
                        help="Directory to save simulation results.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for random number generation.")
    parser.add_argument("--no_scaling", action="store_true", help="Disable finite-size scaling.")
    
    # Parse arguments provided, or from sys.argv if none provided
    parsed_args = parser.parse_args(args=args)

    print("2D Site Percolation Simulation")
    print("=" * 40)

    output_dir = Path(parsed_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists
    output_path = output_dir / "percolation_results.png"
    scaling_path = output_dir / "percolation_scaling.png"

    # Main results
    p_vals = [parsed_args.p] if parsed_args.p is not None else None
    results = simulate(L=parsed_args.L, p_values=p_vals, n_realizations=parsed_args.n_realizations, seeds=[parsed_args.seed])
    plot_results(results, output_path)

    if not parsed_args.no_scaling:
        # Finite-size scaling
        print("\n--- Finite-Size Scaling ---")
        results_by_L = simulate_scaling(L_values=parsed_args.L_values,
                                         n_realizations=parsed_args.n_realizations, seeds=[parsed_args.seed])

        plot_scaling(results_by_L, scaling_path)
    else:
        results_by_L = {}

    # Prepare data for JSON serialization (convert numpy arrays to lists)
    def convert_numpy_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_numpy_to_list(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy_to_list(elem) for elem in obj]
        return obj

    all_results = {
        "params": {
            "L": parsed_args.L,
            "p": parsed_args.p,
            "n_realizations": parsed_args.n_realizations,
            "L_values": parsed_args.L_values,
            "seed": parsed_args.seed
        },
        "main_results": convert_numpy_to_list(results),
        "scaling_results_by_L": {str(k): convert_numpy_to_list(v) for k, v in results_by_L.items()},
        "protocol_metrics": calculate_protocol_metrics(results)
    }
    
    results_json_path = output_dir / "results.json"
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Numerical results saved to {results_json_path}")

    # Save CSV
    import pandas as pd
    df = pd.DataFrame({
        "p": results["p"],
        "order_param": results["order_param"],
        "op_err": results["op_err"],
        "susceptibility": results["susceptibility"],
        "sus_err": results["sus_err"],
        "variance": results["variance"],
        "var_err": results["var_err"],
        "span_prob": results["span_prob"]
    })
    results_csv_path = output_dir / "results.csv"
    df.to_csv(results_csv_path, index=False)
    print(f"CSV results saved to {results_csv_path}")


if __name__ == "__main__":
    run()
