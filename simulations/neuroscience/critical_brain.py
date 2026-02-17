"""
Neural Branching Process — Critical Brain Hypothesis
=====================================================
Simulates a neural branching process on a grid with periodic boundary
conditions. Each active neuron probabilistically activates its neighbors with
branching ratio σ. At the critical point (σ_c = 1.0), avalanches follow
power-law distributions — the hallmark of criticality in neural systems.

Features:
- Sparse avalanche propagation using Python sets (fast for small avalanches)
- Multi-seed ensemble averaging with bootstrap error bars
- Power-law fitting via MLE for avalanche size and duration distributions
- Finite-size scaling across multiple grid sizes
- Size-duration scaling analysis

Usage:
    python simulations/neuroscience/critical_brain.py

Produces:
    simulations/neuroscience/critical_brain_results.png
    simulations/neuroscience/critical_brain_scaling.png
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from pathlib import Path


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


# --- Model (numba-accelerated) ---

@njit
def _run_avalanche(N, p, seed_i, seed_j, max_activations, rand_vals, footprint):
    """Numba-accelerated avalanche propagation using array-based BFS.

    Uses two flat arrays as alternating active-neuron buffers.
    footprint is a pre-allocated N×N boolean array (zeroed on entry).

    Returns (size, duration, rand_consumed).
    """
    # Active buffers (flat: pairs of i,j stored as i*N+j)
    buf_a = np.empty(N * N, dtype=np.int32)
    buf_b = np.empty(N * N, dtype=np.int32)

    footprint[seed_i, seed_j] = True
    buf_a[0] = seed_i * N + seed_j
    n_active = 1
    size = 1
    duration = 0
    rand_idx = 0
    n_rand = len(rand_vals)

    di = np.array([1, -1, 0, 0], dtype=np.int32)
    dj = np.array([0, 0, 1, -1], dtype=np.int32)

    while n_active > 0:
        duration += 1
        n_new = 0

        for k in range(n_active):
            idx = buf_a[k]
            ai = idx // N
            aj = idx % N
            for d in range(4):
                ni = (ai + di[d]) % N
                nj = (aj + dj[d]) % N
                if not footprint[ni, nj]:
                    if rand_vals[rand_idx] < p:
                        footprint[ni, nj] = True
                        buf_b[n_new] = ni * N + nj
                        n_new += 1
                    rand_idx += 1
                    if rand_idx >= n_rand:
                        rand_idx = 0

        size += n_new
        n_active = n_new

        # Swap buffers
        buf_a, buf_b = buf_b, buf_a

        if size >= max_activations:
            break

    return size, duration


def run_avalanche_sparse(N, sigma, rng, max_activations):
    """Run a single avalanche with numba acceleration.

    Returns (size, duration, footprint_array).
    """
    p = sigma / 2.0
    seed_i = int(rng.integers(N))
    seed_j = int(rng.integers(N))
    # Pre-generate random numbers (4 per potential neuron)
    rand_vals = rng.random(N * N)
    footprint = np.zeros((N, N), dtype=np.bool_)
    size, duration = _run_avalanche(N, p, seed_i, seed_j, max_activations,
                                     rand_vals, footprint)
    return size, duration, footprint


def footprint_to_array(footprint, N):
    """Identity for array footprints, convert sets for backward compat."""
    if isinstance(footprint, np.ndarray):
        return footprint
    arr = np.zeros((N, N), dtype=bool)
    for i, j in footprint:
        arr[i, j] = True
    return arr


# --- Simulation ---

SEEDS = [42, 137, 256, 314, 999]


def _simulate_single(N, sigma_values, n_avalanches, seed):
    """Run one critical brain simulation with a given seed."""
    rng = np.random.default_rng(seed)
    max_act = N * N

    sigma_c = 1.0
    snap_targets = [0.5, 1.0, 1.5]
    snapshots = {}

    order_param = np.zeros(len(sigma_values))
    susceptibility = np.zeros(len(sigma_values))
    mean_duration = np.zeros(len(sigma_values))

    # Raw data at sigma_c for distribution analysis
    sizes_at_sc = []
    durations_at_sc = []

    for idx, sigma in enumerate(sigma_values):
        sizes = np.zeros(n_avalanches)
        durations = np.zeros(n_avalanches)

        for a in range(n_avalanches):
            s, d, fp = run_avalanche_sparse(N, sigma, rng, max_act)
            sizes[a] = s
            durations[a] = d

            # Capture the largest avalanche footprint near each target σ
            for s_target in snap_targets:
                if abs(sigma - s_target) < 0.08:
                    if s_target not in snapshots or s > np.sum(snapshots[s_target]):
                        snapshots[s_target] = footprint_to_array(fp, N)

            # Collect raw data near sigma_c
            if abs(sigma - sigma_c) < 0.05:
                sizes_at_sc.append(s)
                durations_at_sc.append(d)

        order_param[idx] = np.mean(sizes) / max_act
        susceptibility[idx] = np.var(sizes)
        mean_duration[idx] = np.mean(durations)

    return {
        "order_param": order_param,
        "susceptibility": susceptibility,
        "mean_duration": mean_duration,
        "snapshots": snapshots,
        "sizes_at_sc": np.array(sizes_at_sc),
        "durations_at_sc": np.array(durations_at_sc),
    }


def simulate(N=100, sigma_values=None, n_avalanches=2000, seeds=None):
    """Run multi-seed ensemble critical brain simulation with bootstrap errors."""
    if sigma_values is None:
        sigma_values = np.concatenate([
            np.linspace(0.2, 0.8, 8, endpoint=False),
            np.linspace(0.8, 1.3, 14, endpoint=False),
            np.linspace(1.3, 2.0, 6),
        ])
    if seeds is None:
        seeds = SEEDS

    sigma_c = 1.0
    snap_targets = [0.5, 1.0, 1.5]

    all_op = []
    all_sus = []
    all_dur = []
    all_sizes_sc = []
    all_durations_sc = []
    snapshots = {}

    for i, seed in enumerate(seeds):
        print(f"  Seed {seed} ({i + 1}/{len(seeds)}), N={N}")
        result = _simulate_single(N, sigma_values, n_avalanches, seed)
        all_op.append(result["order_param"])
        all_sus.append(result["susceptibility"])
        all_dur.append(result["mean_duration"])
        if len(result["sizes_at_sc"]) > 0:
            all_sizes_sc.extend(result["sizes_at_sc"].tolist())
            all_durations_sc.extend(result["durations_at_sc"].tolist())
        if not snapshots:
            snapshots = result["snapshots"]

    all_op = np.array(all_op)
    all_sus = np.array(all_sus)
    all_dur = np.array(all_dur)

    n_sig = len(sigma_values)
    op_mean = np.zeros(n_sig)
    op_err = np.zeros(n_sig)
    sus_mean = np.zeros(n_sig)
    sus_err = np.zeros(n_sig)
    dur_mean = np.zeros(n_sig)
    dur_err = np.zeros(n_sig)

    for t in range(n_sig):
        op_mean[t], op_err[t] = bootstrap_error(all_op[:, t])
        sus_mean[t], sus_err[t] = bootstrap_error(all_sus[:, t])
        dur_mean[t], dur_err[t] = bootstrap_error(all_dur[:, t])

    return {
        "sigma": sigma_values,
        "order_param": op_mean,
        "op_err": op_err,
        "susceptibility": sus_mean,
        "sus_err": sus_err,
        "mean_duration": dur_mean,
        "dur_err": dur_err,
        "snapshots": snapshots,
        "snap_targets": snap_targets,
        "N": N,
        "sizes_at_sc": np.array(all_sizes_sc),
        "durations_at_sc": np.array(all_durations_sc),
    }


def simulate_scaling(N_values=None, sigma_values=None, n_avalanches=2000, seeds=None):
    """Run finite-size scaling: multiple grid sizes."""
    if N_values is None:
        N_values = [50, 100, 200]
    if seeds is None:
        seeds = SEEDS

    results_by_N = {}
    for N in N_values:
        print(f"\n--- N = {N} ---")
        results_by_N[N] = simulate(N=N, sigma_values=sigma_values,
                                    n_avalanches=n_avalanches, seeds=seeds)
    return results_by_N


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
    ax1.errorbar(sigma, results["order_param"], yerr=results["op_err"],
                 fmt="o-", color="#2c7bb6", markersize=4, capsize=2)
    ax1.axvline(sigma_c, color="gray", linestyle="--", alpha=0.7,
                label=r"$\sigma_c$ = 1.0")
    ax1.set_xlabel(r"Branching ratio $\sigma$")
    ax1.set_ylabel(r"$\langle s \rangle / N^2$")
    ax1.set_title("Order Parameter (Mean Avalanche Size)")
    ax1.legend()

    # Panel 2: Susceptibility
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.errorbar(sigma, results["susceptibility"], yerr=results["sus_err"],
                 fmt="o-", color="#d7191c", markersize=4, capsize=2)
    ax2.axvline(sigma_c, color="gray", linestyle="--", alpha=0.7,
                label=r"$\sigma_c$ = 1.0")
    ax2.set_xlabel(r"Branching ratio $\sigma$")
    ax2.set_ylabel(r"Var$(s)$")
    ax2.set_title("Susceptibility (Avalanche Size Variance)")
    ax2.legend()

    # Panel 3: Mean duration
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.errorbar(sigma, results["mean_duration"], yerr=results["dur_err"],
                 fmt="o-", color="#fdae61", markersize=4, capsize=2)
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


def plot_scaling(results_by_N, output_path):
    """Generate 4-panel scaling analysis figure."""
    # Known mean-field branching exponents
    tau_s = 3.0 / 2.0     # P(s) ~ s^{-3/2}
    tau_d = 2.0            # P(d) ~ d^{-2}
    gamma_sd = 2.0         # <s|d> ~ d^{gamma_sd}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Neural Branching Process — Scaling Analysis",
                 fontsize=16, fontweight="bold")

    # Use the largest N for distribution plots
    largest_N = max(results_by_N.keys())
    sizes = results_by_N[largest_N]["sizes_at_sc"]
    durations = results_by_N[largest_N]["durations_at_sc"]

    # Panel 1: Avalanche size distribution at sigma_c
    ax1 = axes[0, 0]
    if len(sizes) > 0:
        sizes_pos = sizes[sizes > 0]
        s_max = int(sizes_pos.max())
        bins = np.logspace(0, np.log10(max(s_max, 2)), 40)
        hist, bin_edges = np.histogram(sizes_pos, bins=bins, density=True)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        mask = hist > 0
        ax1.loglog(bin_centers[mask], hist[mask], "o", color="#2c7bb6", markersize=4)

        alpha_s = powerlaw_mle(sizes_pos, s_min=5)
        if not np.isnan(alpha_s):
            s_fit = np.logspace(np.log10(5), np.log10(max(s_max, 10)), 50)
            p_fit = s_fit ** (-alpha_s)
            # Normalize to data
            ref_mask = bin_centers[mask] >= 5
            if np.any(ref_mask):
                p_fit *= hist[mask][ref_mask][0] / (5 ** (-alpha_s))
            ax1.loglog(s_fit, p_fit, "--", color="gray", alpha=0.7,
                       label=rf"$\alpha_s$ = {alpha_s:.2f} (exact = {tau_s:.2f})")
    ax1.set_xlabel("Avalanche size $s$")
    ax1.set_ylabel("$P(s)$")
    ax1.set_title(f"Size Distribution at $\\sigma_c$ (N={largest_N})")
    ax1.legend()

    # Panel 2: Avalanche duration distribution at sigma_c
    ax2 = axes[0, 1]
    if len(durations) > 0:
        dur_pos = durations[durations > 0]
        d_max = int(dur_pos.max())
        bins_d = np.logspace(0, np.log10(max(d_max, 2)), 30)
        hist_d, bin_edges_d = np.histogram(dur_pos, bins=bins_d, density=True)
        bin_centers_d = np.sqrt(bin_edges_d[:-1] * bin_edges_d[1:])
        mask_d = hist_d > 0
        ax2.loglog(bin_centers_d[mask_d], hist_d[mask_d], "o", color="#d7191c", markersize=4)

        alpha_d = powerlaw_mle(dur_pos, s_min=5)
        if not np.isnan(alpha_d):
            d_fit = np.logspace(np.log10(5), np.log10(max(d_max, 10)), 50)
            p_fit_d = d_fit ** (-alpha_d)
            ref_mask_d = bin_centers_d[mask_d] >= 5
            if np.any(ref_mask_d):
                p_fit_d *= hist_d[mask_d][ref_mask_d][0] / (5 ** (-alpha_d))
            ax2.loglog(d_fit, p_fit_d, "--", color="gray", alpha=0.7,
                       label=rf"$\alpha_d$ = {alpha_d:.2f} (exact = {tau_d:.2f})")
    ax2.set_xlabel("Avalanche duration $d$")
    ax2.set_ylabel("$P(d)$")
    ax2.set_title(f"Duration Distribution at $\\sigma_c$ (N={largest_N})")
    ax2.legend()

    # Panel 3: Size-duration scaling: <s|d> vs d
    ax3 = axes[1, 0]
    if len(sizes) > 0 and len(durations) > 0:
        # Bin by duration, compute mean size per bin
        dur_int = durations.astype(int)
        unique_d = np.unique(dur_int)
        unique_d = unique_d[unique_d > 0]
        mean_s_given_d = []
        d_vals = []
        for d in unique_d:
            mask_d = dur_int == d
            if np.sum(mask_d) >= 3:
                mean_s_given_d.append(np.mean(sizes[mask_d]))
                d_vals.append(d)
        if len(d_vals) >= 3:
            d_vals = np.array(d_vals, dtype=float)
            mean_s_given_d = np.array(mean_s_given_d)
            ax3.loglog(d_vals, mean_s_given_d, "o", color="#fdae61", markersize=4)

            # Fit slope
            log_d = np.log(d_vals)
            log_s = np.log(mean_s_given_d)
            coeffs = np.polyfit(log_d, log_s, 1)
            d_fit = np.logspace(np.log10(d_vals.min()), np.log10(d_vals.max()), 50)
            ax3.loglog(d_fit, np.exp(coeffs[1]) * d_fit ** coeffs[0], "--",
                       color="gray", alpha=0.7,
                       label=rf"slope = {coeffs[0]:.2f} (exact $\gamma_{{sd}}$ = {gamma_sd})")
    ax3.set_xlabel("Duration $d$")
    ax3.set_ylabel(r"$\langle s | d \rangle$")
    ax3.set_title("Size-Duration Scaling")
    ax3.legend()

    # Panel 4: Mean avalanche size at sigma_c vs N (log-log)
    ax4 = axes[1, 1]
    colors = {50: "#2c7bb6", 100: "#fdae61", 200: "#d7191c"}
    N_arr = []
    mean_s_arr = []
    for N, res in results_by_N.items():
        s_sc = res["sizes_at_sc"]
        if len(s_sc) > 0:
            N_arr.append(N)
            mean_s_arr.append(np.mean(s_sc))
    if len(N_arr) >= 2:
        N_arr = np.array(N_arr, dtype=float)
        mean_s_arr = np.array(mean_s_arr)
        ax4.loglog(N_arr, mean_s_arr, "s-", color="#2c7bb6", markersize=8)

        coeffs = np.polyfit(np.log(N_arr), np.log(mean_s_arr), 1)
        N_fit = np.linspace(N_arr.min(), N_arr.max(), 50)
        ax4.loglog(N_fit, np.exp(coeffs[1]) * N_fit ** coeffs[0], "--",
                   color="gray", alpha=0.7,
                   label=rf"slope = {coeffs[0]:.2f}")
    ax4.set_xlabel("Grid size $N$")
    ax4.set_ylabel(r"$\langle s \rangle$ at $\sigma_c$")
    ax4.set_title(r"Mean Avalanche Size vs $N$ (Finite-Size Scaling)")
    ax4.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Scaling results saved to {output_path}")


# --- Main ---

if __name__ == "__main__":
    print("Neural Branching Process Simulation")
    print("=" * 40)

    output_dir = Path(__file__).parent
    output_path = output_dir / "critical_brain_results.png"
    scaling_path = output_dir / "critical_brain_scaling.png"

    # Main results with default N=100
    results = simulate(N=100, n_avalanches=2000)
    plot_results(results, output_path)

    # Finite-size scaling
    print("\n--- Finite-Size Scaling ---")
    results_by_N = simulate_scaling(N_values=[50, 100, 200], n_avalanches=2000)

    plot_scaling(results_by_N, scaling_path)
