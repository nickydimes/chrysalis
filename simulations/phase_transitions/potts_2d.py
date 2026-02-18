"""
2D 3-State Potts Model Phase Transition Simulation
===================================================
Simulates a 2D q=3 Potts model on a square lattice with periodic boundary
conditions using both Metropolis and Wolff cluster algorithms. Sweeps
temperature through the critical point (T_c ≈ 0.995) to observe the phase
transition. Generalizes the Ising model from Z_2 to Z_3 symmetry, belonging
to a different universality class.

Features:
- Wolff cluster algorithm with Fortuin-Kasteleyn bond probability
- Multi-seed ensemble averaging with bootstrap error bars
- Finite-size scaling with Binder cumulant analysis
- Autocorrelation time comparison (Metropolis vs Wolff)

Usage:
    python simulations/phase_transitions/potts_2d.py

Produces:
    simulations/phase_transitions/potts_results.png
    simulations/phase_transitions/potts_scaling.png
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


def autocorrelation_time(series):
    """Integrated autocorrelation time via initial positive sequence estimator."""
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    if var == 0:
        return 1.0
    centered = series - mean
    tau_int = 0.5
    for t in range(1, n // 2):
        c_t = np.mean(centered[: n - t] * centered[t:]) / var
        if c_t < 0:
            break
        tau_int += c_t
    return max(tau_int, 0.5)


# --- Model (numba-accelerated) ---

Q = 3  # number of Potts states


def init_lattice(N, rng):
    """Initialize NxN lattice with random spins in {0, 1, ..., Q-1}."""
    return rng.integers(0, Q, size=(N, N), dtype=np.int8)


@njit
def lattice_energy(spins):
    """Total energy H = -J * sum delta(s_i, s_j) over nearest-neighbor bonds."""
    N = spins.shape[0]
    E = 0
    for i in range(N):
        for j in range(N):
            if spins[i, j] == spins[(i + 1) % N, j]:
                E -= 1
            if spins[i, j] == spins[i, (j + 1) % N]:
                E -= 1
    return E


def order_parameter(spins):
    """Potts order parameter: m = (q * max_fraction - 1) / (q - 1)."""
    N = spins.shape[0]
    counts = np.bincount(spins.ravel(), minlength=Q)
    max_fraction = counts.max() / (N * N)
    return (Q * max_fraction - 1) / (Q - 1)


@njit
def _metropolis_sweep(spins, beta, q, rand_ij, rand_spin, rand_accept):
    """Numba-accelerated Metropolis sweep for Potts model."""
    N = spins.shape[0]
    n = N * N
    for k in range(n):
        i = rand_ij[k, 0]
        j = rand_ij[k, 1]
        old_spin = spins[i, j]
        new_spin = (old_spin + rand_spin[k]) % q

        # Count matching neighbors
        old_matches = 0
        new_matches = 0
        for d in range(4):
            if d == 0:
                ni, nj = (i + 1) % N, j
            elif d == 1:
                ni, nj = (i - 1) % N, j
            elif d == 2:
                ni, nj = i, (j + 1) % N
            else:
                ni, nj = i, (j - 1) % N
            nb = spins[ni, nj]
            if nb == old_spin:
                old_matches += 1
            if nb == new_spin:
                new_matches += 1

        dE = old_matches - new_matches
        if dE <= 0 or rand_accept[k] < np.exp(-beta * dE):
            spins[i, j] = new_spin


def metropolis_sweep(spins, beta, rng):
    """One full sweep: N^2 single-spin-flip Metropolis updates."""
    N = spins.shape[0]
    n = N * N
    rand_ij = rng.integers(0, N, size=(n, 2))
    rand_spin = rng.integers(1, Q, size=n).astype(np.int8)
    rand_accept = rng.random(n)
    _metropolis_sweep(spins, beta, Q, rand_ij, rand_spin, rand_accept)


@njit
def _wolff_step(spins, p_add, seed_i, seed_j, new_spin, rand_vals):
    """Numba-accelerated Wolff cluster flip for Potts model."""
    N = spins.shape[0]
    cluster_spin = spins[seed_i, seed_j]

    stack_i = np.empty(N * N, dtype=np.int32)
    stack_j = np.empty(N * N, dtype=np.int32)
    visited = np.zeros((N, N), dtype=np.bool_)

    stack_i[0] = seed_i
    stack_j[0] = seed_j
    visited[seed_i, seed_j] = True
    stack_top = 1
    flipped = 0
    rand_idx = 0

    di = np.array([1, -1, 0, 0], dtype=np.int32)
    dj = np.array([0, 0, 1, -1], dtype=np.int32)

    while stack_top > 0:
        stack_top -= 1
        ci = stack_i[stack_top]
        cj = stack_j[stack_top]
        flipped += 1

        for d in range(4):
            ni = (ci + di[d]) % N
            nj = (cj + dj[d]) % N
            if not visited[ni, nj] and spins[ni, nj] == cluster_spin:
                if rand_vals[rand_idx] < p_add:
                    visited[ni, nj] = True
                    stack_i[stack_top] = ni
                    stack_j[stack_top] = nj
                    stack_top += 1
                rand_idx += 1
                if rand_idx >= len(rand_vals):
                    rand_idx = 0

    for i in range(N):
        for j in range(N):
            if visited[i, j]:
                spins[i, j] = new_spin

    return flipped


def wolff_step(spins, beta, rng):
    """Single Wolff cluster flip for Potts model. Returns number of spins flipped."""
    N = spins.shape[0]
    p_add = 1 - np.exp(-beta)
    seed_i = int(rng.integers(N))
    seed_j = int(rng.integers(N))
    cluster_spin = spins[seed_i, seed_j]
    new_spin = np.int8((cluster_spin + rng.integers(1, Q)) % Q)
    rand_vals = rng.random(N * N)
    return _wolff_step(spins, p_add, seed_i, seed_j, new_spin, rand_vals)


def wolff_sweep(spins, beta, rng):
    """Enough Wolff steps to flip ~N^2 spins total (one sweep equivalent)."""
    N = spins.shape[0]
    n_target = N * N
    flipped = 0
    while flipped < n_target:
        flipped += wolff_step(spins, beta, rng)


# --- Simulation ---

SEEDS = [42, 137, 256, 314, 999]


def _simulate_single(N, T_values, eq_sweeps, meas_sweeps, seed, use_wolff=True):
    """Run one Potts simulation with a given seed."""
    rng = np.random.default_rng(seed)
    n_spins = N * N

    T_c = 1.0 / np.log(1 + np.sqrt(Q))
    snap_targets = [0.6, T_c, 1.5]
    snapshots = {}

    mag = np.zeros(len(T_values))
    ene = np.zeros(len(T_values))
    sus = np.zeros(len(T_values))
    sheat = np.zeros(len(T_values))
    m2_arr = np.zeros(len(T_values))
    m4_arr = np.zeros(len(T_values))

    spins = init_lattice(N, rng)

    for idx, T in enumerate(T_values):
        beta = 1.0 / T
        sweep_fn = (
            wolff_sweep if use_wolff else lambda s, b, r: metropolis_sweep(s, b, r)
        )

        for _ in range(eq_sweeps):
            sweep_fn(spins, beta, rng)

        m_samples = np.zeros(meas_sweeps)
        e_samples = np.zeros(meas_sweeps)
        for s in range(meas_sweeps):
            sweep_fn(spins, beta, rng)
            m_samples[s] = order_parameter(spins)
            e_samples[s] = lattice_energy(spins) / n_spins

        mag[idx] = np.mean(m_samples)
        ene[idx] = np.mean(e_samples)
        sus[idx] = beta * n_spins * np.var(m_samples)
        sheat[idx] = (beta**2) * n_spins * np.var(e_samples)
        m2_arr[idx] = np.mean(m_samples**2)
        m4_arr[idx] = np.mean(m_samples**4)

        for t_snap in snap_targets:
            if abs(T - t_snap) < 0.05 and t_snap not in snapshots:
                snapshots[t_snap] = spins.copy()

    return {
        "mag": mag,
        "ene": ene,
        "sus": sus,
        "sheat": sheat,
        "m2": m2_arr,
        "m4": m4_arr,
        "snapshots": snapshots,
    }


def simulate(N=50, T_values=None, eq_sweeps=500, meas_sweeps=1000, seeds=None):
    """Run multi-seed ensemble Potts simulation with bootstrap errors."""
    T_c = 1.0 / np.log(1 + np.sqrt(Q))

    if T_values is None:
        T_values = np.concatenate(
            [
                np.linspace(0.4, 0.8, 8, endpoint=False),
                np.linspace(0.8, 1.2, 12, endpoint=False),
                np.linspace(1.2, 2.0, 8),
            ]
        )
    if seeds is None:
        seeds = SEEDS

    snap_targets = [0.6, T_c, 1.5]

    all_mag = []
    all_ene = []
    all_sus = []
    all_sheat = []
    all_m2 = []
    all_m4 = []
    snapshots = {}

    for i, seed in enumerate(seeds):
        print(f"  Seed {seed} ({i + 1}/{len(seeds)}), L={N}")
        result = _simulate_single(N, T_values, eq_sweeps, meas_sweeps, seed)
        all_mag.append(result["mag"])
        all_ene.append(result["ene"])
        all_sus.append(result["sus"])
        all_sheat.append(result["sheat"])
        all_m2.append(result["m2"])
        all_m4.append(result["m4"])
        if not snapshots:
            snapshots = result["snapshots"]

    all_mag = np.array(all_mag)
    all_ene = np.array(all_ene)
    all_sus = np.array(all_sus)
    all_sheat = np.array(all_sheat)
    all_m2 = np.array(all_m2)
    all_m4 = np.array(all_m4)

    n_T = len(T_values)
    mag_mean = np.zeros(n_T)
    mag_err = np.zeros(n_T)
    sus_mean = np.zeros(n_T)
    sus_err = np.zeros(n_T)
    sheat_mean = np.zeros(n_T)
    sheat_err = np.zeros(n_T)

    for t in range(n_T):
        mag_mean[t], mag_err[t] = bootstrap_error(all_mag[:, t])
        sus_mean[t], sus_err[t] = bootstrap_error(all_sus[:, t])
        sheat_mean[t], sheat_err[t] = bootstrap_error(all_sheat[:, t])

    m2_mean = np.mean(all_m2, axis=0)
    m4_mean = np.mean(all_m4, axis=0)

    return {
        "T": T_values,
        "magnetization": mag_mean,
        "mag_err": mag_err,
        "energy": np.mean(all_ene, axis=0),
        "susceptibility": sus_mean,
        "sus_err": sus_err,
        "specific_heat": sheat_mean,
        "sheat_err": sheat_err,
        "m2": m2_mean,
        "m4": m4_mean,
        "snapshots": snapshots,
        "snap_targets": snap_targets,
        "T_c": T_c,
        "N": N,
    }


def simulate_scaling(
    L_values=None, T_values=None, eq_sweeps=500, meas_sweeps=1000, seeds=None
):
    """Run finite-size scaling: multiple L values."""
    if L_values is None:
        L_values = [25, 50, 100]
    if T_values is None:
        T_values = np.linspace(0.7, 1.3, 30)
    if seeds is None:
        seeds = SEEDS

    results_by_L = {}
    for L in L_values:
        print(f"\n--- L = {L} ---")
        results_by_L[L] = simulate(
            N=L,
            T_values=T_values,
            eq_sweeps=eq_sweeps,
            meas_sweeps=meas_sweeps,
            seeds=seeds,
        )
    return results_by_L


def measure_autocorrelation(N=50, T_values=None, n_sweeps=2000, seed=42):
    """Compare autocorrelation times for Metropolis vs Wolff."""
    if T_values is None:
        T_values = np.linspace(0.7, 1.3, 15)

    rng_m = np.random.default_rng(seed)
    rng_w = np.random.default_rng(seed)

    tau_metro = np.zeros(len(T_values))
    tau_wolff = np.zeros(len(T_values))

    for idx, T in enumerate(T_values):
        beta = 1.0 / T
        eq = 200

        # Metropolis
        spins_m = init_lattice(N, rng_m)
        for _ in range(eq):
            metropolis_sweep(spins_m, beta, rng_m)
        m_series = np.zeros(n_sweeps)
        for s in range(n_sweeps):
            metropolis_sweep(spins_m, beta, rng_m)
            m_series[s] = order_parameter(spins_m)
        tau_metro[idx] = autocorrelation_time(m_series)

        # Wolff
        spins_w = init_lattice(N, rng_w)
        for _ in range(eq):
            wolff_sweep(spins_w, beta, rng_w)
        m_series_w = np.zeros(n_sweeps)
        for s in range(n_sweeps):
            wolff_sweep(spins_w, beta, rng_w)
            m_series_w[s] = order_parameter(spins_w)
        tau_wolff[idx] = autocorrelation_time(m_series_w)

        print(
            f"  T={T:.3f}: tau_metro={tau_metro[idx]:.1f}, tau_wolff={tau_wolff[idx]:.1f}"
        )

    return {"T": T_values, "tau_metro": tau_metro, "tau_wolff": tau_wolff}


# --- Visualization ---


def plot_results(results, output_path):
    """Generate 4-panel figure summarizing the phase transition."""
    T = results["T"]
    T_c = results["T_c"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "2D 3-State Potts Model — Phase Transition", fontsize=16, fontweight="bold"
    )

    # Panel 1: Order Parameter
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.errorbar(
        T,
        results["magnetization"],
        yerr=results["mag_err"],
        fmt="o-",
        color="#2c7bb6",
        markersize=4,
        capsize=2,
    )
    ax1.axvline(
        T_c, color="gray", linestyle="--", alpha=0.7, label=f"$T_c$ = {T_c:.3f}"
    )
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel(r"$\langle m \rangle$")
    ax1.set_title("Order Parameter")
    ax1.legend()

    # Panel 2: Susceptibility
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.errorbar(
        T,
        results["susceptibility"],
        yerr=results["sus_err"],
        fmt="o-",
        color="#d7191c",
        markersize=4,
        capsize=2,
    )
    ax2.axvline(
        T_c, color="gray", linestyle="--", alpha=0.7, label=f"$T_c$ = {T_c:.3f}"
    )
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel(r"$\chi$")
    ax2.set_title("Susceptibility")
    ax2.legend()

    # Panel 3: Specific Heat
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.errorbar(
        T,
        results["specific_heat"],
        yerr=results["sheat_err"],
        fmt="o-",
        color="#fdae61",
        markersize=4,
        capsize=2,
    )
    ax3.axvline(
        T_c, color="gray", linestyle="--", alpha=0.7, label=f"$T_c$ = {T_c:.3f}"
    )
    ax3.set_xlabel("Temperature")
    ax3.set_ylabel(r"$C_v$")
    ax3.set_title("Specific Heat")
    ax3.legend()

    # Panel 4: Lattice snapshots
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_axis_off()
    ax4.set_title("Lattice Snapshots")

    snapshots = results["snapshots"]
    snap_targets = results["snap_targets"]
    labels = ["Ordered\n(T=0.6)", f"Critical\n(T≈{T_c:.3f})", "Disordered\n(T=1.5)"]
    potts_cmap = ListedColormap(["#2c7bb6", "#d7191c", "#fdae61"])

    for i, (t_snap, label) in enumerate(zip(snap_targets, labels)):
        if t_snap in snapshots:
            inset = fig.add_axes([0.58 + i * 0.14, 0.12, 0.12, 0.25])
            inset.imshow(
                snapshots[t_snap],
                cmap=potts_cmap,
                vmin=0,
                vmax=Q - 1,
                interpolation="nearest",
            )
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title(label, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nResults saved to {output_path}")


def plot_scaling(results_by_L, autocorr, output_path):
    """Generate 4-panel finite-size scaling figure."""
    T_c = 1.0 / np.log(1 + np.sqrt(Q))
    # Known exact exponents for 2D q=3 Potts
    beta_nu = 2.0 / 15.0  # beta/nu
    gamma_nu = 26.0 / 15.0  # gamma/nu
    nu = 5.0 / 6.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "2D 3-State Potts Model — Finite-Size Scaling", fontsize=16, fontweight="bold"
    )

    colors = {25: "#2c7bb6", 50: "#fdae61", 100: "#d7191c"}

    # Panel 1: Binder cumulant
    ax1 = axes[0, 0]
    for L, res in results_by_L.items():
        T = res["T"]
        m2 = res["m2"]
        m4 = res["m4"]
        U_L = 1 - m4 / (3 * m2**2)
        ax1.plot(
            T, U_L, "o-", color=colors.get(L, "black"), markersize=3, label=f"L={L}"
        )
    ax1.axvline(T_c, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel(r"$U_L = 1 - \langle m^4 \rangle / 3\langle m^2 \rangle^2$")
    ax1.set_title("Binder Cumulant")
    ax1.legend()

    # Panel 2: Order parameter scaling collapse
    ax2 = axes[0, 1]
    for L, res in results_by_L.items():
        T = res["T"]
        m = res["magnetization"]
        x = (T - T_c) * L ** (1.0 / nu)
        y = L**beta_nu * m
        ax2.plot(x, y, "o", color=colors.get(L, "black"), markersize=3, label=f"L={L}")
    ax2.set_xlabel(r"$(T - T_c) \cdot L^{1/\nu}$")
    ax2.set_ylabel(r"$L^{\beta/\nu} \cdot \langle m \rangle$")
    ax2.set_title("Order Parameter Collapse")
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
        ax3.loglog(
            L_fit,
            np.exp(coeffs[1]) * L_fit ** coeffs[0],
            "--",
            color="gray",
            alpha=0.7,
            label=rf"slope = {coeffs[0]:.2f} (exact $\gamma/\nu$ = {gamma_nu:.2f})",
        )
    ax3.set_xlabel("L")
    ax3.set_ylabel(r"$\chi_{\max}$")
    ax3.set_title(r"$\chi_{\max}$ vs $L$ (log-log)")
    ax3.legend()

    # Panel 4: Autocorrelation time comparison
    ax4 = axes[1, 1]
    ax4.plot(
        autocorr["T"],
        autocorr["tau_metro"],
        "o-",
        color="#d7191c",
        markersize=4,
        label="Metropolis",
    )
    ax4.plot(
        autocorr["T"],
        autocorr["tau_wolff"],
        "s-",
        color="#2c7bb6",
        markersize=4,
        label="Wolff",
    )
    ax4.axvline(T_c, color="gray", linestyle="--", alpha=0.5)
    ax4.set_xlabel("Temperature")
    ax4.set_ylabel(r"$\tau_{\mathrm{int}}$ (sweeps)")
    ax4.set_title("Autocorrelation Time")
    ax4.set_yscale("log")
    ax4.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Scaling results saved to {output_path}")


def calculate_protocol_metrics(results):
    """
    Maps multi-state Potts simulation results to the Eight-Step Navigation Protocol.
    """
    mag = np.atleast_1d(results["magnetization"])
    sus = np.atleast_1d(results["susceptibility"])
    sheat = np.atleast_1d(results["specific_heat"])

    # 1. Purification: Order parameter
    purification = mag

    # 2. Containment: Coherence proxy
    if np.max(sheat) > 0:
        containment = 1.0 - (sheat / np.max(sheat))
    else:
        containment = np.zeros_like(sheat)

    # 3. Anchoring: State stability
    anchoring = mag**2

    # 4. Dissolution: Energy fluctuations
    if np.max(sheat) > 0:
        dissolution = sheat / np.max(sheat)
    else:
        dissolution = np.zeros_like(sheat)

    # 5. Liminality: Susceptibility peak
    if np.max(sus) > 0:
        liminality = sus / np.max(sus)
    else:
        liminality = np.zeros_like(sus)

    # 6. Encounter: Correlation length proxy
    encounter = np.sqrt(sus)

    # 7. Integration: Coherent ordering
    integration = mag**2

    # 8. Emergence: Final stable order parameter
    emergence = np.abs(mag)

    metrics = {
        "Purification": purification,
        "Containment": containment,
        "Anchoring": anchoring,
        "Dissolution": dissolution,
        "Liminality": liminality,
        "Encounter": encounter,
        "Integration": integration,
        "Emergence": emergence,
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
    Runs the 2D 3-State Potts Model simulation.
    Args:
        args: A list of arguments (e.g., from sys.argv[1:]). If None, uses default values or argparse.
    """
    parser = argparse.ArgumentParser(
        description="Run 2D 3-State Potts Model simulation."
    )
    parser.add_argument("--N", type=int, default=50, help="Lattice size N")
    parser.add_argument(
        "--T", type=float, default=None, help="Single temperature to simulate"
    )
    parser.add_argument(
        "--eq_sweeps", type=int, default=500, help="Equilibration sweeps"
    )
    parser.add_argument(
        "--meas_sweeps", type=int, default=1000, help="Measurement sweeps"
    )
    parser.add_argument(
        "--L_values",
        type=int,
        nargs="*",
        default=[25, 50, 100],
        help="Lattice sizes for finite-size scaling",
    )
    parser.add_argument(
        "--n_autocorr_sweeps",
        type=int,
        default=2000,
        help="Number of sweeps for autocorrelation measurement",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).parent),
        help="Directory to save simulation results.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Base seed for random number generation."
    )
    parser.add_argument(
        "--no_scaling",
        action="store_true",
        help="Disable finite-size scaling and autocorrelation.",
    )

    # Parse arguments provided, or from sys.argv if none provided
    parsed_args = parser.parse_args(args=args)

    print("2D 3-State Potts Model Simulation")
    print("=" * 40)

    output_dir = Path(parsed_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
    output_path = output_dir / "potts_results.png"
    scaling_path = output_dir / "potts_scaling.png"

    # Main results
    T_vals = [parsed_args.T] if parsed_args.T is not None else None
    results = simulate(
        N=parsed_args.N,
        T_values=T_vals,
        eq_sweeps=parsed_args.eq_sweeps,
        meas_sweeps=parsed_args.meas_sweeps,
        seeds=[parsed_args.seed],
    )
    plot_results(results, output_path)

    if not parsed_args.no_scaling:
        # Finite-size scaling
        print("\n--- Finite-Size Scaling ---")
        results_by_L = simulate_scaling(
            L_values=parsed_args.L_values,
            eq_sweeps=parsed_args.eq_sweeps,
            meas_sweeps=parsed_args.meas_sweeps,
            seeds=[parsed_args.seed],
        )

        # Autocorrelation comparison
        print("\n--- Autocorrelation Comparison ---")
        autocorr = measure_autocorrelation(
            N=parsed_args.N,
            n_sweeps=parsed_args.n_autocorr_sweeps,
            seed=parsed_args.seed,
        )

        plot_scaling(results_by_L, autocorr, scaling_path)
    else:
        results_by_L = {}
        autocorr = {}

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
            "N": parsed_args.N,
            "T": parsed_args.T,
            "eq_sweeps": parsed_args.eq_sweeps,
            "meas_sweeps": parsed_args.meas_sweeps,
            "L_values": parsed_args.L_values,
            "n_autocorr_sweeps": parsed_args.n_autocorr_sweeps,
            "seed": parsed_args.seed,
        },
        "main_results": convert_numpy_to_list(results),
        "scaling_results_by_L": {
            str(k): convert_numpy_to_list(v) for k, v in results_by_L.items()
        },
        "autocorrelation_results": convert_numpy_to_list(autocorr),
        "protocol_metrics": calculate_protocol_metrics(results),
    }

    results_json_path = output_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Numerical results saved to {results_json_path}")

    # Save CSV
    import pandas as pd

    df = pd.DataFrame(
        {
            "T": results["T"],
            "magnetization": results["magnetization"],
            "mag_err": results["mag_err"],
            "susceptibility": results["susceptibility"],
            "sus_err": results["sus_err"],
            "specific_heat": results["specific_heat"],
            "sheat_err": results["sheat_err"],
            "energy": results["energy"],
        }
    )
    results_csv_path = output_dir / "results.csv"
    df.to_csv(results_csv_path, index=False)
    print(f"CSV results saved to {results_csv_path}")


if __name__ == "__main__":
    run()
