"""
2D Ising Model Phase Transition Simulation
===========================================
Simulates a 2D Ising model on a square lattice with periodic boundary
conditions using both Metropolis-Hastings and Wolff cluster algorithms.
Sweeps temperature through the critical point (T_c ≈ 2.269) to observe
the phase transition between ordered (ferromagnetic) and disordered
(paramagnetic) states.

Features:
- Wolff cluster algorithm for reduced critical slowing-down
- Multi-seed ensemble averaging with bootstrap error bars
- Finite-size scaling with Binder cumulant analysis
- Autocorrelation time comparison (Metropolis vs Wolff)

Usage:
    python simulations/phase_transitions/ising_2d.py

Produces:
    simulations/phase_transitions/ising_results.png
    simulations/phase_transitions/ising_scaling.png
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
        c_t = np.mean(centered[:n - t] * centered[t:]) / var
        if c_t < 0:
            break
        tau_int += c_t
    return max(tau_int, 0.5)


# --- Model (numba-accelerated) ---

def init_lattice(N, rng):
    """Initialize NxN lattice with random spins ±1."""
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(N, N)).astype(np.int8)


@njit
def lattice_energy(spins):
    """Total energy with periodic boundary conditions (J=1)."""
    N = spins.shape[0]
    E = 0
    for i in range(N):
        for j in range(N):
            E -= spins[i, j] * (spins[(i + 1) % N, j] + spins[i, (j + 1) % N])
    return E


@njit
def _metropolis_sweep(spins, beta, rand_ij, rand_accept):
    """Numba-accelerated Metropolis sweep. Pre-generated random numbers."""
    N = spins.shape[0]
    n = N * N
    for k in range(n):
        i = rand_ij[k, 0]
        j = rand_ij[k, 1]
        nn_sum = (
            spins[(i + 1) % N, j] + spins[(i - 1) % N, j] +
            spins[i, (j + 1) % N] + spins[i, (j - 1) % N]
        )
        dE = 2 * spins[i, j] * nn_sum
        if dE <= 0 or rand_accept[k] < np.exp(-beta * dE):
            spins[i, j] *= -1


def metropolis_sweep(spins, beta, rng):
    """One full sweep: N^2 single-spin-flip Metropolis updates."""
    N = spins.shape[0]
    n = N * N
    rand_ij = rng.integers(0, N, size=(n, 2))
    rand_accept = rng.random(n)
    _metropolis_sweep(spins, beta, rand_ij, rand_accept)


@njit
def _wolff_step(spins, p_add, seed_i, seed_j, rand_vals):
    """Numba-accelerated Wolff cluster flip using array-based stack.

    rand_vals is a pre-allocated random array (size N*N) — we consume
    entries sequentially and return (flipped_count, rand_consumed).
    """
    N = spins.shape[0]
    cluster_spin = spins[seed_i, seed_j]

    # Stack and visited array
    stack_i = np.empty(N * N, dtype=np.int32)
    stack_j = np.empty(N * N, dtype=np.int32)
    visited = np.zeros((N, N), dtype=np.bool_)

    stack_i[0] = seed_i
    stack_j[0] = seed_j
    visited[seed_i, seed_j] = True
    stack_top = 1
    flipped = 0
    rand_idx = 0

    # Neighbor offsets
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
                    rand_idx = 0  # wrap — very rare, only for huge clusters

    # Flip all visited spins
    for i in range(N):
        for j in range(N):
            if visited[i, j]:
                spins[i, j] *= -1

    return flipped


def wolff_step(spins, beta, rng):
    """Single Wolff cluster flip. Returns number of spins flipped."""
    N = spins.shape[0]
    p_add = 1 - np.exp(-2 * beta)
    seed_i = int(rng.integers(N))
    seed_j = int(rng.integers(N))
    # Pre-generate enough random numbers (4 per spin worst case)
    rand_vals = rng.random(N * N)
    return _wolff_step(spins, p_add, seed_i, seed_j, rand_vals)


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
    """Run one Ising simulation with a given seed."""
    rng = np.random.default_rng(seed)
    n_spins = N * N

    T_c = 2.269
    snap_targets = [1.0, T_c, 3.5]
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
        sweep_fn = wolff_sweep if use_wolff else lambda s, b, r: metropolis_sweep(s, b, r)

        for _ in range(eq_sweeps):
            sweep_fn(spins, beta, rng)

        m_samples = np.zeros(meas_sweeps)
        e_samples = np.zeros(meas_sweeps)
        for s in range(meas_sweeps):
            sweep_fn(spins, beta, rng)
            m_samples[s] = np.abs(np.sum(spins)) / n_spins
            e_samples[s] = lattice_energy(spins) / n_spins

        mag[idx] = np.mean(m_samples)
        ene[idx] = np.mean(e_samples)
        sus[idx] = beta * n_spins * np.var(m_samples)
        sheat[idx] = (beta ** 2) * n_spins * np.var(e_samples)
        m2_arr[idx] = np.mean(m_samples ** 2)
        m4_arr[idx] = np.mean(m_samples ** 4)

        for t_snap in snap_targets:
            if abs(T - t_snap) < 0.05 and t_snap not in snapshots:
                snapshots[t_snap] = spins.copy()

    return {
        "mag": mag, "ene": ene, "sus": sus, "sheat": sheat,
        "m2": m2_arr, "m4": m4_arr, "snapshots": snapshots,
    }


def simulate(N=50, T_values=None, eq_sweeps=500, meas_sweeps=1000, seeds=None):
    """Run multi-seed ensemble Ising simulation with bootstrap errors."""
    if T_values is None:
        T_values = np.concatenate([
            np.linspace(1.0, 2.0, 8, endpoint=False),
            np.linspace(2.0, 2.6, 12, endpoint=False),
            np.linspace(2.6, 3.5, 8),
        ])
    if seeds is None:
        seeds = SEEDS

    T_c = 2.269
    snap_targets = [1.0, T_c, 3.5]

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
        "N": N,
    }


def simulate_scaling(L_values=None, T_values=None, eq_sweeps=500, meas_sweeps=1000,
                      seeds=None):
    """Run finite-size scaling: multiple L values."""
    if L_values is None:
        L_values = [25, 50, 100]
    if T_values is None:
        T_values = np.linspace(1.8, 2.8, 30)
    if seeds is None:
        seeds = SEEDS

    results_by_L = {}
    for L in L_values:
        print(f"\n--- L = {L} ---")
        results_by_L[L] = simulate(N=L, T_values=T_values, eq_sweeps=eq_sweeps,
                                    meas_sweeps=meas_sweeps, seeds=seeds)
    return results_by_L


def measure_autocorrelation(N=50, T_values=None, n_sweeps=2000, seed=42):
    """Compare autocorrelation times for Metropolis vs Wolff."""
    if T_values is None:
        T_values = np.linspace(1.8, 2.8, 15)

    rng_m = np.random.default_rng(seed)
    rng_w = np.random.default_rng(seed)

    tau_metro = np.zeros(len(T_values))
    tau_wolff = np.zeros(len(T_values))

    for idx, T in enumerate(T_values):
        beta = 1.0 / T
        n_spins = N * N
        eq = 200

        # Metropolis
        spins_m = init_lattice(N, rng_m)
        for _ in range(eq):
            metropolis_sweep(spins_m, beta, rng_m)
        m_series = np.zeros(n_sweeps)
        for s in range(n_sweeps):
            metropolis_sweep(spins_m, beta, rng_m)
            m_series[s] = np.abs(np.sum(spins_m)) / n_spins
        tau_metro[idx] = autocorrelation_time(m_series)

        # Wolff
        spins_w = init_lattice(N, rng_w)
        for _ in range(eq):
            wolff_sweep(spins_w, beta, rng_w)
        m_series_w = np.zeros(n_sweeps)
        for s in range(n_sweeps):
            wolff_sweep(spins_w, beta, rng_w)
            m_series_w[s] = np.abs(np.sum(spins_w)) / n_spins
        tau_wolff[idx] = autocorrelation_time(m_series_w)

        print(f"  T={T:.3f}: tau_metro={tau_metro[idx]:.1f}, tau_wolff={tau_wolff[idx]:.1f}")

    return {"T": T_values, "tau_metro": tau_metro, "tau_wolff": tau_wolff}


# --- Visualization ---

def plot_results(results, output_path):
    """Generate 4-panel figure summarizing the phase transition."""
    T = results["T"]
    T_c = 2.269

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("2D Ising Model — Phase Transition", fontsize=16, fontweight="bold")

    # Panel 1: Magnetization
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.errorbar(T, results["magnetization"], yerr=results["mag_err"],
                 fmt="o-", color="#2c7bb6", markersize=4, capsize=2)
    ax1.axvline(T_c, color="gray", linestyle="--", alpha=0.7, label=f"$T_c$ = {T_c}")
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel(r"$\langle |m| \rangle$")
    ax1.set_title("Magnetization (Order Parameter)")
    ax1.legend()

    # Panel 2: Susceptibility
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.errorbar(T, results["susceptibility"], yerr=results["sus_err"],
                 fmt="o-", color="#d7191c", markersize=4, capsize=2)
    ax2.axvline(T_c, color="gray", linestyle="--", alpha=0.7, label=f"$T_c$ = {T_c}")
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel(r"$\chi$")
    ax2.set_title("Magnetic Susceptibility")
    ax2.legend()

    # Panel 3: Specific Heat
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.errorbar(T, results["specific_heat"], yerr=results["sheat_err"],
                 fmt="o-", color="#fdae61", markersize=4, capsize=2)
    ax3.axvline(T_c, color="gray", linestyle="--", alpha=0.7, label=f"$T_c$ = {T_c}")
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
    labels = ["Ordered\n(T=1.0)", f"Critical\n(T≈{T_c})", "Disordered\n(T=3.5)"]

    for i, (t_snap, label) in enumerate(zip(snap_targets, labels)):
        if t_snap in snapshots:
            inset = fig.add_axes([0.58 + i * 0.14, 0.12, 0.12, 0.25])
            inset.imshow(snapshots[t_snap], cmap="coolwarm", vmin=-1, vmax=1,
                         interpolation="nearest")
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title(label, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nResults saved to {output_path}")


def plot_scaling(results_by_L, autocorr, output_path):
    """Generate 4-panel finite-size scaling figure."""
    T_c = 2.269
    # Known exact exponents for 2D Ising
    beta_nu = 1.0 / 8.0   # beta/nu
    gamma_nu = 7.0 / 4.0  # gamma/nu
    nu = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("2D Ising Model — Finite-Size Scaling", fontsize=16, fontweight="bold")

    colors = {25: "#2c7bb6", 50: "#fdae61", 100: "#d7191c"}

    # Panel 1: Binder cumulant
    ax1 = axes[0, 0]
    for L, res in results_by_L.items():
        T = res["T"]
        m2 = res["m2"]
        m4 = res["m4"]
        U_L = 1 - m4 / (3 * m2 ** 2)
        ax1.plot(T, U_L, "o-", color=colors.get(L, "black"), markersize=3,
                 label=f"L={L}")
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
        y = L ** beta_nu * m
        ax2.plot(x, y, "o", color=colors.get(L, "black"), markersize=3,
                 label=f"L={L}")
    ax2.set_xlabel(r"$(T - T_c) \cdot L^{1/\nu}$")
    ax2.set_ylabel(r"$L^{\beta/\nu} \cdot \langle |m| \rangle$")
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
    # Fit slope
    if len(L_arr) >= 2:
        coeffs = np.polyfit(np.log(L_arr), np.log(chi_max_arr), 1)
        L_fit = np.linspace(L_arr.min(), L_arr.max(), 50)
        ax3.loglog(L_fit, np.exp(coeffs[1]) * L_fit ** coeffs[0], "--",
                   color="gray", alpha=0.7,
                   label=rf"slope = {coeffs[0]:.2f} (exact $\gamma/\nu$ = {gamma_nu})")
    ax3.set_xlabel("L")
    ax3.set_ylabel(r"$\chi_{\max}$")
    ax3.set_title(r"$\chi_{\max}$ vs $L$ (log-log)")
    ax3.legend()

    # Panel 4: Autocorrelation time comparison
    ax4 = axes[1, 1]
    ax4.plot(autocorr["T"], autocorr["tau_metro"], "o-", color="#d7191c",
             markersize=4, label="Metropolis")
    ax4.plot(autocorr["T"], autocorr["tau_wolff"], "s-", color="#2c7bb6",
             markersize=4, label="Wolff")
    ax4.axvline(T_c, color="gray", linestyle="--", alpha=0.5)
    ax4.set_xlabel("Temperature")
    ax4.set_ylabel(r"$\tau_{\mathrm{int}}$ (sweeps)")
    ax4.set_title("Autocorrelation Time")
    ax4.set_yscale("log")
    ax4.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Scaling results saved to {output_path}")


# --- Main ---

if __name__ == "__main__":
    print("2D Ising Model Simulation")
    print("=" * 40)

    output_dir = Path(__file__).parent
    output_path = output_dir / "ising_results.png"
    scaling_path = output_dir / "ising_scaling.png"

    # Main results with default L=50
    results = simulate(N=50, eq_sweeps=500, meas_sweeps=1000)
    plot_results(results, output_path)

    # Finite-size scaling
    print("\n--- Finite-Size Scaling ---")
    results_by_L = simulate_scaling(L_values=[25, 50, 100],
                                     eq_sweeps=500, meas_sweeps=1000)

    # Autocorrelation comparison
    print("\n--- Autocorrelation Comparison ---")
    autocorr = measure_autocorrelation(N=50, n_sweeps=2000)

    plot_scaling(results_by_L, autocorr, scaling_path)
