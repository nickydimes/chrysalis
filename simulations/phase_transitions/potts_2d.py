"""
2D 3-State Potts Model Phase Transition Simulation
===================================================
Simulates a 2D q=3 Potts model on a square lattice with periodic boundary
conditions using the Metropolis algorithm. Sweeps temperature through the
critical point (T_c ≈ 0.995) to observe the phase transition. Generalizes the
Ising model from Z_2 to Z_3 symmetry, belonging to a different universality
class but exhibiting the same qualitative critical behavior.

Usage:
    python simulations/phase_transitions/potts_2d.py

Produces: simulations/phase_transitions/potts_results.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path


# --- Model ---

Q = 3  # number of Potts states

def init_lattice(N):
    """Initialize NxN lattice with random spins in {0, 1, 2}."""
    return np.random.randint(0, Q, size=(N, N))


def lattice_energy(spins):
    """Total energy H = -J * sum delta(s_i, s_j) over nearest-neighbor bonds."""
    return -np.sum(
        (spins == np.roll(spins, 1, axis=0)).astype(int) +
        (spins == np.roll(spins, 1, axis=1)).astype(int)
    )


def order_parameter(spins):
    """Potts order parameter: m = (q * max_fraction - 1) / (q - 1)."""
    N = spins.shape[0]
    counts = np.bincount(spins.ravel(), minlength=Q)
    max_fraction = counts.max() / (N * N)
    return (Q * max_fraction - 1) / (Q - 1)


def metropolis_sweep(spins, beta, rng):
    """One full sweep: N^2 single-spin-flip Metropolis updates."""
    N = spins.shape[0]
    for _ in range(N * N):
        i = rng.integers(N)
        j = rng.integers(N)
        old_spin = spins[i, j]
        # Pick a new state different from the current one
        new_spin = (old_spin + rng.integers(1, Q)) % Q

        # Count matching neighbors for old and new spin
        neighbors = [
            spins[(i + 1) % N, j], spins[(i - 1) % N, j],
            spins[i, (j + 1) % N], spins[i, (j - 1) % N],
        ]
        dE = sum(1 for nb in neighbors if nb == old_spin) - sum(1 for nb in neighbors if nb == new_spin)
        if dE <= 0 or rng.random() < np.exp(-beta * dE):
            spins[i, j] = new_spin


# --- Simulation ---

def simulate(N=50, T_values=None, eq_sweeps=200, meas_sweeps=200, seed=42):
    """
    Run the Potts simulation across a range of temperatures.

    Returns dict with arrays: T, magnetization, energy, susceptibility, specific_heat,
    and snapshot lattices at low/critical/high T.
    """
    T_c = 1.0 / np.log(1 + np.sqrt(Q))  # ≈ 0.9950

    if T_values is None:
        T_values = np.concatenate([
            np.linspace(0.4, 0.8, 8, endpoint=False),
            np.linspace(0.8, 1.2, 12, endpoint=False),  # dense near T_c
            np.linspace(1.2, 2.0, 8),
        ])

    rng = np.random.default_rng(seed)
    n_spins = N * N

    mag = np.zeros(len(T_values))
    ene = np.zeros(len(T_values))
    sus = np.zeros(len(T_values))
    sheat = np.zeros(len(T_values))

    # Temperatures at which to capture snapshots
    snap_targets = [0.6, T_c, 1.5]  # ordered, critical, disordered
    snapshots = {}

    spins = init_lattice(N)

    for idx, T in enumerate(T_values):
        beta = 1.0 / T
        print(f"  T = {T:.3f}  ({idx + 1}/{len(T_values)})")

        # Equilibration
        for _ in range(eq_sweeps):
            metropolis_sweep(spins, beta, rng)

        # Measurement
        m_samples = np.zeros(meas_sweeps)
        e_samples = np.zeros(meas_sweeps)
        for s in range(meas_sweeps):
            metropolis_sweep(spins, beta, rng)
            m_samples[s] = order_parameter(spins)
            e_samples[s] = lattice_energy(spins) / n_spins

        mag[idx] = np.mean(m_samples)
        ene[idx] = np.mean(e_samples)
        sus[idx] = beta * n_spins * np.var(m_samples)
        sheat[idx] = (beta ** 2) * n_spins * np.var(e_samples)

        # Capture snapshot if close to a target temperature
        for t_snap in snap_targets:
            if abs(T - t_snap) < 0.05 and t_snap not in snapshots:
                snapshots[t_snap] = spins.copy()

    return {
        "T": T_values,
        "magnetization": mag,
        "energy": ene,
        "susceptibility": sus,
        "specific_heat": sheat,
        "snapshots": snapshots,
        "snap_targets": snap_targets,
        "T_c": T_c,
    }


# --- Visualization ---

def plot_results(results, output_path):
    """Generate 4-panel figure summarizing the phase transition."""
    T = results["T"]
    T_c = results["T_c"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("2D 3-State Potts Model — Phase Transition", fontsize=16, fontweight="bold")

    # Panel 1: Order Parameter
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(T, results["magnetization"], "o-", color="#2c7bb6", markersize=4)
    ax1.axvline(T_c, color="gray", linestyle="--", alpha=0.7, label=f"$T_c$ = {T_c:.3f}")
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel(r"$\langle m \rangle$")
    ax1.set_title("Order Parameter")
    ax1.legend()

    # Panel 2: Susceptibility
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(T, results["susceptibility"], "o-", color="#d7191c", markersize=4)
    ax2.axvline(T_c, color="gray", linestyle="--", alpha=0.7, label=f"$T_c$ = {T_c:.3f}")
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel(r"$\chi$")
    ax2.set_title("Susceptibility")
    ax2.legend()

    # Panel 3: Specific Heat
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(T, results["specific_heat"], "o-", color="#fdae61", markersize=4)
    ax3.axvline(T_c, color="gray", linestyle="--", alpha=0.7, label=f"$T_c$ = {T_c:.3f}")
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
            inset.imshow(snapshots[t_snap], cmap=potts_cmap, vmin=0, vmax=Q - 1,
                         interpolation="nearest")
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title(label, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nResults saved to {output_path}")


# --- Main ---

if __name__ == "__main__":
    print("2D 3-State Potts Model Simulation")
    print("=" * 40)

    output_dir = Path(__file__).parent
    output_path = output_dir / "potts_results.png"

    results = simulate(N=50, eq_sweeps=200, meas_sweeps=200)
    plot_results(results, output_path)
