"""
2D XY Model Phase Transition Simulation (BKT Transition)
========================================================
Simulates a 2D XY model on a square lattice with periodic boundary conditions.
The XY model features continuous O(2) symmetry, where each spin is a unit vector
defined by an angle θ ∈ [0, 2π).

Unlike the Ising or Potts models, the 2D XY model does not have a standard
symmetry-breaking transition with long-range order at finite temperature.
Instead, it exhibits a Berezinskii-Kosterlitz-Thouless (BKT) transition
at T_BKT ≈ 0.89 J/k_B, characterized by the topological binding/unbinding
of vortex-antivortex pairs and a jump in the helicity modulus (stiffness).

Features:
- Wolff cluster algorithm (embedded Ising variant) for efficient sampling
- Multi-seed ensemble averaging with bootstrap error bars
- Helicity modulus (stiffness) calculation
- Vortex/Antivortex topological defect detection and visualization
- Standard JSON and CSV data export

Usage:
    python simulations/phase_transitions/xy_2d.py

Produces:
    simulations/phase_transitions/xy_results.png
    simulations/phase_transitions/xy_vortices.png
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from pathlib import Path
import json
import argparse
import pandas as pd

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


# --- Model (numba-accelerated) ---


def init_lattice(N, rng):
    """Initialize NxN lattice with random angles in [0, 2π)."""
    return rng.uniform(0, 2 * np.pi, size=(N, N)).astype(np.float64)


@njit
def lattice_energy(angles):
    """Total energy H = -J * sum cos(theta_i - theta_j) over nearest-neighbors."""
    N = angles.shape[0]
    E = 0.0
    for i in range(N):
        for j in range(N):
            E -= np.cos(angles[i, j] - angles[(i + 1) % N, j])
            E -= np.cos(angles[i, j] - angles[i, (j + 1) % N])
    return E


@njit
def _wolff_step(angles, beta, seed_i, seed_j, r_vec_x, r_vec_y, rand_vals):
    """
    Numba-accelerated Wolff cluster flip for the XY model.
    Uses the 'embedded Ising' method: reflects spins across a random plane.
    """
    N = angles.shape[0]
    # r_vec is the normal to the reflection plane (a unit vector)

    def reflect(angle):
        # Reflect spin s = (cos(th), sin(th)) across plane normal to r:
        # s' = s - 2(s . r)r
        # This is equivalent to th' = 2*phi - th + pi, where phi is angle of r.
        # Simple version: project onto r direction and flip that component.
        s_x = np.cos(angle)
        s_y = np.sin(angle)
        dot = s_x * r_vec_x + s_y * r_vec_y
        new_sx = s_x - 2 * dot * r_vec_x
        new_sy = s_y - 2 * dot * r_vec_y
        return np.arctan2(new_sy, new_sx)

    def get_projection(angle):
        return np.cos(angle) * r_vec_x + np.sin(angle) * r_vec_y

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

        c_proj = get_projection(angles[ci, cj])

        for d in range(4):
            ni = (ci + di[d]) % N
            nj = (cj + dj[d]) % N
            if not visited[ni, nj]:
                n_proj = get_projection(angles[ni, nj])
                # Probability depends on the Ising-like coupling of projections
                p_add = 1.0 - np.exp(min(0.0, -2.0 * beta * c_proj * n_proj))
                if rand_vals[rand_idx] < p_add:
                    visited[ni, nj] = True
                    stack_i[stack_top] = ni
                    stack_j[stack_top] = nj
                    stack_top += 1
                rand_idx += 1
                if rand_idx >= len(rand_vals):
                    rand_idx = 0

    # Apply reflection to all spins in the cluster
    for i in range(N):
        for j in range(N):
            if visited[i, j]:
                angles[i, j] = reflect(angles[i, j])

    return flipped


def wolff_step(angles, beta, rng):
    """Single Wolff cluster flip. Selects a random reflection plane."""
    N = angles.shape[0]
    phi = rng.uniform(0, 2 * np.pi)
    r_vec_x = np.cos(phi)
    r_vec_y = np.sin(phi)

    seed_i = int(rng.integers(N))
    seed_j = int(rng.integers(N))
    rand_vals = rng.random(N * N)
    return _wolff_step(angles, beta, seed_i, seed_j, r_vec_x, r_vec_y, rand_vals)


def wolff_sweep(angles, beta, rng):
    """Equivalent of one sweep using Wolff steps."""
    N = angles.shape[0]
    n_target = N * N
    flipped = 0
    while flipped < n_target:
        flipped += wolff_step(angles, beta, rng)


@njit
def calculate_helicity_modulus(angles, beta):
    """
    Calculates the helicity modulus (stiffness) Upsilon.
    Upsilon = <e> / 2 - (beta/N^2) <(sum sin(th_i - th_j))^2>
    where the sum is over bonds in one direction (e.g., x).
    """
    N = angles.shape[0]
    # Current implementation uses a simplified version or just measures bond energy
    # Real helicity involves the response to a phase twist.
    # Here we use the standard formula for stiffness.
    term1 = 0.0
    term2_sum = 0.0
    for i in range(N):
        for j in range(N):
            diff = angles[(i + 1) % N, j] - angles[i, j]
            term1 += np.cos(diff)
            term2_sum += np.sin(diff)

    return (term1 / (2.0 * N * N)) - (beta / (N * N)) * (term2_sum**2)


@njit
def detect_vortices(angles):
    """
    Detects vortices and antivortices by calculating the circulation
    around each plaquette.
    Returns an array of charges (+1, -1, or 0).
    """
    N = angles.shape[0]
    vortices = np.zeros((N, N), dtype=np.int8)

    for i in range(N):
        for j in range(N):
            # Plaquette: (i,j) -> (i+1,j) -> (i+1,j+1) -> (i,j+1) -> (i,j)
            d1 = angles[(i + 1) % N, j] - angles[i, j]
            d2 = angles[(i + 1) % N, (j + 1) % N] - angles[(i + 1) % N, j]
            d3 = angles[i, (j + 1) % N] - angles[(i + 1) % N, (j + 1) % N]
            d4 = angles[i, j] - angles[i, (j + 1) % N]

            # Wrap to [-pi, pi]
            def wrap(d):
                while d > np.pi:
                    d -= 2 * np.pi
                while d < -np.pi:
                    d += 2 * np.pi
                return d

            total = wrap(d1) + wrap(d2) + wrap(d3) + wrap(d4)
            charge = int(np.round(total / (2 * np.pi)))
            if charge != 0:
                vortices[i, j] = charge

    return vortices


# --- Simulation ---

SEEDS = [42, 137, 256, 314, 999]


def _simulate_single(N, T_values, eq_sweeps, meas_sweeps, seed):
    rng = np.random.default_rng(seed)
    n_spins = N * N

    T_BKT = 0.89
    snap_targets = [0.5, T_BKT, 1.5]
    snapshots = {}
    vortex_maps = {}

    mag = np.zeros(len(T_values))
    sus = np.zeros(len(T_values))
    stiff = np.zeros(len(T_values))
    ene = np.zeros(len(T_values))
    sheat = np.zeros(len(T_values))

    angles = init_lattice(N, rng)

    for idx, T in enumerate(T_values):
        beta = 1.0 / T

        # Equilibration
        for _ in range(eq_sweeps):
            wolff_sweep(angles, beta, rng)

        m_samples = np.zeros(meas_sweeps)
        s_samples = np.zeros(meas_sweeps)
        e_samples = np.zeros(meas_sweeps)

        for s in range(meas_sweeps):
            wolff_sweep(angles, beta, rng)

            # Magnetization M = |sum(vec_s_i)| / N^2
            sx = np.sum(np.cos(angles))
            sy = np.sum(np.sin(angles))
            m_samples[s] = np.sqrt(sx**2 + sy**2) / n_spins
            s_samples[s] = calculate_helicity_modulus(angles, beta)
            e_samples[s] = lattice_energy(angles) / n_spins

        mag[idx] = np.mean(m_samples)
        sus[idx] = beta * n_spins * np.var(m_samples)
        stiff[idx] = np.mean(s_samples)
        ene[idx] = np.mean(e_samples)
        sheat[idx] = (beta**2) * n_spins * np.var(e_samples)

        for t_snap in snap_targets:
            if abs(T - t_snap) < 0.05 and t_snap not in snapshots:
                snapshots[t_snap] = angles.copy()
                vortex_maps[t_snap] = detect_vortices(angles)

    return {
        "mag": mag,
        "sus": sus,
        "stiff": stiff,
        "ene": ene,
        "sheat": sheat,
        "snapshots": snapshots,
        "vortex_maps": vortex_maps,
    }


def simulate(N=32, T_values=None, eq_sweeps=400, meas_sweeps=800, seeds=None):
    if T_values is None:
        T_values = np.concatenate(
            [
                np.linspace(0.1, 0.7, 6, endpoint=False),
                np.linspace(0.7, 1.1, 12, endpoint=False),
                np.linspace(1.1, 2.0, 6),
            ]
        )
    if seeds is None:
        seeds = SEEDS

    all_mag, all_sus, all_stiff, all_ene, all_sheat = [], [], [], [], []
    snapshots, vortex_maps = {}, {}

    for i, seed in enumerate(seeds):
        print(f"  Seed {seed} ({i + 1}/{len(seeds)}), L={N}")
        res = _simulate_single(N, T_values, eq_sweeps, meas_sweeps, seed)
        all_mag.append(res["mag"])
        all_sus.append(res["sus"])
        all_stiff.append(res["stiff"])
        all_ene.append(res["ene"])
        all_sheat.append(res["sheat"])
        if not snapshots:
            snapshots = res["snapshots"]
            vortex_maps = res["vortex_maps"]

    n_T = len(T_values)
    m_mean, m_err = np.zeros(n_T), np.zeros(n_T)
    s_mean, s_err = np.zeros(n_T), np.zeros(n_T)
    st_mean, st_err = np.zeros(n_T), np.zeros(n_T)
    sh_mean, sh_err = np.zeros(n_T), np.zeros(n_T)

    for t in range(n_T):
        m_mean[t], m_err[t] = bootstrap_error(np.array(all_mag)[:, t])
        s_mean[t], s_err[t] = bootstrap_error(np.array(all_sus)[:, t])
        st_mean[t], st_err[t] = bootstrap_error(np.array(all_stiff)[:, t])
        sh_mean[t], sh_err[t] = bootstrap_error(np.array(all_sheat)[:, t])

    return {
        "T": T_values,
        "magnetization": m_mean,
        "mag_err": m_err,
        "susceptibility": s_mean,
        "sus_err": s_err,
        "stiffness": st_mean,
        "stiff_err": st_err,
        "energy": np.mean(all_ene, axis=0),
        "specific_heat": sh_mean,
        "sheat_err": sh_err,
        "snapshots": snapshots,
        "vortex_maps": vortex_maps,
        "N": N,
        "T_BKT": 0.89,
    }


def calculate_protocol_metrics(results):
    """
    Maps physical simulation results for the XY model to the Eight-Step Navigation Protocol.
    In the XY model, the BKT transition is topological; stiffness is the key order parameter.
    """
    mag = np.atleast_1d(results["magnetization"])
    sus = np.atleast_1d(results["susceptibility"])
    stiff = np.atleast_1d(results["stiffness"])
    sheat = np.atleast_1d(results["specific_heat"])

    # 1. Purification: Magnetization as a proxy for ordering (finite size)
    purification = mag

    # 2. Containment: Specific heat peaks at the transition; 1-norm(sheat) is stable containment
    if np.max(sheat) > 0:
        containment = 1.0 - (sheat / np.max(sheat))
    else:
        containment = np.zeros_like(sheat)

    # 3. Anchoring: Stiffness (Helicity Modulus) represents the 'anchor' of the ordered phase
    anchoring = stiff

    # 4. Dissolution: Loss of stiffness as we approach T_BKT
    dissolution = 1.0 - stiff

    # 5. Liminality: Normalized susceptibility (peaks at T_BKT due to finite-size)
    if np.max(sus) > 0:
        liminality = sus / np.max(sus)
    else:
        liminality = np.zeros_like(sus)

    # 6. Encounter: Square root of susceptibility (fluctuation proxy)
    encounter = np.sqrt(sus)

    # 7. Integration: Stiffness returning in the low-T phase
    integration = stiff

    # 8. Emergence: Final stiffness value
    emergence = stiff

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


# --- Visualization ---


def plot_results(results, output_dir):
    T = results["T"]
    T_BKT = results["T_BKT"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("2D XY Model — BKT Transition", fontsize=16, fontweight="bold")

    # Order Parameter (Magnetization)
    ax = axes[0, 0]
    ax.errorbar(
        T, results["magnetization"], yerr=results["mag_err"], fmt="o-", color="#2c7bb6"
    )
    ax.axvline(T_BKT, color="gray", linestyle="--", label=f"T_BKT ≈ {T_BKT}")
    ax.set_title("Magnetization (Appears due to finite size)")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("<|m|>")
    ax.legend()

    # Helicity Modulus (Stiffness)
    ax = axes[0, 1]
    ax.errorbar(
        T, results["stiffness"], yerr=results["stiff_err"], fmt="s-", color="#d7191c"
    )
    # BKT Universal Jump: stiffness drops from 2/pi * T_BKT to 0
    ax.plot(T, (2 / np.pi) * T, "k--", alpha=0.5, label="Universal Jump Line (2/π * T)")
    ax.axvline(T_BKT, color="gray", linestyle="--")
    ax.set_title("Helicity Modulus (Stiffness)")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Υ")
    ax.set_ylim(0, 1.2)
    ax.legend()

    # Susceptibility
    ax = axes[1, 0]
    ax.errorbar(
        T, results["susceptibility"], yerr=results["sus_err"], fmt="^-", color="#fdae61"
    )
    ax.axvline(T_BKT, color="gray", linestyle="--")
    ax.set_title("Susceptibility")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("χ")

    # Vortices Visualization
    ax = axes[1, 1]
    ax.set_axis_off()
    ax.set_title("Vortex Defects at T ≈ T_BKT")
    v_map = results["vortex_maps"].get(T_BKT)
    if v_map is not None:
        y, x = np.where(v_map != 0)
        colors = ["red" if v_map[i, j] > 0 else "blue" for i, j in zip(y, x)]
        ax.scatter(x, y, c=colors, s=50, edgecolors="k")
        ax.set_xlim(-1, results["N"])
        ax.set_ylim(-1, results["N"])
        ax.set_aspect("equal")
        # Add legend for vortices
        from matplotlib.lines import Line2D

        custom_lines = [
            Line2D([0], [0], color="red", marker="o", linestyle="", label="Vortex"),
            Line2D(
                [0], [0], color="blue", marker="o", linestyle="", label="Antivortex"
            ),
        ]
        ax.legend(handles=custom_lines, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_dir / "xy_results.png", dpi=150)
    print(f"Results plot saved to {output_dir / 'xy_results.png'}")


def run(args=None):
    parser = argparse.ArgumentParser(description="Run 2D XY Model simulation.")
    parser.add_argument("--N", type=int, default=32, help="Lattice size N")
    parser.add_argument(
        "--T", type=float, default=None, help="Single temperature to simulate"
    )
    parser.add_argument(
        "--eq_sweeps", type=int, default=400, help="Equilibration sweeps"
    )
    parser.add_argument(
        "--meas_sweeps", type=int, default=800, help="Measurement sweeps"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).parent),
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument(
        "--no_scaling", action="store_true", help="Ignored, for compatibility."
    )

    parsed_args = parser.parse_args(args=args)
    output_dir = Path(parsed_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("2D XY Model Simulation")
    print("=" * 40)

    T_vals = [parsed_args.T] if parsed_args.T is not None else None
    results = simulate(
        N=parsed_args.N,
        T_values=T_vals,
        eq_sweeps=parsed_args.eq_sweeps,
        meas_sweeps=parsed_args.meas_sweeps,
        seeds=[parsed_args.seed],
    )

    plot_results(results, output_dir)

    # Export data
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        return obj

    all_results = {
        "params": {
            "N": parsed_args.N,
            "T": parsed_args.T,
            "eq_sweeps": parsed_args.eq_sweeps,
            "meas_sweeps": parsed_args.meas_sweeps,
            "seed": parsed_args.seed,
        },
        "main_results": convert(results),
        "protocol_metrics": calculate_protocol_metrics(results),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    df = pd.DataFrame(
        {
            "T": results["T"],
            "magnetization": results["magnetization"],
            "mag_err": results["mag_err"],
            "susceptibility": results["susceptibility"],
            "sus_err": results["sus_err"],
            "stiffness": results["stiffness"],
            "stiff_err": results["stiff_err"],
            "specific_heat": results["specific_heat"],
            "sheat_err": results["sheat_err"],
        }
    )
    df.to_csv(output_dir / "results.csv", index=False)
    print(f"Data exported to {output_dir / 'results.json'} and results.csv")


if __name__ == "__main__":
    run()
