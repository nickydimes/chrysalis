"""
Universality Comparison Dashboard Generator
============================================
Generates cross-system comparison dashboards showing how different microscopic
systems (Ising, Potts, Percolation, Neural Branching) exhibit identical
critical behavior — the hallmark of universality.

Produces two dashboards:
  1. universality_comparison.png — 4-panel overview (one per system)
  2. universality_scaling.png   — 4-panel scaling analysis comparison
  3. universality_exponents.png — quantitative exponent comparison with
     error bars against exact values

Usage:
    python src/tools/universality_plotter.py

Produces: research/observations/universality_comparison.png
          research/observations/universality_scaling.png
          research/observations/universality_exponents.png
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
SIM_DIR = BASE_DIR / "simulations"
OUT_DIR = BASE_DIR / "research" / "observations"

# Ensure the repo root is on sys.path so we can import simulation modules
import sys
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _load_image(ax, path, title):
    """Load an image into an axes, or show a placeholder if missing."""
    if os.path.exists(path):
        img = mpimg.imread(str(path))
        ax.imshow(img)
        ax.set_title(title, fontsize=13, fontweight="bold")
    else:
        ax.text(0.5, 0.5, f"File Not Found:\n{path}", ha="center", va="center",
                fontsize=10, color="gray", transform=ax.transAxes)
        ax.set_title(title, fontsize=13, fontweight="bold", color="gray")
    ax.axis("off")


def generate_dashboard():
    """Generate the 4-panel overview dashboard from *_results.png files."""
    figures = {
        "Ising (Magnetic)": SIM_DIR / "phase_transitions" / "ising_results.png",
        "Potts (Symmetry)": SIM_DIR / "phase_transitions" / "potts_results.png",
        "Percolation (Geometric)": SIM_DIR / "phase_transitions" / "percolation_results.png",
        "Brain (Cognitive)": SIM_DIR / "neuroscience" / "critical_brain_results.png",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.suptitle("Universality Across Systems", fontsize=20, fontweight="bold")

    for i, (name, path) in enumerate(figures.items()):
        _load_image(axes[i // 2, i % 2], path, name)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "universality_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Dashboard saved to {out_path}")


def generate_scaling_dashboard():
    """Generate the 4-panel scaling analysis dashboard from *_scaling.png files."""
    figures = {
        "Ising — Finite-Size Scaling": SIM_DIR / "phase_transitions" / "ising_scaling.png",
        "Potts — Finite-Size Scaling": SIM_DIR / "phase_transitions" / "potts_scaling.png",
        "Percolation — Finite-Size Scaling": SIM_DIR / "phase_transitions" / "percolation_scaling.png",
        "Brain — Scaling Analysis": SIM_DIR / "neuroscience" / "critical_brain_scaling.png",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.suptitle("Finite-Size Scaling Across Systems", fontsize=20, fontweight="bold")

    for i, (name, path) in enumerate(figures.items()):
        _load_image(axes[i // 2, i % 2], path, name)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "universality_scaling.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Scaling dashboard saved to {out_path}")


def generate_exponent_comparison():
    """Generate a quantitative comparison of measured vs exact critical exponents.

    Imports simulation modules and runs with minimal parameters to extract
    scaling exponents. If simulation data is unavailable, uses placeholder values.
    """
    # Exact exponents for each universality class
    exact = {
        "2D Ising": {"beta_nu": 1/8, "gamma_nu": 7/4, "nu": 1.0},
        "2D Potts (q=3)": {"beta_nu": 2/15, "gamma_nu": 26/15, "nu": 5/6},
        "2D Percolation": {"beta_nu": 5/48, "gamma_nu": 43/24, "nu": 4/3},
        "Mean-field\nBranching": {"tau_s": 3/2, "tau_d": 2.0, "gamma_sd": 2.0},
    }

    # Try to extract measured exponents from scaling runs
    measured = {}
    try:
        from simulations.phase_transitions.ising_2d import simulate
        res_25 = simulate(N=15, T_values=np.linspace(1.8, 2.8, 12),
                          eq_sweeps=30, meas_sweeps=50, seeds=[42])
        res_50 = simulate(N=30, T_values=np.linspace(1.8, 2.8, 12),
                          eq_sweeps=30, meas_sweeps=50, seeds=[42])
        chi_25 = np.max(res_25["susceptibility"])
        chi_50 = np.max(res_50["susceptibility"])
        gamma_nu_meas = np.log(chi_50 / chi_25) / np.log(30 / 15)
        measured["2D Ising"] = {"gamma_nu": gamma_nu_meas}
        print(f"  Ising gamma/nu = {gamma_nu_meas:.3f}")
    except Exception as e:
        print(f"  Ising extraction skipped: {e}")

    try:
        from simulations.phase_transitions.potts_2d import simulate
        res_25 = simulate(N=15, T_values=np.linspace(0.7, 1.3, 12),
                          eq_sweeps=30, meas_sweeps=50, seeds=[42])
        res_50 = simulate(N=30, T_values=np.linspace(0.7, 1.3, 12),
                          eq_sweeps=30, meas_sweeps=50, seeds=[42])
        chi_25 = np.max(res_25["susceptibility"])
        chi_50 = np.max(res_50["susceptibility"])
        gamma_nu_meas = np.log(chi_50 / chi_25) / np.log(30 / 15)
        measured["2D Potts (q=3)"] = {"gamma_nu": gamma_nu_meas}
        print(f"  Potts gamma/nu = {gamma_nu_meas:.3f}")
    except Exception as e:
        print(f"  Potts extraction skipped: {e}")

    try:
        from simulations.phase_transitions.percolation_2d import simulate, powerlaw_mle
        res_50 = simulate(L=30, p_values=np.linspace(0.4, 0.8, 12),
                          n_realizations=30, seeds=[42])
        res_100 = simulate(L=60, p_values=np.linspace(0.4, 0.8, 12),
                           n_realizations=30, seeds=[42])
        chi_50 = np.max(res_50["susceptibility"])
        chi_100 = np.max(res_100["susceptibility"])
        gamma_nu_meas = np.log(chi_100 / chi_50) / np.log(60 / 30)
        meas_perc = {"gamma_nu": gamma_nu_meas}
        cs = res_100["cluster_sizes_at_pc"]
        if len(cs) > 10:
            meas_perc["tau"] = powerlaw_mle(cs, s_min=5)
        measured["2D Percolation"] = meas_perc
        print(f"  Percolation gamma/nu = {gamma_nu_meas:.3f}")
    except Exception as e:
        print(f"  Percolation extraction skipped: {e}")

    try:
        from simulations.neuroscience.critical_brain import simulate, powerlaw_mle
        res = simulate(N=50, n_avalanches=500, seeds=[42])
        sizes = res["sizes_at_sc"]
        durs = res["durations_at_sc"]
        meas_brain = {}
        if len(sizes) > 10:
            meas_brain["tau_s"] = powerlaw_mle(sizes, s_min=5)
        if len(durs) > 10:
            meas_brain["tau_d"] = powerlaw_mle(durs, s_min=5)
        measured["Mean-field\nBranching"] = meas_brain
        print(f"  Brain tau_s = {meas_brain.get('tau_s', 'N/A')}")
    except Exception as e:
        print(f"  Brain extraction skipped: {e}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Critical Exponent Comparison: Measured vs Exact",
                 fontsize=16, fontweight="bold")

    # Left panel: gamma/nu for thermal + percolation systems
    ax1 = axes[0]
    systems_gn = ["2D Ising", "2D Potts (q=3)", "2D Percolation"]
    exact_gn = [exact[s]["gamma_nu"] for s in systems_gn]
    meas_gn = [measured.get(s, {}).get("gamma_nu", np.nan) for s in systems_gn]

    x = np.arange(len(systems_gn))
    ax1.bar(x - 0.15, exact_gn, 0.3, label="Exact", color="#2c7bb6", alpha=0.8)
    ax1.bar(x + 0.15, meas_gn, 0.3, label="Measured", color="#d7191c", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems_gn, fontsize=10)
    ax1.set_ylabel(r"$\gamma / \nu$", fontsize=12)
    ax1.set_title(r"Susceptibility Exponent $\gamma/\nu$")
    ax1.legend()
    for i, (e, m) in enumerate(zip(exact_gn, meas_gn)):
        if not np.isnan(m):
            ax1.annotate(f"{m:.2f}", (i + 0.15, m), ha="center", va="bottom", fontsize=9)
        ax1.annotate(f"{e:.2f}", (i - 0.15, e), ha="center", va="bottom", fontsize=9)

    # Right panel: power-law exponents for brain + percolation
    ax2 = axes[1]
    labels = []
    exact_vals = []
    meas_vals = []

    if "tau" in exact.get("2D Percolation", {}):
        pass  # tau not in our exact dict, add it
    # Percolation tau
    labels.append(r"Perc $\tau$")
    exact_vals.append(187 / 91)
    meas_vals.append(measured.get("2D Percolation", {}).get("tau", np.nan))

    # Brain tau_s
    labels.append(r"Brain $\tau_s$")
    exact_vals.append(3 / 2)
    meas_vals.append(measured.get("Mean-field\nBranching", {}).get("tau_s", np.nan))

    # Brain tau_d
    labels.append(r"Brain $\tau_d$")
    exact_vals.append(2.0)
    meas_vals.append(measured.get("Mean-field\nBranching", {}).get("tau_d", np.nan))

    x2 = np.arange(len(labels))
    ax2.bar(x2 - 0.15, exact_vals, 0.3, label="Exact", color="#2c7bb6", alpha=0.8)
    ax2.bar(x2 + 0.15, meas_vals, 0.3, label="Measured", color="#d7191c", alpha=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Exponent value", fontsize=12)
    ax2.set_title("Power-Law Exponents")
    ax2.legend()
    for i, (e, m) in enumerate(zip(exact_vals, meas_vals)):
        if not np.isnan(m):
            ax2.annotate(f"{m:.2f}", (i + 0.15, m), ha="center", va="bottom", fontsize=9)
        ax2.annotate(f"{e:.2f}", (i - 0.15, e), ha="center", va="bottom", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = OUT_DIR / "universality_exponents.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Exponent comparison saved to {out_path}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating overview dashboard...")
    generate_dashboard()

    print("\nGenerating scaling dashboard...")
    generate_scaling_dashboard()

    print("\nGenerating exponent comparison (running mini simulations)...")
    generate_exponent_comparison()
