"""
Universality Comparison Dashboard Generator
============================================
Generates cross-system comparison dashboards showing how different microscopic
systems (Ising, Potts, Percolation, XY, Neural Branching) exhibit identical
critical behavior — the hallmark of universality.

Produces three dashboards:
  1. universality_comparison.png — 6-panel overview (one per system)
  2. universality_scaling.png   — 6-panel scaling analysis comparison
  3. universality_exponents.png — quantitative exponent comparison with
     error bars against exact values
  4. universality_collapse.png  — scaling collapse for symmetry-breaking models

Usage:
    python src/tools/universality_plotter.py
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SIM_DIR = BASE_DIR / "simulations"
OUT_DIR = BASE_DIR / "research" / "observations"

# Ensure the repo root is on sys.path so we can import simulation modules
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))  # noqa: E402


def _load_image(ax, path, title):
    """Load an image into an axes, or show a placeholder if missing."""
    path_obj = Path(path)
    if path_obj.exists():
        img = mpimg.imread(str(path_obj))
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight="bold")
    else:
        ax.text(
            0.5,
            0.5,
            f"File Not Found:\n{path_obj.name}",
            ha="center",
            va="center",
            fontsize=9,
            color="gray",
            transform=ax.transAxes,
        )
        ax.set_title(title, fontsize=11, fontweight="bold", color="gray")
    ax.axis("off")


def generate_dashboard():
    """Generate the overview dashboard from *_results.png files."""
    figures = {
        "Ising (Magnetic)": SIM_DIR / "phase_transitions" / "ising_results.png",
        "Potts (Symmetry)": SIM_DIR / "phase_transitions" / "potts_results.png",
        "Percolation (Geometric)": SIM_DIR
        / "phase_transitions"
        / "percolation_results.png",
        "XY Model (Topological)": SIM_DIR / "phase_transitions" / "xy_results.png",
        "Brain (Cognitive)": SIM_DIR / "neuroscience" / "critical_brain_results.png",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.suptitle("Universality Across Systems", fontsize=20, fontweight="bold")

    for i, (name, path) in enumerate(figures.items()):
        _load_image(axes[i // 3, i % 3], path, name)

    # Hide the empty 6th panel
    axes[1, 2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "universality_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Dashboard saved to {out_path}")


def generate_scaling_dashboard():
    """Generate the scaling analysis dashboard from *_scaling.png files."""
    figures = {
        "Ising — Scaling": SIM_DIR / "phase_transitions" / "ising_scaling.png",
        "Potts — Scaling": SIM_DIR / "phase_transitions" / "potts_scaling.png",
        "Percolation — Scaling": SIM_DIR
        / "phase_transitions"
        / "percolation_scaling.png",
        "XY — Vortices": SIM_DIR
        / "phase_transitions"
        / "xy_vortices.png",  # Placeholder/alt for XY
        "Brain — Scaling": SIM_DIR / "neuroscience" / "critical_brain_scaling.png",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.suptitle("Finite-Size Scaling Across Systems", fontsize=20, fontweight="bold")

    for i, (name, path) in enumerate(figures.items()):
        _load_image(axes[i // 3, i % 3], path, name)

    axes[1, 2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "universality_scaling.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Scaling dashboard saved to {out_path}")


def generate_exponent_comparison():
    """Generate a quantitative comparison of measured vs exact critical exponents."""
    # Exact exponents for each universality class
    exact = {
        "2D Ising": {"beta_nu": 1 / 8, "gamma_nu": 7 / 4, "nu": 1.0},
        "2D Potts (q=3)": {"beta_nu": 2 / 15, "gamma_nu": 26 / 15, "nu": 5 / 6},
        "2D Percolation": {"beta_nu": 5 / 48, "gamma_nu": 43 / 24, "nu": 4 / 3},
        "Mean-field\nBranching": {"tau_s": 3 / 2, "tau_d": 2.0, "gamma_sd": 2.0},
    }

    # Try to extract measured exponents from scaling runs
    measured = {}
    try:
        from simulations.phase_transitions.ising_2d import simulate

        res_25 = simulate(
            N=15,
            T_values=np.linspace(1.8, 2.8, 12),
            eq_sweeps=30,
            meas_sweeps=50,
            seeds=[42],
        )
        res_50 = simulate(
            N=30,
            T_values=np.linspace(1.8, 2.8, 12),
            eq_sweeps=30,
            meas_sweeps=50,
            seeds=[42],
        )
        chi_25 = np.max(res_25["susceptibility"])
        chi_50 = np.max(res_50["susceptibility"])
        gamma_nu_meas = np.log(chi_50 / chi_25) / np.log(30 / 15)
        measured["2D Ising"] = {"gamma_nu": gamma_nu_meas}
    except Exception:
        pass

    try:
        from simulations.phase_transitions.potts_2d import simulate

        res_25 = simulate(
            N=15,
            T_values=np.linspace(0.7, 1.3, 12),
            eq_sweeps=30,
            meas_sweeps=50,
            seeds=[42],
        )
        res_50 = simulate(
            N=30,
            T_values=np.linspace(0.7, 1.3, 12),
            eq_sweeps=30,
            meas_sweeps=50,
            seeds=[42],
        )
        chi_25 = np.max(res_25["susceptibility"])
        chi_50 = np.max(res_50["susceptibility"])
        gamma_nu_meas = np.log(chi_50 / chi_25) / np.log(30 / 15)
        measured["2D Potts (q=3)"] = {"gamma_nu": gamma_nu_meas}
    except Exception:
        pass

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Critical Exponent Comparison: Measured vs Exact",
        fontsize=16,
        fontweight="bold",
    )

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

    ax2 = axes[1]
    labels = [r"Perc $\tau$", r"Brain $\tau_s$", r"Brain $\tau_d$"]
    exact_vals = [187 / 91, 3 / 2, 2.0]
    meas_vals = [np.nan, np.nan, np.nan]  # Placeholder

    x2 = np.arange(len(labels))
    ax2.bar(x2 - 0.15, exact_vals, 0.3, label="Exact", color="#2c7bb6", alpha=0.8)
    ax2.bar(x2 + 0.15, meas_vals, 0.3, label="Measured", color="#d7191c", alpha=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_title("Power-Law Exponents")
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = OUT_DIR / "universality_exponents.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Exponent comparison saved to {out_path}")


def _get_latest_experiment_dir(system_name):
    """Find the most recent experiment directory for a given system."""
    results_root = SIM_DIR / "results"
    if not results_root.exists():
        return None

    candidates = []
    for d in results_root.iterdir():
        if d.is_dir() and system_name.lower() in d.name.lower():
            candidates.append(d)

    if not candidates:
        return None

    return max(candidates, key=os.path.getmtime)


def generate_scaling_collapse_figure():
    """Generates a scaling collapse figure for symmetry-breaking models."""
    systems = {
        "Ising": {
            "T_c": 2.269,
            "beta": 1 / 8,
            "gamma": 7 / 4,
            "nu": 1.0,
            "color": "blue",
            "marker": "o",
            "label": "Ising (T)",
        },
        "Potts": {
            "T_c": 0.995,
            "beta": 2 / 15,
            "gamma": 26 / 15,
            "nu": 5 / 6,
            "color": "green",
            "marker": "s",
            "label": "Potts (T)",
        },
        "Percolation": {
            "p_c": 0.5927,
            "beta": 5 / 48,
            "gamma": 43 / 24,
            "nu": 4 / 3,
            "color": "red",
            "marker": "^",
            "label": "Percolation (p)",
        },
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Universality: Cross-System Scaling Collapse", fontsize=20, fontweight="bold"
    )

    for name, params in systems.items():
        exp_dir = _get_latest_experiment_dir(name)
        if not exp_dir:
            continue

        print(f"  Processing {name} from {exp_dir.name}...")
        t_vals, m_vals, chi_vals, L_vals = [], [], [], []

        for res_file in exp_dir.glob("**/results.json"):
            try:
                with open(res_file, "r") as f:
                    data = json.load(f)
                p_dict = data.get("params", {})
                L = p_dict.get("N", p_dict.get("L"))
                cp = p_dict.get("T", p_dict.get("p"))
                if cp is None:
                    continue

                mres = data.get("main_results", {})
                m = mres.get("magnetization", mres.get("order_param", []))
                chi = mres.get("susceptibility", [])

                vm = m[0] if isinstance(m, list) and m else m
                vchi = chi[0] if isinstance(chi, list) and chi else chi

                t_vals.append(cp)
                m_vals.append(vm)
                chi_vals.append(vchi)
                L_vals.append(L)
            except Exception:
                continue

        if not t_vals:
            continue
        control_param, order_param, susceptibility, L = map(
            np.array, [t_vals, m_vals, chi_vals, L_vals]
        )

        idx = np.argsort(control_param)
        control_param, order_param, susceptibility, L = (
            control_param[idx],
            order_param[idx],
            susceptibility[idx],
            L[idx],
        )

        crit_val = params.get("T_c", params.get("p_c"))
        t = (control_param - crit_val) / crit_val
        y_m = order_param * (L ** (params["beta"] / params["nu"]))
        x_scaled = t * (L ** (1 / params["nu"]))
        y_chi = susceptibility / (L ** (params["gamma"] / params["nu"]))

        ax1.plot(
            x_scaled,
            y_m,
            marker=params["marker"],
            linestyle="",
            label=params["label"],
            color=params["color"],
            alpha=0.6,
        )
        ax2.plot(
            x_scaled,
            y_chi,
            marker=params["marker"],
            linestyle="",
            label=params["label"],
            color=params["color"],
            alpha=0.6,
        )

    ax1.set_xlabel(r"$\epsilon L^{1/\nu}$", fontsize=14)
    ax1.set_ylabel(r"$m L^{\beta/\nu}$", fontsize=14)
    ax1.set_title("Order Parameter Collapse", fontsize=16)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel(r"$\epsilon L^{1/\nu}$", fontsize=14)
    ax2.set_ylabel(r"$\chi L^{-\gamma/\nu}$", fontsize=14)
    ax2.set_title("Susceptibility Collapse", fontsize=16)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = OUT_DIR / "universality_collapse.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Scaling collapse figure saved to {out_path}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating overview dashboard...")
    generate_dashboard()
    print("\nGenerating scaling dashboard...")
    generate_scaling_dashboard()
    print("\nGenerating exponent comparison...")
    generate_exponent_comparison()
    print("\nGenerating scaling collapse figure...")
    generate_scaling_collapse_figure()
