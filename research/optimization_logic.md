# Simulation Optimization: Objectives & Algorithms

The goal of auto-optimization is to find the parameter set $\mathbf{	heta} = \{T, p, ...\}$ that minimizes or maximizes a specific "Criticality Objective" function $f(\mathbf{	heta})$.

## 1. Optimization Objectives (Target Metrics)

| Objective | Target Function | Representation in Chrysalis |
|---|---|---|
| **Max Susceptibility** | $f(T) = \max \chi(T)$ | Find $T_c$ where system responsiveness is highest (Liminality peak). |
| **Max Fluctuations** | $f(T) = \max Var(E)$ | Find $T_c$ where energy/structure breakdown is most volatile (Dissolution peak). |
| **Connectivity Threshold** | $f(p) = 	ext{diff}(Ps, 0.5)$ | Find $p_c$ where spanning probability is 0.5 (Structural transition). |
| **Binder Crossing** | $\min 	ext{Var}(U_L(T, L_i))$ | Find the scale-invariant transition point across different lattice sizes. |

## 2. Iterative Peak Refinement (IPR) Algorithm

Since simulations are expensive, we use an iterative approach to narrow down the critical point:

1.  **Stage 1: Coarse Sweep.** Run a wide-range sweep with few points to identify the approximate region of the peak.
2.  **Stage 2: Gradient Step.** Calculate the numerical derivative of the target metric.
3.  **Stage 3: Fine Refinement.** Zoom in on the identified peak with higher resolution and increased number of seeds for statistical precision.
4.  **Stage 4: Convergence.** Stop when the parameter step size $\Delta 	heta < \epsilon$ or the metric gain is negligible.
