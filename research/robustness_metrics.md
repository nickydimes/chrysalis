# Protocol Robustness & Sensitivity Analysis

This document defines the quantitative framework for measuring the "robustness" of the Eight-Step Navigation Protocol within physical simulations.

## 1. Success Criteria (Protocol Completion)

A simulation run is considered a **Success** if it achieves a stable emergent state.
- **Ising/Potts:** Final Magnetization $|m| > 0.8$.
- **Percolation:** Final Spanning Cluster Fraction $P_\infty > 0.7$.

## 2. Robustness Metrics

| Metric | Definition | Interpretation |
|---|---|---|
| `Success Probability (Ps)` | Fraction of seeds that reach the Emergence state for a given parameter set. | Reliability of the transformation path. |
| `Transition Width (Δ)` | The range of parameters where $0.1 < Ps < 0.9$. | The 'Critical Zone' or 'Window of Vulnerability'. |
| `Robustness Score (R)` | $\int Ps(p) dp$ normalized by the parameter range. | Overall stability of the protocol against noise/variance. |
| `Tipping Point (Tc*)` | The parameter value where $Ps = 0.5$. | The precise threshold where the protocol is likely to fail. |

## 3. Dimensions of Sensitivity
- **Thermal Noise (T):** Represents environmental stress or internal entropy.
- **System Scale (N):** Represents the size of the community or network.
- **Coupling Strength (J):** Represents 'trust' or 'cohesion' between agents.
- **External Field (H):** Represents top-down direction or external 'Encounter' pressure.
