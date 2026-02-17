---
date: 2026-01-20
location: "Simulation Lab"
tags: [physics, percolation, connectivity, spanning-cluster, universality]
source: "Broadbent & Hammersley 1957, Stauffer & Aharony 1994"
---

## Purification
The percolation problem begins with the simplest possible preparation: an empty lattice. Each site is independently occupied with probability p or left vacant with probability (1-p). There is no Hamiltonian, no temperature, no dynamics — only geometry and probability. This radical simplification strips away every complication of interacting systems (energy, entropy, detailed balance) to isolate the pure question: at what density does a random medium become connected? The system is "purified" to its geometric essence.

## Containment
The L x L lattice with defined boundary conditions contains the problem. For percolation, the boundary condition that matters is whether we ask for a spanning cluster — one that connects top to bottom or left to right. The finite lattice ensures every cluster has a maximum possible size (L^2), which rounds the percolation transition. The lattice geometry itself (square, triangular, honeycomb) affects the critical threshold p_c but not the critical exponents — another manifestation of universality within the containment.

## Dissolution
Below p_c, the lattice consists of many small, disconnected clusters. As p increases toward p_c = 0.5927... (site percolation on the square lattice), the clusters grow and begin to merge. The cluster size distribution develops a power-law tail. The "old order" of disconnected small clusters dissolves into increasingly complex, ramified structures. The mean cluster size (excluding the largest) diverges as chi ~ |p - p_c|^{-gamma}, analogous to susceptibility divergence at a thermal phase transition.

## Liminality
At p_c, the incipient infinite cluster first appears — a fractal object with dimension d_f = 91/48 that spans the system but occupies zero fraction of the total area in the thermodynamic limit. The cluster size distribution follows P(s) ~ s^{-tau} with tau = 187/91. The system is at maximal sensitivity: adding or removing a single occupied site near p_c can connect or disconnect macroscopic regions. The correlation length has diverged — the characteristic cluster size is the system size itself. Every scale participates in the structure simultaneously.

## Encounter
The encounter is the spanning cluster itself — a geometric object that has no analogue below p_c. It is not "almost" a spanning cluster; it is qualitatively new. Its fractal structure means it has infinite surface-to-volume ratio, maximum interface with the unoccupied sites. In the language of electrical conductivity: below p_c the medium is an insulator; at p_c a conducting path appears for the first time. This is not a gradual transition — it is a threshold phenomenon. The conducting path was structurally impossible below p_c regardless of how close p was to the critical value.

## Integration
Finite-size scaling integrates the data across different lattice sizes into a coherent picture. The spanning probability — the probability that a spanning cluster exists — transitions from 0 to 1 as L -> infinity, with the transition sharpening around p_c. The collapse of P_infinity * L^{beta/nu} vs (p - p_c) * L^{1/nu} onto a single curve (with beta/nu = 5/48, nu = 4/3) confirms that the same universal scaling function governs all system sizes. The Union-Find algorithm provides the computational integration: efficiently tracking which sites belong to which cluster as bonds are added.

## Emergence
Above p_c, the spanning cluster grows to fill a finite fraction P_infinity ~ (p - p_c)^{beta} of the lattice (beta = 5/36). A macroscopic connected structure has emerged from purely local, independent random events. No site "knows" about distant sites, yet long-range connectivity crystallizes from short-range occupation. The emergent conducting phase has properties (conductivity exponent, backbone dimension, red-bond structure) that cannot be predicted from the properties of individual occupied sites. The whole is genuinely more than the sum of its parts.
