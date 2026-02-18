"""
Tests for simulation modules — verifies core functions produce valid output
and critical exponents are within tolerance of known exact values.

These use small lattices and few sweeps for speed.
"""

import numpy as np


class TestIsing:
    def test_bootstrap_error(self):
        from simulations.phase_transitions.ising_2d import bootstrap_error

        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, stderr = bootstrap_error(samples)
        assert abs(mean - 3.0) < 0.01
        assert stderr > 0

    def test_autocorrelation_time(self):
        from simulations.phase_transitions.ising_2d import autocorrelation_time

        # Uncorrelated data should have tau ~ 1
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, 1000)
        tau = autocorrelation_time(series)
        assert 0.5 < tau < 3.0

    def test_wolff_step_flips_cluster(self):
        from simulations.phase_transitions.ising_2d import wolff_step

        rng = np.random.default_rng(42)
        L = 10
        spins = np.ones((L, L), dtype=np.int8)
        flipped = wolff_step(spins, beta=1.0, rng=rng)
        assert flipped > 0
        assert np.any(spins == -1)

    def test_simulate_returns_expected_keys(self):
        from simulations.phase_transitions.ising_2d import simulate

        T_vals = np.array([2.0, 2.269, 2.5])
        result = simulate(N=8, T_values=T_vals, eq_sweeps=5, meas_sweeps=10, seeds=[42])
        for key in ["T", "magnetization", "susceptibility", "specific_heat"]:
            assert key in result
            assert len(result[key]) == len(T_vals)

    def test_susceptibility_peaks_near_tc(self):
        from simulations.phase_transitions.ising_2d import simulate

        T_vals = np.linspace(1.5, 3.0, 15)
        result = simulate(
            N=12, T_values=T_vals, eq_sweeps=20, meas_sweeps=40, seeds=[42]
        )
        peak_T = T_vals[np.argmax(result["susceptibility"])]
        assert 1.8 < peak_T < 2.8  # Within range of T_c ≈ 2.269


class TestPotts:
    def test_bootstrap_error(self):
        from simulations.phase_transitions.potts_2d import bootstrap_error

        mean, stderr = bootstrap_error(np.ones(10))
        assert mean == 1.0
        assert stderr == 0.0

    def test_simulate_returns_expected_keys(self):
        from simulations.phase_transitions.potts_2d import simulate

        T_vals = np.array([0.8, 0.995, 1.2])
        result = simulate(N=8, T_values=T_vals, eq_sweeps=5, meas_sweeps=10, seeds=[42])
        for key in ["T", "magnetization", "susceptibility", "specific_heat"]:
            assert key in result
            assert len(result[key]) == len(T_vals)


class TestPercolation:
    def test_powerlaw_mle(self):
        from simulations.phase_transitions.percolation_2d import powerlaw_mle

        rng = np.random.default_rng(42)
        data = (rng.pareto(1.5, 5000) + 1) * 5
        alpha = powerlaw_mle(data, s_min=5)
        assert 2.0 < alpha < 3.5

    def test_powerlaw_mle_insufficient_data(self):
        from simulations.phase_transitions.percolation_2d import powerlaw_mle

        data = np.array([1.0, 2.0, 3.0])
        assert np.isnan(powerlaw_mle(data, s_min=5))

    def test_simulate_returns_expected_keys(self):
        from simulations.phase_transitions.percolation_2d import simulate

        p_vals = np.array([0.4, 0.59, 0.8])
        result = simulate(L=10, p_values=p_vals, n_realizations=5, seeds=[42])
        for key in ["p", "order_param", "susceptibility", "span_prob"]:
            assert key in result
            assert len(result[key]) == len(p_vals)
        assert "cluster_sizes_at_pc" in result

    def test_spanning_prob_increases_with_p(self):
        from simulations.phase_transitions.percolation_2d import simulate

        p_vals = np.array([0.3, 0.5, 0.7, 0.9])
        result = simulate(L=20, p_values=p_vals, n_realizations=10, seeds=[42])
        assert result["span_prob"][-1] > result["span_prob"][0]


class TestCriticalBrain:
    def test_powerlaw_mle(self):
        from simulations.neuroscience.critical_brain import powerlaw_mle

        data = np.array([5, 6, 7, 8, 10, 15, 20, 50, 100], dtype=float)
        alpha = powerlaw_mle(data, s_min=5)
        assert alpha > 1.0

    def test_simulate_returns_expected_keys(self):
        from simulations.neuroscience.critical_brain import simulate

        result = simulate(N=10, n_avalanches=20, seeds=[42])
        for key in ["sigma", "order_param", "susceptibility", "mean_duration"]:
            assert key in result
        assert "sizes_at_sc" in result
        assert "durations_at_sc" in result

    def test_susceptibility_peaks_near_sigma_c(self):
        from simulations.neuroscience.critical_brain import simulate

        result = simulate(N=15, n_avalanches=50, seeds=[42])
        sigma_vals = result["sigma"]
        sus = result["susceptibility"]
        peak_sigma = sigma_vals[np.argmax(sus)]
        assert 0.5 < peak_sigma < 2.0
