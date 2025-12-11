# tests/test_core.py
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from xy3d_wolff import core


def test_initialize_lattice_shape_and_norm():
    L = 4
    spins = core.initialize_lattice(L)
    assert spins.shape == (L, L, L, 2)

    norms = np.linalg.norm(spins, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-7)


@pytest.mark.parametrize("update_func_name", ["wolff_update", "wolff_update_new"])
def test_wolff_updates_preserve_shape_and_norm(update_func_name):
    L = 4
    J = 1.0
    T = 2.0
    np.random.seed(0)

    spins = core.initialize_lattice(L)
    spins_before = spins.copy()

    update_func = getattr(core, update_func_name)
    cluster_size = update_func(spins, J, T)

    V = L**3
    # Cluster size reasonable
    assert 1 <= cluster_size <= V

    # Shape unchanged
    assert spins.shape == spins_before.shape

    # Spins still unit length
    norms = np.linalg.norm(spins, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-7)


def test_wolff_update_with_estimator_outputs():
    L = 4
    J = 1.0
    T = 2.0
    np.random.seed(1)

    spins = core.initialize_lattice(L)
    cluster_size, cluster_Sq, q_vectors = core.wolff_update_with_estimator(
        spins, J, T
    )
    V = L**3

    assert 1 <= cluster_size <= V
    assert cluster_Sq.shape == (3,)
    assert len(q_vectors) == 3
    # q vectors should have magnitude ~ 2π / L along axes
    expected_mag = 2 * np.pi / L
    for q in q_vectors:
        assert np.isclose(np.linalg.norm(q), expected_mag, atol=1e-12)


def test_compute_improved_structure_factor_basic():
    L = 4
    # Put a small "cluster" at a few positions
    cluster_positions = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    cluster_Sq, q_vectors = core.compute_improved_structure_factor(
        cluster_positions, L
    )

    assert cluster_Sq.shape == (3,)
    assert len(q_vectors) == 3
    # Structure factor must be non-negative
    assert np.all(cluster_Sq >= 0)


def test_compute_correlation_length_from_Sq_positive_and_inf():
    L = 4
    # simple positive case
    S_q = np.array([1.0, 2.0, 3.0])
    q_vectors = [
        np.array([2 * np.pi / L, 0, 0]),
        np.array([0, 2 * np.pi / L, 0]),
        np.array([0, 0, 2 * np.pi / L]),
    ]
    xi = core.compute_correlation_length_from_Sq(S_q, q_vectors, L)
    assert np.isfinite(xi)
    assert xi > 0

    # degenerate case should return inf
    S_q_zero = np.zeros_like(S_q)
    xi_inf = core.compute_correlation_length_from_Sq(S_q_zero, q_vectors, L)
    assert xi_inf == np.inf


def test_compute_autocorrelation_and_tau_int():
    data = np.array([1.0, 2.0, 3.0, 4.0])

    autocorr = core.compute_autocorrelation(data)
    # Length should match input length
    assert len(autocorr) == len(data)
    # Normalized: value at lag 0 must be 1
    assert autocorr[0] == pytest.approx(1.0)
    # Values bounded between -1 and 1
    assert np.all(autocorr <= 1.0 + 1e-12)
    assert np.all(autocorr >= -1.0 - 1e-12)

    # Simple synthetic autocorr to test tau_int
    fake_autocorr = np.array([1.0, 0.5, 0.25, -0.1])
    tau_int = core.estimate_autocorrelation_time(fake_autocorr)
    # tau_int = 0.5 + (0.5 + 0.25) = 1.25
    assert tau_int == pytest.approx(1.25)


def test_compute_susceptibility_with_estimator_reasonable():
    V = 8
    T = 2.0
    magnetizations = [0.1, 0.2, 0.3, 0.4]
    chi, err = core.compute_susceptibility_with_estimator(
        magnetizations, V, T
    )
    assert chi > 0
    assert err >= 0
    # Error should be smaller than chi for a well-behaved sample
    assert err < chi


def test_compute_F_uniform_spins():
    L = 4
    # All spins pointing along x-axis
    spins = np.ones((L, L, L, 2))
    spins[..., 1] = 0.0

    F = core.compute_F(spins, L)
    # F must be non-negative
    assert F >= 0


def test_compute_xi_second_moment_basic():
    L = 4
    chi = 2.0
    F = 0.5
    xi = core.compute_xi_second_moment(chi, F, L)
    assert np.isfinite(xi)
    assert xi >= 0

    # Also test case where chi < F: abs() should still give finite result
    xi2 = core.compute_xi_second_moment(chi=0.5, F=2.0, L=L)
    assert np.isfinite(xi2)
    assert xi2 >= 0


def test_compute_magnetization_uniform_spins():
    L = 3
    spins = np.zeros((L, L, L, 2))
    spins[..., 0] = 1.0  # all spins along x-axis

    m = core.compute_magnetization(spins)
    # Perfect order => magnetization 1
    assert m == pytest.approx(1.0)


def test_compute_susceptibility_matches_manual():
    V = 2
    T = 1.5
    # magnetization per site: 0, 1/V => total magnetization 0 and 1
    magnetizations = np.array([0.0, 1.0 / V])

    M_abs = magnetizations * V
    M_abs_mean = np.mean(M_abs)
    M2_mean = np.mean(M_abs**2)
    chi_manual = (M2_mean - M_abs_mean**2) / (V * T)

    chi, err = core.compute_susceptibility(magnetizations, V, T)
    assert chi == pytest.approx(chi_manual)
    assert err >= 0


def test_compute_binder_cumulant_constant_M():
    V = 8
    # constant magnetization => Binder cumulant should be 2/3
    m0 = 0.5
    magnetizations = np.full(100, m0)

    U, error_U = core.compute_binder_cumulant(magnetizations, V)
    assert U == pytest.approx(2.0 / 3.0, rel=1e-6)
    assert error_U >= 0


def test_compute_energy_uniform_spins():
    L = 3
    J = 1.0
    # All spins aligned => maximal negative energy
    spins = np.zeros((L, L, L, 2))
    spins[..., 0] = 1.0

    E = core.compute_energy(spins, J)
    V = L**3
    # For this code, uniform configuration gives E = -6 J V
    assert E == pytest.approx(-6.0 * J * V)


def test_compute_specific_heat_manual_C():
    V = 8
    T = 2.0
    # Some synthetic energies per site
    energies = np.linspace(-1.0, 1.0, 100)

    C, sigma_C = core.compute_specific_heat(energies, V, T)

    # Manual C formula used in core.compute_specific_heat
    E = np.array(energies)
    E_mean = np.mean(E)
    E2_mean = np.mean(E**2)
    variance_E = E2_mean - E_mean**2
    C_manual = variance_E / (3 * T**2)

    assert C == pytest.approx(C_manual)
    assert sigma_C >= 0


def test_compute_correlation_length_from_cluster_sizes():
    V = 64
    # cluster_sizes are normalized by V according to the code
    cluster_sizes_normalized = np.array([1, 2, 3, 4, 5]) / V

    xi = core.compute_correlation_length_from_cluster_sizes(
        cluster_sizes_normalized, V, num_bins=5
    )
    assert np.isfinite(xi)
    assert xi > 0


def test_compute_spin_correlation_monotonic_for_ordered_spins():
    L = 4
    max_r = 2
    # Uniform spins => G(r) should be constant and positive
    spins = np.zeros((L, L, L, 2))
    spins[..., 0] = 1.0

    G_r = core.compute_spin_correlation(spins, max_r)
    assert G_r.shape == (max_r,)
    # All correlations equal to 1 for perfect order
    assert np.allclose(G_r, G_r[0], atol=1e-7)
    assert G_r[0] > 0


def test_fit_correlation_length_on_synthetic_data():
    # Synthetic G(r) ~ exp(-r/xi0)
    xi0 = 2.0
    r_max = 6
    r_vals = np.arange(1, r_max + 1)
    G_r = np.exp(-r_vals / xi0)

    xi_fit = core.fit_correlation_length(G_r)
    # Fitted xi should be close to xi0
    assert xi_fit == pytest.approx(xi0, rel=0.2)
