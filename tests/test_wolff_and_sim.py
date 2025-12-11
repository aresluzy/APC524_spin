import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from xy3d_wolff.wolff import WolffParameters, XYLattice, WolffClusterUpdater
from xy3d_wolff.simulation import SimulationConfig, XYSimulation, XYStudy


def test_xylattice_init_and_reset():
    L = 4
    lat = XYLattice(L)
    assert lat.spins.shape == (L, L, L, 2)
    norms = np.linalg.norm(lat.spins, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-7)
    lat.reset_random()
    norms2 = np.linalg.norm(lat.spins, axis=-1)
    assert np.allclose(norms2, 1.0, atol=1e-7)


def test_wolffclusterupdater_step():
    L = 4
    lat = XYLattice(L)
    params = WolffParameters(J=1.0, T=2.2)
    updater = WolffClusterUpdater(params)
    size = updater.step(lat.spins)
    assert isinstance(size, int)
    assert 1 <= size <= L ** 3


def test_wolffclusterupdater_step_new_and_estimator():
    L = 4
    lat = XYLattice(L)
    params = WolffParameters(J=1.0, T=2.2)
    updater = WolffClusterUpdater(params)

    size_new = updater.step_new(lat.spins)
    assert isinstance(size_new, int)
    assert 1 <= size_new <= L ** 3

    size_est, cluster_Sq, q_vectors = updater.step_with_estimator(lat.spins)
    assert isinstance(size_est, int)
    assert cluster_Sq.shape[0] == q_vectors.shape[0]


def test_xysimulation_basic_and_improved():
    L = 4
    T = 2.2
    J = 1.0
    n_steps = 20
    n_equil = 5

    sim = XYSimulation(L=L, T=T, J=J, n_steps=n_steps, n_equil=n_equil)
    res_basic = sim.run_basic()
    res_improved = sim.run_improved()

    for res in (res_basic, res_improved):
        assert "energy_density" in res
        assert "magnetization" in res
        assert "susceptibility" in res
        assert "specific_heat" in res
        assert "binder_cumulant" in res
        assert "energies" in res
        assert "magnetizations" in res


def test_xystudy_run_all():
    J = 1.0
    n_steps = 10
    n_equil = 3
    L_list = [4, 6]
    T_list = np.linspace(2.0, 2.4, 3)

    cfg = SimulationConfig(
        J=J,
        n_steps=n_steps,
        n_equil=n_equil,
        L_list=L_list,
        T_list=T_list,
    )
    study = XYStudy(cfg)
    data_basic = study.run_basic_all()
    data_improved = study.run_improved_all()

    for data in (data_basic, data_improved):
        assert isinstance(data, dict)
        for L in L_list:
            assert L in data
            for T in T_list:
                assert T in data[L]
