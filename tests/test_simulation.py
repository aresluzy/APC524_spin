import numpy as np
import pytest
from src.xy3d_wolff.simulation import XYSimulation, XYStudy


# Helper: valid error bar (allow NaN)
def valid_err(x):
    return np.isnan(x) or x >= 0.0


# --------------------------------------------------------------------
# Basic run_simulation
# --------------------------------------------------------------------
@pytest.mark.parametrize("L,T,n_steps,n_equil", [
    (4, 2.0, 5, 2),
    (6, 1.5, 3, 1),
])
def test_run_simulation_basic(L, T, n_steps, n_equil):
    np.random.seed(0)
    sim = XYSimulation(L, T, J=1.0, n_steps=n_steps, n_equil=n_equil)
    out = sim.run_simulation()

    keys = ["magnetizations","energies","cluster_sizes",
            "susceptibility","specific_heat","binder_cumulant","spin"]
    for k in keys:
        assert k in out

    assert out["spin"].shape == (L,L,L,2)
    assert len(out["magnetizations"]) == n_steps
    assert np.isfinite(out["susceptibility"][0])
    assert valid_err(out["susceptibility"][1])


# --------------------------------------------------------------------
# run_improved
# --------------------------------------------------------------------
def test_run_improved_basic():
    L, T = 4, 2.0
    sim = XYSimulation(L, T, J=1.0, n_steps=4, n_equil=2)
    out = sim.run_improved()

    keys = ["magnetizations","energies","cluster_sizes",
            "m2_improved","autocorrelation_time",
            "susceptibility","specific_heat","binder_cumulant","spins"]
    for k in keys:
        assert k in out

    assert len(out["m2_improved"]) == 4
    assert np.isfinite(out["susceptibility"][0])
    assert valid_err(out["susceptibility"][1])


# --------------------------------------------------------------------
# XYStudy multi-run
# --------------------------------------------------------------------
def test_xystudy_run_simulation_all():
    L_list = [4,6]
    T_list = [1.5, 2.0]
    study = XYStudy(J=1.0, n_steps=3, n_equil=1, L_list=L_list, T_list=T_list)

    out = study.run_simulation_all()
    assert set(out.keys()) == set(L_list)

    for L in L_list:
        for T in T_list:
            assert "magnetizations" in out[L][T]


def test_xystudy_run_improved_all():
    L_list = [4,6]
    T_list = [1.5,2.0]
    study = XYStudy(J=1.0, n_steps=3, n_equil=1, L_list=L_list, T_list=T_list)

    out = study.run_improved_all()
    assert set(out.keys()) == set(L_list)

    for L in L_list:
        for T in T_list:
            assert "m2_improved" in out[L][T]
            assert len(out[L][T]["m2_improved"]) == 3
