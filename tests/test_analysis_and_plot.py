import numpy as np
import pytest


from src.xy3d_wolff.analysis import XYAnalysis
from src.xy3d_wolff.plotting import XYPlotter
from src.xy3d_wolff.simulation import SimulationConfig, XYStudy
from src.xy3d_wolff.wolff import XYLattice


def test_xyanalysis_autocorr_and_structure_factor():
    series = np.random.rand(50)
    ac, tau_int = XYAnalysis.autocorrelation(series)
    assert len(ac) == len(series)
    assert tau_int >= 0.0

    L = 4
    lat = XYLattice(L)
    G_r, S_q = XYAnalysis.structure_factor_from_spins(lat.spins)
    assert G_r.ndim == 1
    assert S_q.ndim == 1


def test_xyanalysis_data_collapse_smoke():
    J = 1.0
    n_steps = 10
    n_equil = 3
    L_list = [4]
    T_list = np.linspace(2.0, 2.4, 3)

    cfg = SimulationConfig(
        J=J,
        n_steps=n_steps,
        n_equil=n_equil,
        L_list=L_list,
        T_list=T_list,
    )
    study = XYStudy(cfg)
    data = study.run_basic_all()

    Tc = 2.2
    alpha, beta, gamma, nu = 0.0, 0.35, 1.3, 0.67

    c_data = XYAnalysis.data_collapse_specific_heat(
        data, L_list, T_list, Tc, alpha, nu, plot=False
    )
    chi_data = XYAnalysis.data_collapse_susceptibility(
        data, L_list, T_list, Tc, gamma, nu, plot=False
    )
    m_data = XYAnalysis.data_collapse_magnetization(
        data, L_list, T_list, Tc, beta, nu, plot=False
    )

    assert isinstance(c_data, dict)
    assert isinstance(chi_data, dict)
    assert isinstance(m_data, dict)


def test_xyplotter_spins_smoke():
    L = 3
    lat = XYLattice(L)
    XYPlotter.plot_spins(lat.spins)
