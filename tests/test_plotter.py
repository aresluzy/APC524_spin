import os
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.xy3d_wolff.plotter import XYPlotter, DataCollapsePlotter


# ======================================================================
# Helper: minimal fake simulation results
# ======================================================================
def make_fake_results(L_list, T_list):
    results = {}
    for L in L_list:
        results[L] = {}
        for T in T_list:
            results[L][T] = {
                "magnetizations": np.array([0.1 * L * T, 0.2 * L * T]),
                "energies": np.array([-L, -0.9 * L]),
                "cluster_sizes": np.array([0.05, 0.07]),
                "susceptibility": (1.0, 0.1),
                "specific_heat": (0.5, 0.05),
                "binder_cumulant": (0.7, 0.02),
                "spin": np.zeros((L, L, L, 2)),
            }
    return results


# ======================================================================
# 1. Test XYPlotter.plot_observables
# ======================================================================
@pytest.mark.parametrize("L_list, T_list", [
    ([4], [1.5, 2.0]),
    ([4, 8], [1.0, 2.0]),
])
def test_plot_observables_pytest(L_list, T_list, monkeypatch):
    results = make_fake_results(L_list, T_list)

    # prevent figure pop-ups
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    # smoke test
    XYPlotter.plot_observables(results, L_list, T_list, T_c=2.2)


# ======================================================================
# 2. Test XYPlotter.plot_spin_orientations
# ======================================================================
def test_plot_spin_orientations_pytest(monkeypatch):
    L = 4
    T = 1.5
    results = make_fake_results([L], [T])

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    XYPlotter.plot_spin_orientations(results, L, T)


# ======================================================================
# 3. Test GIF creation with pytest + full monkeypatching
# ======================================================================
def test_create_spin_orientation_gif_pytest(tmp_path, monkeypatch):
    L = 4
    T_list = [1.0, 2.0]

    results = make_fake_results([L], T_list)

    # mock file I/O so no real files are created
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr("imageio.imread", lambda path: np.zeros((10, 10, 3), dtype=np.uint8))
    monkeypatch.setattr("imageio.mimsave", lambda *args, **kwargs: None)
    monkeypatch.setattr(os, "remove", lambda *args, **kwargs: None)
    monkeypatch.setattr(os, "rmdir", lambda *args, **kwargs: None)

    output_file = tmp_path / "test.gif"

    # run
    XYPlotter.create_spin_orientation_gif(
        simulation_results=results,
        L=L,
        T_list=T_list,
        interval=200,
        output_file=str(output_file)
    )

    # assertion: function completes successfully
    assert True


# ======================================================================
# 4. DataCollapsePlotter: specific heat collapse
# ======================================================================
def test_collapse_specific_heat_pytest(monkeypatch):
    L_list = [4, 8]
    T_list = [1.0, 1.5, 2.0]
    results = make_fake_results(L_list, T_list)

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    collapse = DataCollapsePlotter.data_collapse_specific_heat(
        results, L_list, T_list, Tc=2.2, alpha=0.01, nu=0.67, plot=False
    )

    assert set(collapse.keys()) == set(L_list)
    for L in L_list:
        x, y = collapse[L]
        assert x.shape == (len(T_list),)
        assert y.shape == (len(T_list),)


# ======================================================================
# 5. DataCollapsePlotter: susceptibility collapse
# ======================================================================
def test_collapse_susceptibility_pytest(monkeypatch):
    L_list = [4, 8]
    T_list = [1.0, 2.0]
    results = make_fake_results(L_list, T_list)

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    collapse = DataCollapsePlotter.data_collapse_susceptibility(
        results, L_list, T_list, Tc=2.2, gamma=1.3, nu=0.67, plot=False
    )

    assert set(collapse.keys()) == set(L_list)
    for L in L_list:
        x, y = collapse[L]
        assert x.shape == (len(T_list),)
        assert y.shape == (len(T_list),)


# ======================================================================
# 6. DataCollapsePlotter: magnetization collapse
# ======================================================================
def test_collapse_magnetization_pytest(monkeypatch):
    L_list = [4, 8]
    T_list = [1.0, 1.5]
    results = make_fake_results(L_list, T_list)

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)

    collapse = DataCollapsePlotter.data_collapse_magnetization(
        results, L_list, T_list, Tc=2.2, beta=0.35, nu=0.67, plot=False
    )

    assert set(collapse.keys()) == set(L_list)
    for L in L_list:
        x, y = collapse[L]
        assert x.shape == (len(T_list),)
        assert y.shape == (len(T_list),)
