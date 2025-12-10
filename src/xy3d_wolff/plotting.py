from __future__ import annotations

from typing import Dict, Any, Sequence

from .core import (
    plot_spin_orientations,
    plot_simulation_results,
    data_collapse_specific_heat,
    data_collapse_susceptibility,
    data_collapse_magnetization,
)


def plot_spins(spins) -> None:
    """
    Plot a 3D quiver of the spin orientations.

    Parameters
    ----------
    spins : ndarray
        Array of shape (L, L, L, 2) with XY spins.
    """
    plot_spin_orientations(spins)


def plot_results(
    simulation_results: Dict[int, Dict[float, Any]],
    L_list: Sequence[int],
    T_list,
) -> None:
    """
    Plot main observables vs temperature for multiple system sizes.

    This is a thin wrapper around plot_simulation_results.
    """
    plot_simulation_results(simulation_results, L_list, T_list)


def collapse_all(
    simulation_results: Dict[int, Dict[float, Any]],
    L_list: Sequence[int],
    T_list,
    Tc: float,
    alpha: float,
    beta: float,
    gamma: float,
    nu: float,
):
    """
    Convenience function to call all three data-collapse routines.

    Returns
    -------
    tuple
        (C_data, chi_data, M_data) from the underlying collapse functions.
    """
    c_data = data_collapse_specific_heat(
        simulation_results, L_list, T_list, Tc, alpha, nu
    )
    chi_data = data_collapse_susceptibility(
        simulation_results, L_list, T_list, Tc, gamma, nu
    )
    m_data = data_collapse_magnetization(
        simulation_results, L_list, T_list, Tc, beta, nu
    )
    return c_data, chi_data, m_data
