from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Dict, Any

import numpy as np

from .core import (
    run_simulation,
    run_improved_simulation,
    simulate_all_data,
    improved_simulate_all_data,
    plot_simulation_results,
    data_collapse_specific_heat,
    data_collapse_susceptibility,
    data_collapse_magnetization,
)


@dataclass
class SimulationConfig:
    """
    Configuration for XY model simulations.

    Attributes
    ----------
    J : float
        Coupling constant.
    n_steps : int
        Total Wolff updates per temperature.
    n_equil : int
        Number of equilibration steps discarded.
    L_list : sequence of int
        System sizes to simulate.
    T_list : sequence of float
        Temperatures to simulate.
    """
    J: float
    n_steps: int
    n_equil: int
    L_list: Sequence[int]
    T_list: Sequence[float]


class SimulationRunner:
    """
    High-level interface for running simulations and plotting results.

    This class wraps the original functions in core.py. It does not
    change any computational details.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    @property
    def J(self) -> float:
        return self.config.J

    @property
    def n_steps(self) -> int:
        return self.config.n_steps

    @property
    def n_equil(self) -> int:
        return self.config.n_equil

    @property
    def L_list(self) -> Sequence[int]:
        return self.config.L_list

    @property
    def T_list(self) -> np.ndarray:
        return np.array(self.config.T_list, dtype=float)

    def run_single(self, L: int, T: float) -> Dict[str, Any]:
        """
        Run a single system size and temperature.

        Parameters
        ----------
        L : int
            Linear system size.
        T : float
            Temperature.

        Returns
        -------
        dict
            Result dictionary from run_simulation.
        """
        return run_simulation(L, self.J, T, self.n_steps, self.n_equil)

    def run_single_improved(self, L: int, T: float) -> Dict[str, Any]:
        """
        Run a single system size and temperature using improved estimators.

        Returns
        -------
        dict
            Result dictionary from run_improved_simulation.
        """
        return run_improved_simulation(L, self.J, T, self.n_steps, self.n_equil)

    def run_basic(self) -> Dict[int, Dict[float, Any]]:
        """
        Run the basic simulation over all L and T in the configuration.

        Returns
        -------
        dict
            Nested dictionary as returned by simulate_all_data.
        """
        return simulate_all_data(
            self.L_list,
            self.T_list,
            self.J,
            self.n_steps,
            self.n_equil,
        )

    def run_with_estimator(self) -> Dict[int, Dict[float, Any]]:
        """
        Run the improved-estimator simulation over all L and T.

        Returns
        -------
        dict
            Nested dictionary as returned by improved_simulate_all_data.
        """
        return improved_simulate_all_data(
            self.L_list,
            self.T_list,
            self.J,
            self.n_steps,
            self.n_equil,
        )

    def plot_basic(self, simulation_results: Dict[int, Dict[float, Any]]) -> None:
        """
        Reproduce the main observables plots.

        Parameters
        ----------
        simulation_results : dict
            Output from run_basic or simulate_all_data.
        """
        plot_simulation_results(simulation_results, self.L_list, self.T_list)

    def collapse_specific_heat(
        self,
        simulation_results: Dict[int, Dict[float, Any]],
        Tc: float,
        alpha: float,
        nu: float,
        plot: bool = True,
    ):
        """
        Perform finite-size scaling collapse for specific heat.

        Returns
        -------
        dict
            Collapsed data structure from data_collapse_specific_heat.
        """
        return data_collapse_specific_heat(
            simulation_results,
            self.L_list,
            self.T_list,
            Tc,
            alpha,
            nu,
            plot=plot,
        )

    def collapse_susceptibility(
        self,
        simulation_results: Dict[int, Dict[float, Any]],
        Tc: float,
        gamma: float,
        nu: float,
        plot: bool = True,
    ):
        """
        Perform finite-size scaling collapse for susceptibility.
        """
        return data_collapse_susceptibility(
            simulation_results,
            self.L_list,
            self.T_list,
            Tc,
            gamma,
            nu,
            plot=plot,
        )

    def collapse_magnetization(
        self,
        simulation_results: Dict[int, Dict[float, Any]],
        Tc: float,
        beta: float,
        nu: float,
        plot: bool = True,
    ):
        """
        Perform finite-size scaling collapse for magnetization.
        """
        return data_collapse_magnetization(
            simulation_results,
            self.L_list,
            self.T_list,
            Tc,
            beta,
            nu,
            plot=plot,
        )
