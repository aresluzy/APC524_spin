from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from . import core

@dataclass
class XYLattice:
    """
    3D XY spin lattice with helper methods for initialization and copying.
    """

    def __init__(self, L: int):
        """
        Parameters
        ----------
        L : int
            Linear system size.
        """
        self.L = L
        self.spins = self.initialize_lattice(L)

    @staticmethod
    def _initialize_lattice(L: int) -> np.ndarray:
        """
        Create a random XY spin configuration.

        Each spin is (cosθ, sinθ) with θ ∈ [0, 2π).

        Parameters
        ----------
        L : int
        Linear system size.

        Returns
        -------
        ndarray
        Random spins with shape (L, L, L, 2)
        """
        theta = np.random.uniform(0, 2 * np.pi, (L, L, L))
        spins = np.stack((np.cos(theta), np.sin(theta)), axis=-1)  # Shape: (L, L, L, 2)
        return spins


class WolffClusterUpdater:
    """
    Wolff cluster updates for the 3D XY model.

    Parameters
    ----------
    J : float
        Coupling constant.
    T : float
        Temperature.
    """

    def __init__(self, J: float, T: float):
        self.J = J
        self.T = T

    def step(self, spins: np.ndarray) -> int:
        """
        Perform one Wolff cluster update using the standard update.

        Parameters
        ----------
        spins : ndarray
            Spin configuration array of shape (L, L, L, 2).

        Returns
        -------
        int
            Size of the cluster that was flipped.
        """
        return core.wolff_update(spins, self.J, self.T)

    def step_new(self, spins: np.ndarray) -> int:
        """
        Perform one Wolff update using wolff_update_new.

        Parameters
        ----------
        spins : ndarray
            Spin configuration array.

        Returns
        -------
        int
            Size of the updated cluster.
        """
        return core.wolff_update_new(spins, self.J, self.T)

    def step_with_estimator(
        self, spins: np.ndarray
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Perform one Wolff update and return improved-estimator data.

        Parameters
        ----------
        spins : ndarray
            Spin configuration array.

        Returns
        -------
        tuple
            (cluster_size, cluster_Sq, q_vectors) as in
            wolff_update_with_estimator.
        """
        return core.wolff_update_with_estimator(spins, self.J, self.T)


