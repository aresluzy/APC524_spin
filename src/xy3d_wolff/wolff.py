from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .core import wolff_update, wolff_update_with_estimator


@dataclass
class WolffParameters:
    """
    Parameters for the Wolff cluster algorithm.

    Attributes
    ----------
    J : float
        Coupling constant.
    T : float
        Temperature.
    """
    J: float
    T: float


class WolffClusterUpdater:
    """
    Object-oriented wrapper for Wolff cluster updates.

    This class delegates directly to the original functions in core.py
    so that the physics and results remain unchanged.
    """

    def __init__(self, params: WolffParameters):
        self.params = params

    @property
    def J(self) -> float:
        return self.params.J

    @property
    def T(self) -> float:
        return self.params.T

    def step(self, spins: np.ndarray) -> int:
        """
        Perform a single Wolff cluster update.

        Parameters
        ----------
        spins : ndarray
            Spin configuration of shape (L, L, L, 2).

        Returns
        -------
        int
            Size of the flipped cluster.
        """
        return wolff_update(spins, self.J, self.T)

    def step_with_estimator(
        self, spins: np.ndarray
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Perform a single Wolff cluster update using the improved estimator.

        Parameters
        ----------
        spins : ndarray
            Spin configuration of shape (L, L, L, 2).

        Returns
        -------
        tuple
            (cluster_size, cluster_Sq, q_vectors) exactly as returned by
            wolff_update_with_estimator in core.py.
        """
        return wolff_update_with_estimator(spins, self.J, self.T)

