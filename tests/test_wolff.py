import numpy as np
import pytest
from src.xy3d_wolff.wolff import XYLattice, WolffClusterUpdater


@pytest.mark.parametrize("L", [3,4,6])
def test_xylattice_initialization_norm(L):
    np.random.seed(0)
    lat = XYLattice(L)
    spins = lat.spins
    assert spins.shape == (L,L,L,2)

    norms = np.linalg.norm(spins, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-7)


def test_wolff_update_cluster_properties():
    L,J,T = 4,1.0,2.0
    np.random.seed(1)

    lat = XYLattice(L)
    spins = lat.spins.copy()
    updater = WolffClusterUpdater(J,T)

    cs = updater.wolff_update(spins)
    assert 1 <= cs <= L**3

    norms = np.linalg.norm(spins, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_wolff_update_estimator_outputs():
    L,J,T = 4,1.0,2.0
    np.random.seed(2)

    lat = XYLattice(L)
    spins = lat.spins.copy()
    updater = WolffClusterUpdater(J,T)

    cs, Sq, qv = updater.wolff_update_with_estimator(spins)

    assert 1 <= cs <= L**3
    assert Sq.shape == (3,)
    assert len(qv) == 3
    for q in qv:
        assert q.shape == (3,)
    assert np.all(Sq >= 0)

    norms = np.linalg.norm(spins, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-6)
