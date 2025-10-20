import numpy as np
from apc524_spin.lattice import init_lattice

def test_initialize_shape_and_norm():
    L=8
    S = init_lattice(L, seed=0)
    assert S.shape == (L, L, L, 2)
    norms = np.linalg.norm(S, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-7)
