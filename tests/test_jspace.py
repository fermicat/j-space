# Unit tests for the nonnegative pursuit (no model needed).
import torch

from jspace_repro.jspace import nonneg_pursuit


def _random_unit_atoms(m: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(m, d, generator=g)
    return D / D.norm(dim=-1, keepdim=True)


def test_recovers_planted_sparse_code():
    D = _random_unit_atoms(200, 128)
    h = 2.0 * D[3] + 1.0 * D[7]
    idx, coeff, ev = nonneg_pursuit(h, D, k=25)
    got = dict(zip(idx, coeff.tolist()))
    assert set(idx) >= {3, 7}
    assert abs(got[3] - 2.0) < 0.05 and abs(got[7] - 1.0) < 0.05
    assert ev > 0.99


def test_nonnegative_and_budget():
    D = _random_unit_atoms(300, 64, seed=1)
    g = torch.Generator().manual_seed(2)
    h = torch.randn(64, generator=g)
    idx, coeff, ev = nonneg_pursuit(h, D, k=25)
    assert len(idx) <= 25
    assert (coeff >= 0).all()
    assert 0.0 <= ev <= 1.0


def test_antiparallel_atom_never_selected():
    D = _random_unit_atoms(10, 32, seed=3)
    h = -1.0 * D[0].clone()
    # Restrict dictionary to just the anti-parallel atom: nothing correlates
    # positively, so the pursuit must select nothing.
    idx, coeff, ev = nonneg_pursuit(h, D[:1], k=25)
    assert idx == []
    assert ev <= 1e-6
