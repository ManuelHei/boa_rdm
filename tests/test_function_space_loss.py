"""The function-space loss must equal the integral it claims to compute.

``function_space_loss`` evaluates ``tr(dD S dD S)`` and asserts that this is
``int dr dr' (gamma - gamma_pred)^2``. That identity is the whole justification
for the closed form, and it is cheap to check against the four-index sum it
replaces, so it is checked rather than trusted.
"""

import numpy as np
import pyscf
import pytest
import torch

from boa.data.basis_info import BasisInfo
from boa.model.rdm_module import RDMLightningModule

WATER = [("O", [0.0, 0.0, 0.117]), ("H", [0.0, 0.757, -0.469]), ("H", [0.0, -0.757, -0.469])]


@pytest.fixture(scope="module")
def overlap_and_basis():
    basis_info = BasisInfo.from_atomic_numbers_with_even_tempered_basis(
        atomic_numbers=[1, 8], basis="6-31g*", beta=None, even_tempered=False, uncontracted=False
    )
    mol = pyscf.gto.M(atom=WATER, basis=basis_info.basis_dict, unit="Angstrom")
    return mol.intor("int1e_ovlp"), basis_info


def test_closed_form_equals_the_four_index_integral(overlap_and_basis):
    """tr(dD S dD S) == sum_{uvab} dD_uv dD_ab S_ua S_vb, the definition."""
    overlap, _ = overlap_and_basis
    rng = np.random.default_rng(0)
    n = overlap.shape[0]
    for _ in range(5):
        a = rng.normal(size=(n, n))
        delta = (a + a.T) / 2  # the prediction and reference are both symmetrised
        definition = np.einsum("uv,ab,ua,vb->", delta, delta, overlap, overlap)
        closed_form = np.trace(delta @ overlap @ delta @ overlap)
        assert np.isclose(closed_form, definition, rtol=1e-12)


def test_matches_the_lowdin_form(overlap_and_basis):
    """And equals ||S^1/2 dD S^1/2||_F^2, which is what makes the meaning plain."""
    overlap, _ = overlap_and_basis
    rng = np.random.default_rng(1)
    n = overlap.shape[0]
    a = rng.normal(size=(n, n))
    delta = (a + a.T) / 2
    evals, vectors = np.linalg.eigh(overlap)
    root = (vectors * np.sqrt(np.clip(evals, 0, None))) @ vectors.T
    assert np.isclose(
        np.linalg.norm(root @ delta @ root) ** 2,
        np.trace(delta @ overlap @ delta @ overlap),
        rtol=1e-10,
    )


def test_module_implementation_matches_the_definition(overlap_and_basis):
    """The batched implementation, including the padding mask, against the sum."""
    overlap, _ = overlap_and_basis
    n = overlap.shape[0]
    rng = np.random.default_rng(2)

    # two molecules of different size in one batch: the second is padded, and
    # the padded rows must contribute nothing
    small = n - 4
    ao_mask = torch.zeros(2, n, dtype=torch.bool)
    ao_mask[0, :] = True
    ao_mask[1, :small] = True

    pred = torch.zeros(2, n, n, dtype=torch.float64)
    target = torch.zeros(2, n, n, dtype=torch.float64)
    for b, size in enumerate((n, small)):
        p = rng.normal(size=(size, size))
        t = rng.normal(size=(size, size))
        pred[b, :size, :size] = torch.as_tensor((p + p.T) / 2)
        target[b, :size, :size] = torch.as_tensor((t + t.T) / 2)

    stacked = torch.zeros(2, n, n, dtype=torch.float64)
    stacked[0] = torch.as_tensor(overlap)
    stacked[1, :small, :small] = torch.as_tensor(overlap[:small, :small])

    class Batch:
        message_passing_matrix = stacked

    # borrow just the two methods rather than building a LightningModule, which
    # would need a full hydra config; they touch no other state
    class Module:
        overlap = RDMLightningModule.overlap
        function_space_loss = RDMLightningModule.function_space_loss

    loss = Module().function_space_loss(Batch(), pred, target, ao_mask)

    expected = []
    for b, size in enumerate((n, small)):
        d = (pred[b, :size, :size] - target[b, :size, :size]).numpy()
        s = overlap[:size, :size]
        expected.append(np.einsum("uv,ab,ua,vb->", d, d, s, s))
    assert np.isclose(float(loss), float(np.mean(expected)), rtol=1e-10)
