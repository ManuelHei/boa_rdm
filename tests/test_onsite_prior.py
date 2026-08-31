"""The on-site prior has to be a constant the model may add without losing equivariance.

Adding anything to a predicted density matrix risks breaking the property the
whole architecture is built to have: rotate the molecule, and the matrix has to
rotate with it. A constant survives that only if it commutes with the AO
transform, which for a diagonal block means being a multiple of the identity on
each same-``l`` shell pair and zero everywhere else.

:func:`test_prior_is_rotation_invariant` is the check that matters -- it takes
the prior measured from arbitrary blocks and rotates it with the same AO
transform ``test_helm_rdm`` pins against pyscf. The rest fix the layout: which
shell pairs exist, that the measured value is the mean of the invariant part,
and that the placement puts each atom's prior on its own diagonal block.
"""

import numpy as np
import pytest
import torch
from e3nn import o3
from scipy.linalg import block_diag
from types import SimpleNamespace

from boa.data.basis_info import BasisInfo
from boa.model.net.helm_net import pyscf_shell_permutation
from boa.model.onsite import OnsiteL0Statistics, same_l_shell_pairs
from boa.model.rdm_module import RDMLightningModule

#: physical (x, y, z) -> the axis order e3nn's spherical harmonics are built in.
XYZ_TO_E3NN = torch.tensor([[0.0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float64)


@pytest.fixture(scope="module")
def basis_info():
    """H and O in 6-31G*: two shells on H, six on O, so both l=0-only and l up to 2."""
    return BasisInfo.from_atomic_numbers_with_even_tempered_basis(
        atomic_numbers=[1, 8],
        basis="6-31g*",
        beta=None,
        even_tempered=False,
        uncontracted=False,
    )


def atom_rotation(basis_info: BasisInfo, atomic_number: int, rotation: np.ndarray) -> np.ndarray:
    """Block-diagonal AO transform for one atom, in pyscf order.

    The same construction as ``tests/test_helm_rdm.atom_rotation``, which is
    checked there against pyscf's own overlap matrix; duplicated rather than
    imported so neither test file depends on the other.
    """
    irreps = basis_info.irreps_per_atom[basis_info.atomic_number_to_atom_index[atomic_number]]
    ls = [ir.l for mul, ir in irreps for _ in range(mul)]
    rotation_e3nn = XYZ_TO_E3NN @ torch.as_tensor(rotation, dtype=torch.float64) @ XYZ_TO_E3NN.T
    blocks = []
    for l in ls:
        wigner = o3.Irrep(l, (-1) ** l).D_from_matrix(rotation_e3nn).numpy()
        q = pyscf_shell_permutation(l)
        blocks.append(q @ wigner @ q.T)
    return block_diag(*blocks)


def random_blocks(basis_info, atom_index, n_atom, seed=0):
    """Symmetric on-site blocks with no structure at all, for the invariance check."""
    rng = np.random.default_rng(seed)
    dim = int(max(basis_info.basis_dim_per_atom))
    blocks = rng.normal(size=(n_atom, dim, dim))
    return 0.5 * (blocks + blocks.transpose(0, 2, 1))


def test_shell_pairs_cover_the_basis(basis_info):
    """Every AO is in exactly one shell, and pairs are formed only within an l."""
    for t, z in enumerate(basis_info.atomic_numbers):
        pairs = same_l_shell_pairs(basis_info, t)
        diagonal = [(l, rows) for l, rows, cols in pairs if rows[0] == cols[0]]
        covered = np.concatenate([rows for _, rows in diagonal])
        assert sorted(covered) == list(range(int(basis_info.basis_dim_per_atom[t]))), z
        for l, rows, cols in pairs:
            assert len(rows) == len(cols) == 2 * l + 1

    # H in 6-31G* is 2s: two shells, four ordered pairs. O is 3s2p1d: six
    # shells, 9 + 4 + 1 = 14 pairs.
    assert len(same_l_shell_pairs(basis_info, 0)) == 4
    assert len(same_l_shell_pairs(basis_info, 1)) == 14


def test_prior_is_rotation_invariant(basis_info):
    """``T M T^T == M``: the prior is the same constant in every orientation."""
    stats = OnsiteL0Statistics(basis_info)
    atom_index = np.array([0, 1, 1, 0])
    stats.update(atom_index, random_blocks(basis_info, atom_index, len(atom_index)))
    bias = stats.bias().numpy()

    torch.manual_seed(0)
    rotation = o3.rand_matrix().numpy().astype(np.float64)
    for t, z in enumerate(basis_info.atomic_numbers):
        dim = int(basis_info.basis_dim_per_atom[t])
        prior = bias[t, :dim, :dim]
        transform = atom_rotation(basis_info, int(z), rotation)
        assert np.allclose(transform @ prior @ transform.T, prior, atol=1e-10), z


def test_prior_is_the_mean_invariant_part(basis_info):
    """The measured value is the mean shell-pair trace, and only that."""
    stats = OnsiteL0Statistics(basis_info)
    dim = int(max(basis_info.basis_dim_per_atom))
    pairs = same_l_shell_pairs(basis_info, 1)

    # One oxygen whose block is a known multiple of the identity on every
    # same-l pair, plus a traceless part the prior has to ignore.
    block = np.zeros((dim, dim))
    for k, (l, rows, cols) in enumerate(pairs):
        block[rows, cols] = 0.5 * k
    traceless = np.zeros((dim, dim))
    l1_rows = pairs[[i for i, (l, r, c) in enumerate(pairs) if l == 1 and r[0] == c[0]][0]][1]
    traceless[l1_rows[0], l1_rows[0]] = 1.0
    traceless[l1_rows[1], l1_rows[1]] = -1.0

    stats.update(np.array([1]), (block + traceless)[None])
    assert np.allclose(stats.means(1), [0.5 * k for k in range(len(pairs))], atol=1e-12)

    bias = stats.bias().numpy()[1, : int(basis_info.basis_dim_per_atom[1]), :]
    assert np.allclose(bias, 0.5 * (block + block.T), atol=1e-6)


def test_bias_is_zero_outside_the_element(basis_info):
    """An element with fewer AOs than ``max_basis_dim`` gets nothing in the padding."""
    stats = OnsiteL0Statistics(basis_info)
    atom_index = np.array([0, 1])
    stats.update(atom_index, random_blocks(basis_info, atom_index, 2, seed=1))
    bias = stats.bias().numpy()
    dim_h = int(basis_info.basis_dim_per_atom[0])
    assert np.allclose(bias[0, dim_h:, :], 0) and np.allclose(bias[0, :, dim_h:], 0)


def test_prior_lands_on_the_right_diagonal_block(basis_info):
    """Each atom's prior goes to its own block, in its own molecule."""
    dim = int(max(basis_info.basis_dim_per_atom))
    n_types = len(basis_info.atomic_numbers)
    onsite_bias = torch.zeros(n_types, dim, dim)
    onsite_bias[0, :2, :2] = torch.eye(2)  # H, marked
    onsite_bias[1] = 2.0 * torch.eye(dim)  # O, marked differently

    # Two molecules: H2O and H2, so the AO counts differ and one is padded.
    batch = SimpleNamespace(
        atomic_numbers=torch.tensor([8, 1, 1, 1, 1]),
        batch=torch.tensor([0, 0, 0, 1, 1]),
    )
    module = SimpleNamespace(
        onsite_bias=onsite_bias,
        atomic_number_to_atom_index=torch.as_tensor(
            basis_info.atomic_number_to_atom_index, dtype=torch.long
        ),
    )
    n_ao = 18  # O(14) + H(2) + H(2)
    ao_index = torch.tensor(
        [
            list(range(14)) + [n_ao] * (dim - 14),   # O, AOs 0..13
            [14, 15] + [n_ao] * (dim - 2),           # H
            [16, 17] + [n_ao] * (dim - 2),           # H
            [0, 1] + [n_ao] * (dim - 2),             # H of the second molecule
            [2, 3] + [n_ao] * (dim - 2),             # H
        ]
    )
    prior = RDMLightningModule.onsite_prior(
        module, batch, ao_index, n_mol=2, n_ao=n_ao, like=torch.zeros(())
    )

    assert prior.shape == (2, n_ao, n_ao)
    assert torch.allclose(prior[0, :14, :14], 2.0 * torch.eye(14))
    assert torch.allclose(prior[0, 14:18, 14:18], torch.eye(4))
    assert torch.allclose(prior[1, :4, :4], torch.eye(4))
    assert torch.allclose(prior[1, 4:, :], torch.zeros(n_ao - 4, n_ao))
    # Nothing off the diagonal blocks.
    assert torch.allclose(prior[0, :14, 14:], torch.zeros(14, 4))
