"""Convention and equivariance checks for the HELM density-matrix head.

The head assembles matrix blocks from irreps with Wigner 3j symbols, which is
only correct if two orderings agree: e3nn's axis convention for spherical
harmonics, and pyscf's ordering of the AOs within a shell. Neither shows up as a
crash if it is wrong -- the model trains, just towards a target it cannot
represent -- so both are pinned down here.

:func:`test_ao_rotation_matches_pyscf` checks the conventions against pyscf
itself, with no network involved: the rotation matrix built from them has to
carry the overlap matrix of a molecule onto that of the rotated molecule.
:func:`test_blocks_are_equivariant` then checks that the network's blocks
actually rotate that way.
"""

import numpy as np
import pyscf
import pytest
import torch
from e3nn import o3
from scipy.linalg import block_diag

from boa.data.basis_info import BasisInfo
from boa.model.net.helm_net import HelmRDM, pyscf_shell_permutation

#: physical (x, y, z) -> the axis order e3nn's spherical harmonics are built in.
XYZ_TO_E3NN = torch.tensor([[0.0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float64)


@pytest.fixture(scope="module")
def basis_info():
    """H and O in 6-31G*, the basis the QMLearn hackathon db was built with."""
    return BasisInfo.from_atomic_numbers_with_even_tempered_basis(
        atomic_numbers=[1, 8],
        basis="6-31g*",
        beta=None,
        even_tempered=False,
        uncontracted=False,
    )


def shell_rotation(l: int, rotation: np.ndarray) -> np.ndarray:
    """How the AOs of one shell of degree ``l`` transform under a rotation, in pyscf order."""
    rotation_e3nn = XYZ_TO_E3NN @ torch.as_tensor(rotation, dtype=torch.float64) @ XYZ_TO_E3NN.T
    wigner = o3.Irrep(l, 1).D_from_matrix(rotation_e3nn).numpy()
    q = pyscf_shell_permutation(l)
    return q @ wigner @ q.T


def atom_rotation(basis_info: BasisInfo, atomic_number: int, rotation: np.ndarray) -> np.ndarray:
    """Block-diagonal AO rotation for one atom, over its shells in pyscf order."""
    irreps = basis_info.irreps_per_atom[basis_info.atomic_number_to_atom_index[atomic_number]]
    ls = [ir.l for mul, ir in irreps for _ in range(mul)]
    return block_diag(*[shell_rotation(l, rotation) for l in ls])


WATER = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]])
WATER_Z = [8, 1, 1]


def test_ao_rotation_matches_pyscf(basis_info):
    """``T S T^T`` has to be the overlap matrix of the rotated molecule.

    This is the ground truth for both conventions at once. The overlap matrix
    involves every shell in the basis, d functions included, so a wrong
    permutation or sign anywhere shows up here.
    """
    torch.manual_seed(0)
    rotation = o3.rand_matrix().numpy().astype(np.float64)
    symbols = [pyscf.data.elements.ELEMENTS[z] for z in WATER_Z]

    mol = pyscf.gto.M(atom=list(zip(symbols, WATER)), basis=basis_info.basis_dict, unit="Angstrom")
    mol_rotated = pyscf.gto.M(
        atom=list(zip(symbols, WATER @ rotation.T)),
        basis=basis_info.basis_dict,
        unit="Angstrom",
    )

    transform = block_diag(*[atom_rotation(basis_info, z, rotation) for z in WATER_Z])
    overlap = mol.intor("int1e_ovlp")
    assert np.allclose(
        transform @ overlap @ transform.T, mol_rotated.intor("int1e_ovlp"), atol=1e-4
    )


def _batch(positions: np.ndarray):
    """The fields :class:`HelmRDM` reads, for a single molecule."""
    n = len(positions)

    class Batch:
        pos = torch.as_tensor(positions, dtype=torch.get_default_dtype())
        atomic_numbers = torch.as_tensor(WATER_Z)
        batch = torch.zeros(n, dtype=torch.long)
        num_graphs = 1
        # Dense graph with self-loops, as the radius transform produces for a
        # molecule this small. Written out rather than recomputed so the edge
        # order is identical before and after the rotation.
        edge_index = torch.tensor(
            [[i for i in range(n) for _ in range(n)], [j for _ in range(n) for j in range(n)]]
        )

    return Batch()


def test_blocks_are_equivariant(basis_info):
    """Rotating the molecule rotates every predicted block: ``B_ij -> T_i B_ij T_j^T``."""
    torch.manual_seed(0)
    net = HelmRDM(
        basis_info,
        sphere_channels=16,
        hidden_channels=16,
        edge_channels=16,
        num_distance_basis=16,
        num_layers=2,
    ).double()
    net.eval()
    torch.set_default_dtype(torch.float64)
    try:
        rotation = o3.rand_matrix().numpy().astype(np.float64)

        with torch.no_grad():
            blocks, edge_index = net(_batch(WATER))
            blocks_rotated, edge_index_rotated = net(_batch(WATER @ rotation.T))
        assert torch.equal(edge_index, edge_index_rotated)

        transforms = [torch.as_tensor(atom_rotation(basis_info, z, rotation)) for z in WATER_Z]
        for k, (i, j) in enumerate(edge_index.T.tolist()):
            n_i, n_j = transforms[i].shape[0], transforms[j].shape[0]
            block = blocks[k, :n_i, :n_j].double()
            expected = transforms[i] @ block @ transforms[j].T
            assert torch.allclose(blocks_rotated[k, :n_i, :n_j].double(), expected, atol=1e-8), (
                f"block {(i, j)} is not equivariant"
            )
    finally:
        torch.set_default_dtype(torch.float32)


def test_head_spans_every_block_entry(basis_info):
    """The assembly has to reach every entry of a block, or part of D is unreachable.

    A missing shell pair or a dropped irrep would leave rows of the block
    permanently zero, capping the achievable error at something that looks like
    a training problem rather than a modelling one.
    """
    net = HelmRDM(
        basis_info,
        sphere_channels=8,
        hidden_channels=8,
        edge_channels=8,
        num_distance_basis=8,
        num_layers=1,
    )

    reach = net.edge_head.assembly.abs().sum(dim=0)
    assert (reach > 0).all(), "edge head cannot reach every entry of a block"

    # The node head is symmetric by construction, so it spans the symmetric
    # matrices only -- every entry is still reachable.
    reach = net.node_head.assembly.abs().sum(dim=0)
    assert (reach > 0).all(), "node head cannot reach every entry of a block"
    blocks = net.node_head.assembly
    assert torch.allclose(blocks, blocks.transpose(1, 2), atol=1e-6)
