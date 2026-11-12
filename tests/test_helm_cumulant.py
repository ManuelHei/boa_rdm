"""The joint 1-RDM / cumulant model: aux-basis conventions, equivariance, losses.

The cumulant lives on a second basis, cc-pvdz-jkfit, with f functions the AO
basis never had. The head assembles its blocks with the same Wigner-3j
construction as for ``D``, which is only right if pyscf orders the aux functions
within a shell the way the head assumes -- for f as well as s, p and d. As with
``D``, nothing crashes if it is wrong, so it is checked against pyscf directly:
the transform built from the conventions has to carry the aux Coulomb metric of
a molecule onto that of the rotated molecule.
"""

import numpy as np
import pyscf
import pytest
import torch
from e3nn import o3
from pyscf import df
from scipy.linalg import block_diag

from boa.data.aux_basis import AuxBasisLayout
from boa.data.basis_info import BasisInfo
from boa.model.helm_cumulant_module import HelmCumulantLightningModule
from boa.model.net.helm_cumulant import HelmRDMCumulant
from boa.model.net.helm_net import pyscf_shell_permutation

XYZ_TO_E3NN = torch.tensor([[0.0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float64)

# Non-planar, so no rotation accidentally maps the molecule onto itself.
WATER = np.array([[0.0, 0.0, 0.117], [0.25, 0.757, -0.469], [-0.1, -0.757, -0.469]])
WATER_Z = [8, 1, 1]


@pytest.fixture(scope="module")
def basis_info():
    return BasisInfo.from_atomic_numbers_with_even_tempered_basis(
        atomic_numbers=[1, 8], basis="6-31g*", beta=None, even_tempered=False, uncontracted=False
    )


@pytest.fixture(scope="module")
def aux_basis():
    return AuxBasisLayout([1, 8], auxbasis="cc-pvdz-jkfit", basis="6-31g*")


def shell_rotation(l: int, rotation: np.ndarray) -> np.ndarray:
    rotation_e3nn = XYZ_TO_E3NN @ torch.as_tensor(rotation, dtype=torch.float64) @ XYZ_TO_E3NN.T
    wigner = o3.Irrep(l, (-1) ** l).D_from_matrix(rotation_e3nn).numpy()
    q = pyscf_shell_permutation(l)
    return q @ wigner @ q.T


def aux_atom_rotation(aux_basis, z, rotation):
    ls = aux_basis.shell_degrees[aux_basis.atomic_numbers.index(z)]
    return block_diag(*[shell_rotation(l, rotation) for l in ls])


def _mol(positions):
    symbols = [pyscf.data.elements.ELEMENTS[z] for z in WATER_Z]
    return pyscf.gto.M(atom=list(zip(symbols, positions)), basis="6-31g*", unit="Angstrom")


def test_layout_matches_pyscf(aux_basis):
    """Function counts and shells per element are pyscf's own."""
    assert aux_basis.basis_dim_per_atom.tolist() == [23, 70]
    assert aux_basis.shell_degrees[1] == [0] * 10 + [1] * 7 + [2] * 5 + [3] * 2
    assert df.addons.make_auxmol(_mol(WATER), "cc-pvdz-jkfit").nao_nr() == 70 + 2 * 23


def _aux_rotation_error(aux_basis, rotation, shell_rotation_fn=shell_rotation):
    metric = aux_basis.coulomb_metric(_mol(WATER))
    metric_rotated = aux_basis.coulomb_metric(_mol(WATER @ rotation.T))
    transform = block_diag(
        *[
            block_diag(*[shell_rotation_fn(l, rotation) for l in aux_basis.shell_degrees[
                aux_basis.atomic_numbers.index(z)]])
            for z in WATER_Z
        ]
    )
    return np.abs(transform @ metric @ transform.T - metric_rotated).max() / np.abs(metric).max()


def test_aux_rotation_matches_pyscf(aux_basis):
    """``T M T^T`` has to be the aux Coulomb metric of the rotated molecule, f included.

    The tolerance is e3nn's, not pyscf's: ``D_from_matrix`` satisfies
    ``D(R1 R2) = D(R1) D(R2)`` only to ~1e-6 even in float64 (measured 5e-7 at
    l = 1 up to 1.5e-6 at l = 3), while the s-s blocks, which involve no Wigner
    matrix, come out exact. A convention error is orders of magnitude larger,
    which :func:`test_aux_rotation_check_is_sensitive` pins down.
    """
    torch.manual_seed(0)
    rotation = o3.rand_matrix(dtype=torch.float64).numpy()
    assert _aux_rotation_error(aux_basis, rotation) < 1e-6


def test_aux_rotation_check_is_sensitive(aux_basis):
    """With pyscf's p ordering ignored, the same check fails by orders of magnitude."""

    def wrong(l, rotation):
        rotation_e3nn = (
            XYZ_TO_E3NN @ torch.as_tensor(rotation, dtype=torch.float64) @ XYZ_TO_E3NN.T
        )
        return o3.Irrep(l, (-1) ** l).D_from_matrix(rotation_e3nn).numpy()

    torch.manual_seed(0)
    rotation = o3.rand_matrix(dtype=torch.float64).numpy()
    assert _aux_rotation_error(aux_basis, rotation, wrong) > 1e-2


def _batch(positions):
    n = len(positions)

    class Batch:
        pos = torch.as_tensor(positions, dtype=torch.get_default_dtype())
        atomic_numbers = torch.as_tensor(WATER_Z)
        batch = torch.zeros(n, dtype=torch.long)
        num_graphs = 1
        edge_index = torch.tensor(
            [[i for i in range(n) for _ in range(n)], [j for _ in range(n) for j in range(n)]]
        )

    return Batch()


def _net(basis_info, aux_basis, zero_init=False):
    """Random cumulant heads by default: zero ones would pass every equivariance
    test trivially, since a zero block rotates to a zero block."""
    torch.manual_seed(0)
    return (
        HelmRDMCumulant(
            basis_info=basis_info,
            aux_basis=aux_basis,
            zero_init_cumulant=zero_init,
            sphere_channels=8,
            hidden_channels=8,
            edge_channels=8,
            num_distance_basis=8,
            num_layers=1,
        )
        .double()
        .eval()
    )


def test_defaults_to_lmax_six(basis_info, aux_basis):
    """The f-f aux blocks need L = 6; the AO basis alone would have settled for 4."""
    net = _net(basis_info, aux_basis)
    assert net.lmax == 6
    assert net.aux_dim == 70 and net.dim == 14


def test_cumulant_blocks_are_rotation_equivariant(basis_info, aux_basis):
    """Every cumulant block rotates as ``B_ij -> T_i B_ij T_j^T`` in the aux basis."""
    torch.set_default_dtype(torch.float64)
    try:
        net = _net(basis_info, aux_basis)
        torch.manual_seed(1)
        for _ in range(3):
            rotation = o3.rand_matrix().double().numpy()
            with torch.no_grad():
                blocks, edge_index = net(_batch(WATER))["cumulant"]
                moved, edge_index_moved = net(_batch(WATER @ rotation.T))["cumulant"]
            assert torch.equal(edge_index, edge_index_moved)
            transforms = [
                torch.as_tensor(aux_atom_rotation(aux_basis, z, rotation)) for z in WATER_Z
            ]
            worst = 0.0
            for k, (i, j) in enumerate(edge_index.T.tolist()):
                n_i, n_j = transforms[i].shape[0], transforms[j].shape[0]
                expected = transforms[i] @ blocks[k, :n_i, :n_j] @ transforms[j].T
                scale = max(float(expected.abs().max()), 1e-12)
                worst = max(worst, float((moved[k, :n_i, :n_j] - expected).abs().max()) / scale)
            assert blocks.abs().max() > 1e-6, "trivially equivariant: the blocks are zero"
            assert worst < 1e-8, worst
    finally:
        torch.set_default_dtype(torch.float32)


def test_rdm_head_unchanged_by_the_second_head(basis_info, aux_basis):
    """The 1-RDM output keeps HelmRDM's form: same blocks, same edges, same AO slots."""
    torch.set_default_dtype(torch.float64)
    try:
        net = _net(basis_info, aux_basis)
        with torch.no_grad():
            out = net(_batch(WATER))
    finally:
        torch.set_default_dtype(torch.float32)
    blocks, edge_index = out["rdm"]
    assert blocks.shape == (3 + 6, 14, 14)
    assert edge_index.shape == (2, 9)
    # hydrogen carries 2 AOs: its padded slots must be exactly zero
    h = 1
    assert blocks[h, 2:, :].abs().max() == 0 and blocks[h, :, 2:].abs().max() == 0
    cum, cum_edges = out["cumulant"]
    assert cum.shape == (3 + 6, 70, 70) and torch.equal(cum_edges, edge_index)
    assert cum[h, 23:, :].abs().max() == 0 and cum[h, :, 23:].abs().max() == 0


def test_cumulant_head_spans_every_block_entry(basis_info, aux_basis):
    """Both cumulant heads are orthonormal bases of their block space, so nothing is unreachable."""
    net = _net(basis_info, aux_basis)
    for head, n_expected in (
        (net.cumulant_edge_head, 70 * 70),
        (net.cumulant_node_head, 70 * 71 // 2),
    ):
        a = head.assembly.double().reshape(head.assembly.shape[0], -1)
        assert a.shape[0] == n_expected
        gram = a @ a.T
        # orthogonal; norms 1, or 2 for mirrored off-diagonal pairs of the node head.
        # The buffer is float32, hence 1e-6 rather than machine precision.
        off = gram - torch.diag(torch.diagonal(gram))
        assert off.abs().max() < 1e-6
        assert set(torch.diagonal(gram).round(decimals=4).tolist()) <= {1.0, 2.0}


def test_coulomb_norm_is_the_metric_norm(aux_basis):
    """``tr(dG M dG M) = ||M^1/2 dG M^1/2||_F^2`` and the energy is ``0.5 <G, M>``."""
    torch.manual_seed(0)
    metric = torch.as_tensor(aux_basis.coulomb_metric(_mol(WATER)))
    n = metric.shape[0]
    delta = torch.randn(n, n, dtype=torch.float64)
    delta = 0.5 * (delta + delta.T)
    w, u = torch.linalg.eigh(metric)
    root = u @ torch.diag(w.sqrt()) @ u.T
    expected = (root @ delta @ root).pow(2).sum()
    got = HelmCumulantLightningModule.coulomb_norm2(delta[None], metric[None])[0]
    assert torch.allclose(got, expected, rtol=1e-8)
    energy = HelmCumulantLightningModule.cumulant_energy(delta[None], metric[None])[0]
    assert torch.allclose(energy, 0.5 * torch.einsum("ab,ab->", delta, metric))


@pytest.mark.parametrize("symmetric", [True, False])
def test_grouped_linear_is_the_same_function(aux_basis, symmetric):
    """Grouping the head's output irreps by degree changes the layout, not the function.

    Copy each per-entry weight of the ungrouped layer into the matching copy of
    the grouped one, and the blocks must agree exactly -- so the speed-up (one
    matmul per degree instead of one op per entry) cannot have changed what the
    head can represent or how it is initialised.
    """
    from boa.model.net.helm_net import ShellPairHead, union_shell_degrees

    union = union_shell_degrees(aux_basis.shell_degrees)
    torch.manual_seed(0)
    flat = ShellPairHead(union, 4, 6, symmetric=symmetric, grouped_linear=False).double()
    grouped = ShellPairHead(union, 4, 6, symmetric=symmetric, grouped_linear=True).double()
    assert flat.linear.weight.numel() == grouped.linear.weight.numel()

    # per-degree queue of the flat layer's (C, 1) weights, in entry order
    queues: dict[int, list] = {}
    for _, ins, w in flat.linear.weight_views(yield_instruction=True):
        queues.setdefault(flat.irreps_out[ins.i_out].ir.l, []).append(w)
    with torch.no_grad():
        for _, ins, w in grouped.linear.weight_views(yield_instruction=True):
            w.copy_(torch.cat(queues[grouped.irreps_out[ins.i_out].ir.l], dim=1))

    x = torch.randn(5, 49, 4, dtype=torch.float64)
    assert torch.allclose(flat(x), grouped(x), atol=1e-12)


def test_cumulant_starts_at_zero_and_can_leave_it(basis_info, aux_basis):
    """Zero-initialised G heads predict G = 0 exactly, and still receive a gradient."""
    torch.set_default_dtype(torch.float64)
    try:
        net = _net(basis_info, aux_basis, zero_init=True)
        blocks, _ = net(_batch(WATER))["cumulant"]
        assert blocks.abs().max() == 0
        blocks.sum().backward()
        assert net.cumulant_edge_head.linear.weight.grad.abs().max() > 0
        assert net.cumulant_node_head.linear.weight.grad.abs().max() > 0
        rdm, _ = net(_batch(WATER))["rdm"]
        assert rdm.abs().max() > 0, "only the cumulant heads are zeroed"
    finally:
        torch.set_default_dtype(torch.float32)


def test_coulomb_whitening_matches_the_loss(aux_basis):
    """The weighted image ``M^1/2 dG M^1/2`` has exactly the loss as its squared norm."""
    torch.manual_seed(1)
    metric = torch.as_tensor(aux_basis.coulomb_metric(_mol(WATER)))
    delta = torch.randn(metric.shape, dtype=torch.float64)
    delta = 0.5 * (delta + delta.T)
    weighted = HelmCumulantLightningModule.coulomb_whiten(delta, metric).double()
    loss = HelmCumulantLightningModule.coulomb_norm2(delta[None], metric[None])[0]
    assert torch.allclose(weighted.pow(2).sum(), loss, rtol=1e-5)
