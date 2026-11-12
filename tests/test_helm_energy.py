"""The density-matrix embedding and the energy model built on it.

Three things have to hold, and none of them shows up as a crash if it is wrong.

The embedding must be the **exact inverse** of the head that writes the density
matrix, not an approximation of it: the Wigner-3j ``assembly`` matrices are
orthogonal and there are as many of them as the block has degrees of freedom, so
a projection recovers the coefficients exactly. If that failed, the energy model
would be reading a lossy view of its input and nobody would notice.

Cutting the blocks back out of a dense matrix must land on the same AO slots the
head wrote them to. Get the padding wrong and a hydrogen block quietly picks up a
carbon's d functions.

The predicted energy must be **rotation-invariant**. It is read from the ``l = 0``
channels for exactly this reason, and a scalar head that accidentally admitted
higher-``l`` components would still train -- towards a quantity that depends on
how the molecule happens to be oriented.
"""

import numpy as np
import pytest
import torch
from e3nn import o3

from boa.data.basis_info import BasisInfo
from boa.model.net.helm_energy import HelmEnergy
from boa.model.net.helm_net import ShellPairHead, union_shell_degrees
from boa.model.net.rdm_embedding import ShellPairEmbedding

WATER_Z = [8, 1, 1]
WATER = np.array([[0.0, 0.0, 0.0], [0.758, 0.587, 0.0], [-0.758, 0.587, 0.0]])


@pytest.fixture(scope="module")
def basis_info():
    return BasisInfo.from_atomic_numbers_with_even_tempered_basis(
        atomic_numbers=[1, 8],
        basis="6-31g*",
        beta=None,
        even_tempered=False,
        uncontracted=False,
    )


def _batch(positions: np.ndarray):
    """The fields :class:`HelmEnergy` reads, for a single molecule."""
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


def _energy_net(basis_info, seed=0):
    """A small net at the *current* default dtype.

    Construct under ``torch.set_default_dtype(torch.float64)`` for equivariance
    checks: the Wigner tables and the assembly basis are built as buffers at
    construction, so a net built in float32 and cast afterwards keeps
    float32-precision constants and drifts by ~1e-3 under rotation -- an
    artefact of the harness that looks exactly like a broken model.
    """
    torch.manual_seed(seed)
    return HelmEnergy(
        basis_info,
        # Deliberately all different: equal widths let a channel-count bug
        # broadcast silently. `num_distance_basis != sphere_channels` in
        # particular is what the real config uses and what caught a wrong
        # injection width.
        sphere_channels=16,
        hidden_channels=12,
        edge_channels=10,
        num_distance_basis=8,
        num_layers=1,
        head_hidden=14,
    ).eval()


@pytest.mark.parametrize("symmetric", [True, False])
def test_embedding_inverts_the_head(basis_info, symmetric):
    """Projection recovers the coefficients the head assembled a block from."""
    union_ls = union_shell_degrees(basis_info)
    lmax = 2 * max(union_ls)
    head = ShellPairHead(union_ls, 8, lmax, symmetric)
    embedding = ShellPairEmbedding(union_ls, 8, lmax, symmetric)

    torch.manual_seed(0)
    coefficients = torch.randn(6, head.irreps_out.dim)
    blocks = torch.einsum("nc,cij->nij", coefficients, head.assembly)

    recovered = embedding.project(blocks)
    assert torch.allclose(recovered, coefficients, atol=1e-5)


@pytest.mark.parametrize("symmetric", [True, False])
def test_projection_spans_the_block_space(basis_info, symmetric):
    """An arbitrary block survives block -> coefficients -> block.

    This is what makes the embedding lossless rather than a compression: the
    basis has exactly as many members as the block has free entries, so nothing
    is thrown away. A symmetric block for the on-site basis, a general one for
    the off-site basis.
    """
    union_ls = union_shell_degrees(basis_info)
    lmax = 2 * max(union_ls)
    embedding = ShellPairEmbedding(union_ls, 8, lmax, symmetric)

    torch.manual_seed(1)
    block = torch.randn(4, embedding.dim, embedding.dim)
    if symmetric:
        block = 0.5 * (block + block.transpose(1, 2))

    coefficients = embedding.project(block)
    rebuilt = torch.einsum("nc,cij->nij", coefficients, embedding.assembly)
    assert torch.allclose(rebuilt, block, atol=1e-5)


def test_stack_and_unstack_are_inverse(basis_info):
    """The l-major <-> e3nn-flat conversions must compose to the identity."""
    union_ls = union_shell_degrees(basis_info)
    lmax = 2 * max(union_ls)
    head = ShellPairHead(union_ls, 8, lmax, symmetric=True)
    embedding = ShellPairEmbedding(union_ls, 8, lmax, symmetric=True)

    torch.manual_seed(2)
    x = torch.randn(5, (lmax + 1) ** 2, 8)
    assert torch.equal(embedding.unstack_irreps(head.stack_irreps(x)), x)


def test_blocks_are_cut_from_the_slots_the_head_wrote(basis_info):
    """Round trip: union block -> dense matrix -> union block.

    Uses the energy net's own AO bookkeeping on both sides, so a change to the
    padding convention has to break this rather than silently shift which AOs a
    block covers.
    """
    net = _energy_net(basis_info)
    batch = _batch(WATER)
    n_atom = len(WATER_Z)
    atom_index = net.atomic_number_to_atom_index[batch.atomic_numbers.long()]

    # AO layout: each atom's AOs laid out consecutively, padding marked by n_ao.
    counts = torch.tensor([len([s for s in net.ao_selection[a] if s < net.dim]) for a in atom_index])
    starts = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])
    n_ao = int(counts.sum())
    ao_index = torch.full((n_atom, net.max_basis_dim), n_ao, dtype=torch.long)
    for a in range(n_atom):
        ao_index[a, : counts[a]] = starts[a] + torch.arange(counts[a])

    torch.manual_seed(3)
    blocks = torch.randn(n_atom, net.dim, net.dim)
    blocks = 0.5 * (blocks + blocks.transpose(1, 2))
    # Zero the union slots this element does not own, which is the invariant the
    # scatter relies on and the gather reproduces.
    for a in range(n_atom):
        own = net.ao_selection[atom_index[a]]
        own = own[own < net.dim]
        keep = torch.zeros(net.dim, dtype=torch.bool)
        keep[own] = True
        blocks[a][~keep, :] = 0
        blocks[a][:, ~keep] = 0

    # Scatter the diagonal blocks into a dense matrix, then cut them back out.
    dense = torch.zeros(1, n_ao, n_ao)
    for a in range(n_atom):
        rows = ao_index[a][ao_index[a] < n_ao]
        own = net.ao_selection[atom_index[a]]
        own = own[own < net.dim]
        dense[0][rows[:, None], rows[None, :]] = blocks[a][own[:, None], own[None, :]]

    ar = torch.arange(n_atom)
    recovered = net.blocks_from_matrix(dense, batch, ao_index, ar, ar, atom_index, atom_index, n_ao)
    assert torch.allclose(recovered, blocks, atol=1e-6)


def test_energy_is_rotation_invariant(basis_info):
    """The predicted energy must not depend on the molecule's orientation.

    The density matrix is not invariant -- it rotates with the molecule -- so
    the input is rotated properly too, by the same AO transform the head's
    equivariance test uses. An energy that changed here would mean the readout
    is picking up l > 0 components.
    """
    torch.set_default_dtype(torch.float64)
    try:
        _check_rotation_invariance(basis_info)
    finally:
        torch.set_default_dtype(torch.float32)


def _check_rotation_invariance(basis_info):
    from scipy.linalg import block_diag

    from test_helm_rdm import atom_rotation  # tests dir is on sys.path under pytest

    net = _energy_net(basis_info)
    torch.manual_seed(4)
    rotation = o3.rand_matrix(dtype=torch.float64).numpy()

    n_atom = len(WATER_Z)
    atom_index = net.atomic_number_to_atom_index[torch.as_tensor(WATER_Z)]
    counts = torch.tensor([len([s for s in net.ao_selection[a] if s < net.dim]) for a in atom_index])
    starts = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])
    n_ao = int(counts.sum())
    ao_index = torch.full((n_atom, net.max_basis_dim), n_ao, dtype=torch.long)
    for a in range(n_atom):
        ao_index[a, : counts[a]] = starts[a] + torch.arange(counts[a])

    transform = torch.as_tensor(
        block_diag(*[atom_rotation(basis_info, z, rotation) for z in WATER_Z]),
        dtype=torch.float64,
    )
    torch.manual_seed(5)
    dm = torch.randn(1, n_ao, n_ao, dtype=torch.float64)
    dm = 0.5 * (dm + dm.transpose(1, 2))

    plain = _batch(WATER)
    turned = _batch(WATER @ rotation.T)

    with torch.no_grad():
        e_plain = net(plain, dm, ao_index, n_ao)
        e_turned = net(turned, transform @ dm @ transform.T, ao_index, n_ao)

    assert torch.allclose(e_plain, e_turned, atol=1e-8), (
        f"energy changed by {(e_plain - e_turned).abs().max():.2e} under rotation"
    )


def test_energy_shape_and_finiteness(basis_info):
    """One finite scalar per molecule, and the graph is actually consumed."""
    net = _energy_net(basis_info)
    batch = _batch(WATER)
    n_atom = len(WATER_Z)
    atom_index = net.atomic_number_to_atom_index[batch.atomic_numbers.long()]
    counts = torch.tensor([len([s for s in net.ao_selection[a] if s < net.dim]) for a in atom_index])
    starts = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])
    n_ao = int(counts.sum())
    ao_index = torch.full((n_atom, net.max_basis_dim), n_ao, dtype=torch.long)
    for a in range(n_atom):
        ao_index[a, : counts[a]] = starts[a] + torch.arange(counts[a])

    torch.manual_seed(6)
    dm = torch.randn(1, n_ao, n_ao)
    dm = 0.5 * (dm + dm.transpose(1, 2))
    energy = net(batch, dm, ao_index, n_ao)

    assert energy.shape == (1,)
    assert torch.isfinite(energy).all()

    # The density matrix must actually reach the output. The readout is
    # zero-initialised on purpose, so at step 0 the model returns 0.0 for every
    # input *and* passes no gradient upstream -- both correct, and both useless
    # for testing the wiring. Give the last layer real weights first; that is the
    # state after one optimiser step anyway.
    torch.nn.init.normal_(net.node_head.mlp[-1].weight, std=0.1)
    torch.nn.init.normal_(net.edge_head.mlp[-1].weight, std=0.1)
    dm = dm.clone().requires_grad_(True)
    net(batch, dm, ao_index, n_ao).sum().backward()
    assert dm.grad is not None and dm.grad.abs().max() > 0, (
        "energy does not depend on its density-matrix input"
    )
