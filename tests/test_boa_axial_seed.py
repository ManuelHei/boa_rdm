"""The pseudovector seed must break parity and nothing else.

``AxialSeed`` exists to lift BOA's out-of-plane blindness (see
``boa.model.net.boa_net.AxialSeed``). It is only correct if it breaks exactly
one symmetry and preserves the others, so all four properties are pinned here:

1. off by default, the coefficients are untouched -- the published path is safe;
2. on, the out-of-plane coefficient of a planar molecule is non-zero;
3. rotation equivariance survives -- that is the valuable symmetry;
4. parity does not -- that is the deliberate break;
5. the seed vanishes at an exactly-C2 geometry, where a *proper* rotation
   forces the coefficient to zero and no parity break may rescue it.
"""

import numpy as np
import pytest
import torch
from e3nn import o3

from boa.data.basis_info import BasisInfo
from boa.model.net.boa_block import BoaBlock
from boa.model.net.boa_block_stack import BoaBlockStack
from boa.model.net.boa_net import BOA, ReducedEdgeEmbedding

#: O 2px and O 3px -- the coefficient slots that vanish for a planar molecule
PX_SLOTS = (3, 6)
#: bent, and deliberately NOT C2-symmetric: the two O-H bonds differ
WATER = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.900, -0.560]])
WATER_C2 = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]])
WATER_Z = np.array([8, 1, 1])


@pytest.fixture(scope="module")
def basis_info():
    return BasisInfo.from_atomic_numbers_with_even_tempered_basis(
        atomic_numbers=[1, 8],
        basis="6-31g*",
        beta=None,
        even_tempered=False,
        uncontracted=False,
    )


def _net(basis_info, axial_seed, seed=0):
    torch.manual_seed(seed)
    from functools import partial

    stack = BoaBlockStack(
        block=partial(BoaBlock), n_blocks=2, basis_info=basis_info, hidden_channels=8
    )
    return (
        BOA(
            basis_info=basis_info,
            boa_stack=stack,
            initial_guess_module=ReducedEdgeEmbedding(basis_info, channels=1),
            num_orbitals=4,
            axial_seed=axial_seed,
        )
        .double()
        .eval()
    )


def _batch(basis_info, positions):
    """One molecule, through the same transform chain training uses."""
    from mldft.ml.data.components.of_data import Representation
    from torch.utils.data import Dataset

    from boa.data.dataloader import OFDataLoader
    from boa.data.mldft_units import build_molecule, sample_from_molecule
    from boa.data.transforms import (
        AddEdgeMatrices,
        AddMessagePassingMatrix,
        AddRadiusEdgeIndex,
        MasterTransform,
        ToTorch,
    )

    # Same chain as configs/data/datamodule/dataset/transform/qmlearn.yaml,
    # built directly because BasisInfo is not an OmegaConf-representable value.
    transform = MasterTransform(
        [
            ToTorch(),
            AddMessagePassingMatrix(basis_info, type="overlap", remove_diagonal=False),
            AddRadiusEdgeIndex(radius=6.0, name="message_edge_index"),
            AddEdgeMatrices(
                basis_info, edge_name="message_edge_index", name="message_edge_matrices"
            ),
            AddRadiusEdgeIndex(radius=3.0, name="edge_index"),
            AddEdgeMatrices(basis_info, edge_name="edge_index", name="edge_matrices"),
            ToTorch(),
        ]
    )

    class One(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, _):
            mol = build_molecule(charges=WATER_Z, positions=positions, basis=basis_info.basis_dict)
            s = sample_from_molecule(mol, basis_info)
            s.add_item("n_atom", torch.tensor([3]), Representation.NONE)
            s.add_item("atom_types", torch.as_tensor(WATER_Z), Representation.NONE)
            s.id = "probe"
            return transform(s)

    return next(iter(OFDataLoader(One(), batch_size=1, shuffle=False)))


def _coeffs(net, basis_info, positions):
    with torch.no_grad():
        coeffs, _ = net(_batch(basis_info, positions))
    return coeffs


def test_off_by_default_leaves_the_out_of_plane_slots_empty(basis_info):
    """Without the seed, BOA is exactly O(3)-equivariant and px is structurally zero."""
    torch.set_default_dtype(torch.float64)
    try:
        coeffs = _coeffs(_net(basis_info, axial_seed=False), basis_info, WATER)
        for slot in PX_SLOTS:
            assert coeffs[:, slot, :].abs().max() < 1e-12
    finally:
        torch.set_default_dtype(torch.float32)


def test_seed_reaches_the_out_of_plane_slots(basis_info):
    """With the seed, px is populated on the same planar geometry."""
    torch.set_default_dtype(torch.float64)
    try:
        coeffs = _coeffs(_net(basis_info, axial_seed=True), basis_info, WATER)
        reference = coeffs[:, 1, :].abs().max()  # the O 2s slot, always populated
        for slot in PX_SLOTS:
            assert coeffs[:, slot, :].abs().max() > 1e-3 * reference
    finally:
        torch.set_default_dtype(torch.float32)


def test_seed_keeps_rotation_equivariance(basis_info):
    """The seed is an l=1 object, so rotating the molecule rotates the coefficients."""
    torch.set_default_dtype(torch.float64)
    try:
        net = _net(basis_info, axial_seed=True)
        torch.manual_seed(2)
        for _ in range(3):
            rotation = o3.rand_matrix().double().numpy()
            base = net.node_embedding.axial.pseudovector(
                torch.as_tensor(WATER), _batch(basis_info, WATER).edge_index
            )
            moved = net.node_embedding.axial.pseudovector(
                torch.as_tensor(WATER @ rotation.T),
                _batch(basis_info, WATER @ rotation.T).edge_index,
            )
            expected = torch.einsum("ij,ajc->aic", torch.as_tensor(rotation), base)
            assert torch.allclose(moved, expected, atol=1e-10)
    finally:
        torch.set_default_dtype(torch.float32)


def test_seed_breaks_parity(basis_info):
    """A polar vector would flip under a mirror; an axial one does not. That is the break."""
    torch.set_default_dtype(torch.float64)
    try:
        net = _net(basis_info, axial_seed=True)
        mirror = np.diag([-1.0, 1.0, 1.0])
        base = net.node_embedding.axial.pseudovector(
            torch.as_tensor(WATER), _batch(basis_info, WATER).edge_index
        )
        moved = net.node_embedding.axial.pseudovector(
            torch.as_tensor(WATER @ mirror.T), _batch(basis_info, WATER @ mirror.T).edge_index
        )
        polar = torch.einsum("ij,ajc->aic", torch.as_tensor(mirror), base)
        assert torch.allclose(moved, -polar, atol=1e-10), "should transform as an AXIAL vector"
        assert not torch.allclose(moved, polar, atol=1e-6), "parity must be broken, not preserved"
    finally:
        torch.set_default_dtype(torch.float32)


def test_seed_vanishes_when_c2_is_exact(basis_info):
    """At an exactly C2-symmetric geometry a proper rotation forces zero, and it must.

    A seed that survived here would be predicting something symmetry forbids.
    """
    torch.set_default_dtype(torch.float64)
    try:
        net = _net(basis_info, axial_seed=True)
        symmetric = net.node_embedding.axial.pseudovector(
            torch.as_tensor(WATER_C2), _batch(basis_info, WATER_C2).edge_index
        )
        assert symmetric[0].abs().max() < 1e-10, "oxygen's seed must vanish under exact C2"

        asymmetric = net.node_embedding.axial.pseudovector(
            torch.as_tensor(WATER), _batch(basis_info, WATER).edge_index
        )
        assert asymmetric[0].abs().max() > 1e-6, "and must not vanish once C2 is broken"
    finally:
        torch.set_default_dtype(torch.float32)
