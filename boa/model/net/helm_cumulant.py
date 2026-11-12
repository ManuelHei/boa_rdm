"""One HELM backbone, two heads: the 1-RDM and the density-fitted cumulant 2-RDM.

Both targets have the same structure, which is why one network can carry both.
Each is a symmetric matrix over atom-centred functions, so it splits into an
on-site ``(i, i)`` block per atom and an ``(i, j)`` block per pair, and each
block between shells of degree ``l1`` and ``l2`` is the ``l1 (x) l2`` irreps.
The same :class:`~boa.model.net.helm_net.ShellPairHead` therefore writes either.

What differs is the basis, and with it the size:

=========================  ==========================  ===========================
                           ``D`` (1-RDM)               ``G`` (cumulant, DF)
=========================  ==========================  ===========================
basis                      6-31g* AOs                  cc-pvdz-jkfit aux functions
functions per H / C,N,O,F  2 / 14                      23 / 70
union shells               3s 2p 1d (14)               10s 7p 5d 2f (70)
heavy-atom pair block      196 entries                 4900 entries
highest ``L`` in a block   4                           6
median matrix (df_db_mp2)  128 x 128                   744 x 744
=========================  ==========================  ===========================

So ``G`` is ~35x the entries of ``D``, and its f-f and d-f shell pairs need
irreps up to ``L = 6``. A linear head cannot create an irrep the backbone does
not carry, so the backbone runs at ``lmax = 6`` by default -- the lowest degree
that reaches every block of both matrices. Lifting a ``lmax = 4`` backbone to 6
inside the head (a tensor product with the edge direction) would be cheaper, and
is the obvious next step if the backbone cost dominates; it is not done here.

Each head reads the *shared* node and edge features, so the 1-RDM and the
cumulant are predicted from one description of the molecule. That is the point
of predicting them jointly: the two are not independent (the partial trace of the
cumulant is fixed by the 1-RDM for an N-representable state), and a shared trunk
lets what one target teaches about the electronic structure reach the other.
"""

from __future__ import annotations

import torch
from torch import Tensor

from boa.data.aux_basis import AuxBasisLayout
from boa.data.basis_info import BasisInfo
from boa.model.net.helm_net import (
    HelmRDM,
    ShellPairHead,
    element_ao_selection,
    selection_table,
    union_shell_degrees,
)


class HelmRDMCumulant(HelmRDM):
    """:class:`~boa.model.net.helm_net.HelmRDM` with a second pair of heads for the cumulant.

    ``forward`` returns two ``(blocks, edge_index)`` pairs in a dict, ``"rdm"``
    and ``"cumulant"``, each in the form
    :meth:`~boa.model.rdm_module.RDMLightningModule.scatter_blocks` consumes.
    The cumulant blocks are ``(n, max_aux_dim, max_aux_dim)`` over the
    auxiliary basis, padded beyond an atom's own function count.

    Args:
        basis_info: the orbital basis of ``D``.
        aux_basis: the auxiliary basis of ``G``. Its element order must be
            ``basis_info.atomic_numbers``, so one element index serves both.
        grouped_linear: on by default here (see :class:`~boa.model.net.helm_net.ShellPairHead`);
        there
            are no older checkpoints of this model to stay compatible with.
        zero_init_cumulant: start the cumulant heads at zero weight, so the
            first prediction is G = 0 (no correlation).
        lmax: defaults to ``2 * max(l)`` over *both* bases -- 6 for 6-31g* with
            cc-pvdz-jkfit, set by the f functions of the latter.
        **kwargs: passed on to :class:`~boa.model.net.helm_net.HelmRDM`.
    """

    def __init__(
        self,
        basis_info: BasisInfo,
        aux_basis: AuxBasisLayout,
        lmax: int | None = None,
        grouped_linear: bool = True,
        zero_init_cumulant: bool = True,
        **kwargs,
    ) -> None:
        if [int(z) for z in basis_info.atomic_numbers] != list(aux_basis.atomic_numbers):
            raise ValueError(
                f"aux_basis elements {aux_basis.atomic_numbers} are not ordered like "
                f"basis_info's {list(basis_info.atomic_numbers)}; the element index would "
                f"pick the wrong aux layout."
            )
        aux_union = union_shell_degrees(aux_basis.shell_degrees)
        ao_union = union_shell_degrees(basis_info)
        if lmax is None:
            lmax = 2 * max(max(ao_union), max(aux_union))
        super().__init__(
            basis_info=basis_info, lmax=lmax, grouped_linear=grouped_linear, **kwargs
        )

        self.aux_basis = aux_basis
        channels = self.node_head.sphere_channels
        self.cumulant_node_head = ShellPairHead(
            aux_union, channels, self.lmax, symmetric=True, grouped_linear=grouped_linear
        )
        self.cumulant_edge_head = ShellPairHead(
            aux_union, channels, self.lmax, symmetric=False, grouped_linear=grouped_linear
        )
        self.aux_dim = self.cumulant_node_head.dim
        if zero_init_cumulant:
            # Untrained, the G head comes out ~650x the size of the target:
            # a cumulant loss of ~1e9 against ~1 for D, which
            # would hand the shared trunk entirely to G. Zero weights start G at
            # exactly 0 -- the mean-field answer, no correlation -- so L_G starts
            # at the reference's own size. The gradient with respect to these
            # weights is the loss gradient times the features, nonzero, so they
            # leave zero on the first step; only the gradient G sends back into
            # the trunk is zero until then.
            with torch.no_grad():
                self.cumulant_node_head.linear.weight.zero_()
                self.cumulant_edge_head.linear.weight.zero_()

        table = selection_table(
            element_ao_selection(aux_basis.shell_degrees, aux_union), sink=self.aux_dim
        )
        if table.shape[1] != aux_basis.max_basis_dim:
            raise AssertionError((table.shape, aux_basis.max_basis_dim))
        self.register_buffer("aux_selection", torch.as_tensor(table))

    def forward(self, batch) -> dict[str, tuple[Tensor, Tensor]]:
        embeddings, edge_index, atom_index = self.embed(batch)
        return {
            "rdm": self.blocks(
                embeddings, edge_index, atom_index,
                self.node_head, self.edge_head, self.ao_selection,
            ),
            "cumulant": self.blocks(
                embeddings, edge_index, atom_index,
                self.cumulant_node_head, self.cumulant_edge_head, self.aux_selection,
            ),
        }
