"""Predict a molecular energy from a density matrix, with the HELM backbone.

The counterpart of :class:`~boa.model.net.helm_net.HelmRDM`: that maps geometry to a density
matrix,
this maps a density matrix (plus geometry) to a scalar.

Structure
---------
1. Split the input matrix into the same blocks the RDM head emits -- one on-site
   block per atom, one off-site block per graph edge.
2. Project each onto the head's own Wigner-3j basis
   (:class:`~boa.model.net.rdm_embedding.ShellPairEmbedding`), which is exact and lossless, and
   map the
   coefficients into backbone irreps.
3. Add those to the backbone's node and edge messages *before* the message
   passing, so every layer sees the density matrix rather than only the readout.
4. Sum a per-node and a per-edge scalar into one energy per molecule.

Why a node **and** an edge readout: to leading order the energy is linear in the
density matrix with pairwise structure, ``E ~ sum_ij tr(D_ij F_ji)``, so a sum of
atom terms and pair terms matches the shape of the target rather than
approximating it. Dropping the edge term would force pair contributions through
the node embeddings, which only see them after message passing.

The scalar is read off the ``l = 0`` channels alone. Higher-``l`` components are
equivariant, so admitting them into an invariant output would break rotational
invariance of the energy -- the one symmetry this model must not violate.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from boa.data.basis_info import BasisInfo
from boa.model.net.helm._compat import conditional_grad
from boa.model.net.helm.esen_osh import eSEN_Backbone
from boa.model.net.helm_net import (
    element_ao_selection,
    union_shell_degrees,
)
from boa.model.net.rdm_embedding import ShellPairEmbedding


class EnergyHead(nn.Module):
    """Invariant scalar per node or edge, from the ``l = 0`` channels."""

    def __init__(self, sphere_channels: int, hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(sphere_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # The molecular energy is a sum over every node *and* every edge, and a
        # molecule here has ~15 atoms but ~200 edges. With ordinary
        # initialisation that sum starts at O(100), against a delta target of
        # ~0.08 Ha -- measured at 49.6 Ha MAE before this. Zeroing the last layer
        # starts the model at exactly zero, so it learns increments on a target
        # it already brackets rather than first unwinding a large random offset.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        """``(N, (lmax+1)^2, C)`` features to ``(N,)`` scalars."""
        return self.mlp(x[:, 0, :]).squeeze(-1)


class HelmEnergy(nn.Module):
    """Density matrix (plus geometry) to a scalar energy.

    Args:
        basis_info: the AO basis the input matrix is written in. Must be the one
            the density-matrix model used, or the blocks are cut in the wrong
            places.
        lmax: defaults to ``2 * max(union_ls)``, the lowest degree that can
            represent every shell-pair block.
        mmax: defaults to ``lmax``.
        head_hidden: width of the invariant readout MLP.
        edge_index_key: which graph the blocks and messages use. Must match the
            graph the density matrix was predicted on, or off-site blocks land
            on edges that were never predicted.

    ``sphere_channels``, ``hidden_channels``, ``edge_channels``,
    ``num_distance_basis``, ``num_layers``, ``gaussian_cutoff``,
    ``gaussian_width``, ``norm_type``, ``wigner_backend`` and
    ``max_num_elements`` are as in :class:`~boa.model.net.helm_net.HelmRDM`.
    """

    def __init__(
        self,
        basis_info: BasisInfo,
        sphere_channels: int = 128,
        hidden_channels: int = 128,
        edge_channels: int = 128,
        num_distance_basis: int = 64,
        num_layers: int = 3,
        lmax: int | None = None,
        mmax: int | None = None,
        gaussian_cutoff: float = 8.0,
        gaussian_width: float = 1.0,
        head_hidden: int = 128,
        edge_index_key: str = "edge_index",
        wigner_backend: str = "torch",
        norm_type: str = "rms_norm_sh",
        max_num_elements: int = 100,
    ) -> None:
        super().__init__()
        self.basis_info = basis_info
        self.edge_index_key = edge_index_key

        union_ls = union_shell_degrees(basis_info)
        self.lmax = 2 * max(union_ls) if lmax is None else int(lmax)
        self.mmax = self.lmax if mmax is None else int(mmax)

        self.node_embedding = ShellPairEmbedding(
            union_ls, sphere_channels, self.lmax, symmetric=True
        )
        self.edge_embedding = ShellPairEmbedding(
            union_ls, sphere_channels, self.lmax, symmetric=False
        )
        self.dim = self.node_embedding.dim

        self.backbone = eSEN_Backbone(
            irreps_out=None,
            max_num_elements=max_num_elements,
            sphere_channels=sphere_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            cutoff=gaussian_cutoff,
            edge_channels=edge_channels,
            num_distance_basis=num_distance_basis,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            norm_type=norm_type,
            gaussian_width=gaussian_width,
            regress_forces=False,
            include_edges=True,
            wigner_backend=wigner_backend,
        )

        self.node_head = EnergyHead(sphere_channels, head_hidden)
        self.edge_head = EnergyHead(sphere_channels, head_hidden)

        # Same AO gather/scatter bookkeeping the RDM head uses, so a block is
        # cut from the dense matrix exactly where the head would have written it.
        selections = element_ao_selection(basis_info, union_ls)
        max_basis_dim = max(len(s) for s in selections)
        selection = np.full((len(selections), max_basis_dim), self.dim, dtype=np.int64)
        for i, sel in enumerate(selections):
            selection[i, : len(sel)] = sel
        self.register_buffer("ao_selection", torch.as_tensor(selection))
        # AO count per element, for the shared ao_layout: it must agree with the
        # RDM module's, or the blocks are cut from shifted positions.
        self.register_buffer(
            "basis_dim_per_atom",
            torch.as_tensor([len(s) for s in selections], dtype=torch.long),
        )
        self.register_buffer(
            "atomic_number_to_atom_index",
            torch.as_tensor(basis_info.atomic_number_to_atom_index, dtype=torch.long),
        )
        self.max_basis_dim = max_basis_dim

    def graph(self, batch) -> Tensor:
        """The message-passing graph: the configured radius graph without self-loops.

        Same rule as :meth:`~boa.model.net.helm_net.HelmRDM.graph`, and for the same reason: a
        self-loop
        has zero displacement, and the edge frame is built by rotating that
        displacement onto the z axis, which is undefined for the zero vector. A
        degenerate frame does not raise -- it silently costs rotational
        invariance of the energy, measured at 2e-02 hartree before this was
        fixed. The on-site blocks travel the node path, so nothing is lost.
        """
        edge_index = getattr(batch, self.edge_index_key)
        return edge_index[:, edge_index[0] != edge_index[1]]

    @conditional_grad(torch.enable_grad())
    def forward(self, batch, rdm: Tensor, ao_index: Tensor, n_ao: int) -> Tensor:
        """Energy per molecule, ``(n_mol,)``.

        Args:
            batch: the graph, for positions, elements and edges.
            rdm: ``(n_mol, n_ao, n_ao)`` dense density matrices, padded.
            ao_index: ``(n_atom, max_basis_dim)`` AO positions per atom, from
                ``RDMLightningModule.ao_layout`` -- the same bookkeeping the RDM
                model scatters with.
            n_ao: padded AO count of the batch.
        """
        edge_index = self.graph(batch)
        atom_index = self.atomic_number_to_atom_index[batch.atomic_numbers.long()]

        node_blocks = self.blocks_from_matrix(
            rdm, batch, ao_index, torch.arange(ao_index.shape[0], device=rdm.device),
            torch.arange(ao_index.shape[0], device=rdm.device), atom_index, atom_index, n_ao
        )
        edge_blocks = self.blocks_from_matrix(
            rdm, batch, ao_index, edge_index[0], edge_index[1],
            atom_index[edge_index[0]], atom_index[edge_index[1]], n_ao
        )

        node_features = self.node_embedding(node_blocks)
        # `x_message_edge` is at `sphere_channels` by the injection point: it is
        # created at `num_distance_basis` width but `edge_degree_embedding`
        # replaces it before the layers, so the embedding's own width is right.
        edge_features = self.edge_embedding(edge_blocks)

        pos = batch.pos
        vec = pos[edge_index[1]] - pos[edge_index[0]]
        distance = vec.norm(dim=-1, keepdim=True)
        n_atom_per_mol = torch.bincount(batch.batch, minlength=int(batch.num_graphs))

        from boa.model.net.helm_net import _HelmBatch

        helm_batch = _HelmBatch(
            pos=pos,
            edge_index=edge_index,
            edge_attr=torch.cat([distance, vec], dim=-1),
            atomic_numbers=batch.atomic_numbers.long(),
            charge=torch.zeros(int(batch.num_graphs), dtype=torch.long, device=pos.device),
            spin_multiplicity=torch.ones(
                int(batch.num_graphs), dtype=torch.long, device=pos.device
            ),
            num_atoms_in_molecule=n_atom_per_mol,
        )
        embeddings = self.backbone(
            helm_batch,
            extra_node_features=node_features,
            extra_edge_features=edge_features,
        )

        n_mol = int(batch.num_graphs)
        energy = pos.new_zeros(n_mol)
        energy = energy.index_add(
            0, batch.batch, self.node_head(embeddings["node_embeddings"])
        )
        energy = energy.index_add(
            0, batch.batch[edge_index[0]], self.edge_head(embeddings["edge_embeddings"])
        )
        return energy

    def blocks_from_matrix(
        self, rdm: Tensor, batch, ao_index: Tensor, atoms_row: Tensor, atoms_col: Tensor,
        elem_row: Tensor, elem_col: Tensor, n_ao: int
    ) -> Tensor:
        """Cut ``(n, dim, dim)`` union-layout blocks for the given atom pairs.

        Padded AO slots of an element with fewer AOs than ``max_basis_dim`` are
        gathered from a clamped index and then masked to zero, and the union-slot
        positions they would occupy are left at zero -- so a hydrogen block does
        not pick up a carbon's d functions.
        """
        mol = batch.batch[atoms_row]
        rows_ao = ao_index[atoms_row]  # (n, max_basis_dim), n_ao marks padding
        cols_ao = ao_index[atoms_col]
        valid_r = rows_ao < n_ao
        valid_c = cols_ao < n_ao
        safe_r = rows_ao.clamp(max=n_ao - 1)
        safe_c = cols_ao.clamp(max=n_ao - 1)
        values = rdm[mol[:, None, None], safe_r[:, :, None], safe_c[:, None, :]]
        values = values * (valid_r[:, :, None] & valid_c[:, None, :]).to(values.dtype)

        union_r = self.ao_selection[elem_row]  # (n, max_basis_dim), self.dim marks padding
        union_c = self.ao_selection[elem_col]
        blocks = rdm.new_zeros(atoms_row.shape[0], self.dim + 1, self.dim + 1)
        idx = torch.arange(atoms_row.shape[0], device=rdm.device)
        blocks[idx[:, None, None], union_r[:, :, None], union_c[:, None, :]] = values
        return blocks[:, : self.dim, : self.dim]
