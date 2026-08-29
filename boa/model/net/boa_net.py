from typing import Tuple

import pyscf
import torch
from torch import Tensor, nn
from torch_geometric.utils import to_dense_batch

from boa.data.basis_info import BasisInfo
from boa.model.net.boa_block_stack import BoaBlockStack
from scdp.model.scn.smearing import GaussianSmearing

numbers_to_element_symbols = pyscf.data.elements.ELEMENTS


class AxialSeed(nn.Module):
    r"""A per-atom pseudovector, seeded into the ``l = 1`` AO slots.

    Why this exists
    ---------------
    BOA seeds features only on the ``l = 0`` slots and reaches everything else
    through the overlap matrices, which makes the out-of-plane p coefficient of
    a planar molecule *exactly* zero -- see
    ``EXP-2026-08-29-03`` in the research protocol. The cause is a fixed-point
    argument: BOA is exactly O(3)-equivariant, a planar molecule is fixed by the
    mirror :math:`\sigma` through its own plane, so every coefficient must
    satisfy :math:`a = \Gamma(\sigma) a` and the odd AOs are the ``-1``
    eigenspace. Since a triatomic is planar by construction, that costs 42% of
    :math:`\|D\|_F` on the H2O set, and it is not fixable by capacity or
    training.

    The construction
    ----------------
    For atom ``k`` with neighbours ``j``, bond vectors :math:`v_j`, and a
    learned radial weight :math:`g(d_j)`:

    .. math::

        n_k = 2 \Big( \sum_j g(d_j)\, v_j \Big) \times \Big( \sum_j v_j \Big)

    which is the neighbour-ordering-independent form of
    :math:`\sum_{j \neq j'} [g(d_j) - g(d_{j'})]\, v_j \times v_{j'}` -- the
    antisymmetric weighting is what makes it independent of how the neighbours
    happen to be indexed, and it collapses to two sums, so this is O(edges) and
    not O(degree^2).

    Three properties, all of them the point:

    * Under a **rotation** it transforms as an ``l = 1`` object, so writing it
      into the ``(px, py, pz)`` slots leaves SO(3) equivariance intact.
    * It is an **axial** vector, so under the mirror that fixes a planar
      molecule it is *invariant* where a polar p coefficient must flip. That
      mismatch is the deliberate parity break, and it is what removes the fixed
      point. The model becomes SO(3)- but not O(3)-equivariant, the same trade
      HELM already makes.
    * It **vanishes when the neighbours are equivalent** -- exactly-C2 water,
      where :math:`g(d_1) = g(d_2)`. That is correct rather than a limitation:
      C2 is a *proper* rotation, so it forces the coefficient to zero for a
      parity-breaking model too, and a seed that survived there would be wrong.
      In this dataset ``|d1 - d2|`` averages 0.07 A, so it is generically far
      from zero.

    Only the p slots are seeded. The odd d functions (``dxy``, ``dxz``) are
    equally unreachable but carry ~0.003 of the target against 0.82 for ``2px``,
    so they are left for later.
    """

    def __init__(
        self,
        basis_info: BasisInfo,
        channels: int = 32,
        num_gaussians: int = 16,
        cutoff: float = 3.0,
    ):
        super().__init__()
        self.channels = channels
        self.smearing = GaussianSmearing(0.0, cutoff, num_gaussians, 1.0)
        self.radial = nn.Sequential(
            nn.Linear(num_gaussians, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

        p_slots, n_shells = [], []
        for index in basis_info.atom_ind_to_basis_function_ind:
            ls = basis_info.l_per_basis_func[index]
            slots = torch.as_tensor((ls == 1).nonzero()[0], dtype=torch.long)
            p_slots.append(slots)
            n_shells.append(slots.numel() // 3)
        self.n_shells = n_shells
        for i, slots in enumerate(p_slots):
            # pyscf orders a p shell (px, py, pz), which is what the seed writes
            self.register_buffer(f"p_slots_{i}", slots, persistent=False)
        self.scale = nn.Parameter(torch.randn(len(p_slots), max(max(n_shells), 1), channels))
        self.register_buffer(
            "atomic_numbers", torch.as_tensor(basis_info.atomic_numbers, dtype=torch.long)
        )

    def pseudovector(self, pos: Tensor, edge_index: Tensor) -> Tensor:
        """``(n_atom, 3, channels)``, the axial vector at every atom."""
        src, dst = edge_index[0], edge_index[1]
        keep = src != dst  # a self-loop has no direction
        src, dst = src[keep], dst[keep]

        bond = pos[dst] - pos[src]
        weight = self.radial(self.smearing(bond.norm(dim=-1)))  # (n_edge, channels)

        n_atom = pos.shape[0]
        total = torch.zeros(n_atom, 3, device=pos.device, dtype=pos.dtype)
        total.index_add_(0, src, bond)
        weighted = torch.zeros(n_atom, 3, self.channels, device=pos.device, dtype=pos.dtype)
        weighted.index_add_(0, src, bond[:, :, None] * weight[:, None, :])

        return 2.0 * torch.cross(weighted, total[:, :, None].expand_as(weighted), dim=1)

    def forward(
        self, x: Tensor, atomic_numbers: Tensor, pos: Tensor, edge_index: Tensor
    ) -> Tensor:
        """Add the seed to a dense ``(n_atom, max_basis_dim, channels)`` feature tensor."""
        axial = self.pseudovector(pos, edge_index).to(x.dtype)
        seed = torch.zeros_like(x)
        for i, z in enumerate(self.atomic_numbers.tolist()):
            shells = self.n_shells[i]
            if shells == 0:
                continue
            rows = (atomic_numbers == z).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            slots = getattr(self, f"p_slots_{i}")
            # (n_sel, 3, shells, C) -> shell-major, matching the slot order
            value = axial[rows][:, :, None, :] * self.scale[i, :shells][None, None, :, :]
            value = value.permute(0, 2, 1, 3).reshape(rows.numel(), 3 * shells, self.channels)
            seed[rows[:, None], slots[None, :], :] = value
        return x + seed


class NodeEmbedding(nn.Module):
    def __init__(self, basis_info: BasisInfo, channels: int = 32, axial_seed: bool = False):
        super().__init__()
        self.basis_info = basis_info
        self.channels = channels
        self.axial = AxialSeed(basis_info, channels=channels) if axial_seed else None

        self.register_buffer(
            "basis_dim_per_atom",
            torch.tensor(basis_info.basis_dim_per_atom, dtype=torch.long),
        )
        self.register_buffer(
            "atomic_number_to_atom_index",
            torch.tensor(basis_info.atomic_number_to_atom_index, dtype=torch.long),
        )
        self.is_scalar_mask = basis_info.l_per_basis_func == 0
        self.scalar_dims = []
        for index in basis_info.atom_ind_to_basis_function_ind:
            self.scalar_dims.append(self.is_scalar_mask[index].sum().item())
        self.scalar_dims = torch.tensor(self.scalar_dims, dtype=torch.long)
        self.embedding = nn.Embedding(
            len(basis_info.atomic_numbers), channels * self.scalar_dims.max().item()
        )

    def forward(
        self,
        atomic_numbers: Tensor,
        coeff_ind_to_node_ind: Tensor,
        pos: Tensor | None = None,
        edge_index: Tensor | None = None,
    ) -> Tensor:
        """
        :param atomic_numbers: Tensor of shape (batch_size, n_atoms)
        :param x: Tensor of shape (batch_size, n_channels)
        :param pos: atom positions, required only when the axial seed is on
        :param edge_index: the radius graph, required only when the axial seed is on
        :return: Tensor of shape (batch_size, n_channels + n_scalar_features)
        """
        x = torch.zeros(
            self.basis_dim_per_atom[self.atomic_number_to_atom_index[atomic_numbers]].sum().item(),
            self.channels,
            device=atomic_numbers.device,
            dtype=self.embedding.weight.dtype,
        )
        x, mask = to_dense_batch(
            x, coeff_ind_to_node_ind, max_num_nodes=max(self.basis_dim_per_atom)
        )
        batch_size = atomic_numbers.shape[0]
        scalar_features = self.embedding(self.atomic_number_to_atom_index[atomic_numbers]).view(
            batch_size, -1, self.channels
        )
        for i, a in enumerate(self.basis_info.atomic_numbers):
            x[atomic_numbers == a, : self.scalar_dims[i]] = scalar_features[
                atomic_numbers == a, : self.scalar_dims[i]
            ]

        if self.axial is not None:
            if pos is None or edge_index is None:
                raise ValueError(
                    "axial_seed is on but NodeEmbedding.forward got no pos/edge_index; the "
                    "pseudovector is built from the geometry."
                )
            x = self.axial(x, atomic_numbers, pos, edge_index)

        x = x[mask]

        return x


class ReducedEdgeEmbedding(nn.Module):
    def __init__(self, basis_info: BasisInfo, channels: int = 32, axial_seed: bool = False):
        super().__init__()
        self.basis_info = basis_info
        self.channels = channels

        self.node_embedding_a = NodeEmbedding(basis_info, channels=channels, axial_seed=axial_seed)
        self.node_embedding_b = NodeEmbedding(basis_info, channels=channels, axial_seed=axial_seed)

    def forward(self, batch) -> Tensor:
        """
        :param batch
        :return: Tensor of shape (n_edges, n_channels)
        """

        edge_index = batch.edge_index
        atomic_numbers = batch.atomic_numbers
        coeff_ind_to_node_ind = batch.coeff_ind_to_node_ind

        node_features_a = self.node_embedding_a(
            atomic_numbers, coeff_ind_to_node_ind, batch.pos, edge_index
        )
        node_features_b = self.node_embedding_b(
            atomic_numbers, coeff_ind_to_node_ind, batch.pos, edge_index
        )

        node_features_a = to_dense_batch(
            node_features_a,
            coeff_ind_to_node_ind,
            max_num_nodes=max(self.basis_info.basis_dim_per_atom),
        )[0]
        node_features_b = to_dense_batch(
            node_features_b,
            coeff_ind_to_node_ind,
            max_num_nodes=max(self.basis_info.basis_dim_per_atom),
        )[0]

        edge_features_a = torch.zeros(
            edge_index.shape[1],
            node_features_a.shape[1],
            self.channels,
            device=edge_index.device,
            dtype=node_features_a.dtype,
        )
        edge_features_b = torch.zeros(
            edge_index.shape[1],
            node_features_b.shape[1],
            self.channels,
            device=edge_index.device,
            dtype=node_features_b.dtype,
        )

        edge_features_a[edge_index[0] == edge_index[1]] = node_features_a
        edge_features_b[edge_index[0] == edge_index[1]] = node_features_b

        return torch.cat([edge_features_a, edge_features_b], dim=-1), edge_index


class EdgeEmbedding(nn.Module):
    """EdgeEmbedding class for embedding edge features based on atomic numbers. Each edge feature is initialized
    with a learnable embedding vector corresponding to the atomic numbers of the nodes it connects.
    """

    def __init__(self, basis_info: BasisInfo, channels: int = 32):
        super().__init__()
        self.basis_info = basis_info
        self.channels = channels

        self.register_buffer(
            "basis_dim_per_atom",
            torch.tensor(basis_info.basis_dim_per_atom, dtype=torch.long),
        )
        self.register_buffer(
            "atomic_number_to_atom_index",
            torch.tensor(basis_info.atomic_number_to_atom_index, dtype=torch.long),
        )
        self.is_scalar_mask = basis_info.l_per_basis_func == 0
        self.scalar_dims = []
        for index in basis_info.atom_ind_to_basis_function_ind:
            self.scalar_dims.append(self.is_scalar_mask[index].sum().item())
        self.scalar_dims = torch.tensor(self.scalar_dims, dtype=torch.long)

        self.embedding = nn.Embedding(
            len(basis_info.atomic_numbers) ** 2, channels * self.scalar_dims.max().item()
        )

        edge_to_embedding_index = torch.arange(
            len(basis_info.atomic_numbers) ** 2,
            dtype=torch.long,
        ).view(len(basis_info.atomic_numbers), len(basis_info.atomic_numbers))
        self.register_buffer("edge_to_embedding_index", edge_to_embedding_index)

    def forward(self, batch) -> Tensor:
        atomic_numbers = batch.atomic_numbers
        edge_index = batch.edge_index

        atomic_numbers_a = atomic_numbers[edge_index[0]]
        atomic_numbers_b = atomic_numbers[edge_index[1]]

        edge_a = torch.zeros(
            edge_index.shape[1],
            self.basis_dim_per_atom.max().item(),
            self.channels,
            device=edge_index.device,
            dtype=self.embedding.weight.dtype,
        )
        edge_b = torch.zeros(
            edge_index.shape[1],
            self.basis_dim_per_atom.max().item(),
            self.channels,
            device=edge_index.device,
            dtype=self.embedding.weight.dtype,
        )

        scalar_features_a = self.embedding(
            self.edge_to_embedding_index[
                self.atomic_number_to_atom_index[atomic_numbers_a],
                self.atomic_number_to_atom_index[atomic_numbers_b],
            ]
        ).view(-1, self.scalar_dims.max().item(), self.channels)
        scalar_features_b = self.embedding(
            self.edge_to_embedding_index[
                self.atomic_number_to_atom_index[atomic_numbers_b],
                self.atomic_number_to_atom_index[atomic_numbers_a],
            ]
        ).view(-1, self.scalar_dims.max().item(), self.channels)
        for i, a in enumerate(self.basis_info.atomic_numbers):
            for j, b in enumerate(self.basis_info.atomic_numbers):
                edge_index_mask = (atomic_numbers_a == a) & (atomic_numbers_b == b)
                edge_a[edge_index_mask, : self.scalar_dims[i]] = scalar_features_a[
                    edge_index_mask, : self.scalar_dims[i]
                ]
                edge_b[edge_index_mask, : self.scalar_dims[j]] = scalar_features_b[
                    edge_index_mask, : self.scalar_dims[j]
                ]

        return torch.cat([edge_a, edge_b], dim=-1), edge_index


class EdgeEmbeddingV2(nn.Module):
    def __init__(self, basis_info: BasisInfo, channels: int = 32):
        super().__init__()
        self.basis_info = basis_info
        self.channels = channels

        self.edge_embedding = EdgeEmbedding(basis_info, channels=channels)
        self.node_cor = EdgeEmbedding(basis_info, channels=channels)

    def forward(self, batch) -> Tensor:
        edge_index = batch.edge_index

        edge_features = self.edge_embedding(batch)[0]
        node_cor_per_edge = self.node_cor(batch)[0]

        edge_features_a, edge_features_b = (
            edge_features[..., : self.channels],
            edge_features[..., self.channels :],
        )
        node_cor_a_per_edge, node_cor_b_per_edge = (
            node_cor_per_edge[..., : self.channels],
            node_cor_per_edge[..., self.channels :],
        )

        self_loop_mask = edge_index[0] == edge_index[1]

        node_cor_a = torch.zeros(
            self_loop_mask.sum(),
            node_cor_a_per_edge.shape[1],
            self.channels,
            device=edge_index.device,
            dtype=edge_features_a.dtype,
        )
        node_cor_b = torch.zeros(
            self_loop_mask.sum(),
            node_cor_b_per_edge.shape[1],
            self.channels,
            device=edge_index.device,
            dtype=edge_features_b.dtype,
        )

        node_cor_a.index_add(0, edge_index[0], node_cor_a_per_edge)
        node_cor_b.index_add(0, edge_index[0], node_cor_b_per_edge)

        out_features_a = torch.empty_like(edge_features_a)
        out_features_b = torch.empty_like(edge_features_b)

        # takes only is correct if the edge_index is sorted
        out_features_a[~self_loop_mask] = edge_features_a[~self_loop_mask]
        out_features_b[~self_loop_mask] = edge_features_b[~self_loop_mask]
        out_features_a[self_loop_mask] = edge_features_a[self_loop_mask] + node_cor_a
        out_features_b[self_loop_mask] = edge_features_b[self_loop_mask] + node_cor_b

        return torch.cat([out_features_a, out_features_b], dim=-1), edge_index


class EdgeDistanceEmbedding(nn.Module):
    def __init__(
        self,
        basis_info: BasisInfo,
        channels: int = 32,
        n_embeddings: int = 5,
        num_gaussians: int = 32,
    ):
        super().__init__()
        self.basis_info = basis_info
        self.channels = channels
        self.n_embeddings = n_embeddings

        # one EdgeEmbedding for n_embeddings
        self.edge_embedding = EdgeEmbedding(basis_info, channels=channels * n_embeddings)

        self.gaussian_smearing = GaussianSmearing(
            0.0,
            3.0,
            num_gaussians,
            1.0,
        )

        self.distance_mlp_a = torch.nn.Sequential(
            nn.Linear(num_gaussians, num_gaussians),
            nn.SiLU(),
            nn.Linear(num_gaussians, n_embeddings),
        )
        self.distance_mlp_b = torch.nn.Sequential(
            nn.Linear(num_gaussians, num_gaussians),
            nn.SiLU(),
            nn.Linear(num_gaussians, n_embeddings),
        )

    def forward(self, batch) -> Tensor:
        coords = batch.pos
        edge_index = batch.edge_index

        distances = torch.norm(coords[edge_index[0]] - coords[edge_index[1]], dim=-1)
        distance_embedding = self.gaussian_smearing(distances)
        distance_factors_a = self.distance_mlp_a(distance_embedding)
        distance_factors_b = self.distance_mlp_b(distance_embedding)
        edge_features = self.edge_embedding(batch)[0]
        edge_features_a, edge_features_b = (
            edge_features[..., : self.channels * self.n_embeddings],
            edge_features[..., self.channels * self.n_embeddings :],
        )
        edge_features_a = edge_features_a.view(
            edge_features_a.shape[0], edge_features_a.shape[1], self.channels, self.n_embeddings
        )
        edge_features_b = edge_features_b.view(
            edge_features_b.shape[0], edge_features_b.shape[1], self.channels, self.n_embeddings
        )
        edge_features_a = edge_features_a * distance_factors_a[:, None, None, :]
        edge_features_b = edge_features_b * distance_factors_b[:, None, None, :]
        edge_features_a = torch.sum(edge_features_a, dim=-1)  # sum over the n_embeddings dimension
        edge_features_b = torch.sum(edge_features_b, dim=-1)  # sum over the n_embeddings dimension
        return torch.cat([edge_features_a, edge_features_b], dim=-1), edge_index


class BOA(nn.Module):
    def __init__(
        self,
        basis_info: BasisInfo,
        boa_stack: BoaBlockStack,
        initial_guess_module: nn.Module,
        direct_gs_prediction: bool = False,
        num_orbitals: int = 0,
        axial_seed: bool = False,
    ) -> None:
        super().__init__()
        self.basis_info = basis_info
        self.boa_stack = boa_stack

        self.num_channels = boa_stack.blocks[0].in_channels

        self.direct_gs_prediction = direct_gs_prediction
        self.num_orbitals = num_orbitals
        self.node_embedding = NodeEmbedding(
            basis_info, channels=self.num_channels, axial_seed=axial_seed
        )
        self.edge_embedding = ReducedEdgeEmbedding(
            basis_info, channels=self.num_channels, axial_seed=axial_seed
        )

        self.initial_guess_module = initial_guess_module

    def forward(self, batch) -> Tuple[Tensor, Tensor]:
        if self.node_embedding is not None:
            x = self.node_embedding(
                batch.atomic_numbers, batch.coeff_ind_to_node_ind, batch.pos, batch.edge_index
            )

        edge_features = self.edge_embedding(batch)[0]
        edge_features_a, edge_features_b = (
            edge_features[..., : self.num_channels],
            edge_features[..., self.num_channels :],
        )

        init_guess_delta, edge_features_a, edge_features_b = self.boa_stack(
            x,
            batch.coeff_ind_to_node_ind,
            batch.atomic_numbers,
            batch.edge_index,
            batch.message_edge_index,
            edge_features_a=edge_features_a,
            edge_features_b=edge_features_b,
            edge_matrices=batch.edge_matrices,
            message_edge_matrices=batch.message_edge_matrices,
        )

        full_edge_index = batch.edge_index
        edge_features_a = edge_features_a.view(
            edge_features_a.shape[0], edge_features_a.shape[1], -1, self.num_orbitals
        ).mean(-2)
        edge_features_b = edge_features_b.view(
            edge_features_b.shape[0], edge_features_b.shape[1], -1, self.num_orbitals
        ).mean(-2)

        init_guess_edge = self.initial_guess_module(batch)[0]
        init_guess_edge_a, init_guess_edge_b = (
            init_guess_edge[..., 0][..., None],
            init_guess_edge[..., 1][..., None],
        )
        init_guess_delta = torch.cat(
            [edge_features_a, init_guess_edge_a, edge_features_b, init_guess_edge_b], dim=-1
        )

        return init_guess_delta, full_edge_index
