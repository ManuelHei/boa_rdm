"""HELM as a density-matrix predictor, in BOA's basis and AO ordering.

What HELM gives us
------------------
:class:`~boa.model.net.helm.esen_osh.eSEN_Backbone` (vendored from maloq, see
``boa/model/net/helm/__init__.py``) is an SO(2)-equivariant message-passing
network that carries *two* sets of SO(3) irreps: one per atom and one per
directed edge. Both come back in the global frame -- each block rotates its
messages into the edge frame, convolves there, and rotates back -- so a head can
read them directly, with shapes ``(n_atom, (lmax+1)^2, C)`` and
``(n_edge, (lmax+1)^2, C)``.

That is exactly the shape of the density matrix problem. :math:`D` is block
structured: the ``(i, i)`` block is a function of atom ``i`` alone, and the
``(i, j)`` block of the ordered pair, so node features feed the diagonal blocks
and edge features the off-diagonal ones.

From irreps to matrix blocks
----------------------------
A block between a shell of degree :math:`l_1` on one atom and a shell of degree
:math:`l_2` on the other is a :math:`(2l_1+1) \\times (2l_2+1)` array that
transforms as :math:`l_1 \\otimes l_2 = \\bigoplus_{L=|l_1-l_2|}^{l_1+l_2} L`.
So the head is a single ``o3.Linear`` from the backbone irreps to one copy of
every :math:`L` in that sum, for every shell pair, followed by a contraction
with the Wigner 3j symbols::

    block = sum_L sum_m c_{Lm} * sqrt(2L+1) * w3j(l1, l2, L)[:, :, m]

The :math:`\\sqrt{2L+1}` makes the basis matrices orthonormal in the Frobenius
norm (e3nn normalises ``wigner_3j`` over all three indices, so a single
:math:`m` slice has squared norm :math:`1/(2L+1)`), which keeps the coefficients
and the block on the same scale. The contraction is a fixed linear map, so it is
precomputed once into a single ``(n_coefficients, dim, dim)`` buffer and applied
with one ``einsum``. This is the same construction HELM's ``Fock_Irreps_Head``
performs; what is left out are its symmetry *reductions* over the alpha/beta
edge subspaces, which save parameters but change no predictions.

Conventions
-----------
Two orderings have to line up, and getting either wrong breaks equivariance
silently -- the model would still train, just towards a target it cannot
represent.

* **Axes.** e3nn's spherical harmonics are built in a frame whose ``(x, y, z)``
  is the physical ``(y, z, x)``; feeding permuted vectors makes component
  ``m + l`` of each irrep the standard real spherical harmonic :math:`Y_{lm}`.
  maloq does this permutation inside the backbone (``edge_dist[:, [2, 3, 1]]``),
  so this module hands it ``[dist, x, y, z]`` and lets it permute.
* **Shell layout.** pyscf orders the AOs of a ``p`` shell as
  ``(px, py, pz) = (m=+1, m=-1, m=0)``, not by ``m``; every other degree is
  ordered ``m = -l .. +l``, matching e3nn. :data:`PYSCF_SHELL_ORDER` holds that
  single permutation and it is folded into the assembly buffer.

Both were checked against pyscf directly: building the block-diagonal rotation
from these conventions reproduces ``mol.intor("int1e_ovlp")`` of a rotated
molecule from the unrotated one, d functions included.

Parity, and why every irrep here is declared even
-------------------------------------------------
The physical parity of the ``L`` component of the block between shells ``l1``
and ``l2`` is :math:`(-1)^{l_1 + l_2}` -- it comes from the two basis functions,
not from ``L``. So the s-p and p-d blocks are genuinely *odd*, and declaring
them ``(L, 1)`` as this head does is not the physically correct label.

It is, however, the only self-consistent one, because the backbone has no parity
to offer. eSEN carries a single feature per ``(l, m)``, shape
``(n, (lmax+1)^2, C)``, with no parity index -- it is an SO(3) architecture, as
eSCN and EquiformerV2 are. Declaring the head's outputs with their true parity
while the input stays parity-less does not *add* a symmetry; ``o3.Linear`` finds
no path from an even input to an odd output and silently emits zero. Measured:
every s-p and p-d component becomes structurally unreachable, and those blocks
carry 31% of the norm of the H2O target -- the same failure that caps BOA,
relocated.

Enforcing parity properly is an architecture change, not a relabelling: the
features would have to carry both parities at each degree (``0e + 0o + 1e + 1o +
...``, as NequIP and MACE do) so that both output parities have a path. What is
lost by not doing it is a constraint, not coverage: SO(3)-equivariance is
strictly weaker, so nothing in the target is unreachable, but the model must
learn parity from data instead of being built with it. The trained H2O model
obeys the physical law to 3.3e-5 against its own error of 1.8e-4 -- so roughly a
fifth of what is left, and a bounded prize rather than a blocker.

Shell padding
-------------
Elements carry different shells (in 6-31G*, H has ``2s`` and O has
``3s 2p 1d``), but the head must emit one fixed layout. As in HELM, that layout
is the *union* over elements -- the largest number of shells of each degree that
any element has -- and each atom selects the rows and columns of its own shells
out of it. An element with fewer ``s`` shells than the union takes the first
ones; which ones is arbitrary and consistent, and the network learns around it.
"""

import numpy as np
import torch
from e3nn import o3
from torch import Tensor, nn

from boa.data.basis_info import BasisInfo
from boa.model.net.helm import eSEN_Backbone

#: For each degree ``l``, the e3nn component that sits at each pyscf AO slot.
#: pyscf writes a ``p`` shell as ``(px, py, pz)`` while e3nn orders it by
#: ``m``, i.e. ``(y, z, x)``; every other degree agrees.
PYSCF_SHELL_ORDER = {1: [2, 0, 1]}


def pyscf_shell_permutation(l: int) -> np.ndarray:
    """``Q`` with ``Q @ v`` reordering an e3nn irrep of degree ``l`` into pyscf's AO order."""
    order = PYSCF_SHELL_ORDER.get(l, list(range(2 * l + 1)))
    q = np.zeros((2 * l + 1, 2 * l + 1))
    q[np.arange(2 * l + 1), order] = 1.0
    return q


def union_shell_degrees(basis_info: BasisInfo) -> list[int]:
    """The degrees of the union shell layout: the most shells of each ``l`` any element has.

    Returned ascending in ``l``, which is also the order pyscf lays out an
    atom's AOs, so an element's ``k``-th shell of degree ``l`` maps to the
    ``k``-th union shell of degree ``l``.
    """
    per_l: dict[int, int] = {}
    for irreps in basis_info.irreps_per_atom:
        counts: dict[int, int] = {}
        for mul, ir in irreps:
            counts[ir.l] = counts.get(ir.l, 0) + mul
        for l, count in counts.items():
            per_l[l] = max(per_l.get(l, 0), count)
    return [l for l in sorted(per_l) for _ in range(per_l[l])]


def element_ao_selection(basis_info: BasisInfo, union_ls: list[int]) -> list[list[int]]:
    """For each element type, the union-layout AO indices holding its own AOs, in pyscf order."""
    union_offset, offset = [], 0
    for l in union_ls:
        union_offset.append(offset)
        offset += 2 * l + 1

    selections = []
    for irreps in basis_info.irreps_per_atom:
        element_ls = [ir.l for mul, ir in irreps for _ in range(mul)]
        used: dict[int, int] = {}
        selection: list[int] = []
        for l in element_ls:
            k = used.get(l, 0)
            used[l] = k + 1
            # the k-th union shell of this degree
            shell = [i for i, ul in enumerate(union_ls) if ul == l][k]
            selection.extend(range(union_offset[shell], union_offset[shell] + 2 * l + 1))
        selections.append(selection)
    return selections


class ShellPairHead(nn.Module):
    """Map backbone irreps to a dense block over the union shell layout.

    Args:
        union_ls: degrees of the union shells, ascending.
        sphere_channels: channel count of the backbone features.
        lmax: highest degree the backbone carries. Must be at least
            ``2 * max(union_ls)``, or the tensor-product irreps of the widest
            shell pair have no path through the linear layer and those blocks
            are stuck at zero.
        symmetric: build a symmetric block, for the ``(i, i)`` diagonal. Only
            shell pairs ``a <= b`` are predicted and mirrored by transposition,
            and a shell paired with itself keeps only even ``L`` -- the odd
            terms of ``l (x) l`` are the antisymmetric ones, which cannot appear
            in a symmetric block.
    """

    def __init__(
        self, union_ls: list[int], sphere_channels: int, lmax: int, symmetric: bool
    ) -> None:
        super().__init__()
        max_l = max(union_ls)
        if lmax < 2 * max_l:
            raise ValueError(
                f"lmax={lmax} is below 2*max(l)={2 * max_l} required by the basis: the "
                f"l={max_l} shell pairs need irreps up to L={2 * max_l}, and a linear layer "
                f"cannot create them out of nothing."
            )
        self.sphere_channels = sphere_channels
        self.lmax = lmax

        offsets, offset = [], 0
        for l in union_ls:
            offsets.append(offset)
            offset += 2 * l + 1
        self.dim = offset

        pairs = [
            (a, b)
            for a in range(len(union_ls))
            for b in range(len(union_ls))
            if not symmetric or a <= b
        ]

        irreps_out, assembly = [], []
        for a, b in pairs:
            la, lb = union_ls[a], union_ls[b]
            q_a = pyscf_shell_permutation(la)
            q_b = pyscf_shell_permutation(lb)
            rows = slice(offsets[a], offsets[a] + 2 * la + 1)
            cols = slice(offsets[b], offsets[b] + 2 * lb + 1)
            for L in range(abs(la - lb), la + lb + 1):
                if symmetric and a == b and L % 2 == 1:
                    continue
                irreps_out.append((1, (L, 1)))
                # (2la+1, 2lb+1, 2L+1), rescaled so each m slice is a unit-norm
                # matrix, then permuted from e3nn's shell order into pyscf's.
                w3j = o3.wigner_3j(la, lb, L).numpy() * np.sqrt(2 * L + 1)
                w3j = np.einsum("pi,ijm,qj->pqm", q_a, w3j, q_b)
                for m in range(2 * L + 1):
                    block = np.zeros((self.dim, self.dim))
                    block[rows, cols] = w3j[:, :, m]
                    # a == b already lands on the diagonal shell block, and the
                    # even-L slices are symmetric there, so mirroring would
                    # double it; mirror the strict upper triangle only.
                    if symmetric and a != b:
                        block[cols, rows] = w3j[:, :, m].T
                    assembly.append(block)

        self.irreps_in = o3.Irreps([(sphere_channels, (l, 1)) for l in range(lmax + 1)])
        self.irreps_out = o3.Irreps(irreps_out)
        self.linear = o3.Linear(self.irreps_in, self.irreps_out)
        self.register_buffer(
            "assembly", torch.as_tensor(np.stack(assembly), dtype=torch.get_default_dtype())
        )

    def stack_irreps(self, x: Tensor) -> Tensor:
        """``(N, (lmax+1)^2, C)`` in l-major layout to e3nn's ``C x 0e + C x 1e + ...``."""
        x = x.transpose(1, 2)  # (N, C, (lmax+1)^2)
        return torch.cat(
            [x[:, :, l**2 : (l + 1) ** 2].reshape(x.shape[0], -1) for l in range(self.lmax + 1)],
            dim=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        """``(N, (lmax+1)^2, C)`` features to ``(N, dim, dim)`` blocks."""
        coefficients = self.linear(self.stack_irreps(x))
        return torch.einsum("nc,cij->nij", coefficients, self.assembly.to(coefficients.dtype))


class _HelmBatch:
    """The field names ``eSEN_Backbone.forward`` reads, over a BOA batch.

    The backbone was written against maloq's own loaders, so it wants ``charge``
    and ``spin_multiplicity`` per molecule, ``num_atoms_in_molecule``, and
    ``edge_attr`` holding ``[distance, x, y, z]``. It also probes the batch with
    ``in``, which is why this is a small class and not a namespace.
    """

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def __contains__(self, key):
        return key in self.__dict__


class HelmRDM(nn.Module):
    """Predict per-pair density-matrix blocks with the HELM backbone.

    ``forward`` returns ``(blocks, edge_index)``, the same interface
    :meth:`boa.model.rdm_module.RDMLightningModule.scatter_blocks` consumes:
    one ``(max_basis_dim, max_basis_dim)`` block per entry of ``edge_index``,
    padded with zeros beyond an atom's own AO count.

    Args:
        basis_info: the AO basis. Fixes the shell layout and, with it, the
            head's output irreps and the minimum ``lmax``.
        sphere_channels: width of the equivariant node/edge features.
        hidden_channels: width inside the SO(2) convolutions.
        edge_channels: width of the invariant edge embedding.
        num_distance_basis: number of gaussians in the distance expansion.
        num_layers: message-passing blocks. Each updates nodes and then edges.
        gaussian_cutoff: upper end of the distance expansion. HELM's configs put
            it at roughly twice the graph cutoff, so the gaussians still resolve
            the longest edge in the graph.
        gaussian_width: width of those gaussians.
        lmax: highest degree carried. Defaults to ``2 * max(l)`` of the basis,
            the smallest value that can represent every block.
        mmax: order truncation of the SO(2) convolutions. Defaults to ``lmax``.
        edge_index_key: which graph on the batch to use, for both message
            passing and the off-diagonal blocks. Atom pairs outside it get a
            structurally zero block, exactly as in the BOA density-matrix path.
        wigner_backend: ``torch`` or ``triton``.
        norm_type: equivariant normalisation used between blocks.
        max_num_elements: size of the element embedding table.
    """

    def __init__(
        self,
        basis_info: BasisInfo,
        sphere_channels: int = 64,
        hidden_channels: int = 64,
        edge_channels: int = 64,
        num_distance_basis: int = 64,
        num_layers: int = 3,
        gaussian_cutoff: float = 6.0,
        gaussian_width: float = 1.0,
        lmax: int | None = None,
        mmax: int | None = None,
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

        self.backbone = eSEN_Backbone(
            irreps_out=None,  # only the head's own irreps matter; unused by the backbone
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

        self.node_head = ShellPairHead(union_ls, sphere_channels, self.lmax, symmetric=True)
        self.edge_head = ShellPairHead(union_ls, sphere_channels, self.lmax, symmetric=False)

        # Union-layout AO indices per element, padded to max_basis_dim with a
        # sink row that the scatter later drops.
        selections = element_ao_selection(basis_info, union_ls)
        max_basis_dim = int(max(basis_info.basis_dim_per_atom))
        self.dim = self.node_head.dim
        selection = np.full((len(selections), max_basis_dim), self.dim, dtype=np.int64)
        for i, sel in enumerate(selections):
            selection[i, : len(sel)] = sel
        self.register_buffer("ao_selection", torch.as_tensor(selection))
        self.register_buffer(
            "atomic_number_to_atom_index",
            torch.as_tensor(basis_info.atomic_number_to_atom_index, dtype=torch.long),
        )

    def graph(self, batch) -> Tensor:
        """The message-passing graph: the configured radius graph without self-loops.

        Self-loops carry a zero displacement, and the edge frame is built by
        rotating that displacement onto the z axis -- undefined for the zero
        vector. The diagonal blocks come from the node features instead, so
        dropping the loops here loses nothing.
        """
        edge_index = getattr(batch, self.edge_index_key)
        return edge_index[:, edge_index[0] != edge_index[1]]

    def select(self, blocks: Tensor, atom_index_row: Tensor, atom_index_col: Tensor) -> Tensor:
        """Cut each union-layout block down to the AOs the two atoms actually carry."""
        padded = torch.zeros(
            blocks.shape[0],
            self.dim + 1,
            self.dim + 1,
            dtype=blocks.dtype,
            device=blocks.device,
        )
        padded[:, : self.dim, : self.dim] = blocks
        rows = self.ao_selection[atom_index_row]  # (n, max_basis_dim)
        cols = self.ao_selection[atom_index_col]
        n = torch.arange(blocks.shape[0], device=blocks.device)
        return padded[n[:, None, None], rows[:, :, None], cols[:, None, :]]

    def forward(self, batch) -> tuple[Tensor, Tensor]:
        edge_index = self.graph(batch)
        pos = batch.pos
        vec = pos[edge_index[1]] - pos[edge_index[0]]
        distance = vec.norm(dim=-1, keepdim=True)

        n_atom_per_mol = torch.bincount(batch.batch, minlength=int(batch.num_graphs))
        helm_batch = _HelmBatch(
            pos=pos,
            edge_index=edge_index,
            edge_attr=torch.cat([distance, vec], dim=-1),
            atomic_numbers=batch.atomic_numbers.long(),
            # Neutral closed-shell throughout: the datasets here carry no
            # charge or multiplicity, and the embeddings for them are constant.
            charge=torch.zeros(int(batch.num_graphs), dtype=torch.long, device=pos.device),
            spin_multiplicity=torch.ones(
                int(batch.num_graphs), dtype=torch.long, device=pos.device
            ),
            num_atoms_in_molecule=n_atom_per_mol,
        )

        embeddings = self.backbone(helm_batch)
        atom_index = self.atomic_number_to_atom_index[batch.atomic_numbers.long()]

        node_blocks = self.node_head(embeddings["node_embeddings"])
        node_blocks = self.select(node_blocks, atom_index, atom_index)

        edge_blocks = self.edge_head(embeddings["edge_embeddings"])
        edge_blocks = self.select(
            edge_blocks, atom_index[edge_index[0]], atom_index[edge_index[1]]
        )

        n_atom = pos.shape[0]
        self_loops = torch.arange(n_atom, device=pos.device).expand(2, n_atom)
        return (
            torch.cat([node_blocks, edge_blocks], dim=0),
            torch.cat([self_loops, edge_index], dim=1),
        )
