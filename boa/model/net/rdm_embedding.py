"""Read a density matrix back into backbone irreps -- the inverse of the head.

``ShellPairHead`` turns backbone irreps into a dense block::

    block = sum_c  coefficient_c * assembly_c

with ``assembly_c`` a Wigner 3j slice placed at one shell pair. This module runs
that backwards: project a block onto the same basis to recover the coefficients,
then map them into the backbone's ``(N, (lmax+1)^2, C)`` layout.

The projection is exact rather than a fit, which is the reason to do it this way
at all. The ``assembly`` matrices are orthogonal in the Frobenius inner product
and there are exactly as many of them as the block has degrees of freedom -- 105
for a symmetric 14x14 block, 196 for a general one -- so

    coefficient_c = <block, assembly_c> / ||assembly_c||^2

recovers the coefficients with no information lost. Verified both ways to float32
precision (1.7e-07): coefficients -> block -> coefficients, and an arbitrary
block -> coefficients -> block.

``||assembly_c||^2`` is 1 for most entries and 2 for the off-diagonal shell pairs
of the symmetric head, which carry a mirrored copy at the transposed position.
Dividing by it is what makes this a projection rather than merely a contraction.
"""

from __future__ import annotations

import torch
from e3nn import o3
from torch import Tensor, nn

from boa.model.net.helm_net import ShellPairHead


class ShellPairEmbedding(nn.Module):
    """Dense shell-pair blocks to backbone irreps.

    Mirrors :class:`ShellPairHead` argument for argument, and builds the same
    ``assembly`` basis, so the two are guaranteed to agree about the shell
    layout, the pyscf ordering and which ``L`` each pair contributes.

    Args:
        union_ls: degrees of the union shells, ascending.
        sphere_channels: channel count of the backbone features to produce.
        lmax: highest degree the backbone carries.
        symmetric: as in the head -- ``True`` for the on-site ``(i, i)`` blocks,
            whose basis spans only symmetric matrices, ``False`` for the general
            off-diagonal blocks.
    """

    def __init__(
        self, union_ls: list[int], sphere_channels: int, lmax: int, symmetric: bool
    ) -> None:
        super().__init__()
        # Built through the head so there is exactly one definition of the
        # basis: any change to the block convention moves both together.
        head = ShellPairHead(union_ls, sphere_channels, lmax, symmetric)
        self.dim = head.dim
        self.lmax = head.lmax
        self.sphere_channels = sphere_channels
        self.irreps_in = head.irreps_out  # the head's output is our input
        self.irreps_out = head.irreps_in  # ... and vice versa

        assembly = head.assembly
        self.register_buffer("assembly", assembly)
        self.register_buffer(
            "assembly_norm2", torch.einsum("cij,cij->c", assembly, assembly)
        )
        self.linear = o3.Linear(self.irreps_in, self.irreps_out)

    def project(self, blocks: Tensor) -> Tensor:
        """``(N, dim, dim)`` blocks to ``(N, n_coeff)`` irrep coefficients.

        The exact inverse of the head's assembly step. A block outside the span
        of the basis -- an antisymmetric part handed to the symmetric embedding,
        say -- is silently projected onto it, because that is what a projection
        does; the caller is responsible for passing the right kind of block.
        """
        assembly = self.assembly.to(blocks.dtype)
        return torch.einsum("nij,cij->nc", blocks, assembly) / self.assembly_norm2.to(
            blocks.dtype
        )

    def unstack_irreps(self, x: Tensor) -> Tensor:
        """e3nn's ``C x 0e + C x 1e + ...`` to ``(N, (lmax+1)^2, C)`` l-major.

        The inverse of :meth:`ShellPairHead.stack_irreps`, and the reason this
        class exists as more than a projection: the backbone wants a dense
        ``(N, (lmax+1)^2, C)`` tensor, while ``o3.Linear`` emits a flat
        irreps-major one.
        """
        n, c = x.shape[0], self.sphere_channels
        out = x.new_zeros(n, (self.lmax + 1) ** 2, c)
        offset = 0
        for l in range(self.lmax + 1):
            width = c * (2 * l + 1)
            # (N, C, 2l+1) within the flat layout, transposed into l-major.
            block = x[:, offset : offset + width].reshape(n, c, 2 * l + 1)
            out[:, l**2 : (l + 1) ** 2, :] = block.transpose(1, 2)
            offset += width
        return out

    def forward(self, blocks: Tensor) -> Tensor:
        """``(N, dim, dim)`` blocks to ``(N, (lmax+1)^2, C)`` backbone features."""
        return self.unstack_irreps(self.linear(self.project(blocks)))
