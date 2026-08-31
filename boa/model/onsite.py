"""The per-element on-site prior: the rotation-invariant part of a diagonal block.

Why
---
A diagonal block ``D_ii`` of the density matrix is dominated by a part that
depends on nothing but the element -- a carbon 1s is doubly occupied in every
molecule. Predicting that from scratch spends capacity on a constant and hides
the part that actually varies: ``rel_fro`` can look respectable while the
chemistry-carrying residual is poorly resolved, and an energy error that does
not grow with molecule size is what that looks like from the outside.

maloq handles this by shifting the ``l = 0`` components of the on-site blocks by
a per-element mean before training and adding it back afterwards
(``src/maloq/dataset_utils/get_scale_shift.py``, applied in
``fock_utils/fock_targets_batched.py``). This module is the same idea in our AO
basis, applied as a **bias on the prediction** rather than a shift of the
target: the model output stays a density matrix, so the energy, the dipole,
``tr(DS)``, ``rel_fro`` and the logged images all keep meaning what they did, and
there is no un-shifting step to forget at inference.

What may be added
-----------------
Rotating the molecule sends :math:`D_{ii} \\to R D_{ii} R^T` with :math:`R` block
diagonal over shells (one Wigner matrix per shell). A constant added to the
prediction has to survive that for the model to stay equivariant, which leaves
exactly the shell-pair trace parts, and only for pairs of equal ``l``:

.. math::

    D_{ab} \;\\longrightarrow\; \\frac{\\operatorname{tr} D_{ab}}{2l+1}\\, I_{2l+1},
    \\qquad l_a = l_b .

One scalar per same-``l`` shell pair -- maloq's "``l = 0`` shift", written in the
AO basis instead of the irrep basis. Off-diagonal (edge) blocks get nothing:
their mean is geometry dependent, and a constant there is not invariant.

Both the trace and the identity are unchanged by any orthogonal transformation
inside a shell, so this prior is independent of the ``m`` ordering convention.
Unlike the rest of the block basis it cannot be broken by an AO-ordering
mistake, which also makes it a poor diagnostic for one.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

pylogger = logging.getLogger(__name__)


def same_l_shell_pairs(basis_info, atom_index: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Ordered shell pairs of one element that share an ``l``.

    Args:
        basis_info: the basis the density matrix is written in.
        atom_index: the element's index in ``basis_info`` (not its atomic number).

    Returns:
        One ``(l, rows, cols)`` per ordered pair, with ``rows`` and ``cols``
        atom-local AO indices of length ``2l + 1``. Reading
        ``block[rows, cols].sum()`` is the trace of that shell-pair sub-block.
    """
    local = basis_info.atom_ind_to_basis_function_ind[atom_index]
    start, dim = int(local[0]), len(local)
    ls = basis_info.l_per_basis_func[start : start + dim]
    begins = np.flatnonzero(basis_info.shell_beginning_mask[start : start + dim])
    shells = [(int(ls[b]), int(b)) for b in begins]

    if sum(2 * l + 1 for l, _ in shells) != dim:
        raise ValueError(
            f"shells of atom type {atom_index} cover "
            f"{sum(2 * l + 1 for l, _ in shells)} of {dim} basis functions; "
            "the shell layout in basis_info is inconsistent."
        )

    pairs = []
    for l_a, off_a in shells:
        for l_b, off_b in shells:
            if l_a != l_b:
                continue
            step = np.arange(2 * l_a + 1)
            pairs.append((l_a, off_a + step, off_b + step))
    return pairs


class OnsiteL0Statistics:
    """Running per-element mean of the ``l = 0`` part of the on-site blocks.

    Streaming rather than collecting: a full pass over a training split is tens
    of thousands of matrices, and only the pair-wise sums are needed.
    """

    def __init__(self, basis_info):
        self.basis_info = basis_info
        self.n_types = len(basis_info.atomic_numbers)
        self.max_basis_dim = int(max(basis_info.basis_dim_per_atom))
        self.pairs = [same_l_shell_pairs(basis_info, t) for t in range(self.n_types)]
        self.sum = [np.zeros(len(p)) for p in self.pairs]
        self.sum_sq = [np.zeros(len(p)) for p in self.pairs]
        self.count = np.zeros(self.n_types, dtype=np.int64)
        self.n_molecules = 0

    def update(self, atom_index: np.ndarray, blocks: np.ndarray) -> None:
        """Accumulate the on-site blocks of one batch.

        Args:
            atom_index: ``(n_atom,)`` element index of every atom.
            blocks: ``(n_atom, max_basis_dim, max_basis_dim)`` diagonal blocks.
                Entries beyond an atom's own AO count are never read, so they may
                be anything.
        """
        atom_index = np.asarray(atom_index)
        blocks = np.asarray(blocks, dtype=np.float64)
        for t in np.unique(atom_index):
            sel = blocks[atom_index == t]
            for k, (l, rows, cols) in enumerate(self.pairs[t]):
                # tr(D_ab) / (2l+1): the only rotation-invariant part of the pair.
                values = sel[:, rows, cols].sum(axis=1) / (2 * l + 1)
                self.sum[t][k] += values.sum()
                self.sum_sq[t][k] += np.square(values).sum()
            self.count[t] += len(sel)

    def means(self, atom_index: int) -> np.ndarray:
        n = max(int(self.count[atom_index]), 1)
        return self.sum[atom_index] / n

    def stds(self, atom_index: int) -> np.ndarray:
        n = max(int(self.count[atom_index]), 1)
        var = self.sum_sq[atom_index] / n - np.square(self.means(atom_index))
        return np.sqrt(np.clip(var, 0.0, None))

    def bias(self) -> torch.Tensor:
        """The prior as ``(n_types, max_basis_dim, max_basis_dim)``.

        Elements never seen get zeros, which makes the prior a no-op for them
        rather than a guess.
        """
        out = np.zeros((self.n_types, self.max_basis_dim, self.max_basis_dim))
        for t in range(self.n_types):
            if self.count[t] == 0:
                pylogger.warning(
                    f"no atom of type {t} (Z={self.basis_info.atomic_numbers[t]}) in the "
                    "statistics pass; its on-site prior stays zero."
                )
                continue
            means = self.means(t)
            for k, (_, rows, cols) in enumerate(self.pairs[t]):
                out[t, rows, cols] = means[k]
            # (a,b) and (b,a) are accumulated separately and agree only to
            # sampling noise; the prior has to be symmetric like the matrix.
            out[t] = 0.5 * (out[t] + out[t].T)
        return torch.as_tensor(out, dtype=torch.get_default_dtype())

    def summary(self) -> list[str]:
        """Per-element table of the shell-pair means, for the training log."""
        lines = [
            f"on-site l=0 prior from {self.n_molecules} molecules "
            f"({int(self.count.sum())} atoms)"
        ]
        for t in range(self.n_types):
            z = int(self.basis_info.atomic_numbers[t])
            if self.count[t] == 0:
                lines.append(f"  Z={z:<3} unseen")
                continue
            means, stds = self.means(t), self.stds(t)
            sem = stds / np.sqrt(int(self.count[t]))
            lines.append(f"  Z={z:<3} {int(self.count[t])} atoms")
            for k, (l, rows, cols) in enumerate(self.pairs[t]):
                # Only the diagonal shell pairs are worth a line each; the
                # cross terms follow the same pattern and would treble the table.
                if rows[0] != cols[0]:
                    continue
                lines.append(
                    f"      l={l} shell@{rows[0]:<3} mean {means[k]:>10.5f}  "
                    f"std {stds[k]:>9.5f}  sem {sem[k]:.2e}"
                )
        return lines


def save_onsite_bias(path, stats: OnsiteL0Statistics, metadata: dict) -> None:
    """Write the prior plus enough provenance to tell a stale cache from a fresh one."""
    torch.save(
        {
            "bias": stats.bias(),
            "means": {t: stats.means(t) for t in range(stats.n_types)},
            "stds": {t: stats.stds(t) for t in range(stats.n_types)},
            "count": stats.count,
            "n_molecules": stats.n_molecules,
            "atomic_numbers": np.asarray(stats.basis_info.atomic_numbers),
            "basis_dim_per_atom": np.asarray(stats.basis_info.basis_dim_per_atom),
            **metadata,
        },
        path,
    )


def load_onsite_bias(path, basis_info) -> torch.Tensor:
    """Read a saved prior, refusing one computed in a different basis.

    A prior from another basis or element set would place numbers in the wrong
    AO slots and be silently wrong, so the layout is checked rather than trusted.
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    stored_z = np.asarray(data["atomic_numbers"])
    stored_dim = np.asarray(data["basis_dim_per_atom"])
    if not np.array_equal(stored_z, np.asarray(basis_info.atomic_numbers)) or not np.array_equal(
        stored_dim, np.asarray(basis_info.basis_dim_per_atom)
    ):
        raise ValueError(
            f"{path} was computed for elements {stored_z.tolist()} with AO counts "
            f"{stored_dim.tolist()}, but this run uses "
            f"{np.asarray(basis_info.atomic_numbers).tolist()} / "
            f"{np.asarray(basis_info.basis_dim_per_atom).tolist()}."
        )
    return data["bias"].to(torch.get_default_dtype())
