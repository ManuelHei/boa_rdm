"""Direct one-particle density matrix (1-RDM) prediction with BOA.

The construction
----------------
``BOA.forward`` returns ``coeffs`` of shape ``(n_edge, max_basis_dim, 2R)`` and
the (self-loop-inclusive) ``edge_index``. Splitting the last axis in half gives
two coefficient sets per edge, ``a`` and ``b``, each ``(n_edge, max_basis_dim, R)``
with ``R = num_orbitals + 1``. The grid module evaluates

.. math::

    \\rho(x) = \\sum_{e=(i,j)} \\sum_{r=1}^{R}
        \\Big[ \\sum_{\\mu \\in i} a_{e\\mu r}\\, \\phi_\\mu(x) \\Big]
        \\Big[ \\sum_{\\nu \\in j} b_{e\\nu r}\\, \\phi_\\nu(x) \\Big].

Because that is bilinear in the basis functions, it is exactly
:math:`\\rho(x) = \\sum_{\\mu\\nu} D_{\\mu\\nu} \\phi_\\mu(x) \\phi_\\nu(x)` with

.. math::

    D_{\\mu\\nu} = \\sum_{e=(i,j)} \\sum_r a_{e\\mu r} b_{e\\nu r},
    \\qquad \\mu \\in i,\\; \\nu \\in j.

So every edge contributes one rank-``R`` block at the ``(atom_i, atom_j)``
position of the matrix, and the whole density matrix is a scatter-add of those
blocks. No gaussians are ever evaluated. This identity is exact, not an
approximation: contracting the ``D`` built here against the basis reproduces the
grid module's prediction to float32 round-off.

``D`` is expressed in the plain AO basis of ``basis_info.basis_dict``, in pyscf's
AO ordering (atoms in order, and within an atom the basis functions in the order
``basis_dict`` lists them) -- the same ordering as ``coeff_ind_to_node_ind`` and
the overlap matrix that ``AddMessagePassingMatrix`` attaches. A reference 1-RDM
from pyscf in that basis is therefore directly comparable, elementwise.

Two caveats inherited from the grid module do *not* apply here, because no
gaussian is evaluated: ``use_radial_correction`` (which would otherwise deform
the basis away from plain GTOs) and ``orb_cutoff`` (a spatial truncation of the
evaluation) are both irrelevant. What *does* still matter is ``edge_index``:
atom pairs outside the graph radius contribute no block at all, so their entries
of ``D`` are structurally zero.

Scope
-----
This first version targets batches of different geometries of *the same*
molecule, which makes the density matrices uniformly shaped and cheap to batch.
Mixed batches are still handled -- matrices are zero-padded to the largest
molecule and an AO mask excludes the padding from the loss -- but that path is
untested.
"""

import logging

from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from lightning import LightningModule
from torch import Tensor
from torch_ema import ExponentialMovingAverage

from boa.model.onsite import OnsiteL0Statistics, load_onsite_bias, save_onsite_bias

pylogger = logging.getLogger(__name__)


def atom_layout(
    batch, dim_per_element: Tensor, atomic_number_to_atom_index: Tensor, max_dim: int
) -> tuple[Tensor, Tensor, int]:
    """Map every (atom, padded function slot) pair to a function index in its molecule.

    Shared by every matrix over atom-centred functions -- the AO density matrix
    and the aux-basis cumulant alike -- which differ only in how many functions
    each element carries.

    Returns:
        index: ``(n_atom, max_dim)``. Entry ``[k, p]`` is the index of slot ``p``
            of atom ``k`` within its own molecule, or ``n`` (a scatter sink for
            the padding) if the slot is unused.
        n_per_mol: ``(n_mol,)`` number of functions of each molecule.
        n: the largest of those, i.e. the size the matrices are padded to.
    """
    atom_index = atomic_number_to_atom_index[batch.atomic_numbers.long()]
    n_bas_atom = dim_per_element[atom_index]  # (n_atom,)

    n_mol = int(batch.num_graphs)
    n_per_mol = torch.zeros(n_mol, dtype=torch.long, device=n_bas_atom.device)
    n_per_mol.index_add_(0, batch.batch, n_bas_atom)
    n = int(n_per_mol.max())

    # Offset of each atom, first globally then relative to its molecule.
    global_offset = torch.cumsum(n_bas_atom, dim=0) - n_bas_atom
    mol_start = torch.cumsum(n_per_mol, dim=0) - n_per_mol
    offset = global_offset - mol_start[batch.batch]

    slot = torch.arange(max_dim, device=n_bas_atom.device)
    index = torch.where(
        slot[None, :] < n_bas_atom[:, None],
        offset[:, None] + slot[None, :],
        torch.full_like(n_bas_atom[:, None], n),
    )
    return index, n_per_mol, n


def scatter_dense(batch, blocks: Tensor, edge_index: Tensor, index: Tensor, n: int) -> Tensor:
    """Scatter-add one block per edge into ``(n_mol, n, n)``; repeated pairs accumulate.

    ``index`` is :func:`atom_layout`'s; its sink value ``n`` absorbs the padded
    slots into a row and column that are sliced off.
    """
    rows = index[edge_index[0]]  # (n_edge, max_dim)
    cols = index[edge_index[1]]
    mol = batch.batch[edge_index[0]]  # (n_edge,)
    n_mol = int(batch.num_graphs)
    pad = n + 1
    flat = (mol[:, None, None] * pad * pad + rows[:, :, None] * pad + cols[:, None, :]).reshape(-1)
    out = torch.zeros(n_mol * pad * pad, dtype=blocks.dtype, device=blocks.device)
    out.index_add_(0, flat, blocks.reshape(-1))
    return out.view(n_mol, pad, pad)[:, :n, :n]


def pad_matrices(matrices, n_mol: int, n: int, like: Tensor, name: str) -> Tensor:
    """A collated per-molecule matrix field as a zero-padded ``(n_mol, n, n)`` tensor.

    Accepts the three shapes a collater plausibly produces: an already stacked
    ``(n_mol, n, n)`` tensor, a list of per-molecule matrices (``list_keys``
    collation), or a default-collated ``(n_mol * n, n)`` block stack.
    """
    if isinstance(matrices, (list, tuple)):
        stacked = torch.zeros(n_mol, n, n, dtype=like.dtype, device=like.device)
        for i, matrix in enumerate(matrices):
            matrix = torch.as_tensor(matrix, dtype=like.dtype, device=like.device)
            stacked[i, : matrix.shape[0], : matrix.shape[1]] = matrix
        return stacked

    matrices = torch.as_tensor(matrices, dtype=like.dtype, device=like.device)
    if matrices.dim() == 2 and matrices.shape[0] == n_mol * matrices.shape[1]:
        matrices = matrices.view(n_mol, matrices.shape[1], matrices.shape[1])
    elif matrices.dim() != 3:
        raise ValueError(
            f"Cannot interpret '{name}' of shape {tuple(matrices.shape)} as {n_mol} matrices."
        )
    if matrices.shape[-2:] != (n, n):
        raise ValueError(
            f"'{name}' is {tuple(matrices.shape[-2:])} but the basis implies {(n, n)} "
            f"functions. Check that it was built in the basis the model is configured with."
        )
    return matrices


class RDMLightningModule(LightningModule):
    """Predict the one-particle density matrix directly, with no grid evaluation."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.basis_info = instantiate(self.hparams.basis_info)
        self.model = instantiate(self.hparams.net)

        self.target_key = self._hp("target_key", "rdm")
        self.criterion = self._hp("criterion", "mse")
        self.symmetrize = self._hp("symmetrize", True)
        self.huber_delta = float(self._hp("huber_delta", 1.0))
        self.log_electron_count = self._hp("log_electron_count", False)
        self.log_matrix_images = self._hp("log_matrix_images", True)
        self.log_matrix_images_every_n_val = int(self._hp("log_matrix_images_every_n_val", 1))
        self.log_occupation_figures = self._hp("log_occupation_figures", True)
        self.log_matrix_images_n_molecules = int(
            self._hp("log_matrix_images_n_molecules", 3)
        )
        self.log_energy = self._hp("log_energy", True)
        # The functional the targets were computed with. df_db_pbe is named
        # `pbe` but its qmmol group says `lda,vwn_rpa`; the stored value is what
        # reproduces the stored energy, so trust the group, not the name.
        self.energy_xc = self._hp("energy_xc", "lda,vwn_rpa")

        if self.criterion not in ("mse", "mae", "huber", "function"):
            raise ValueError(
                f"Unknown criterion '{self.criterion}'; expected one of 'mse', 'mae', "
                f"'huber', 'function'."
            )

        # Number of basis functions per atom is a pure function of the element,
        # so the whole AO layout can be derived from basis_info alone.
        self.register_buffer(
            "basis_dim_per_atom",
            torch.as_tensor(self.basis_info.basis_dim_per_atom, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "atomic_number_to_atom_index",
            torch.as_tensor(self.basis_info.atomic_number_to_atom_index, dtype=torch.long),
            persistent=False,
        )
        self.max_basis_dim = int(self.basis_dim_per_atom.max())

        self.register_buffer("scale", torch.tensor(float(self._hp("scale_factor", 1.0))))

        # The per-element on-site prior (see :mod:`boa.model.onsite`). Registered
        # only when it is switched on, so a checkpoint from a run without it
        # still loads into a model built from that run's config.
        self.onsite_bias_cfg = self._hp("onsite_bias", None)
        self.onsite_bias_max_molecules = int(self._hp("onsite_bias_max_molecules", 2000))
        self.onsite_bias_min_atoms = int(self._hp("onsite_bias_min_atoms", 32))
        if self.onsite_bias_cfg:
            self.register_buffer(
                "onsite_bias",
                torch.zeros(
                    len(self.basis_info.atomic_numbers),
                    self.max_basis_dim,
                    self.max_basis_dim,
                ),
            )
        else:
            self.onsite_bias = None

        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.hparams.train.ema.decay)
        trainer_cfg = self.hparams.train.trainer
        self.distributed = (
            trainer_cfg.get("strategy", "auto") == "ddp" and trainer_cfg.get("devices", 1) > 1
        )

    def _hp(self, name: str, default):
        """Read an optional hyperparameter, tolerating both dict and attribute access."""
        if hasattr(self.hparams, "get"):
            value = self.hparams.get(name, default)
        else:  # pragma: no cover - defensive
            value = getattr(self.hparams, name, default)
        return default if value is None else value

    # ------------------------------------------------------------------
    # AO bookkeeping
    # ------------------------------------------------------------------
    def ao_layout(self, batch) -> tuple[Tensor, Tensor, int]:
        """:func:`atom_layout` for the AO basis.

        Returns:
            ao_index: ``(n_atom, max_basis_dim)``. Entry ``[k, p]`` is the AO
                index of slot ``p`` of atom ``k`` within its own molecule, or
                ``n_ao`` (a scatter sink for the padding) if the slot is unused.
            n_ao_per_mol: ``(n_mol,)`` number of AOs of each molecule.
            n_ao: the largest of those, i.e. the size the matrices are padded to.
        """
        return atom_layout(
            batch, self.basis_dim_per_atom, self.atomic_number_to_atom_index, self.max_basis_dim
        )

    # ------------------------------------------------------------------
    # The density matrix
    # ------------------------------------------------------------------
    def build_rdm(self, batch, coeffs: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """Contract the per-edge low-rank blocks into a batched density matrix.

        Args:
            coeffs: ``(n_edge, max_basis_dim, 2R)`` as returned by ``BOA.forward``.
            edge_index: ``(2, n_edge)``, including self-loops.

        Returns:
            rdm: ``(n_mol, n_ao, n_ao)``.
            ao_mask: ``(n_mol, n_ao)``, ``True`` on real AOs. All ``True`` when
                every molecule in the batch is the same size.
        """
        if coeffs.shape[1] != self.max_basis_dim:
            raise ValueError(
                f"coeffs has {coeffs.shape[1]} basis slots but basis_info implies "
                f"{self.max_basis_dim}. Model and basis_info disagree."
            )
        if coeffs.shape[-1] % 2 != 0:
            raise ValueError(
                f"coeffs last dim is {coeffs.shape[-1]}, which is odd; it must split into "
                "the 'a' and 'b' halves of the bilinear expansion."
            )

        a, b = coeffs.chunk(2, dim=-1)  # each (n_edge, max_basis_dim, R)
        blocks = torch.einsum("emr,enr->emn", a, b)  # (n_edge, max_basis_dim, max_basis_dim)
        return self.scatter_blocks(batch, blocks, edge_index)

    def scatter_blocks(self, batch, blocks: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """Place one dense block per edge into a batched density matrix.

        Split out of :meth:`build_rdm` so a net that predicts the blocks
        directly -- :class:`boa.model.net.helm_net.HelmRDM` -- can reuse the
        placement, the padding sink and the symmetrisation without going through
        the bilinear coefficient construction.

        Args:
            blocks: ``(n_edge, max_basis_dim, max_basis_dim)``. Entries beyond an
                atom's own AO count are ignored, not required to be zero.
            edge_index: ``(2, n_edge)``. Repeated pairs accumulate.

        Returns:
            rdm: ``(n_mol, n_ao, n_ao)``.
            ao_mask: ``(n_mol, n_ao)``, ``True`` on real AOs.
        """
        ao_index, n_ao_per_mol, n_ao = self.ao_layout(batch)
        n_mol = int(batch.num_graphs)
        rdm = scatter_dense(batch, blocks, edge_index, ao_index, n_ao)

        if self.symmetrize:
            rdm = 0.5 * (rdm + rdm.transpose(1, 2))

        rdm = rdm * self.scale

        # After the scale, because the prior is in the units of the matrix and
        # not something the net's output scale should stretch.
        if self.onsite_bias is not None:
            rdm = rdm + self.onsite_prior(batch, ao_index, n_mol, n_ao, rdm)

        ao_mask = torch.arange(n_ao, device=rdm.device)[None, :] < n_ao_per_mol[:, None]
        return rdm, ao_mask

    # ------------------------------------------------------------------
    # The on-site prior
    # ------------------------------------------------------------------
    def onsite_prior(self, batch, ao_index: Tensor, n_mol: int, n_ao: int, like: Tensor) -> Tensor:
        """The per-element prior, placed on the diagonal blocks.

        Same placement as :meth:`scatter_blocks`, including the padding sink, so
        an atom whose element has fewer AOs than ``max_basis_dim`` contributes
        nothing outside its own block.
        """
        atom_index = self.atomic_number_to_atom_index[batch.atomic_numbers.long()]
        blocks = self.onsite_bias[atom_index].to(like.dtype)
        pad = n_ao + 1
        flat = (
            batch.batch[:, None, None] * pad * pad
            + ao_index[:, :, None] * pad
            + ao_index[:, None, :]
        ).reshape(-1)
        buffer = torch.zeros(n_mol * pad * pad, dtype=like.dtype, device=like.device)
        buffer.index_add_(0, flat, blocks.reshape(-1))
        return buffer.view(n_mol, pad, pad)[:, :n_ao, :n_ao]

    @staticmethod
    def onsite_blocks(batch, matrix: Tensor, ao_index: Tensor, n_ao: int) -> Tensor:
        """The diagonal ``(atom, atom)`` blocks of a batched matrix, one per atom.

        Padded coefficient slots are clamped onto a real AO rather than sent to
        the sink, because gathering has no sink -- the values they pick up are
        junk, and every consumer reads only the slots below the element's own AO
        count.
        """
        safe = ao_index.clamp(max=n_ao - 1)
        return matrix[batch.batch[:, None, None], safe[:, :, None], safe[:, None, :]]

    def collect_onsite_statistics(self, loader) -> OnsiteL0Statistics:
        """Mean on-site block per element, over the *targets* of ``loader``.

        The molecule cap is a floor as well as a ceiling: a rare element (six
        fluorines in the first two hundred df_db molecules) would otherwise get
        a prior from a handful of atoms, or none at all, purely because the pass
        stopped early. The scan continues past the cap until every element has
        ``onsite_bias_min_atoms``, and ends at the split either way.
        """
        stats = OnsiteL0Statistics(self.basis_info)
        cap = self.onsite_bias_max_molecules
        like = torch.zeros((), dtype=torch.get_default_dtype(), device=self.device)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                ao_index, _, n_ao = self.ao_layout(batch)
                n_mol = int(batch.num_graphs)
                target = self.get_target(batch, n_ao, n_mol, like=like)
                blocks = self.onsite_blocks(batch, target, ao_index, n_ao)
                atom_index = self.atomic_number_to_atom_index[batch.atomic_numbers.long()]
                stats.update(atom_index.cpu().numpy(), blocks.cpu().numpy())
                stats.n_molecules += n_mol
                if (
                    cap
                    and stats.n_molecules >= cap
                    and int(stats.count.min()) >= self.onsite_bias_min_atoms
                ):
                    break
        if cap and stats.n_molecules > cap:
            pylogger.info(
                f"read {stats.n_molecules} molecules rather than {cap}: the rarest "
                f"element needed {self.onsite_bias_min_atoms} atoms"
            )
        return stats

    def _onsite_cache_path(self, n_train: int) -> Path | None:
        """Where an ``auto`` prior is cached, or None if the config does not say.

        The training-split size is in the file name: a prior is a property of
        the split it was measured on, and silently reusing one from a different
        split is the failure this is meant to avoid.
        """
        try:
            root = Path(self.hparams.train.paths.data_dir) / self.hparams.train.data.dataset_name
        except Exception:  # noqa: BLE001 - a config without paths is a valid one
            return None
        if not root.is_dir():
            return None
        cap = self.onsite_bias_max_molecules
        return root / f"onsite_l0_bias__ntrain{n_train}__cap{cap}.pt"

    def _train_loader(self):
        """The training loader, however this run was started.

        ``train.py`` hands ``trainer.fit`` the dataloaders rather than the
        datamodule, so ``trainer.datamodule`` is None there; a run driven by a
        datamodule has it and no ``trainer.train_dataloader`` until later.
        """
        loader = getattr(self.trainer, "train_dataloader", None)
        if loader is not None:
            return loader
        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None:
            return datamodule.train_dataloader()
        raise RuntimeError(
            "onsite_bias needs the training data to measure the prior, and the "
            "trainer exposes neither a dataloader nor a datamodule. Point "
            "model.onsite_bias at a file written by an earlier run instead."
        )

    def on_train_start(self):
        """Fill the on-site prior before the first step, from cache or from the data.

        ``on_train_start`` rather than ``setup``: the training dataloader does
        not exist yet at setup when ``fit`` was handed loaders instead of a
        datamodule, and by here a resumed run has already loaded its checkpoint,
        so the measurement is skipped rather than redone and thrown away.
        """
        if self.onsite_bias is None:
            return
        if bool(self.onsite_bias.any()):
            return  # a resumed run already carries it

        cfg = self.onsite_bias_cfg
        if not isinstance(cfg, bool) and str(cfg) != "auto":
            path = Path(str(cfg))
            self.onsite_bias.copy_(load_onsite_bias(path, self.basis_info))
            pylogger.info(f"on-site l=0 prior loaded from {path}")
            return

        loader = self._train_loader()
        try:
            n_train = len(loader.dataset)
        except (TypeError, AttributeError):  # a dataset without __len__ is still trainable
            n_train = 0
        cache = self._onsite_cache_path(n_train)

        if cache is not None and cache.is_file():
            self.onsite_bias.copy_(load_onsite_bias(cache, self.basis_info))
            pylogger.info(f"on-site l=0 prior loaded from {cache}")
            return

        pylogger.info(
            f"measuring the on-site l=0 prior over the training split "
            f"(cap {self.onsite_bias_max_molecules or n_train} molecules)"
        )
        stats = self.collect_onsite_statistics(loader)
        for line in stats.summary():
            pylogger.info(line)
        self.onsite_bias.copy_(stats.bias().to(self.onsite_bias.dtype))
        if cache is not None:
            save_onsite_bias(
                cache,
                stats,
                {"n_train": n_train, "cap": self.onsite_bias_max_molecules},
            )
            pylogger.info(f"on-site l=0 prior written to {cache}")

    def predict_rdm(self, batch) -> tuple[Tensor, Tensor]:
        """Density matrix for a batch, without needing a target."""
        coeffs, edge_index = self.model(batch)
        return self.build_rdm(batch, coeffs, edge_index)

    # ------------------------------------------------------------------
    # Target handling
    # ------------------------------------------------------------------
    def get_target(self, batch, n_ao: int, n_mol: int, like: Tensor) -> Tensor:
        """Fetch the reference density matrix and normalise it to ``(n_mol, n_ao, n_ao)``.

        Accepts the three shapes a collater plausibly produces: an already
        stacked ``(n_mol, n, n)`` tensor (what ``OverlapMatrix`` collation
        gives), a list of per-molecule matrices (``list_keys`` collation), or a
        default-collated ``(n_mol * n, n)`` block stack.
        """
        target = None
        if hasattr(batch, self.target_key):
            target = getattr(batch, self.target_key)
        elif self.target_key in getattr(batch, "keys", list)():
            target = batch[self.target_key]

        if target is None:
            raise KeyError(
                f"No reference density matrix found on the batch under "
                f"'{self.target_key}'. Attach one per sample (shape (n_ao, n_ao), same AO "
                f"ordering as basis_info.basis_dict), or set model.target_key to the key you "
                f"used."
            )

        return pad_matrices(target, n_mol, n_ao, like, self.target_key)

    # ------------------------------------------------------------------
    # Loss and metrics
    # ------------------------------------------------------------------
    def overlap(self, batch, like: Tensor) -> Tensor | None:
        """The AO overlap as a dense ``(n_mol, n_ao, n_ao)`` tensor, or None.

        Only usable when ``AddMessagePassingMatrix`` ran with
        ``remove_diagonal: False``. With the diagonal blocks zeroed this is not
        the overlap at all and every quantity built on it is wrong, so the
        callers check the result rather than assuming it.
        """
        matrix = getattr(batch, "message_passing_matrix", None)
        if matrix is None:
            return None
        matrix = torch.as_tensor(matrix, dtype=like.dtype, device=like.device)
        if matrix.dim() != 3 or matrix.shape[-2:] != like.shape[-2:]:
            return None
        return matrix

    def function_space_loss(self, batch, pred: Tensor, target: Tensor, ao_mask: Tensor) -> Tensor:
        r"""The squared :math:`L^2` distance between the density matrices *as functions*.

        .. math::

            \int dr\, dr'\, \big[\gamma(r,r') - \tilde\gamma(r,r')\big]^2

        With :math:`\gamma(r,r') = \sum_{\mu\nu} D_{\mu\nu}\phi_\mu(r)\phi_\nu(r')`
        the integral factorises into two overlap matrices,

        .. math::

            \sum_{\mu\nu\alpha\beta} \Delta D_{\mu\nu} \Delta D_{\alpha\beta}
            S_{\mu\alpha} S_{\nu\beta}
            = \operatorname{tr}(\Delta D\, S\, \Delta D\, S)
            = \lVert S^{1/2} \Delta D\, S^{1/2} \rVert_F^2 ,

        so no grid and no eigendecomposition are needed -- one matmul per
        molecule. Verified against the four-index sum to 4e-15.

        The middle form is what is evaluated. The Löwdin form on the right is
        equal but needs :math:`S^{1/2}`, and the last expression makes the
        meaning plain: this is the elementwise error measured in the metric the
        basis actually induces, rather than in the coordinates it happens to be
        written in. Directions where the basis is nearly linearly dependent
        barely change the function and are penalised accordingly, which
        elementwise MSE cannot express.

        Averaged over molecules rather than over elements, so a 156-AO molecule
        does not outweigh a 50-AO one -- unlike :meth:`compute_loss`.
        """
        overlap = self.overlap(batch, like=pred)
        if overlap is None:
            raise ValueError(
                "criterion 'function' needs the AO overlap on the batch as "
                "'message_passing_matrix', shaped (n_mol, n_ao, n_ao). Add "
                "AddMessagePassingMatrix with remove_diagonal: False to the transform."
            )
        mask = ao_mask[:, :, None] & ao_mask[:, None, :]
        delta = (pred - target) * mask
        product = delta @ (overlap * mask)
        # tr(A A) = sum_ij A_ij A_ji; non-negative in exact arithmetic, and
        # clamped because round-off can take it just below zero near convergence
        return (product * product.transpose(1, 2)).sum(dim=(1, 2)).clamp(min=0).mean()

    def compute_loss(self, pred: Tensor, target: Tensor, ao_mask: Tensor) -> Tensor:
        mask = ao_mask[:, :, None] & ao_mask[:, None, :]
        n = mask.sum().clamp(min=1)
        diff = (pred - target) * mask
        if self.criterion == "mse":
            return diff.pow(2).sum() / n
        if self.criterion == "mae":
            return diff.abs().sum() / n
        return (
            F.huber_loss(pred * mask, target * mask, reduction="sum", delta=self.huber_delta) / n
        )

    @staticmethod
    def relative_frobenius(pred: Tensor, target: Tensor, ao_mask: Tensor) -> Tensor:
        """Per-molecule ``||D - D_ref||_F / ||D_ref||_F`` -- the RDM analogue of NMAPE."""
        mask = ao_mask[:, :, None] & ao_mask[:, None, :]
        num = (((pred - target) * mask) ** 2).sum(dim=(1, 2)).sqrt()
        den = ((target * mask) ** 2).sum(dim=(1, 2)).sqrt().clamp(min=1e-12)
        return num / den

    def off_graph_fraction(self, batch, target: Tensor, ao_mask: Tensor) -> Tensor:
        """Share of ``||D_ref||_F`` on AO pairs no edge reaches -- a floor on ``rel_fro``.

        Both models predict blocks per graph edge, so an atom pair outside the
        radius graph has no block at all and its entries of ``D`` are
        structurally zero. Whatever reference density sits there is unreachable
        error that no amount of training removes, and it is invisible in the
        loss: the prediction is zero, the gradient is zero, and only the metric
        moves.

        On a single molecule this was always 0 and the cutoff never mattered. On
        a set with 5-24 atoms it is 11.9% of ``||D||_F`` at the inherited 3.0 A
        and 0.01% at 8.0 A, so it is logged rather than assumed -- if someone
        lowers the radius, the floor appears in the metrics instead of being
        quietly absorbed into the reported error.
        """
        ao_index, _, n_ao = self.ao_layout(batch)
        edge_index = batch.edge_index
        rows = ao_index[edge_index[0]]
        cols = ao_index[edge_index[1]]
        mol = batch.batch[edge_index[0]]

        n_mol = int(batch.num_graphs)
        pad = n_ao + 1
        flat = (
            mol[:, None, None] * pad * pad + rows[:, :, None] * pad + cols[:, None, :]
        ).reshape(-1)
        covered = torch.zeros(n_mol * pad * pad, dtype=torch.bool, device=target.device)
        covered[flat] = True
        covered = covered.view(n_mol, pad, pad)[:, :n_ao, :n_ao]
        # the prediction is symmetrised, so a pair covered either way is reachable
        covered = covered | covered.transpose(1, 2)

        mask = ao_mask[:, :, None] & ao_mask[:, None, :]
        total = (target * mask).pow(2).sum().sqrt().clamp(min=1e-12)
        return (target * (mask & ~covered)).pow(2).sum().sqrt() / total

    def electron_count(self, batch, rdm: Tensor) -> Tensor | None:
        """``tr(D S)`` per molecule, if an untouched overlap matrix is on the batch.

        Only meaningful when ``AddMessagePassingMatrix`` ran with
        ``remove_diagonal: False``; otherwise the diagonal blocks of ``S`` are
        zeroed and the trace is not the electron count.
        """
        overlap = getattr(batch, "message_passing_matrix", None)
        if overlap is None:
            return None
        overlap = torch.as_tensor(overlap, dtype=rdm.dtype, device=rdm.device)
        if overlap.dim() != 3 or overlap.shape[-2:] != rdm.shape[-2:]:
            return None
        return torch.einsum("bmn,bmn->b", rdm, overlap)

    @staticmethod
    def occupation_numbers(rdm: Tensor, overlap: Tensor) -> Tensor:
        """Eigenvalues of ``S^{1/2} γ S^{1/2}``, largest first.

        Eigenvalues only. The eigenvectors of this sandwich live in the Löwdin
        basis, not the AO basis, so they are not returned.
        """
        # On CPU deliberately. These are at most 162x162, so the GPU buys nothing,
        # and `torch.linalg.eigh` there goes through cuSOLVER, which fails with
        # CUSOLVER_STATUS_INTERNAL_ERROR when it cannot allocate a handle on a
        # busy card -- killing a 20 000-step run for the sake of a diagnostic.
        gamma = rdm.detach().cpu().to(dtype=torch.float64)
        s = overlap.detach().cpu().to(dtype=torch.float64)
        gamma = 0.5 * (gamma + gamma.T)
        s = 0.5 * (s + s.T)
        evals_s, u = torch.linalg.eigh(s)
        s12 = (u * evals_s.clamp(min=1e-12).sqrt()) @ u.T
        gtilde = s12 @ gamma @ s12
        gtilde = 0.5 * (gtilde + gtilde.T)
        return torch.linalg.eigvalsh(gtilde).flip(0)

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    def _tb_writer(self):
        """The TensorBoard SummaryWriter, or None if we are not logging to one."""
        writer = getattr(getattr(self, "logger", None), "experiment", None)
        return writer if hasattr(writer, "add_figure") else None

    @staticmethod
    def _hill_formula(atomic_numbers) -> str:
        """Molecular formula in Hill notation: C, then H, then the rest A-Z.

        Without carbon the whole thing is alphabetical, hydrogen included --
        that is the convention, not an oversight (H2O, not OH2).
        """
        import ase.data

        counts = Counter(ase.data.chemical_symbols[int(z)] for z in atomic_numbers)
        if "C" in counts:
            order = ["C"] + (["H"] if "H" in counts else [])
            order += sorted(k for k in counts if k not in ("C", "H"))
        else:
            order = sorted(counts)
        return "".join(f"{s}{counts[s] if counts[s] > 1 else ''}" for s in order)

    @staticmethod
    def _molecule(batch, index: int = 0):
        """``(id, atomic numbers, positions)`` for molecule ``index`` of the batch.

        Every figure in this module goes through here, so the structure image and
        the matrices beside it cannot drift onto different molecules. Returns
        ``None`` if the batch does not carry the fields (the H2O QMLearn path).
        """
        pos = getattr(batch, "pos", None)
        types = getattr(batch, "atom_types", None)
        if pos is None or types is None:
            return None
        assign = getattr(batch, "batch", None)
        sel = slice(None) if assign is None else (assign == index)
        mol_id = getattr(batch, "id", None)
        if isinstance(mol_id, (list, tuple)):
            mol_id = mol_id[index] if index < len(mol_id) else None
        return mol_id, types[sel].detach().cpu(), pos[sel].detach().cpu()

    def _molecule_label(self, batch, index: int = 0) -> str:
        """``"C6H6 (mol_1)"`` -- what is actually in the panels, for the title.

        An unlabelled heatmap is not reproducible: with a shuffled loader or a
        changed split the same tag silently starts showing a different molecule.
        """
        info = self._molecule(batch, index)
        if info is None:
            return ""
        mol_id, types, _ = info
        label = self._hill_formula(types)
        if mol_id is not None:
            label += f" ({mol_id})"
        return label

    @staticmethod
    def _structure_figure(atomic_numbers, positions, title):
        """The molecule itself, so the matrices can be read against a geometry.

        Bonds are drawn where the separation is under 1.3x the sum of covalent
        radii -- a display heuristic, not chemistry, and it says nothing about
        which pairs the model's graph actually connects (that is
        ``edge_radius_ang``, currently 8.0 A, which reaches essentially every
        pair at this molecule size).
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from ase.data import chemical_symbols, covalent_radii
        from ase.data.colors import jmol_colors

        z = np.asarray(atomic_numbers, dtype=int)
        xyz = np.asarray(positions, dtype=float)

        fig = plt.figure(figsize=(4.2, 3.6), dpi=110)
        ax = fig.add_subplot(111, projection="3d")

        d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
        cutoff = 1.3 * (covalent_radii[z][:, None] + covalent_radii[z][None, :])
        for i, j in zip(*np.triu(np.logical_and(d < cutoff, d > 0), k=1).nonzero()):
            ax.plot(*zip(xyz[i], xyz[j]), color="0.45", linewidth=1.4, zorder=1)

        ax.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            s=[max(28.0, 190.0 * covalent_radii[n]) for n in z],
            c=[jmol_colors[n] for n in z],
            edgecolors="0.25", linewidths=0.5, depthshade=False, zorder=2,
        )
        for (x, y, zz), n in zip(xyz, z):
            ax.text(x, y, zz, chemical_symbols[n], fontsize=6, zorder=3)

        # Equal aspect: an anisotropically scaled molecule reads as the wrong one.
        # Padding is a fraction of the extent, not an absolute number of angstrom,
        # so a 4-atom molecule fills the panel as well as a 24-atom one does.
        # Padding and zoom are tied: zoom magnifies about the box centre, so an
        # atom sitting close to the box edge leaves the panel. 1.45 keeps every
        # atom at <=0.9 of the axis half-width after a 1.3x zoom, corners of a
        # rotated projection included.
        extent = float((xyz.max(axis=0) - xyz.min(axis=0)).max())
        span = 0.5 * max(extent, 1.0) * 1.45
        mid = 0.5 * (xyz.max(axis=0) + xyz.min(axis=0))
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)
        # zoom= is what actually fills the panel; the 3D projection is drawn well
        # inside its axes and set_position alone leaves the molecule small.
        ax.set_box_aspect((1, 1, 1), zoom=1.3)
        ax.set_title(title, fontsize=9, pad=0)
        ax.set_axis_off()
        # 3D axes reserve room for decorations that set_axis_off only hides, so
        # the panel has to be pushed out to the figure edges by hand.
        ax.set_position((0.0, 0.0, 1.0, 0.94))
        return fig

    @staticmethod
    def _matrix_figure(matrix, title, vmax=None):
        """One density-matrix panel on a zero-centred diverging scale.

        The scale is symmetric about zero because the sign of a density-matrix
        element is meaningful; a sequential map would hide it. ``vmax`` is put in
        the title so the colour scale is readable off the image itself -- it
        changes from step to step, and an unlabelled heatmap of a shrinking
        error looks identical to one of a constant error.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        matrix = matrix.detach().float().cpu().numpy()
        if vmax is None:
            vmax = float(np.abs(matrix).max())
        vmax = max(vmax, 1e-12)

        fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=110)
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{title}  (max |x| = {vmax:.3g})", fontsize=9)
        ax.set_xlabel("AO index", fontsize=8)
        ax.set_ylabel("AO index", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        return fig

    def log_matrix_figures(self, batch, pred, target, ao_mask):
        """Log target, prediction, error and structure for the first molecules.

        ``log_matrix_images_n_molecules`` of them (default 3), on rank zero. One
        molecule is a poor sample of a set spanning 3-27 atoms and 18-162 AOs:
        the first validation molecule happens to be 4-atom ammonia, so every
        figure logged before this showed an 18x18 matrix standing in for a
        typical 131x131 one.

        Within a molecule, target and prediction share one colour scale so they
        are directly comparable; the error gets its own, since it is typically
        far smaller and would be invisible on the shared one. Scales are **not**
        shared across molecules -- these are different systems, and a common
        scale would flatten the smaller one to nothing.

        Each image type gets its **own** top-level tag prefix, with one tag per
        molecule inside it. TensorBoard groups the Images tab by the text before
        the first ``/``, so the previous ``rdm/target``, ``rdm/prediction``,
        ``rdm/error`` collapsed into one section; the ``rdm_`` stem keeps the
        sections adjacent in the alphabetical list.
        """
        writer = self._tb_writer()
        if writer is None:
            return
        import matplotlib.pyplot as plt

        n_mol = min(int(self.log_matrix_images_n_molecules), int(pred.shape[0]))
        for i in range(n_mol):
            n_ao = int(ao_mask[i].sum())
            t = target[i, :n_ao, :n_ao]
            p = pred[i, :n_ao, :n_ao]
            shared = max(float(t.abs().max()), float(p.abs().max()))
            who = self._molecule_label(batch, i)
            suffix = f" -- {who}" if who else ""
            # The slot, not the molecule id, indexes the tag: the id would start a
            # fresh series whenever the loader order changed, and these are time
            # series that have to stay comparable across steps.
            slot = f"mol{i}"

            figures = {
                f"rdm_reference/{slot}": self._matrix_figure(
                    t, f"reference D{suffix}", vmax=shared
                ),
                f"rdm_prediction/{slot}": self._matrix_figure(
                    p, f"predicted D{suffix}", vmax=shared
                ),
                f"rdm_error/{slot}": self._matrix_figure(
                    p - t, f"predicted - reference{suffix}"
                ),
            }
            info = self._molecule(batch, i)
            if info is not None:
                _, types, positions = info
                figures[f"rdm_structure/{slot}"] = self._structure_figure(
                    types.numpy(), positions.numpy(), who or "structure"
                )
            for tag, fig in figures.items():
                writer.add_figure(tag, fig, global_step=self.global_step)
                plt.close(fig)

    def log_property_errors(self, batch, pred, target, ao_mask):
        """Physical observables from the predicted RDM against stored references.

        Total energy, kinetic energy and dipole, for the same molecules the
        figures show, through the same
        :func:`boa.model.properties.property_errors` the standalone sweep
        (``boa/eval_energy.py``) uses, so the two cannot drift apart.

        These are the only metrics here that are not basis-relative. ``rel_fro``
        weights an error by the size of the matrix element; these weight it by
        how much it changes the system, and the energy comes with a scale
        everyone knows -- 1 kcal/mol is chemical accuracy.

        References come from the batch, so the dataset must carry ``energy``,
        ``ke`` and ``dipole`` in ``extra_props``. Whichever are present get
        checked; if none are, this skips once and quietly.
        """
        if not self.log_energy:
            return
        available = {
            key: getattr(batch, attr, None)
            for key, attr in (("energy", "energy"), ("kinetic", "ke"), ("dipole", "dipole"))
        }
        available = {k: v for k, v in available.items() if v is not None}
        if not available:
            if not getattr(self, "_energy_warned", False):
                pylogger.warning(
                    "log_energy is on but the batch carries none of energy/ke/dipole; "
                    "add them to the dataset's extra_props. Skipping the check."
                )
                self._energy_warned = True
            return

        from boa.model.properties import property_errors, references_from_batch

        basis = self._hp("dft_basis_set", "6-31g*")
        n_mol = min(int(self.log_matrix_images_n_molecules), int(pred.shape[0]))
        collected = {}
        for i in range(n_mol):
            info = self._molecule(batch, i)
            if info is None:
                return
            mol_id, types, positions = info
            n_ao = int(ao_mask[i].sum())
            refs = references_from_batch(batch, i, int(pred.shape[0]))
            if not refs:
                return
            try:
                res = property_errors(
                    types.numpy(),
                    positions.numpy(),
                    pred[i, :n_ao, :n_ao].detach().float().cpu().numpy(),
                    refs,
                    basis=basis,
                    xc=self.energy_xc,
                )
            except Exception as exc:  # pyscf failure must not kill training
                pylogger.warning(f"property check failed for molecule {i} ({mol_id}): {exc}")
                continue
            parts = []
            if "energy_abs_error" in res:
                collected.setdefault("energy", []).append(res["energy_abs_error"])
                parts.append(
                    f"E {res['energy_pred']:.4f} vs {res['energy_ref']:.4f} Ha "
                    f"({res['energy_abs_error_kcal']:.2f} kcal/mol)"
                )
            if "kinetic_abs_error" in res:
                collected.setdefault("kinetic", []).append(res["kinetic_abs_error"])
                parts.append(
                    f"T {res['kinetic_pred']:.4f} vs {res['kinetic_ref']:.4f} Ha"
                )
            if "dipole_abs_error" in res:
                collected.setdefault("dipole", []).append(res["dipole_abs_error"])
                parts.append(
                    f"|Δmu| {res['dipole_abs_error']:.4f} a.u. "
                    f"(|mu_ref| {res['dipole_ref_norm']:.4f})"
                )
            pylogger.info(f"props {mol_id}: " + "; ".join(parts))

        writer = self._tb_writer()
        if writer is None or not collected or not hasattr(writer, "add_scalar"):
            return
        for key, errs in collected.items():
            mean = float(np.mean(errs))
            # Per slot as well as averaged: the molecules differ ~6x in size, so a
            # mean over three hides which one is failing.
            for i, e in enumerate(errs):
                writer.add_scalar(f"{key}/abs_err_mol{i}", e, self.global_step)
            writer.add_scalar(f"{key}/abs_err_mean", mean, self.global_step)
            if key == "energy":
                writer.add_scalar(
                    "energy/abs_err_mean_kcal", mean * 627.509474, self.global_step
                )

    def log_electron_traces(self, batch, pred, target):
        """Log ``tr(D S)`` for the first molecule of the batch.

        Same molecule as the heatmaps. Skips if the overlap is missing or is
        not the full (un-zeroed) matrix.
        """
        n_pred = self.electron_count(batch, pred)
        n_ref = self.electron_count(batch, target)
        if n_pred is None or n_ref is None:
            return
        n_pred = float(n_pred[0])
        n_ref = float(n_ref[0])
        err = abs(n_pred - n_ref)
        pylogger.info(f"tr(D S)  pred={n_pred:.6f}  ref={n_ref:.6f}  |err|={err:.6f}")
        writer = self._tb_writer()
        if writer is None or not hasattr(writer, "add_scalar"):
            return
        step = self.global_step
        writer.add_scalar("n_electrons/pred", n_pred, step)
        writer.add_scalar("n_electrons/ref", n_ref, step)
        writer.add_scalar("n_electrons/err", err, step)

    @staticmethod
    def _occupation_figure(occ_pred, occ_ref):
        """Sorted occupations of predicted vs reference, with the [0, 2] bounds."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        occ_pred = occ_pred.detach().float().cpu().numpy()
        occ_ref = occ_ref.detach().float().cpu().numpy()
        index = np.arange(len(occ_pred))

        fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=110)
        ax.plot(index, occ_ref, "o-", label="reference", markersize=3.5)
        ax.plot(index, occ_pred, "s-", label="predicted", markersize=3.5)
        ax.axhline(0.0, color="0.4", lw=0.8, ls="--")
        ax.axhline(2.0, color="0.4", lw=0.8, ls="--")
        ax.set_title(r"occupations of $S^{1/2} \gamma S^{1/2}$", fontsize=9)
        ax.set_xlabel("index (sorted)", fontsize=8)
        ax.set_ylabel("occupation", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def log_occupation_spectrum(self, batch, pred, target, ao_mask):
        """Log the occupation spectrum of the first molecule of the batch.

        Same molecule as the heatmaps and the trace. Skips if the overlap is
        missing or is not the full (un-zeroed) matrix.
        """
        overlap = getattr(batch, "message_passing_matrix", None)
        if overlap is None:
            return
        overlap = torch.as_tensor(overlap, dtype=pred.dtype, device=pred.device)
        if overlap.dim() != 3 or overlap.shape[-2:] != pred.shape[-2:]:
            return

        n_ao = int(ao_mask[0].sum())
        s = overlap[0, :n_ao, :n_ao]
        occ_pred = self.occupation_numbers(pred[0, :n_ao, :n_ao], s)
        occ_ref = self.occupation_numbers(target[0, :n_ao, :n_ao], s)
        # The scalars below stay on molecule 0 so their history is one series;
        # only the figures fan out over several molecules.

        def _extrema(occ):
            n_elec = float(occ.sum())
            lo = float(occ.min())
            hi = float(occ.max())
            return n_elec, lo, hi, max(0.0, -lo), max(0.0, hi - 2.0)

        n_p, lo_p, hi_p, below_p, above_p = _extrema(occ_pred)
        n_r, lo_r, hi_r, below_r, above_r = _extrema(occ_ref)
        pylogger.info(
            f"pred: N = {n_p:.4f}, min occ = {lo_p:.3f}, max occ = {hi_p:.3f}  "
            f"(below 0 = {below_p:.3f}, above 2 = {above_p:.3f})"
        )
        pylogger.info(
            f"ref : N = {n_r:.4f}, min occ = {lo_r:.3f}, max occ = {hi_r:.3f}  "
            f"(below 0 = {below_r:.3f}, above 2 = {above_r:.3f})"
        )

        writer = self._tb_writer()
        if writer is None:
            return
        if hasattr(writer, "add_scalar"):
            step = self.global_step
            writer.add_scalar("occ/pred_min", lo_p, step)
            writer.add_scalar("occ/pred_max", hi_p, step)
            writer.add_scalar("occ/pred_below_0", below_p, step)
            writer.add_scalar("occ/pred_above_2", above_p, step)
            writer.add_scalar("occ/ref_min", lo_r, step)
            writer.add_scalar("occ/ref_max", hi_r, step)
            writer.add_scalar("occ/ref_below_0", below_r, step)
            writer.add_scalar("occ/ref_above_2", above_r, step)

        import matplotlib.pyplot as plt

        n_show = min(int(self.log_matrix_images_n_molecules), int(pred.shape[0]))
        for i in range(n_show):
            n_ao_i = int(ao_mask[i].sum())
            s_i = overlap[i, :n_ao_i, :n_ao_i]
            fig = self._occupation_figure(
                self.occupation_numbers(pred[i, :n_ao_i, :n_ao_i], s_i),
                self.occupation_numbers(target[i, :n_ao_i, :n_ao_i], s_i),
            )
            who = self._molecule_label(batch, i)
            if who:
                fig.suptitle(who, fontsize=8)
            writer.add_figure(
                f"rdm_occupations/mol{i}", fig, global_step=self.global_step
            )
            plt.close(fig)

    # ------------------------------------------------------------------
    # Lightning plumbing
    # ------------------------------------------------------------------
    def forward(self, batch):
        rdm, ao_mask = self.predict_rdm(batch)
        target = self.get_target(batch, rdm.shape[-1], rdm.shape[0], like=rdm)
        if self.criterion == "function":
            loss = self.function_space_loss(batch, rdm, target, ao_mask)
        else:
            loss = self.compute_loss(rdm, target, ao_mask)
        return loss, rdm, target, ao_mask

    def training_step(self, batch, batch_idx):
        loss, rdm, target, ao_mask = self(batch)
        n_mol = rdm.shape[0]
        self.log_dict({"loss/train": loss}, batch_size=n_mol, sync_dist=self.distributed)

        interval = self._hp("log_train_metrics_interval", 10)
        if batch_idx % interval == 0:
            self.log_dict(
                {"rel_fro/train": self.relative_frobenius(rdm, target, ao_mask).mean()},
                batch_size=n_mol,
                sync_dist=self.distributed,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, rdm, target, ao_mask = self(batch)
        n_mol = rdm.shape[0]
        metrics = {
            "loss/val": loss,
            "rel_fro/val": self.relative_frobenius(rdm, target, ao_mask).mean(),
            # the unreachable floor, logged so a cutoff change cannot hide in it
            "off_graph/val": self.off_graph_fraction(batch, target, ao_mask),
        }
        if self.log_electron_count:
            n_elec = self.electron_count(batch, rdm)
            n_elec_ref = self.electron_count(batch, target)
            if n_elec is not None and n_elec_ref is not None:
                metrics["n_electrons_err/val"] = (n_elec - n_elec_ref).abs().mean()
        self.log_dict(metrics, batch_size=n_mol, sync_dist=self.distributed)

        if (
            batch_idx == 0
            and self.trainer.is_global_zero
            and self.trainer.sanity_checking is False
            and (self.current_epoch % self.log_matrix_images_every_n_val == 0)
        ):
            if self.log_matrix_images:
                self.log_matrix_figures(batch, rdm, target, ao_mask)
            self.log_electron_traces(batch, rdm, target)
            self.log_property_errors(batch, rdm, target, ao_mask)
            if self.log_occupation_figures:
                try:
                    self.log_occupation_spectrum(batch, rdm, target, ao_mask)
                except Exception as exc:
                    # Diagnostics must never end a run. The energy check already
                    # guards itself; this one used to raise straight through.
                    pylogger.warning(f"occupation spectrum failed: {exc}")

        return loss

    def test_step(self, batch, batch_idx):
        loss, rdm, target, ao_mask = self(batch)
        n_mol = rdm.shape[0]
        rel_fro = self.relative_frobenius(rdm, target, ao_mask)
        self.test_rel_fro.extend(rel_fro.tolist())
        metrics = {"loss/test": loss, "rel_fro/test": rel_fro.mean()}
        if self.log_electron_count:
            n_elec = self.electron_count(batch, rdm)
            n_elec_ref = self.electron_count(batch, target)
            if n_elec is not None and n_elec_ref is not None:
                metrics["n_electrons_err/test"] = (n_elec - n_elec_ref).abs().mean()
        self.log_dict(metrics, batch_size=n_mol, sync_dist=self.distributed)
        return rel_fro.mean()

    def on_test_start(self):
        self.test_rel_fro = []

    def on_test_end(self):
        if not self.test_rel_fro:
            return
        values = torch.tensor(self.test_rel_fro)
        pylogger.info(
            f"Test relative Frobenius error: {values.mean():.4e} +/- {values.std():.4e} "
            f"over {len(values)} molecules."
        )

    def configure_optimizers(self):
        opt = instantiate(self.hparams.train.optim, params=self.parameters(), _convert_="partial")
        scheduler = instantiate(self.hparams.train.lr_scheduler, optimizer=opt)

        if "lr_schedule_freq" in self.hparams.train:
            scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": self.hparams.train.lr_schedule_freq,
                "monitor": self.hparams.train.monitor.metric,
            }

        return {
            "optimizer": opt,
            "lr_scheduler": scheduler,
            "monitor": self.hparams.train.monitor.metric,
        }

    def on_fit_start(self):
        self.ema.to(self.device)

    def on_save_checkpoint(self, checkpoint):
        with self.ema.average_parameters():
            checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        try:
            if "ema_state_dict" in checkpoint:
                self.ema.load_state_dict(checkpoint["ema_state_dict"])
        except Exception as e:  # noqa: BLE001 - mirrors ChgLightningModule
            print(e)
            print("Failed to load EMA state dict. Please make sure this was intended.")

    def on_validation_epoch_start(self):
        self.ema.store()
        self.ema.copy_to(self.parameters())

    def on_validation_epoch_end(self):
        self.ema.restore()
        if isinstance(self.lr_schedulers(), torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_schedulers().step(
                self.trainer.callback_metrics[self.hparams.train.monitor.metric]
            )

    def on_before_zero_grad(self, optimizer):
        self.ema.update(self.parameters())

    def on_after_backward(self):
        total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), float("inf"), norm_type=2.0)
        self.log("trainer/grad_norm", total_norm)
