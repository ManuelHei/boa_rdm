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

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from lightning import LightningModule
from torch import Tensor
from torch_ema import ExponentialMovingAverage

pylogger = logging.getLogger(__name__)


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

        if self.criterion not in ("mse", "mae", "huber"):
            raise ValueError(
                f"Unknown criterion '{self.criterion}'; expected one of 'mse', 'mae', 'huber'."
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
        """Map every (atom, padded coefficient slot) pair to an AO index.

        Returns:
            ao_index: ``(n_atom, max_basis_dim)``. Entry ``[k, p]`` is the AO
                index of slot ``p`` of atom ``k`` within its own molecule, or
                ``n_ao`` (a scatter sink for the padding) if the slot is unused.
            n_ao_per_mol: ``(n_mol,)`` number of AOs of each molecule.
            n_ao: the largest of those, i.e. the size the matrices are padded to.
        """
        atom_index = self.atomic_number_to_atom_index[batch.atomic_numbers.long()]
        n_bas_atom = self.basis_dim_per_atom[atom_index]  # (n_atom,)

        n_mol = int(batch.num_graphs)
        n_ao_per_mol = torch.zeros(n_mol, dtype=torch.long, device=n_bas_atom.device)
        n_ao_per_mol.index_add_(0, batch.batch, n_bas_atom)
        n_ao = int(n_ao_per_mol.max())

        # AO offset of each atom, first globally then relative to its molecule.
        global_offset = torch.cumsum(n_bas_atom, dim=0) - n_bas_atom
        mol_start = torch.cumsum(n_ao_per_mol, dim=0) - n_ao_per_mol
        ao_offset = global_offset - mol_start[batch.batch]

        slot = torch.arange(self.max_basis_dim, device=n_bas_atom.device)
        ao_index = torch.where(
            slot[None, :] < n_bas_atom[:, None],
            ao_offset[:, None] + slot[None, :],
            torch.full_like(n_bas_atom[:, None], n_ao),
        )
        return ao_index, n_ao_per_mol, n_ao

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

        rows = ao_index[edge_index[0]]  # (n_edge, max_basis_dim)
        cols = ao_index[edge_index[1]]
        mol = batch.batch[edge_index[0]]  # (n_edge,)

        # Scatter into a (n_mol, n_ao + 1, n_ao + 1) buffer; index n_ao is the
        # sink that absorbs the padded coefficient slots, and is sliced off.
        n_mol = int(batch.num_graphs)
        pad = n_ao + 1
        flat = (
            mol[:, None, None] * pad * pad + rows[:, :, None] * pad + cols[:, None, :]
        ).reshape(-1)
        rdm = torch.zeros(n_mol * pad * pad, dtype=blocks.dtype, device=blocks.device)
        rdm.index_add_(0, flat, blocks.reshape(-1))
        rdm = rdm.view(n_mol, pad, pad)[:, :n_ao, :n_ao]

        if self.symmetrize:
            rdm = 0.5 * (rdm + rdm.transpose(1, 2))

        rdm = rdm * self.scale

        ao_mask = torch.arange(n_ao, device=rdm.device)[None, :] < n_ao_per_mol[:, None]
        return rdm, ao_mask

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

        if isinstance(target, (list, tuple)):
            stacked = torch.zeros(n_mol, n_ao, n_ao, dtype=like.dtype, device=like.device)
            for i, matrix in enumerate(target):
                matrix = torch.as_tensor(matrix, dtype=like.dtype, device=like.device)
                stacked[i, : matrix.shape[0], : matrix.shape[1]] = matrix
            return stacked

        target = torch.as_tensor(target, dtype=like.dtype, device=like.device)
        if target.dim() == 3:
            pass
        elif target.dim() == 2 and target.shape[0] == n_mol * target.shape[1]:
            target = target.view(n_mol, target.shape[1], target.shape[1])
        else:
            raise ValueError(
                f"Cannot interpret '{self.target_key}' of shape {tuple(target.shape)} as "
                f"{n_mol} density matrices."
            )

        if target.shape[-2:] != (n_ao, n_ao):
            raise ValueError(
                f"Reference density matrix is {tuple(target.shape[-2:])} but the basis implies "
                f"{(n_ao, n_ao)} AOs. Check that the target was built with "
                f"basis_info.basis_dict."
            )
        return target

    # ------------------------------------------------------------------
    # Loss and metrics
    # ------------------------------------------------------------------
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
        gamma = rdm.detach().to(dtype=torch.float64)
        s = overlap.detach().to(dtype=torch.float64)
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

    def log_matrix_figures(self, pred, target, ao_mask):
        """Log the target, the prediction and their difference as images.

        Only the first molecule of the batch, on rank zero. Target and
        prediction share one colour scale so they are directly comparable; the
        error gets its own, since it is typically far smaller and would be
        invisible on the shared one.
        """
        writer = self._tb_writer()
        if writer is None:
            return
        import matplotlib.pyplot as plt

        n_ao = int(ao_mask[0].sum())
        t = target[0, :n_ao, :n_ao]
        p = pred[0, :n_ao, :n_ao]
        shared = max(float(t.abs().max()), float(p.abs().max()))

        figures = {
            "rdm/target": self._matrix_figure(t, "reference D", vmax=shared),
            "rdm/prediction": self._matrix_figure(p, "predicted D", vmax=shared),
            "rdm/error": self._matrix_figure(p - t, "predicted - reference"),
        }
        for tag, fig in figures.items():
            writer.add_figure(tag, fig, global_step=self.global_step)
            plt.close(fig)

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
        pylogger.info(
            f"tr(D S)  pred={n_pred:.6f}  ref={n_ref:.6f}  |err|={err:.6f}"
        )
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

        fig = self._occupation_figure(occ_pred, occ_ref)
        writer.add_figure("rdm/occupations", fig, global_step=self.global_step)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Lightning plumbing
    # ------------------------------------------------------------------
    def forward(self, batch):
        rdm, ao_mask = self.predict_rdm(batch)
        target = self.get_target(batch, rdm.shape[-1], rdm.shape[0], like=rdm)
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
                self.log_matrix_figures(rdm, target, ao_mask)
            self.log_electron_traces(batch, rdm, target)
            if self.log_occupation_figures:
                self.log_occupation_spectrum(batch, rdm, target, ao_mask)

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
