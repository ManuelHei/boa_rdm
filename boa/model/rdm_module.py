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

        ao_index, n_ao_per_mol, n_ao = self.ao_layout(batch)

        a, b = coeffs.chunk(2, dim=-1)  # each (n_edge, max_basis_dim, R)
        blocks = torch.einsum("emr,enr->emn", a, b)  # (n_edge, max_basis_dim, max_basis_dim)

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
            # Only the symmetric part is physical: rho(x) sees D and D^T alike,
            # so the antisymmetric part is unobservable and the reference is
            # symmetric by construction.
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

    # ------------------------------------------------------------------
    # Lightning plumbing
    # ------------------------------------------------------------------
    def forward(self, batch):
        coeffs, edge_index = self.model(batch)
        rdm, ao_mask = self.build_rdm(batch, coeffs, edge_index)
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
