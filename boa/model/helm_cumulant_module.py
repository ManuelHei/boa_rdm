"""Joint training of the 1-RDM and the density-fitted cumulant 2-RDM.

:class:`~boa.model.net.helm_cumulant.HelmRDMCumulant` predicts both from one
backbone. This module turns its two block sets into dense matrices, scores them
and adds the losses::

    loss = rdm_loss_weight * L_D  +  cumulant_loss_weight * L_G

``L_D`` is the density-matrix loss of :class:`~boa.model.rdm_module.RDMLightningModule` unchanged
(``criterion``: ``mse``, ``function``, ...), so every 1-RDM metric and figure
stays comparable with the single-head runs.

The cumulant loss, and why it is not elementwise
------------------------------------------------
``G`` is the cumulant pair density on the auxiliary basis, and the auxiliary
basis is far from orthonormal: on CH4 its Coulomb metric ``M = (A|B)`` has
condition number 3.2e6. In ``M``'s eigenbasis, 64% of ``||G||_F`` sits in the
softest quarter of directions, which carry 0.2% of the cumulant energy; the
stiffest quarter carries 95.8%. An elementwise MSE would
spend most of its effort where the energy does not care.

The fitting basis was built for the Coulomb metric, so ``L_G`` is measured in it:

.. math::

    L_G = \\operatorname{tr}(\\Delta G\\, M\\, \\Delta G\\, M)
        = \\lVert M^{1/2} \\Delta G\\, M^{1/2} \\rVert_F^2 ,

averaged over molecules -- the exact analogue of the ``function`` criterion
``tr(dD S dD S)`` for ``D``, with the Coulomb operator in place of the overlap.
The cumulant energy ``0.5 <G, M>`` is linear in ``G`` with weights ``M``, so this
is the norm that controls it. ``cumulant_criterion: mse`` is kept for comparison.

Metrics, per stage: ``loss_rdm``, ``loss_cumulant``, ``cum_rel_fro`` (plain
Frobenius), ``cum_rel_coulomb`` (the same ratio in the metric), and
``cum_energy_err_kcal``, the error of the cumulant energy ``0.5 <G, M>`` --
which is what the MP2 energy is reconstructed from, so it is the number that
says whether a prediction is chemically useful.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import Tensor

from boa.model.helm_rdm_module import HelmRDMLightningModule
from boa.model.rdm_module import atom_layout, pad_matrices, scatter_dense

HARTREE_TO_KCAL = 627.509474

pylogger = logging.getLogger(__name__)


class HelmCumulantLightningModule(HelmRDMLightningModule):
    """1-RDM and cumulant from one :class:`~boa.model.net.helm_cumulant.HelmRDMCumulant`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        aux = self.model.aux_basis
        self.register_buffer(
            "aux_dim_per_atom",
            torch.as_tensor(aux.basis_dim_per_atom, dtype=torch.long),
            persistent=False,
        )
        self.max_aux_dim = aux.max_basis_dim

        self.cumulant_key = self._hp("cumulant_key", "gamma2cum")
        self.metric_key = self._hp("metric_key", "aux_metric")
        self.cumulant_criterion = self._hp("cumulant_criterion", "coulomb")
        if self.cumulant_criterion not in ("coulomb", "mse"):
            raise ValueError(
                f"Unknown cumulant_criterion '{self.cumulant_criterion}'; expected 'coulomb' "
                f"or 'mse'."
            )
        self.rdm_loss_weight = float(self._hp("rdm_loss_weight", 1.0))
        self.cumulant_loss_weight = float(self._hp("cumulant_loss_weight", 1.0))
        self.register_buffer(
            "cumulant_scale", torch.tensor(float(self._hp("cumulant_scale_factor", 1.0)))
        )
        # Results of the last forward, for the step methods to log. Lightning's
        # steps unpack a 4-tuple from `self(batch)`; keeping that contract is what
        # lets the parent's 1-RDM logging, figures and test bookkeeping run
        # unchanged, so the cumulant side travels alongside instead.
        self._cumulant: dict | None = None
        self.log_cumulant_images = bool(self._hp("log_cumulant_images", True))

    # ------------------------------------------------------------------
    # The cumulant matrix
    # ------------------------------------------------------------------
    def aux_layout(self, batch) -> tuple[Tensor, Tensor, int]:
        """:func:`~boa.model.rdm_module.atom_layout` for the auxiliary basis."""
        return atom_layout(
            batch, self.aux_dim_per_atom, self.atomic_number_to_atom_index, self.max_aux_dim
        )

    def scatter_cumulant(self, batch, blocks: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """``(n_mol, n_aux, n_aux)`` cumulant from its blocks, and the aux mask.

        Always symmetrised: ``G`` is symmetric by construction, and both
        directions of every edge are predicted, so the mean of the two is the
        estimate. No on-site prior -- that is a statement about ``D``.
        """
        index, n_per_mol, n = self.aux_layout(batch)
        cumulant = scatter_dense(batch, blocks, edge_index, index, n)
        cumulant = 0.5 * (cumulant + cumulant.transpose(1, 2)) * self.cumulant_scale
        mask = torch.arange(n, device=cumulant.device)[None, :] < n_per_mol[:, None]
        return cumulant, mask

    def _field(self, batch, key: str):
        if hasattr(batch, key):
            return getattr(batch, key)
        if key in getattr(batch, "keys", list)():
            return batch[key]
        raise KeyError(
            f"'{key}' is not on the batch. The cumulant model needs DfDbCumulantDataset, "
            f"with '{key}' in the loader's list_keys."
        )

    # ------------------------------------------------------------------
    # Loss and metrics
    # ------------------------------------------------------------------
    @staticmethod
    def coulomb_norm2(delta: Tensor, metric: Tensor) -> Tensor:
        """Per-molecule ``tr(dG M dG M)``; ``delta`` and ``metric`` already masked."""
        product = delta @ metric
        return (product * product.transpose(1, 2)).sum(dim=(1, 2)).clamp(min=0)

    @staticmethod
    def cumulant_energy(cumulant: Tensor, metric: Tensor) -> Tensor:
        """Per-molecule ``0.5 <G, M>``, the cumulant's contribution to the energy."""
        return 0.5 * (cumulant * metric).sum(dim=(1, 2))

    def cumulant_terms(self, batch, cumulant: Tensor, aux_mask: Tensor) -> dict:
        n_mol, n = cumulant.shape[0], cumulant.shape[-1]
        target = pad_matrices(
            self._field(batch, self.cumulant_key), n_mol, n, cumulant, self.cumulant_key
        )
        metric = pad_matrices(
            self._field(batch, self.metric_key), n_mol, n, cumulant, self.metric_key
        )
        mask = (aux_mask[:, :, None] & aux_mask[:, None, :]).to(cumulant.dtype)
        delta = (cumulant - target) * mask
        metric = metric * mask

        delta_norm2 = self.coulomb_norm2(delta, metric)
        if self.cumulant_criterion == "coulomb":
            loss = delta_norm2.mean()
        else:
            loss = delta.pow(2).sum() / mask.sum().clamp(min=1)

        energy_err = self.cumulant_energy(delta, metric)  # linear in G
        ref_norm2 = self.coulomb_norm2(target * mask, metric)
        rel_fro = delta.pow(2).sum(dim=(1, 2)).sqrt() / (
            (target * mask).pow(2).sum(dim=(1, 2)).sqrt().clamp(min=1e-12)
        )
        terms = {
            # detached, for the figures; the losses above keep their graph
            "cumulant": cumulant.detach(),
            "target": target.detach(),
            "metric": metric.detach(),
            "mask": aux_mask,
            "loss": loss,
            "rel_fro": rel_fro,
            "rel_coulomb": (delta_norm2 / ref_norm2.clamp(min=1e-20)).sqrt(),
            "energy_err_kcal": energy_err.abs() * HARTREE_TO_KCAL,
        }
        # The stored `energycum_df` must equal 0.5 <G_ref, M>; if it does not, the
        # metric or the aux ordering is wrong and every number above is meaningless.
        stored = getattr(batch, "energycum_df", None)
        if stored is not None:
            stored = torch.as_tensor(stored, dtype=cumulant.dtype, device=cumulant.device)
            terms["ref_energy_consistency_kcal"] = (
                (self.cumulant_energy(target * mask, metric) - stored.reshape(-1)).abs()
                * HARTREE_TO_KCAL
            )
        return terms

    # ------------------------------------------------------------------
    # Lightning plumbing
    # ------------------------------------------------------------------
    def predict_rdm(self, batch) -> tuple[Tensor, Tensor]:
        """The 1-RDM alone, for callers that know nothing of the cumulant."""
        out = self.model(batch)
        return self.scatter_blocks(batch, *out["rdm"])

    def predict(self, batch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(rdm, ao_mask, cumulant, aux_mask)`` from one backbone pass."""
        out = self.model(batch)
        rdm, ao_mask = self.scatter_blocks(batch, *out["rdm"])
        cumulant, aux_mask = self.scatter_cumulant(batch, *out["cumulant"])
        return rdm, ao_mask, cumulant, aux_mask

    def forward(self, batch):
        rdm, ao_mask, cumulant, aux_mask = self.predict(batch)
        target = self.get_target(batch, rdm.shape[-1], rdm.shape[0], like=rdm)
        if self.criterion == "function":
            loss_rdm = self.function_space_loss(batch, rdm, target, ao_mask)
        else:
            loss_rdm = self.compute_loss(rdm, target, ao_mask)
        terms = self.cumulant_terms(batch, cumulant, aux_mask)
        loss = self.rdm_loss_weight * loss_rdm + self.cumulant_loss_weight * terms["loss"]
        self._cumulant = {"loss_rdm": loss_rdm.detach(), **terms}
        return loss, rdm, target, ao_mask

    def _log_cumulant(self, stage: str, n_mol: int) -> None:
        c = self._cumulant
        metrics = {
            f"loss_rdm/{stage}": c["loss_rdm"],
            f"loss_cumulant/{stage}": c["loss"].detach(),
            f"cum_rel_fro/{stage}": c["rel_fro"].mean(),
            f"cum_rel_coulomb/{stage}": c["rel_coulomb"].mean(),
            f"cum_energy_err_kcal/{stage}": c["energy_err_kcal"].mean(),
        }
        if "ref_energy_consistency_kcal" in c and stage != "train":
            metrics[f"cum_ref_consistency_kcal/{stage}"] = c["ref_energy_consistency_kcal"].max()
        self.log_dict(metrics, batch_size=n_mol, sync_dist=self.distributed)

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        if batch_idx % self._hp("log_train_metrics_interval", 10) == 0:
            self._log_cumulant("train", int(batch.num_graphs))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = super().validation_step(batch, batch_idx)
        self._log_cumulant("val", int(batch.num_graphs))
        if (
            self.log_cumulant_images
            and batch_idx == 0
            and self.trainer.is_global_zero
            and not self.trainer.sanity_checking
            and self.current_epoch % self.log_matrix_images_every_n_val == 0
        ):
            try:
                self.log_cumulant_figures(batch)
            except Exception as exc:  # noqa: BLE001 - diagnostics must never end a run
                pylogger.warning(f"cumulant figures failed: {exc}")
        return loss

    @staticmethod
    def coulomb_whiten(matrix: Tensor, metric: Tensor) -> Tensor:
        """``M^1/2 X M^1/2``: the matrix as the Coulomb metric sees it.

        The cumulant loss is the squared Frobenius norm of this for ``X = dG``,
        and the cumulant energy is linear in it, so an image of it shows the
        error that matters. Computed in float64; negative round-off
        eigenvalues of ``M`` are clamped to zero.
        """
        m = metric.double().cpu()
        w, u = torch.linalg.eigh(0.5 * (m + m.T))
        root = (u * w.clamp(min=0).sqrt()) @ u.T
        return (root @ matrix.double().cpu() @ root).float()

    def _atom_boundaries(self, batch, index: int) -> list[int]:
        """Aux-function offsets where each atom of molecule ``index`` starts."""
        z = batch.atomic_numbers[batch.batch == index].long()
        dims = self.aux_dim_per_atom[self.atomic_number_to_atom_index[z]]
        return torch.cumsum(dims, 0)[:-1].tolist()

    @staticmethod
    def _cumulant_figure(matrix, what, who, vmax=None, boundaries=()):
        """One cumulant panel: symmetric-log colour scale, atom blocks outlined.

        A linear scale shows almost nothing here: ``G`` is a handful of entries
        near its maximum on a background three orders of magnitude smaller. The
        symmetric-log scale is linear within 1e-3 of the maximum and logarithmic
        beyond, so both are visible and the sign stays readable. Thin lines mark
        where each atom's auxiliary functions start.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm

        matrix = matrix.detach().float().cpu().numpy()
        vmax = max(float(np.abs(matrix).max()) if vmax is None else vmax, 1e-12)
        norm = SymLogNorm(linthresh=1e-3 * vmax, vmin=-vmax, vmax=vmax, base=10)

        fig, ax = plt.subplots(figsize=(4.4, 4.0), dpi=110)
        im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, interpolation="nearest")
        for b in boundaries:
            ax.axhline(b - 0.5, color="0.6", linewidth=0.3)
            ax.axvline(b - 0.5, color="0.6", linewidth=0.3)
        ax.set_title(f"{what}\n{who}  (max |x| = {vmax:.3g}, symlog)", fontsize=8)
        ax.set_xlabel("aux index", fontsize=8)
        ax.set_ylabel("aux index", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        return fig

    def log_cumulant_figures(self, batch, writer=None, step: int | None = None) -> None:
        """Reference, prediction and error of ``G`` for the first molecules, twice.

        Once as stored, on the auxiliary basis, and once Coulomb-weighted
        (:meth:`coulomb_whiten`). The plain view is dominated by directions the
        energy barely sees: under the Coulomb loss its relative error stays near
        1 by design. The weighted view is what the loss and the energy measure.
        As for the 1-RDM, reference and prediction share a colour scale per
        molecule and the error gets its own. Uses the forward pass just run on
        ``batch``; ``writer`` and ``step`` default to the run's TensorBoard and
        global step, and can be given to render figures from a checkpoint.
        """
        writer = writer if writer is not None else self._tb_writer()
        if writer is None or self._cumulant is None:
            return
        import matplotlib.pyplot as plt

        step = self.global_step if step is None else step
        c = self._cumulant
        n_mol = min(int(self.log_matrix_images_n_molecules), int(c["cumulant"].shape[0]))
        for i in range(n_mol):
            n = int(c["mask"][i].sum())
            ref, pred = c["target"][i, :n, :n], c["cumulant"][i, :n, :n]
            metric = c["metric"][i, :n, :n]
            who = self._molecule_label(batch, i)
            slot = f"mol{i}"
            bounds = self._atom_boundaries(batch, i)
            views = {"cum": (ref, pred), "cum_weighted": (
                self.coulomb_whiten(ref, metric), self.coulomb_whiten(pred, metric)
            )}
            figures = {}
            for stem, (r, p) in views.items():
                name = "G" if stem == "cum" else "M^1/2 G M^1/2"
                shared = max(float(r.abs().max()), float(p.abs().max()))
                for kind, matrix, vmax in (
                    ("reference", r, shared),
                    ("prediction", p, shared),
                    ("error", p - r, None),
                ):
                    what = f"{name}: predicted - reference" if kind == "error" else (
                        f"{'reference' if kind == 'reference' else 'predicted'} {name}"
                    )
                    figures[f"{stem}_{kind}/{slot}"] = self._cumulant_figure(
                        matrix, what, who, vmax=vmax, boundaries=bounds
                    )
            for tag, fig in figures.items():
                writer.add_figure(tag, fig, global_step=step)
                plt.close(fig)

    def test_step(self, batch, batch_idx):
        out = super().test_step(batch, batch_idx)
        self._log_cumulant("test", int(batch.num_graphs))
        self.test_cum_energy_err.extend(self._cumulant["energy_err_kcal"].tolist())
        return out

    def on_test_start(self):
        super().on_test_start()
        self.test_cum_energy_err = []

    def on_test_end(self):
        super().on_test_end()
        if self.test_cum_energy_err:
            values = torch.tensor(self.test_cum_energy_err)
            within = (values < 1.0).float().mean()
            pylogger.info(
                f"Test cumulant energy error: {values.mean():.3f} kcal/mol mean, "
                f"{values.median():.3f} median, {100 * within:.1f}% within 1 kcal/mol, "
                f"over {len(values)} molecules."
            )
