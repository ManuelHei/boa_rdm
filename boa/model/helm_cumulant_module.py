"""Joint training of the 1-RDM and the density-fitted cumulant 2-RDM.

:class:`~boa.model.net.helm_cumulant.HelmRDMCumulant` predicts both from one
backbone. This module turns its two block sets into dense matrices, scores them
and adds the losses::

    loss = rdm_loss_weight * L_D  +  cumulant_loss_weight * L_G

``L_D`` is the density-matrix loss of :class:`RDMLightningModule` unchanged
(``criterion``: ``mse``, ``function``, ...), so every 1-RDM metric and figure
stays comparable with the single-head runs.

The cumulant loss, and why it is not elementwise
------------------------------------------------
``G`` is the cumulant pair density on the auxiliary basis, and the auxiliary
basis is far from orthonormal: on CH4 its Coulomb metric ``M = (A|B)`` has
condition number 3.2e6. In ``M``'s eigenbasis, 64% of ``||G||_F`` sits in the
softest quarter of directions, which carry 0.2% of the cumulant energy; the
stiffest quarter carries 95.8% (``EXP-2026-11-12-03``). An elementwise MSE would
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

import torch
from torch import Tensor

from boa.model.helm_rdm_module import HelmRDMLightningModule
from boa.model.rdm_module import atom_layout, pad_matrices, scatter_dense

HARTREE_TO_KCAL = 627.509474

pylogger = logging.getLogger(__name__)


class HelmCumulantLightningModule(HelmRDMLightningModule):
    """1-RDM and cumulant from one :class:`HelmRDMCumulant`."""

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

    # ------------------------------------------------------------------
    # The cumulant matrix
    # ------------------------------------------------------------------
    def aux_layout(self, batch) -> tuple[Tensor, Tensor, int]:
        """:func:`atom_layout` for the auxiliary basis."""
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
        return loss

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
