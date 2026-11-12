"""Train :class:`HelmEnergy` on a predicted density matrix.

The input is a density matrix a *frozen* RDM checkpoint produced, so this module
never runs that model: the matrices were generated once by
:mod:`boa.make_energy_dataset` and are read from disk like any other input. That
is deliberate -- it keeps the two trainings independent, makes the energy set
reproducible from a named checkpoint, and avoids paying for the RDM forward pass
every epoch.

Errors are reported in **kcal/mol** as well as hartree. 1 kcal/mol is chemical
accuracy and it is the only scale these numbers mean anything against; a hartree
figure alone hides whether a model is close or two orders away.
"""

from __future__ import annotations

import logging

import torch
from hydra.utils import instantiate
from lightning import LightningModule
from torch import Tensor

pylogger = logging.getLogger(__name__)

HARTREE_TO_KCAL = 627.509474


class EnergyLightningModule(LightningModule):
    """Predict a molecular energy from a stored predicted density matrix."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.basis_info = instantiate(self.hparams.basis_info)
        self.model = instantiate(self.hparams.net)
        self.criterion = self._hp("criterion", "mse")
        self.target_key = self._hp("target_key", "energy_target")
        self.input_key = self._hp("input_key", "rdm_pred")
        # A target of ~0.02 Ha (a delta) and one of ~-400 Ha (a total energy)
        # need very different output scales. The shift is subtracted from the
        # target and added back for reporting, so the network always sees an
        # O(1) quantity whichever mode is in use.
        self.target_shift = float(self._hp("target_shift", 0.0))
        self.target_scale = float(self._hp("target_scale", 1.0))

    def _hp(self, name, default=None):
        if hasattr(self.hparams, "get"):
            return self.hparams.get(name, default)
        return getattr(self.hparams, name, default)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def ao_layout(self, batch) -> tuple[Tensor, Tensor, int]:
        """Per-atom AO positions and the padded AO count.

        Reuses :class:`RDMLightningModule`'s implementation so the energy model
        cuts blocks from exactly the slots the RDM model scattered them into.
        Sharing the code rather than restating it is the point: a divergence
        here would misalign every block without raising.
        """
        from boa.model.rdm_module import RDMLightningModule

        return RDMLightningModule.ao_layout(self, batch)

    @property
    def atomic_number_to_atom_index(self):
        return self.model.atomic_number_to_atom_index

    @property
    def basis_dim_per_atom(self):
        return self.model.basis_dim_per_atom

    @property
    def max_basis_dim(self):
        return self.model.max_basis_dim

    def dense_input(self, batch, n_ao: int, n_mol: int, like: Tensor) -> Tensor:
        """The stored predicted matrices, padded into ``(n_mol, n_ao, n_ao)``."""
        matrices = getattr(batch, self.input_key)
        dense = like.new_zeros(n_mol, n_ao, n_ao)
        for i, matrix in enumerate(matrices):
            m = matrix.to(dense.dtype).to(dense.device)
            dense[i, : m.shape[0], : m.shape[1]] = m
        return dense

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def forward(self, batch):
        n_mol = int(batch.num_graphs)
        ao_index, _, n_ao = self.ao_layout(batch)
        like = batch.pos
        rdm = self.dense_input(batch, n_ao, n_mol, like)
        prediction = self.model(batch, rdm, ao_index, n_ao)

        target = getattr(batch, self.target_key).reshape(-1).to(prediction.dtype)
        scaled = (target - self.target_shift) / self.target_scale
        if self.criterion == "mae":
            loss = (prediction - scaled).abs().mean()
        else:
            loss = (prediction - scaled).pow(2).mean()
        return loss, prediction * self.target_scale + self.target_shift, target

    def _log_errors(self, stage: str, predicted: Tensor, target: Tensor, n_mol: int) -> None:
        error = (predicted - target).abs()
        # `loss/train` is logged by `training_step` itself; repeating it here
        # would be a second `self.log` of the same key and Lightning refuses.
        metrics = {} if stage == "train" else {f"loss/{stage}": error.pow(2).mean()}
        self.log_dict(
            {
                **metrics,
                f"mae/{stage}": error.mean(),
                f"mae_kcal/{stage}": error.mean() * HARTREE_TO_KCAL,
                # The share within chemical accuracy says more than a mean, which
                # a few bad molecules dominate.
                f"within_1kcal/{stage}": (error * HARTREE_TO_KCAL < 1).float().mean(),
            },
            batch_size=n_mol,
            sync_dist=True,
        )

    def training_step(self, batch, batch_idx):
        loss, predicted, target = self(batch)
        n_mol = int(batch.num_graphs)
        self.log("loss/train", loss, batch_size=n_mol, prog_bar=True)
        if batch_idx % int(self._hp("log_train_metrics_interval", 50)) == 0:
            self._log_errors("train", predicted.detach(), target, n_mol)
        return loss

    def validation_step(self, batch, batch_idx):
        _, predicted, target = self(batch)
        self._log_errors("val", predicted, target, int(batch.num_graphs))

    def test_step(self, batch, batch_idx):
        _, predicted, target = self(batch)
        self._log_errors("test", predicted, target, int(batch.num_graphs))

    def configure_optimizers(self):
        """As :class:`BOA`'s: the optimiser and schedule live in the top-level
        config, which ``train.py`` hands the module as ``train``, not in the
        model subtree."""
        train = self.hparams.train
        optimizer = instantiate(train.optim, params=self.parameters(), _convert_="partial")
        scheduler = instantiate(train.lr_scheduler, optimizer=optimizer)
        if "lr_schedule_freq" in train:
            scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": train.lr_schedule_freq,
                "monitor": train.monitor.metric,
            }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": train.monitor.metric,
        }
