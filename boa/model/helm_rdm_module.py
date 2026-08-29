"""Density-matrix training with the HELM backbone instead of BOA's.

Everything downstream of the prediction -- the loss, ``rel_fro``, the
electron-count check, the TensorBoard matrix images, EMA, the optimiser --
is the same as for BOA, so this is :class:`~boa.model.rdm_module.RDMLightningModule`
with one method replaced.

The difference is where the blocks come from. BOA emits per-edge coefficient
pairs and the matrix is their outer product, :math:`D_{\\mu\\nu} = \\sum_r a_{\\mu r}
b_{\\nu r}`, which is what makes it a *rank-R* prediction per block and what
makes its scale compound quadratically. :class:`~boa.model.net.helm_net.HelmRDM`
emits each block directly as a linear map of equivariant features, so the blocks
are full rank and the output is linear in the head's coefficients. That also
removes the reason BOA needed the initial-guess pre-training: there is no
product of two learned factors to run away, so
``initial_guess_pre_training_steps`` should be 0 here (and the net has no
``initial_guess_module`` for it to train anyway).
"""

from torch import Tensor

from boa.model.rdm_module import RDMLightningModule


class HelmRDMLightningModule(RDMLightningModule):
    """:class:`RDMLightningModule` over a net that predicts the blocks itself."""

    def predict_rdm(self, batch) -> tuple[Tensor, Tensor]:
        """Density matrix for a batch.

        The net returns the blocks and the pairs they belong to -- self-loops
        for the diagonal blocks, both directions of every graph edge for the
        rest -- and the placement, padding and symmetrisation are shared with
        the BOA path.
        """
        blocks, edge_index = self.model(batch)
        return self.scatter_blocks(batch, blocks, edge_index)
