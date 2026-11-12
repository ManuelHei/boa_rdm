"""Pair a source molecule with a predicted density matrix and an energy target.

The geometry, the basis and the graph all come from the original dataset -- this
adds only what :mod:`boa.make_energy_dataset` produced: the density matrix a
particular RDM checkpoint predicted, and the energy to train against.

Keeping the geometry in one place matters for more than tidiness. The energy
model's blocks are cut on the same graph the RDM model predicted on, so the two
must agree about positions and about the radius cutoff; re-deriving the geometry
from a second copy is exactly how they would silently drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from boa.data.dataset import DfDbDataset
from mldft.ml.data.components.of_data import Representation


class EnergyDataset(DfDbDataset):
    """A source molecule plus ``rdm_pred`` and the energy target.

    Args:
        path: the *source* dataset directory (geometry and reference values).
        energy_path: the directory :mod:`boa.make_energy_dataset` wrote.
        target_mode: ``delta`` trains on ``e_ref - E[rdm_pred]``, which needs the
            functional energy and so only exists for a DFT source; ``total``
            trains on ``e_ref`` itself, which is the only option for a
            correlated (MP2) source.

    The remaining arguments are as in :class:`~boa.data.dataset.DfDbDataset`.

    The molecules are the intersection of the two directories, in sorted order,
    so a partially generated energy set is usable rather than an error -- but the
    split file must then come from the energy directory, which
    ``make_energy_dataset`` writes for that reason.
    """

    def __init__(
        self,
        path,
        energy_path,
        basis_info,
        transform=None,
        rdm_key: str = "gamma",
        extra_props=("energy",),
        target_mode: str = "delta",
        pattern: str = "*.hdf5",
    ) -> None:
        super().__init__(
            path=path,
            basis_info=basis_info,
            transform=transform,
            rdm_key=rdm_key,
            extra_props=extra_props,
            pattern=pattern,
        )
        self.energy_root = Path(energy_path)
        if target_mode not in ("delta", "total"):
            raise ValueError(f"target_mode must be 'delta' or 'total', not {target_mode!r}")
        self.target_mode = target_mode

        available = {p.stem for p in self.energy_root.glob(pattern)}
        if not available:
            raise FileNotFoundError(
                f"no predicted density matrices in {self.energy_root}. Generate them "
                f"with boa/make_energy_dataset.py before training an energy model."
            )
        keep = [i for i, p in enumerate(self.paths) if p.stem in available]
        if not keep:
            raise ValueError(
                f"{self.energy_root} shares no molecule ids with {path}; the energy set "
                f"was generated from a different source dataset."
            )
        if len(keep) < len(self.paths):
            # Not an error: generating a subset is a normal way to iterate.
            self.paths = [self.paths[i] for i in keep]

        provenance = self.energy_root / "provenance.json"
        self.provenance = json.loads(provenance.read_text()) if provenance.is_file() else {}
        if self.target_mode == "delta" and self.provenance.get("energy_of_prediction") is False:
            raise ValueError(
                f"target_mode='delta' needs E[rdm_pred], which {self.energy_root} does not "
                f"carry (correlated source: the MP2 energy is not a functional of the "
                f"1-RDM). Use target_mode='total'."
            )

    def __getitem__(self, idx):
        of_data = super().__getitem__(idx)
        stem = self.paths[idx].stem
        with h5py.File(self.energy_root / f"{stem}.hdf5", "r") as h5:
            rdm_pred = np.asarray(h5["rdm_pred"][()])
            values = {k: float(np.asarray(h5[k][()])) for k in h5 if k != "rdm_pred"}

        if self.target_mode == "delta":
            if "delta" not in values:
                raise KeyError(
                    f"{stem} carries no 'delta'; it was generated without the functional "
                    f"energy. Use target_mode='total'."
                )
            target = values["delta"]
        else:
            target = values["e_ref"]

        # `rdm_pred` is the model's *input*, so it is a matrix like `rdm` and has
        # to travel the same list_keys path: it differs in size per molecule and
        # cannot be stacked by the collater.
        of_data.add_item(
            "rdm_pred",
            torch.as_tensor(rdm_pred, dtype=torch.get_default_dtype()),
            Representation.NONE,
        )
        of_data.add_item(
            "energy_target",
            torch.tensor([target], dtype=torch.get_default_dtype()),
            Representation.NONE,
        )
        for name, value in values.items():
            of_data.add_item(
                f"energy_{name}",
                torch.tensor([value], dtype=torch.get_default_dtype()),
                Representation.NONE,
            )
        return of_data
