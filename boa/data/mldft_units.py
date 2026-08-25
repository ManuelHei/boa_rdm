"""Unit-safe wrappers around the vendored ``mldft`` molecule API.

BOA works in Ångström throughout: probe grids, atom positions, the message
passing and orbital cutoffs, and the densities (see ``BOHR_TO_ANGSTROM`` in
``boa/data/dataset.py``). The vendored ``structures25`` release of ``mldft`` is
Bohr-native:

* ``mldft.utils.molecules.build_molecule_np`` defaults to ``unit="Bohr"``,
* ``mldft.utils.molecules.build_molecule_ofdata`` hardcodes ``unit="Bohr"``,
* ``OFData.minimal_sample_from_mol`` sets ``pos`` from ``mol.atom_coords()``,
  which returns Bohr.

Handing BOA's Ångström coordinates to any of those builds the molecule 1.889x
too small. It fails *silently*: ``pos`` comes back numerically unchanged (Bohr
in, Bohr out) so the GTO evaluation at the probe points still lines up, and only
the pyscf-derived overlap and edge matrices are computed on the compressed
geometry. The prediction stays correlated with the reference and integrates to
roughly the right electron count while the error grows by more than an order of
magnitude.

These wrappers pin the unit at every boundary. **Import molecules and OFData
samples through this module, never from ``mldft`` directly**, so a new call site
cannot quietly reintroduce the bug.
"""

import numpy as np
import torch

from mldft.ml.data.components.of_data import OFData
from mldft.utils.molecules import build_molecule_np as _build_molecule_np

#: The unit BOA works in. Every call into mldft's molecule API is pinned to it.
UNIT = "Angstrom"


def _as_numpy(x):
    return x if isinstance(x, np.ndarray) else x.numpy(force=True)


def build_molecule(charges, positions, basis=None, spin=None, output=None):
    """Build a pyscf molecule from Ångström positions.

    Mirrors :func:`mldft.utils.molecules.build_molecule_np` but pins ``unit``,
    which otherwise defaults to Bohr.
    """
    return _build_molecule_np(
        charges=_as_numpy(charges),
        positions=_as_numpy(positions),
        basis=basis,
        unit=UNIT,
        spin=spin,
        output=output,
    )


def build_molecule_from_sample(sample, basis=None, spin=None):
    """Build a pyscf molecule from an :class:`OFData` sample whose ``pos`` is Ångström.

    Replaces :func:`mldft.utils.molecules.build_molecule_ofdata`, which takes no
    ``unit`` argument and hardcodes Bohr. That function is a thin wrapper around
    ``build_molecule_np``, so calling it directly with the right unit is
    equivalent and avoids patching the vendored package.
    """
    return build_molecule(sample.atomic_numbers, sample.pos, basis=basis, spin=spin)


def sample_from_molecule(mol, basis_info, **kwargs):
    """``OFData.minimal_sample_from_mol`` with ``pos`` returned in Ångström.

    The upstream method exposes no unit hook and stores ``mol.atom_coords()``
    (Bohr). ``pos`` is registered ``Representation.NONE``, i.e. a plain field, so
    overwriting it after construction is safe.
    """
    sample = OFData.minimal_sample_from_mol(mol, basis_info, **kwargs)
    pos = mol.atom_coords(unit=UNIT)
    if isinstance(sample.pos, torch.Tensor):
        sample.pos = torch.as_tensor(pos, dtype=sample.pos.dtype)
    else:
        sample.pos = np.asarray(pos, dtype=sample.pos.dtype)
    return sample
