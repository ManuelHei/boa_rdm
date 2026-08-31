"""Physical observables from a density matrix, via pyscf.

The one-particle density matrix is the model's target, but ``rel_fro`` measures
it in a basis-dependent norm that says nothing directly about physics. These
observables do: they are what the density matrix exists to produce, they are
stored alongside every molecule in the dataset, and they weight an error by how
much it changes the system rather than by how large the matrix element is.

Three, each reproducing its stored value exactly from the reference matrix:

===========  =========================================  ==================
quantity     from the density matrix                    agreement
===========  =========================================  ==================
energy       ``RKS.energy_tot(dm)``, no SCF              9e-09 - 2e-07 Ha
kinetic      ``tr(D T)``, ``T = int1e_kin``              0.0 (exact)
dipole       ``RKS.dip_moment(dm, unit="AU")``           4e-15 a.u.
===========  =========================================  ==================

**The stored dipole is in atomic units, not Debye.** pyscf's ``dip_moment``
defaults to Debye, so taking the default would be wrong by a factor of 2.54 and
would look like a plausible model error rather than a unit bug.

Reproducing the stored energy from the *reference* matrix is exact to
1e-8 - 3e-7 Ha in float64 (checked on molecules of 20, 122 and 126 AOs), so any
residual against a predicted matrix is the model's error and not the method's.

Through the training loader everything is float32 and that floor rises to about
1e-5 Ha = 0.01 kcal/mol, dominated by the rounding of the *stored energy* rather
than of the matrix. Still three orders of magnitude below any model error seen so
far, but it is the reason the baseline check tolerates 1e-3 Ha rather than 1e-7.

Both the in-training validation logging
(:meth:`RDMLightningModule.log_property_errors`) and the standalone sweep
(``boa/eval_energy.py``) call :func:`compute_properties` here, so the two can
never disagree about what any of these means.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

pylogger = logging.getLogger(__name__)

# Building the XC grid dominates the cost (~1 s at 126 AOs, ~0.2-0.5 s per
# evaluation afterwards). Validation re-evaluates the *same* molecules every
# epoch, so the mean-field object is cached on geometry and method.
_MF_CACHE: dict = {}
_CACHE_LIMIT = 64


def _cache_key(atomic_numbers, positions, basis, xc, charge, spin) -> tuple:
    return (
        tuple(int(z) for z in atomic_numbers),
        np.asarray(positions, dtype=float).round(8).tobytes(),
        str(basis),
        str(xc),
        int(charge),
        int(spin),
    )


def _mean_field(atomic_numbers, positions, basis, xc, charge, spin):
    """A cached pyscf RKS object for this geometry, basis and functional."""
    from pyscf import dft, gto
    from pyscf.data.elements import ELEMENTS

    key = _cache_key(atomic_numbers, positions, basis, xc, charge, spin)
    hit = _MF_CACHE.get(key)
    if hit is not None:
        return hit

    mol = gto.Mole()
    mol.atom = [
        [ELEMENTS[int(z)], tuple(float(c) for c in xyz)]
        for z, xyz in zip(atomic_numbers, np.asarray(positions, dtype=float))
    ]
    mol.basis = basis
    mol.charge = int(charge)
    mol.spin = int(spin)
    # The dataset stores positions in angstrom (ASE convention); pyscf's default
    # is also angstrom, but stating it stops a silent 1.89x error if that changes.
    mol.unit = "Angstrom"
    mol.verbose = 0
    mol.build()

    mf = dft.RKS(mol)
    mf.xc = xc

    if len(_MF_CACHE) >= _CACHE_LIMIT:
        _MF_CACHE.clear()
    _MF_CACHE[key] = mf
    return mf


def total_energy(
    atomic_numbers: Sequence[int],
    positions: np.ndarray,
    dm: np.ndarray,
    basis: str = "6-31g*",
    xc: str = "lda,vwn_rpa",
    charge: int = 0,
    spin: int = 0,
) -> float:
    """Total energy of ``dm``, including nuclear repulsion, in hartree.

    No SCF is run: this evaluates the energy functional at the density matrix it
    is given, which is what makes it a fair test of a *predicted* matrix. A
    prediction that is not N-representable still yields a number -- an unphysical
    one, which is the point of measuring it.

    Args:
        atomic_numbers: nuclear charges, one per atom.
        positions: ``(n_atom, 3)`` in angstrom.
        dm: ``(n_ao, n_ao)`` density matrix in pyscf AO ordering. The model's
            output already uses that ordering; see
            ``the-density-matrix-block-basis-is-fixed-by-two-orderings``.
        basis: pyscf basis name, matching the one the target was computed in.
        xc: pyscf functional string. The df_db_pbe set is ``lda,vwn_rpa``
            despite the ``pbe`` in its name.
        charge, spin: as the reference calculation used.

    Raises:
        ValueError: if ``dm`` does not match the basis's AO count -- that means
            the basis or the ordering is wrong, and a silently returned number
            would be worse than a failure.
    """
    mf = _mean_field(atomic_numbers, positions, basis, xc, charge, spin)
    n_ao = mf.mol.nao_nr()
    dm = np.asarray(dm, dtype=float)
    if dm.shape != (n_ao, n_ao):
        raise ValueError(
            f"density matrix is {dm.shape} but basis '{basis}' gives {n_ao} AOs "
            f"for this molecule; the basis or the AO ordering is wrong."
        )
    # Symmetrise: the energy functional assumes a symmetric matrix, and a
    # predicted one is only symmetric to within float error.
    dm = 0.5 * (dm + dm.T)
    return float(mf.energy_tot(dm=dm))


def compute_properties(
    atomic_numbers: Sequence[int],
    positions: np.ndarray,
    dm: np.ndarray,
    basis: str = "6-31g*",
    xc: str = "lda,vwn_rpa",
    charge: int = 0,
    spin: int = 0,
    which: Sequence[str] = ("energy", "kinetic", "dipole"),
) -> dict:
    """Observables of ``dm``: total energy (Ha), kinetic energy (Ha), dipole (a.u.).

    One shared entry point so the in-training check and the standalone sweep
    cannot drift apart. The mean-field object is cached, so asking for all three
    costs barely more than asking for the energy alone -- the XC grid dominates
    and only the energy needs it.
    """
    mf = _mean_field(atomic_numbers, positions, basis, xc, charge, spin)
    n_ao = mf.mol.nao_nr()
    dm = np.asarray(dm, dtype=float)
    if dm.shape != (n_ao, n_ao):
        raise ValueError(
            f"density matrix is {dm.shape} but basis '{basis}' gives {n_ao} AOs "
            f"for this molecule; the basis or the AO ordering is wrong."
        )
    dm = 0.5 * (dm + dm.T)

    out = {}
    if "energy" in which:
        out["energy"] = float(mf.energy_tot(dm=dm))
    if "kinetic" in which:
        # tr(D T): the electronic kinetic energy, exact against the stored value.
        out["kinetic"] = float(np.einsum("ij,ji->", dm, mf.mol.intor("int1e_kin")))
    if "dipole" in which:
        # unit="AU" is not the pyscf default; the dataset stores atomic units.
        out["dipole"] = np.asarray(
            mf.dip_moment(dm=dm, unit="AU", verbose=0), dtype=float
        )
    return out


def references_from_batch(batch, index: int, n_mol: int) -> dict:
    """Stored reference observables for molecule ``index`` of a collated batch.

    Shared by both consumers because getting this wrong is silent. The collater
    concatenates along dim 0 with no molecule axis, so per-molecule **scalars**
    (``energy``, ``ke``) arrive as ``(n_mol,)`` and the **vector** ``dipole``
    arrives as ``(3 * n_mol,)`` -- flat, not ``(n_mol, 3)``. Indexing that
    directly yields one component of one molecule's dipole, which numpy then
    broadcasts against the predicted 3-vector without complaint, producing an
    error larger than the dipole itself and looking exactly like a model failure.

    ``ke`` is the dataset's name for the kinetic energy.
    """
    refs = {}
    for key, attr in (("energy", "energy"), ("kinetic", "ke"), ("dipole", "dipole")):
        val = getattr(batch, attr, None)
        if val is None:
            continue
        if key == "dipole":
            flat = np.asarray(val.detach().cpu().numpy()).reshape(-1)
            if flat.size != 3 * n_mol:
                pylogger.warning(
                    f"dipole has {flat.size} values for {n_mol} molecules; expected "
                    f"{3 * n_mol}. Skipping the dipole check."
                )
                continue
            refs[key] = flat.reshape(n_mol, 3)[index]
        else:
            refs[key] = float(val.reshape(-1)[index])
    return refs


def property_errors(
    atomic_numbers: Sequence[int],
    positions: np.ndarray,
    dm_pred: np.ndarray,
    refs: dict,
    **kwargs,
) -> dict:
    """Predicted observables against stored references.

    ``refs`` maps any of ``energy`` / ``kinetic`` / ``dipole`` to its reference
    value; only the keys present are computed and compared. The dipole error is
    the norm of the vector difference, which is what "how wrong is the dipole"
    means -- comparing magnitudes alone would call an exactly reversed dipole
    perfect.
    """
    which = tuple(k for k in ("energy", "kinetic", "dipole") if k in refs)
    got = compute_properties(atomic_numbers, positions, dm_pred, which=which, **kwargs)
    out = {}
    for key in which:
        ref = refs[key]
        if key == "dipole":
            pred = np.asarray(got[key], dtype=float)
            ref = np.asarray(ref, dtype=float)
            out["dipole_pred"] = pred
            out["dipole_ref"] = ref
            out["dipole_abs_error"] = float(np.linalg.norm(pred - ref))
            out["dipole_ref_norm"] = float(np.linalg.norm(ref))
        else:
            pred = float(got[key])
            out[f"{key}_pred"] = pred
            out[f"{key}_ref"] = float(ref)
            out[f"{key}_error"] = pred - float(ref)
            out[f"{key}_abs_error"] = abs(pred - float(ref))
    if "energy" in which:
        # Chemical accuracy is 1 kcal/mol; hartree hides that scale.
        out["energy_abs_error_kcal"] = out["energy_abs_error"] * 627.509474
    return out


def energy_error(
    atomic_numbers: Sequence[int],
    positions: np.ndarray,
    dm_pred: np.ndarray,
    e_ref: float,
    **kwargs,
) -> dict:
    """``total_energy(dm_pred)`` against a stored reference energy.

    Returns the predicted and reference energies, the signed error and its
    absolute value, all in hartree, plus the error in kcal/mol -- chemical
    accuracy is 1 kcal/mol, which is the scale this has to be read against and
    is invisible in hartree.
    """
    e_pred = total_energy(atomic_numbers, positions, dm_pred, **kwargs)
    err = e_pred - float(e_ref)
    return {
        "e_pred": e_pred,
        "e_ref": float(e_ref),
        "error": err,
        "abs_error": abs(err),
        "abs_error_kcal": abs(err) * 627.509474,
    }
