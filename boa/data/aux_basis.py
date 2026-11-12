"""The auxiliary (density-fitting) basis the cumulant 2-RDM is written in.

``gamma2cum`` in the cumulant release of ``df_db_mp2`` is not an AO matrix. It is
the cumulant pair density fitted onto an auxiliary basis,

.. math::

    \\rho_2^{\\mathrm{cum}}(r_1, r_2) \\approx
        \\sum_{AB} G_{AB}\\, \\chi_A(r_1)\\, \\chi_B(r_2),

with :math:`\\chi` the ``cc-pvdz-jkfit`` functions of ``pyscf.df.addons.make_auxmol``,
in that function's ordering. Its energy is :math:`\\tfrac12 \\langle G, M \\rangle`
with :math:`M = (A|B)` the aux-basis Coulomb metric (``int2c2e``). Both facts were
checked against the stored ``energycum_df`` (exact) and against the stored MP2
energy (7.5e-4 Ha on CH4, the density-fitting error); see ``EXP-2026-11-12-03``.

There is no :class:`~boa.data.basis_info.BasisInfo` for this basis, and it does
not need one. The density-matrix head needs only the shell degrees of each
element, in pyscf order, and the number of functions per element -- this class
supplies exactly those, ordered like ``basis_info.atomic_numbers`` so the same
``atomic_number_to_atom_index`` indexes both.
"""

from __future__ import annotations

import numpy as np
from pyscf import df, gto
from pyscf.data.elements import ELEMENTS

#: The auxiliary basis of the released ``gamma2cum``.
DEFAULT_AUXBASIS = "cc-pvdz-jkfit"


def _element_mol(atomic_number: int, basis: str) -> gto.Mole:
    symbol = ELEMENTS[int(atomic_number)]
    return gto.M(atom=f"{symbol} 0 0 0", basis=basis, spin=int(atomic_number) % 2, verbose=0)


class AuxBasisLayout:
    """Per-element shell layout of an auxiliary basis.

    Args:
        atomic_numbers: the elements, in the order of ``basis_info.atomic_numbers``.
        auxbasis: the auxiliary basis name, as pyscf knows it.
        basis: the orbital basis. ``make_auxmol`` takes a molecule, and for a
            named jkfit set the orbital basis does not change the result; it is
            asked for only so the layout is built exactly the way the release was.
    """

    def __init__(
        self,
        atomic_numbers,
        auxbasis: str = DEFAULT_AUXBASIS,
        basis: str = "6-31g*",
    ) -> None:
        self.atomic_numbers = [int(z) for z in atomic_numbers]
        self.auxbasis = auxbasis
        self.basis = basis

        self.shell_degrees: list[list[int]] = []
        for z in self.atomic_numbers:
            aux = df.addons.make_auxmol(_element_mol(z, basis), auxbasis)
            if aux.cart:
                raise ValueError(f"{auxbasis} came out cartesian; the head assumes spherical.")
            ls = []
            for shell in range(aux.nbas):
                ls.extend([aux.bas_angular(shell)] * aux.bas_nctr(shell))
            if ls != sorted(ls):
                # union_shell_degrees maps an element's k-th shell of degree l to
                # the k-th union shell of degree l, which needs ascending l.
                raise ValueError(f"{auxbasis} shells of Z={z} are not ascending in l: {ls}")
            self.shell_degrees.append(ls)

        self.basis_dim_per_atom = np.array(
            [sum(2 * l + 1 for l in ls) for ls in self.shell_degrees], dtype=np.int64
        )

    @property
    def max_basis_dim(self) -> int:
        return int(self.basis_dim_per_atom.max())

    def auxmol(self, mol: gto.Mole) -> gto.Mole:
        """The auxiliary molecule of ``mol``, in the release's function ordering."""
        return df.addons.make_auxmol(mol, self.auxbasis)

    def coulomb_metric(self, mol: gto.Mole) -> np.ndarray:
        """``(A|B)``, the two-centre Coulomb metric the cumulant energy contracts with."""
        return self.auxmol(mol).intor("int2c2e")

    def __repr__(self) -> str:
        dims = dict(zip(self.atomic_numbers, self.basis_dim_per_atom.tolist()))
        return f"AuxBasisLayout({self.auxbasis}, functions per element {dims})"
