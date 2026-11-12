"""HELM -- the equivariant backbone from MALOQ, vendored for density-matrix prediction.

Source
------
https://github.com/manasakani/maloq, ``src/maloq/helm`` at commit
``e946a4727d6bfec1bae5632a3fa4c66d6a7e44ff``. Licences of maloq and of the
upstream projects it adapts (fairchem, e3nn, DeepH-E3, SchNetPack) are in
``LICENSES/``.

What was changed
----------------
The files are otherwise verbatim. The imports were adjusted so the package
builds in this environment, and there is one functional addition:

* ``eSEN_Backbone.forward`` takes optional ``extra_node_features`` and
  ``extra_edge_features``, added to the node and edge messages before the first
  layer. ``HelmEnergy`` uses them to inject a density matrix. Left as ``None``
  they change nothing, so the geometry-only path is identical to upstream.

The import changes:

* ``from mpi4py import MPI`` was dropped from three modules. Every use of it
  upstream is already commented out.
* ``fairchem.core``'s ``registry`` and ``conditional_grad`` were replaced by the
  equivalents in :mod:`._compat`, which keeps ``fairchem-core`` out of the
  dependency tree. ``registry.register_model`` is a name table used by
  fairchem's config loader and has no effect here; ``conditional_grad`` only
  matters when regressing forces.
* ``nn/communication.py`` (cupy + NCCL node exchange for distributed graph
  training) was removed and its two entry points stubbed in :mod:`._compat`.
  They are reachable only when a block is given a ``partition``, which this
  repo never does.

The papers to cite are HELM (arXiv:2510.00224) and MALOQ (arXiv:2606.28911).
"""

from .esen_osh import eSEN_Backbone  # noqa: F401
