"""Stand-ins for the two fairchem helpers the vendored HELM code imports.

Upstream ``maloq`` pulls ``conditional_grad`` and the model ``registry`` from
``fairchem.core``. Neither carries any state we need here -- the registry is a
name -> class table used by fairchem's config loader, and ``conditional_grad``
only wraps ``forward`` in ``torch.enable_grad()`` when forces are being
regressed, which never happens for density-matrix prediction. Reimplementing
them locally keeps ``fairchem-core`` (and the whole OCP dependency tree) out of
this environment.
"""

from functools import wraps


def conditional_grad(dec):
    """Apply ``dec`` to ``forward`` only when the module regresses forces.

    Same contract as ``fairchem.core.common.utils.conditional_grad``.
    """

    def decorator(func):
        @wraps(func)
        def cls_method(self, *args, **kwargs):
            f = func
            if getattr(self, "regress_forces", False) and not getattr(self, "direct_forces", 0):
                f = dec(func)
            return f(self, *args, **kwargs)

        return cls_method

    return decorator


class _Registry:
    """No-op replacement for fairchem's global model registry."""

    def register_model(self, name):
        def wrapper(cls):
            return cls

        return wrapper


registry = _Registry()


class ExchangeNodes:
    """Placeholder for the cupy/NCCL node exchange used in distributed graph training.

    ``esen_block`` only touches this when a block is handed a ``partition``,
    which happens exclusively under maloq's distributed graph training. That
    path needs cupy and mpi4py, neither of which is installed here, so it is
    stubbed rather than vendored -- a graph is never partitioned in this repo.
    """

    @staticmethod
    def apply(*args, **kwargs):
        raise NotImplementedError(
            "Distributed graph training is not vendored into boa; build the model with "
            "`distributed_graph_training=False` (the default)."
        )


def exchange_nodes(*args, **kwargs):
    raise NotImplementedError(
        "Distributed graph training is not vendored into boa; build the model with "
        "`distributed_graph_training=False` (the default)."
    )
