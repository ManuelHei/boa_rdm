"""Cap the OpenMP and BLAS thread pools before anything loads them.

pyscf takes every core it can see, and on a 128-core node that is both
antisocial and **slower**. Measured on one 126-AO molecule, ``energy_tot``:

===========  ==========  ===========  ===================
threads      time        speedup      molecules/s *per core*
===========  ==========  ===========  ===================
1            4244 ms     1.0x         **0.236**
2            2305 ms     1.8x         0.118
4            1308 ms     3.2x         0.081
8             833 ms     5.1x         0.150
16            598 ms     7.1x         0.074
32            499 ms     8.5x         0.063
64            527 ms     8.1x         0.031
128          1013 ms     4.2x         0.008
===========  ==========  ===========  ===================

Two things follow. **Uncapped is the worst setting available** -- 128 threads is
twice as slow as 32 and barely better than 8, because the thread pool costs more
to marshal than the work saves at this molecule size. And for *throughput* over
many molecules, parallel efficiency collapses well before the core count does:
one thread per process is the most core-efficient point by 1.6x over eight, so N
independent single-threaded processes beat one wide process by a wide margin.

This has to run before ``numpy``, ``torch`` or ``pyscf`` are imported: OpenMP and
the BLAS libraries size their pools when they load, and nothing afterwards
shrinks a pool that is already up. That is why the flag is read by hand rather
than through argparse, which runs far too late.
"""

from __future__ import annotations

import os
import sys

#: Threads and dataloader workers a script may use unless told otherwise.
DEFAULT_CPUS = 8

#: The variables every BLAS and OpenMP runtime reads *when it loads*.
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def cpus_from_argv(default: int = DEFAULT_CPUS) -> int:
    """``--cpus`` read straight out of ``sys.argv``.

    argparse still declares the option, so ``--help`` documents it and a bad
    value is rejected properly a moment later; this only needs the number early
    enough to matter.
    """
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--cpus" and i + 1 < len(argv):
            arg = f"--cpus={argv[i + 1]}"
        if arg.startswith("--cpus="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                return default
    return default


def limit_threads(cpus: int) -> None:
    """Cap the OpenMP and BLAS thread pools. ``0`` leaves the environment alone."""
    if cpus <= 0:
        return
    for var in _THREAD_VARS:
        os.environ[var] = str(cpus)


def limit_from_argv(default: int = DEFAULT_CPUS) -> int:
    """Read ``--cpus`` and apply it. Returns the value, for logging."""
    cpus = cpus_from_argv(default)
    limit_threads(cpus)
    return cpus
