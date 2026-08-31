"""Energy check for a trained RDM model over many validation molecules.

The in-training version (:meth:`RDMLightningModule.log_energy_errors`) covers
the same three molecules the figures show, which is enough to watch a trend but
far too few to characterise a model: this dataset spans 3-27 atoms and 18-162
AOs, and the first validation molecule is 4-atom ammonia.

Both paths call :func:`boa.model.energy.total_energy`, so "the energy" means the
same thing in each.

Usage::

    .venv/bin/python boa/eval_energy.py \\
        --run /export/data/.../115_dfdb_helm_w128_... \\
        --n 100 --split validation

``--plot res_50.png`` additionally writes an error histogram and an
error-versus-size scatter for each of the three observables the run computes;
``--plot-props energy`` narrows that. ``--from-json res_50.json`` draws the same
figure from a results file written earlier, without loading the model or the
dataset.

``--run`` takes a run directory (its ``config.yaml`` and best checkpoint are read
from it). ``--baseline`` additionally reports the energy of the *reference*
matrix, which should reproduce the stored value to ~1e-7 Ha; if it does not,
something about the basis or the AO ordering is wrong and the model numbers below
it mean nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("eval_energy")

HARTREE_TO_KCAL = 627.509474


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, help="run directory")
    p.add_argument("--n", type=int, default=50, help="number of molecules (0 = all)")
    p.add_argument(
        "--split", default="validation", choices=("validation", "test", "train")
    )
    p.add_argument("--ckpt", type=Path, default=None, help="override the checkpoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--baseline",
        action="store_true",
        help="also evaluate the reference matrix, as a sanity check on basis/ordering",
    )
    p.add_argument("--out", type=Path, default=None, help="write per-molecule JSON here")
    p.add_argument("--plot", type=Path, default=None, help="write the error figure here")
    p.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="plot an existing results JSON and exit; no checkpoint or data needed",
    )
    p.add_argument(
        "--plot-props",
        nargs="+",
        default=list(_ERROR_COLUMNS),
        choices=tuple(_ERROR_COLUMNS),
        help="which observables to plot (default: all three)",
    )
    args = p.parse_args()
    if args.run is None and args.from_json is None:
        p.error("--run is required (or --from-json to plot an existing results file)")
    return args


def pick_checkpoint(run_dir: Path) -> Path:
    best = run_dir / "best_model_path.txt"
    if best.is_file():
        p = Path(best.read_text().strip())
        if p.is_file():
            return p
        # The recorded path is absolute and from the training host; fall back to
        # the same basename inside this run directory.
        local = run_dir / p.name
        if local.is_file():
            return local
    ckpts = sorted(q for q in run_dir.glob("*.ckpt") if q.name != "last.ckpt")
    if ckpts:
        return ckpts[-1]
    if (run_dir / "last.ckpt").is_file():
        return run_dir / "last.ckpt"
    raise FileNotFoundError(f"no checkpoint in {run_dir}")


# A muted blue for the data and an orange for the mean: the one pair that stays
# distinguishable under every common colour-vision deficiency. Dark ink for the
# median, which is a line rather than a colour.
_FILL, _INK, _ACCENT = "#4c78a8", "#2f3337", "#f58518"

# The column each observable's absolute error lands in, the factor that takes it
# to the unit it is plotted in, and that unit. Both energies are plotted in
# kcal/mol -- hartree hides the scale a chemist reads them against. The dipole is
# not an energy and has no kcal/mol, so it stays in atomic units.
_ERROR_COLUMNS = {
    "energy": ("energy_abs_error_kcal", 1.0, "kcal/mol"),
    "kinetic": ("kinetic_abs_error", HARTREE_TO_KCAL, "kcal/mol"),
    "dipole": ("dipole_abs_error", 1.0, "a.u."),
}


def _plot_property(ax_hist, ax_scatter, err, n_atom, prop, unit, log):
    """One observable's two panels: error histogram, and error against size."""
    positive = err[err > 0]
    use_log = (
        (positive.size > 1 and positive.max() / positive.min() > 100)
        if log == "auto"
        else bool(log)
    )
    bins = (
        np.geomspace(positive.min(), positive.max(), 21)
        if use_log and positive.size > 1
        else np.linspace(0.0, max(err.max(), 1e-12), 21)
    )
    label = f"|Δ{prop}|  [{unit}]"
    mean, median = float(err.mean()), float(np.median(err))

    ax_hist.hist(err, bins=bins, color=_FILL, edgecolor="white", linewidth=0.6)
    # Both, because they answer different questions: a mean well above the
    # median is a tail, and a tail is what an outlier molecule looks like.
    ax_hist.axvline(mean, color=_ACCENT, linewidth=1.6, label=f"mean {mean:.3g}")
    ax_hist.axvline(median, color=_INK, linestyle="--", linewidth=1.2,
                    label=f"median {median:.3g}")
    if use_log:
        ax_hist.set_xscale("log")
    ax_hist.set_xlabel(label)
    ax_hist.set_ylabel("molecules")
    ax_hist.set_title(prop, fontsize=9, color=_INK, loc="left")
    ax_hist.legend(frameon=False, fontsize=8)

    # Sizes are integers, so points at the same size would sit on one vertical
    # line; a fixed-seed jitter separates them without moving anything visibly.
    jitter = np.random.default_rng(0).uniform(-0.18, 0.18, n_atom.size)
    ax_scatter.scatter(n_atom + jitter, err, s=24, color=_FILL, alpha=0.75,
                       edgecolor="white", linewidth=0.5)
    if use_log:
        ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel("atoms per molecule")
    ax_scatter.set_ylabel(label)

    for ax in (ax_hist, ax_scatter):
        ax.grid(axis="y", color=_INK, alpha=0.15, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)


def plot_errors(rows, out=None, props=tuple(_ERROR_COLUMNS), title=None, log="auto"):
    """Error histogram and error-vs-size scatter from an ``--out`` JSON file.

    One row of panels per observable, and by default all three that
    :func:`~boa.model.properties.property_errors` computes -- the energy alone
    can look respectable while the dipole is badly wrong, since an energy is a
    rotation-invariant scalar and cannot see a fault the dipole would expose.

    Two panels per row, because the two questions are different: the histogram
    says how large the errors are, the scatter says whether they grow with the
    molecule -- a model that is fine on ammonia and hopeless at 20 atoms has the
    same mean as one that is uniformly mediocre, and only the second panel tells
    them apart.

    Args:
        rows: parsed rows, or a path to the JSON file ``--out`` wrote.
        out: where to write the figure; ``None`` just returns it.
        props: which of energy / kinetic / dipole to draw; any absent from the
            rows is skipped rather than raising, since a run whose dataset
            carried no dipole simply has no such column. Both energies are
            drawn in kcal/mol; the dipole is not an energy and stays in a.u.
        title: figure title; defaults to the file name when a path was given.
        log: log axes for the error; ``"auto"`` turns them on past two decades,
            which is where a linear histogram collapses into one bar. Decided
            per observable, as the three span very different ranges.

    Returns:
        The matplotlib figure.
    """
    import sys

    import matplotlib

    # Only claim the backend if pyplot has not already been set up -- forcing Agg
    # inside a notebook would silently stop figures from showing.
    if out is not None and "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(rows, (str, Path)):
        title = title or Path(rows).name
        rows = json.loads(Path(rows).read_text())

    ok = [r for r in rows if "error" not in r]
    n_failed = len(rows) - len(ok)

    panels = []
    for prop in props:
        col, to_unit, unit = _ERROR_COLUMNS[prop]
        sel = [r for r in ok if col in r and "n_atom" in r]
        if not sel:
            logging.getLogger("eval_energy").warning(
                f"no {col} in these rows; skipping the {prop} panels"
            )
            continue
        panels.append((
            prop,
            unit,
            np.array([r[col] for r in sel], dtype=float) * to_unit,
            np.array([r["n_atom"] for r in sel], dtype=float),
        ))
    if not panels:
        raise KeyError(
            f"none of {list(props)} is in these rows; expected columns "
            f"{[_ERROR_COLUMNS[p][0] for p in props]}"
        )

    fig, axes = plt.subplots(
        len(panels), 2, figsize=(9.5, 3.6 * len(panels)),
        squeeze=False, constrained_layout=True,
    )
    for (prop, unit, err, n_atom), (ax_h, ax_s) in zip(panels, axes):
        _plot_property(ax_h, ax_s, err, n_atom, prop, unit, log)

    head = f"{panels[0][2].size} molecules"
    if n_failed:
        head += f"   ({n_failed} failed)"
    fig.suptitle(f"{title}   {head}" if title else head, fontsize=10, color=_INK)

    if out is not None:
        fig.savefig(out, dpi=200)
        plt.close(fig)
    return fig


def main():
    args = parse_args()

    # Plotting a file that already exists needs neither the checkpoint nor the
    # dataset, so take that path before anything heavy is imported.
    if args.from_json is not None:
        out = args.plot or args.from_json.with_suffix(".png")
        plot_errors(args.from_json, out, props=args.plot_props)
        log.info(f"wrote {out}")
        return

    import hydra
    from dotenv import load_dotenv
    from omegaconf import OmegaConf

    from boa.model.properties import property_errors, references_from_batch

    # The stored config interpolates ${oc.env:PROJECT_ROOT} for `paths.root_dir`.
    # Training gets it from the launching environment; a standalone run of this
    # script need not have it, and the repo root is what it always means.
    os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    # `paths.output_dir` / `paths.work_dir` interpolate ${hydra:runtime.*}, which
    # only resolves inside a hydra run. Nothing the datamodule or model needs
    # touches them, but a whole-config resolve walks them anyway -- so stub the
    # resolver rather than deleting keys, and point it at this run's directory.
    OmegaConf.register_new_resolver(
        "hydra",
        lambda key: {
            "runtime.output_dir": str(args.run),
            "runtime.cwd": os.getcwd(),
        }.get(key, ""),
        replace=True,
    )

    cfg = OmegaConf.load(args.run / "config.yaml")
    OmegaConf.resolve(cfg)

    # The energy needs the reference value; a run trained before `extra_props`
    # existed will not have it in its stored config.
    ds_cfg = cfg.data.datamodule.dataset
    props = list(ds_cfg.get("extra_props", []) or [])
    wanted = [k for k in ("energy", "ke", "dipole") if k not in props]
    if wanted:
        ds_cfg.extra_props = props + wanted
        log.info(f"added {wanted} to extra_props (absent from the run's own config)")

    datamodule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
    datamodule.setup("fit" if args.split != "test" else "test")
    loader = {
        "validation": datamodule.val_dataloader,
        "test": datamodule.test_dataloader,
        "train": datamodule.train_dataloader,
    }[args.split]()

    # Same call train.py makes: the module takes the whole config as `train` and
    # the datamodule's `metadata`, not just its own subtree.
    model = hydra.utils.instantiate(
        cfg.model,
        train=cfg,
        _recursive_=False,
        metadata=getattr(datamodule, "metadata", None),
    )
    ckpt_path = args.ckpt or pick_checkpoint(args.run)
    log.info(f"checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval().to(args.device)

    basis = cfg.model.get("dft_basis_set", "6-31g*")
    xc = cfg.model.get("energy_xc", "lda,vwn_rpa")
    log.info(f"basis {basis}, xc {xc}, split {args.split}")

    rows, done = [], 0
    with torch.no_grad():
        for batch in loader:
            if args.n and done >= args.n:
                break
            batch = batch.to(args.device)
            _, pred, target, ao_mask = model(batch)
            refs_all = {
                key: getattr(batch, attr, None)
                for key, attr in (("energy", "energy"), ("kinetic", "ke"), ("dipole", "dipole"))
            }
            refs_all = {k: v for k, v in refs_all.items() if v is not None}
            if not refs_all:
                raise RuntimeError(
                    "batch carries none of energy/ke/dipole; check extra_props"
                )
            for i in range(pred.shape[0]):
                if args.n and done >= args.n:
                    break
                info = model._molecule(batch, i)
                if info is None:
                    raise RuntimeError("batch carries no positions/atom_types")
                mol_id, types, positions = info
                z, xyz = types.numpy(), positions.numpy()
                n_ao = int(ao_mask[i].sum())
                refs = references_from_batch(batch, i, int(pred.shape[0]))
                row = {"id": mol_id, "n_atom": int(len(z)), "n_ao": n_ao}
                try:
                    res = property_errors(
                        z, xyz, pred[i, :n_ao, :n_ao].float().cpu().numpy(),
                        refs, basis=basis, xc=xc,
                    )
                    for k, v in res.items():
                        row[k] = v.tolist() if isinstance(v, np.ndarray) else v
                    if args.baseline:
                        base = property_errors(
                            z, xyz, target[i, :n_ao, :n_ao].float().cpu().numpy(),
                            refs, basis=basis, xc=xc,
                        )
                        for k in ("energy", "kinetic", "dipole"):
                            if f"{k}_abs_error" in base:
                                row[f"baseline_{k}_abs_error"] = base[f"{k}_abs_error"]
                except Exception as exc:
                    row["error"] = str(exc)
                    log.warning(f"{mol_id}: {exc}")
                rows.append(row)
                done += 1
            if done % 20 < pred.shape[0]:
                log.info(f"  {done} molecules")

    ok = [r for r in rows if "error" not in r]
    if not ok:
        log.error("no molecule evaluated successfully")
        return
    log.info(f"\n{len(ok)} molecules ({len(rows) - len(ok)} failed)")
    units = {"energy": "Ha", "kinetic": "Ha", "dipole": "a.u."}
    for key in ("energy", "kinetic", "dipole"):
        col = f"{key}_abs_error"
        vals = np.array([r[col] for r in ok if col in r])
        if not vals.size:
            continue
        log.info(f"\n{key}  |Δ| in {units[key]}")
        for name, f in (("mean", np.mean), ("median", np.median), ("max", np.max)):
            log.info(f"  {name:>6} {f(vals):>12.6f}")
        if key == "energy":
            kcal = vals * HARTREE_TO_KCAL
            log.info(f"  {'mean':>6} {kcal.mean():>12.3f} kcal/mol")
            # Chemical accuracy is the threshold anyone reads energies against.
            log.info(f"  within 1 kcal/mol : {(kcal < 1).sum()}/{len(vals)}")
            log.info(f"  within 10 kcal/mol: {(kcal < 10).sum()}/{len(vals)}")
        if key == "dipole":
            # Absolute dipole error is meaningless without the scale it sits on.
            ref = np.array([r["dipole_ref_norm"] for r in ok if "dipole_ref_norm" in r])
            log.info(f"  mean |mu_ref| {ref.mean():>12.6f}  "
                     f"-> relative {vals.mean()/max(ref.mean(), 1e-12):.1%}")

    if args.baseline:
        # Every observable gets a baseline, not just the energy. The dipole
        # baseline is what catches a *frame* or *collation* fault, which the
        # rotation-invariant energy cannot see: a flattened (3*n_mol,) dipole
        # indexed as if it were (n_mol, 3) once produced a 332% "model error"
        # that was entirely a reference-handling bug.
        tol = {"energy": 1e-3, "kinetic": 1e-3, "dipole": 1e-3}
        # Through the loader everything is float32 and the stored reference is
        # rounded harder than the matrix (8.7e-6 vs 2.2e-6 Ha on a 126-AO
        # molecule), so the floor is ~1e-5, not the ~1e-7 float64 gives straight
        # from the HDF5. Near 1e-3 means a real fault, not precision.
        log.info("\nreference-matrix baseline (should be float32 noise, ~1e-5):")
        for key in ("energy", "kinetic", "dipole"):
            col = f"baseline_{key}_abs_error"
            vals = np.array([r[col] for r in ok if col in r])
            if not vals.size:
                continue
            verdict = "ok" if vals.max() < tol[key] else "TOO LARGE -- results above are meaningless"
            log.info(f"  {key:<8} max {vals.max():.2e}  {verdict}")

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))
        log.info(f"\nwrote {args.out}")

    if args.plot:
        plot_errors(rows, args.plot, props=args.plot_props, title=args.run.name)
        log.info(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
