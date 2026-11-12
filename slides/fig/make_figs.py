"""Figures for the density-matrix slide. All data real: runs 121/122 on df_db_mp2.

    CUDA_VISIBLE_DEVICES= .venv/bin/python slides/fig/make_figs.py
"""
import glob, os, sys
sys.path.insert(0, os.getcwd()); os.environ.setdefault("PROJECT_ROOT", os.getcwd())
import numpy as np, torch, hydra, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.gridspec import GridSpec
from omegaconf import OmegaConf
from pathlib import Path
from ase.data import covalent_radii
from ase.data.colors import jmol_colors

R = "/export/data/mklockow/boa_rdm_models/train/runs"
OUT = Path("slides/fig")
GREEN, RED, GREY = "#2b7a4b", "#b03a3a", "0.35"
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif"})
OmegaConf.register_new_resolver("hydra", lambda k: "", replace=True)


def load(name):
    d = [x for x in glob.glob(f"{R}/*")
         if os.path.basename(x).split("_devices")[0].split("_", 1)[1] == name][0]
    cfg = OmegaConf.load(Path(d) / "config.yaml"); OmegaConf.resolve(cfg)
    dm = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False); dm.setup("fit")
    m = hydra.utils.instantiate(cfg.model, train=cfg, _recursive_=False,
                                metadata=getattr(dm, "metadata", None))
    ck = sorted(glob.glob(d + "/epoch=*.ckpt"))[-1]
    m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"])
    return m.eval(), dm


model_fn, dmod = load("mp2_helm_w128")
batch = next(iter(dmod.val_dataloader()))
with torch.no_grad():
    _, pred_fn, target, ao_mask = model_fn(batch)
model_mse, _ = load("mp2_helm_w128_mse")
with torch.no_grad():
    _, pred_mse, _, _ = model_mse(batch)

sizes = [int(ao_mask[i].sum()) for i in range(pred_fn.shape[0])]
i = int(np.argsort(sizes)[len(sizes) // 2]); n = sizes[i]
mol_id, z, xyz = model_fn._molecule(batch, i)
D_ref = target[i, :n, :n].numpy(); D_fn = pred_fn[i, :n, :n].numpy()
D_mse = pred_mse[i, :n, :n].numpy()
print(f"molecule {mol_id}: {len(z)} atoms, {n} AOs")

# ------------------------------------------------------------------ figure 1
# Symlog, not a clipped linear scale: |D| spans median 0.002 to max 2.09, so a
# linear scale either shows only the core diagonal or clips it away. Symlog keeps
# both, and nothing is discarded.
fig = plt.figure(figsize=(5.8, 1.75))
gs = GridSpec(1, 3, width_ratios=[0.95, 1.25, 1.25], wspace=0.28,
              left=0.01, right=0.99, top=0.86, bottom=0.04)
ax = fig.add_subplot(gs[0], projection="3d")
zz = np.asarray(z, int); pos = np.asarray(xyz, float)
# Orient by principal axes so the molecule is seen across its widest face
# rather than down an arbitrary lab axis: the two largest-variance directions go
# into the screen plane, and the smallest points at the viewer. Combined with the
# near-overhead view below this gives the most legible projection of any
# molecule, without hand-picking an angle per figure.
_centred = pos - pos.mean(0)
_evals, _evecs = np.linalg.eigh(_centred.T @ _centred)
pos = _centred @ _evecs[:, ::-1]          # columns ordered by descending spread
d = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
cut = 1.3 * (covalent_radii[zz][:, None] + covalent_radii[zz][None, :])
for a, b in zip(*np.triu(np.logical_and(d < cut, d > 0), 1).nonzero()):
    ax.plot(*zip(pos[a], pos[b]), color="0.45", lw=1.5, zorder=1)
ax.scatter(*pos.T, s=[max(30, 185 * covalent_radii[q]) for q in zz],
           c=[jmol_colors[q] for q in zz], edgecolors="0.25", lw=0.5,
           depthshade=False, zorder=2)
ext = float((pos.max(0) - pos.min(0)).max()); mid = 0.5 * (pos.max(0) + pos.min(0))
sp = 0.5 * max(ext, 1.0) * 1.45
ax.set_xlim(mid[0]-sp, mid[0]+sp); ax.set_ylim(mid[1]-sp, mid[1]+sp); ax.set_zlim(mid[2]-sp, mid[2]+sp)
ax.set_box_aspect((1, 1, 1), zoom=1.4); ax.set_axis_off()
# Near-overhead on the principal plane: flat enough to read, tilted enough to
# keep the depth cue that tells a ring from a chain.
ax.view_init(elev=68, azim=-90)
ax.set_title("geometry", fontsize=9.5, pad=-4)
# Every panel on the slide is this one molecule: the same index drives the
# geometry, both heatmaps and all three occupation spectra.
from collections import Counter
from ase.data import chemical_symbols
_c = Counter(chemical_symbols[q] for q in zz)
_order = (["C"] + (["H"] if "H" in _c else []) + sorted(k for k in _c if k not in ("C", "H"))) \
    if "C" in _c else sorted(_c)
_formula = "".join(f"$\\mathrm{{{e}}}_{{{_c[e]}}}$" if _c[e] > 1 else f"$\\mathrm{{{e}}}$"
                   for e in _order)
ax.text2D(0.5, -0.02, f"{_formula}  ({mol_id})", transform=ax.transAxes,
          ha="center", va="top", fontsize=8)

norm = SymLogNorm(linthresh=0.01, vmin=-2.1, vmax=2.1)
for k, (M, lab) in enumerate([(D_ref, "reference $D$"), (D_fn, "predicted $D$")]):
    a = fig.add_subplot(gs[k + 1])
    im = a.imshow(M, cmap="RdBu_r", norm=norm, interpolation="nearest")
    a.set_title(lab, fontsize=9.5, pad=3)
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_linewidth(0.6)
cb = fig.colorbar(im, ax=fig.axes[1:], fraction=0.032, pad=0.02, aspect=18)
cb.ax.tick_params(labelsize=7)
fig.savefig(OUT / "pipeline.pdf", bbox_inches="tight", transparent=True)
print("wrote pipeline.pdf")

# ------------------------------------------------------------------ figure 2
S = model_fn.overlap(batch, like=pred_fn)[i, :n, :n].numpy().astype(float)
ev, U = np.linalg.eigh(0.5 * (S + S.T))
s12 = (U * np.sqrt(np.clip(ev, 1e-12, None))) @ U.T
def occ(D):
    return np.linalg.eigvalsh(s12 @ (0.5 * (D + D.T)) @ s12)[::-1]
o_ref, o_fn, o_mse = occ(D_ref), occ(D_fn), occ(D_mse)

fig, (axo, axb) = plt.subplots(1, 2, figsize=(6.8, 1.9),
                               gridspec_kw={"width_ratios": [1.0, 1.32], "wspace": 0.30})

# Full range from 0: that is the physical picture -- occupations fall from 2 to
# 0 across the spectrum. The discriminating detail lives in a 0.08-wide band at
# the top, which is invisible at this scale, so it gets an inset.
axo.axhline(2.0, color="0.45", lw=0.9, ls="--")
axo.plot(o_ref, "-", lw=1.3, color="0.25", label="MP2 reference")
axo.plot(o_fn, "-", lw=1.3, color=GREEN, label="function-space loss")
axo.plot(o_mse, "-", lw=1.3, color=RED, label="elementwise MSE")
axo.set_ylim(0, 2.13)
axo.set_yticks([0, 1, 2])
axo.set_xlabel("orbital", fontsize=8.5)
axo.set_ylabel("occupation", fontsize=8.5)
axo.tick_params(labelsize=8)
for s in ("top", "right"): axo.spines[s].set_visible(False)
axo.legend(frameon=False, fontsize=7.5, loc="upper left", handlelength=1.3,
           labelspacing=0.3, bbox_to_anchor=(0.30, 0.38))
axo.set_title("occupation numbers", fontsize=9)

ins = axo.inset_axes([0.40, 0.42, 0.58, 0.44])
m = 26
ins.axhline(2.0, color="0.45", lw=0.9, ls="--")
ins.plot(o_ref[:m], "-", lw=1.2, color="0.25")
ins.plot(o_fn[:m], "-", lw=1.2, color=GREEN)
ins.plot(o_mse[:m], "-", lw=1.2, color=RED)
ins.set_ylim(1.996, 2.086)
ins.set_yticks([2.00, 2.04, 2.08])
ins.tick_params(labelsize=6.5, pad=1.5)
ins.annotate(f"+{o_mse.max()-2:.3f}", xy=(int(np.argmax(o_mse[:m])), o_mse.max()),
             xytext=(6, 2.062), fontsize=7.5, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
ins.annotate(f"+{o_fn.max()-2:.4f}", xy=(int(np.argmax(o_fn[:m])), o_fn.max()),
             xytext=(9, 2.022), fontsize=7.5, color=GREEN,
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.8))
for sp in ("top", "right"): ins.spines[sp].set_visible(False)

# Units on the axis, not on every bar: the two bars in a group share them.
metrics = ["rel. Frob.", "kinetic\n(Ha)", "dipole\n(a.u.)", "electrons\n(e)"]
fn = np.array([0.1055, 0.0283, 0.0724, 0.0086]); mse = np.array([0.0403, 0.0635, 0.1802, 0.0612])
nrm = np.maximum(fn, mse); x = np.arange(4); w = 0.36
axb.bar(x - w/2, fn/nrm, w, color=GREEN); axb.bar(x + w/2, mse/nrm, w, color=RED)
for k in x:
    axb.text(k - w/2, fn[k]/nrm[k] + 0.03, f"{fn[k]:.3f}", ha="center", fontsize=7.5)
    axb.text(k + w/2, mse[k]/nrm[k] + 0.03, f"{mse[k]:.3f}", ha="center", fontsize=7.5)
axb.axvline(0.5, color="0.6", lw=0.8, ls=":")
axb.text(0.0, 1.18, "matrix norm", ha="center", fontsize=8, style="italic", color=GREY)
axb.text(2.0, 1.18, "physical observables", ha="center", fontsize=8, style="italic", color=GREY)
axb.set_xticks(x); axb.set_xticklabels(metrics, fontsize=8, linespacing=1.35)
axb.set_yticks([]); axb.set_ylim(0, 1.30)
axb.set_ylabel("error  (lower is better)", fontsize=8.5)
for s in ("top", "right", "left"): axb.spines[s].set_visible(False)
axb.set_title("held-out test error", fontsize=9)
fig.savefig(OUT / "results.pdf", bbox_inches="tight", transparent=True)
print("wrote results.pdf")
print(f"occ max: ref {o_ref.max():.4f}  fn {o_fn.max():.4f}  mse {o_mse.max():.4f}")
