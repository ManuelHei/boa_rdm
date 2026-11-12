"""Render the cumulant figures of a joint 1-RDM / cumulant run from a checkpoint.

Training logs these figures itself at every validation (``log_cumulant_images``).
This script is for runs started before that existed, or for a fresh look at any
checkpoint. It takes the first validation batch, the same molecules the
training-time figures show, and writes TensorBoard events in a directory of its
own, at the checkpoint's global step, so successive checkpoints form a series::

    .venv/bin/python boa/render_cumulant_figures.py --run 42
    tensorboard --logdir $BOA_MODELS/eval/cumulant_figures

The EMA weights are used, as in validation, so the figures match the logged
``*/val`` numbers.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="run id, directory name, or full path")
    p.add_argument("--ckpt", type=Path, default=None, help="default: the run's last.ckpt")
    p.add_argument("--out", type=Path, default=None,
                   help="default: $BOA_MODELS/eval/cumulant_figures/<run name>")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n", type=int, default=3, help="molecules to draw")
    return p.parse_args()


def main():
    args = parse_args()
    import hydra
    from dotenv import load_dotenv
    from omegaconf import OmegaConf
    from torch.utils.tensorboard import SummaryWriter

    from boa.eval_energy import resolve_run

    repo = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PROJECT_ROOT", str(repo))
    load_dotenv(repo / ".env")
    run = resolve_run(args.run)
    OmegaConf.register_new_resolver(
        "hydra",
        lambda key: {"runtime.output_dir": str(run), "runtime.cwd": os.getcwd()}.get(key, ""),
        replace=True,
    )
    cfg = OmegaConf.load(run / "config.yaml")
    OmegaConf.resolve(cfg)

    datamodule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
    datamodule.setup("fit")
    model = hydra.utils.instantiate(cfg.model, train=cfg, _recursive_=False)
    ckpt_path = args.ckpt or run / "last.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.on_load_checkpoint(ckpt)
    model.ema.copy_to(model.parameters())
    model.log_matrix_images_n_molecules = args.n
    model.eval().to(args.device)

    batch = next(iter(datamodule.val_dataloader())).to(args.device)
    with torch.no_grad():
        model(batch)

    out = args.out or Path(os.environ["BOA_MODELS"]) / "eval" / "cumulant_figures" / run.name
    writer = SummaryWriter(log_dir=str(out))
    step = int(ckpt.get("global_step", 0))
    model.log_cumulant_figures(batch, writer=writer, step=step)
    writer.close()
    print(f"{ckpt_path.name} (step {step}) -> {out}")


if __name__ == "__main__":
    main()
