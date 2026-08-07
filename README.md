# BOA
This repository contains the code to reproduce the results in the ICLR paper [A Function-Centric Graph Neural Network Approach For Predicting Electron Densities](https://openreview.net/pdf?id=HDdkFjFEZd).

## Installation
For environment management, we use UV:
```bash
uv venv --python=3.11
source .venv/bin/activate
uv pip install -r requirements.txt --index-strategy unsafe-best-match
uv pip install -e scdp/ -e structures25/ -e .
```

BOA builds on two bundled packages, both installed by the last line above:

- **`scdp/`** — the charge density prediction codebase of
  [Fu et al. (2024)](https://openreview.net/forum?id=b7REKaNUTv) (MIT). 
- **`structures25/`** — the `mldft` package of
  [Remme et al., *JACS* **147**, 28851 (2025)](https://doi.org/10.1021/jacs.5c06219)
  (LGPL-3.0).

## Environment Variables

BOA resolves two paths from the environment. Create a `.env` file in the
repository root:

```bash
# Where the datasets live — resolved as `data_dir` in configs/paths/default.yaml
BOA_DATA="/path/to/data"

# Where run outputs go: checkpoints, logs and TensorBoard events (`log_dir`)
BOA_MODELS="/path/to/models"
```

The file is gitignored, so your local paths stay out of version control.

## Data setup

Each dataset lives under `$BOA_DATA/<dataset_name>/`, with `<dataset_name>`
matching a config in `configs/data/`:

```
$BOA_DATA/<dataset_name>/
├── data/             # the dataset itself
└── datasplits.json   # train/val/test split (unused by `md`)
```

| `dataset_name` | Loader | `data/` contains | Source |
| --- | --- | --- | --- |
| `qm9_vasp` | `LmdbDataset` | `*.lmdb` | QM9 densities computed with VASP from [Jørgensen & Bhowmik, *npj Comput. Mater.* **8**, 183 (2022)](https://doi.org/10.1038/s41524-022-00863-y), distributed at [DTU Data](https://doi.org/10.11583/DTU.16794500) |
| `qm9_pyscf` | `PyscfDataset` | `dsgdb9nsd_<index:06d>/` per molecule | QM9 densities on real-space grids from [Li et al., *Nat. Commun.* **16**, 4811 (2025)](https://doi.org/10.1038/s41467-025-60095-8) |
| `md` | `SmallDensityDataset` | `<mol>/<mol>_<split>/{structures,dft_densities}.npy`, for benzene, ethane, ethanol, malonaldehyde, phenol and resorcinol | MD-sampled geometries from [Brockherde et al., *Nat. Commun.* **8**, 872 (2017)](https://doi.org/10.1038/s41467-017-00839-3) and [Bogojeski et al., *Nat. Commun.* **11**, 5223 (2020)](https://doi.org/10.1038/s41467-020-19093-1), curated by [Cheng & Peng (2023)](https://openreview.net/forum?id=EjiA3uWpnc) and distributed at [quantum-machine.org](http://www.quantum-machine.org/datasets/) |

Both QM9 datasets are built on the geometries of
[Ramakrishnan et al., *Sci. Data* **1**, 140022 (2014)](https://doi.org/10.1038/sdata.2014.22)
and [Ruddigkeit et al., *J. Chem. Inf. Model.* **52**, 2864 (2012)](https://doi.org/10.1021/ci300415d),
and differ only in how the reference density was computed. The train/validation/test
split follows [Fu et al. (2024)](https://openreview.net/forum?id=b7REKaNUTv) for
`qm9_vasp` and Li et al. (2025) for `qm9_pyscf`.

The downloaded `qm9_vasp` tarballs are converted to LMDB with the bundled
preprocessing script (paths in `scdp/README.md` are relative to that
subdirectory, so prefix them with `scdp/`):

```bash
python scdp/scdp/scripts/preprocess.py \
  --data_path <dir with the downloaded .tar files> \
  --out_path $BOA_DATA/qm9_vasp/data \
  --tar --disable_pbc --device cpu --atom_cutoff 6 --vnode_method none
```

The `md` loader is adapted from
[InfGCN](https://github.com/ccr-cheng/InfGCN-pytorch).

## Training

```bash
python boa/train.py experiment=<your_experiment>
```

## Testing

```bash
python boa/test.py eval=<your_eval>
```