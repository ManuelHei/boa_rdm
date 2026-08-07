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

## Environment Variables

BOA resolves two paths from the environment. Create a `.env` file in the
repository root:

```bash
# Where the datasets live — resolved as `data_dir` in configs/paths/default.yaml
BOA_DATA="/path/to/data"

# Where run outputs go: checkpoints, logs and TensorBoard events (`log_dir`)
BOA_MODELS="/path/to/models"
```

`.env` is loaded automatically — `boa/train.py` and `boa/test.py` call
`rootutils.setup_root(..., indicator=".project-root")`, which finds the
repository root and reads the file from there. No `source` or `export` needed.

The file is gitignored, so your local paths stay out of version control.

## Data setup

Detailed instructions for MD and QM9 data setup will follow soon.

## Training

```bash
python boa/train.py experiment=<your_experiment>
```

## Testing

```bash
python boa/test.py eval=<your_eval>
```