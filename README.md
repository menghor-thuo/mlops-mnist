# MLOps MNIST

A small CNN classifier trained on the corrupted MNIST dataset with PyTorch.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync
```

Put the raw corrupted MNIST files in `data/raw/`:

```txt
data/raw/
├── train_images_{0..5}.pt
├── train_target_{0..5}.pt
├── test_images.pt
└── test_target.pt
```

## Usage

Run the commands from the project root.

```bash
# 1. Preprocess: merge the training shards, normalize, and save to data/processed/
uv run src/mnist/data.py data/raw data/processed

# 2. Train: saves models/model.pth and reports/figures/training_statistics.png
uv run src/mnist/train.py --lr 1e-3 --batch-size 32 --epochs 10

# 3. Evaluate: prints test accuracy
uv run src/mnist/evaluate.py models/model.pth

# 4. Visualize: t-SNE of the learned features, saved to reports/figures/embedding.png
uv run src/mnist/visualize.py models/model.pth
```

Every script uses [Typer](https://typer.tiangolo.com/), so `--help` lists its options.

Steps 1 and 2 also have [invoke](https://www.pyinvoke.org/) tasks:

```bash
uv run invoke preprocess-data
uv run invoke train
```

## Project structure

```txt
├── data/                     # Data (git-ignored)
│   ├── raw/
│   └── processed/
├── models/                   # Trained weights (git-ignored)
├── reports/
│   └── figures/              # Generated plots (git-ignored)
├── src/
│   └── mnist/
│       ├── __init__.py
│       ├── data.py           # Dataset loading and preprocessing
│       ├── model.py          # CNN definition
│       ├── train.py          # Training loop
│       ├── evaluate.py       # Test-set evaluation
│       └── visualize.py      # t-SNE embedding plot
├── .gitignore
├── pyproject.toml            # Project metadata and dependencies
├── uv.lock
├── README.md
└── tasks.py                  # Invoke tasks
```
