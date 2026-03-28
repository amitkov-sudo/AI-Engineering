# AI-Engineering

Personal workspace for **deep learning**, **machine learning**, and **AI** work: notes, hands-on exercises, and standalone projects.

This repository is meant to organize experiments, coursework, and independent builds in one place.

## Repository layout

| Folder | Purpose |
|--------|---------|
| `Notes/` | Concepts, references, and short write-ups |
| `Exercises/` | Focused notebooks and small practice tasks |
| `Projects/` | Larger or end-to-end projects |

## What’s here (examples)

The `Exercises/` folder currently includes PyTorch notebooks such as:

- **`benchmarking_exercise_1.ipynb`** — MLP setup; model size (parameters + buffers); forward-pass latency across batch sizes.
- **`mlps_exercise_2.ipynb`** — Binary classification with an MLP on tabular booking data; training loop with BCE-with-logits loss and Adam.

Paths and dependencies inside notebooks (e.g. `datasets/bookings_train.csv`, optional `custom_torchinfo` for model summaries) match whatever environment you run in; adjust paths or swap helpers (e.g. `torchinfo.summary`) when working locally. Currently datasets are not provided in this directory.

## Local setup

Needed stack for the notebooks above:

- Python 3.9+
- Jupyter (`jupyter lab` or `jupyter notebook`)
- PyTorch, NumPy, Pandas

Create a virtual environment, install what each notebook imports, and place any expected data files where the notebook points (or update the paths).


