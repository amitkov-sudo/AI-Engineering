# AI Engineer Career Path (Codecademy) — March 2026 Snapshot

This repository is a personal collection of notes, exercises, and projects from Codecademy’s **AI Engineer Career Path**, captured and organized during **March 2026**.

At the moment, the snapshot contains the exercise notebooks; the `Notes/` and `Projects/` folders are reserved for future additions.

## Repository Structure

- `Notes/` — conceptual summaries and reference snippets (active in this snapshot)
- `Exercises/` — Notebook exercises (active in this snapshot)
- `Projects/` — larger capstone-style work (currently empty in this snapshot)

## Included Exercises (Notebooks)

### `Exercises/benchmarking_exercise_1.ipynb`

Build and analyze a simple PyTorch MLP:

- Define a `SimpleMLP` network with fully connected layers and ReLU activations.
- Implement utilities to estimate:
  - **Model size in bytes** (parameters + buffers)
  - **Latency per forward pass** in milliseconds using repeated forward passes
- Compare latency behavior across **different batch sizes** using synthetic input.

### `Exercises/mlps_exercise_2.ipynb`

Train an MLP to predict hotel cancellations using real-world booking data:

- Use PyTorch to implement an MLP (`SimpleMLP`) for binary classification.
- Load training data from `datasets/bookings_train.csv`.
- Implement training components:
  - Loss: `BCEWithLogitsLoss`
  - Optimizer: Adam (`lr=0.0001`)
  - Training loop over `10` epochs with tracked loss and accuracy.

## Local Setup (If Running Notebooks Locally)

### Requirements

- Python 3.9+ (recommended)
- Jupyter (e.g., `jupyter lab` or `jupyter notebook`)
- PyTorch
- Common data utilities used in notebooks:
  - `numpy`
  - `pandas`

### Data file expected by `mlps_exercise_2.ipynb`

The notebook expects a local CSV at:

`datasets/bookings_train.csv`

### Model summary helper

`benchmarking_exercise_1.ipynb` imports `custom_torchinfo`. This helper is typically available in Codecademy’s runtime.
If it is not available locally, you may need to provide an equivalent (for example, replace the summary call with `torchinfo.summary` if appropriate).

## How to Use

Open the notebook(s) under `Exercises/` and run cells top-to-bottom.
Several checkpoints instruct you to add your solution into cells labeled with `## YOUR SOLUTION HERE ##`. 
For recruiters: Those checkpoints demonstrate my original work and logic. Note that databases are not provided for exerceses at this point in time. 

## Disclaimer

This repo contains work based on Codecademy course material. 
