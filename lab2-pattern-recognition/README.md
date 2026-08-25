# Lab 2: Pattern Recognition

Work in progress for COMP3710 Lab Demonstration 2 (2026). The lab covers four
connected areas:

1. discrete Fourier transforms and PyTorch GPU timing;
2. PCA eigenfaces and random-forest classification;
3. an LFW CNN classifier and a CIFAR-10 ResNet-18 challenge; and
4. an advanced recognition task using the OASIS brain MRI dataset.

## Planned structure

```text
lab2-pattern-recognition/
|-- notebooks/       # Exploratory work and demonstration notebooks
|-- pattern_recognition_demo2/
|   `-- part1_fourier.py
|-- src/             # Reusable Python modules
|-- results/         # Generated plots, metrics, and timing summaries
|-- requirements.txt
`-- README.md
```

Generated datasets, model checkpoints, and cluster outputs should not be
committed. Record reproducible commands and concise results here as each part is
completed.

## Setup

Create a virtual environment and install the initial dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The exact PyTorch build used on the Rangpur cluster may differ from the local
environment. Keep cluster setup instructions separate when they are added.

## Progress checklist

- [x] Part 1: NumPy square-wave Fourier reconstruction started
- [ ] Part 1: NumPy/PyTorch DFT timing comparison
- [ ] Part 2: Eigenfaces, compactness plot, and random-forest classification
- [ ] Part 3.1: Two-layer CNN classifier for LFW
- [ ] Part 3.2: CIFAR-10 ResNet-18 training and Rangpur demonstration
- [ ] Part 4: Selected OASIS recognition task
- [ ] Add result plots and demonstration notes
- [ ] Document AI assistance and external sources
