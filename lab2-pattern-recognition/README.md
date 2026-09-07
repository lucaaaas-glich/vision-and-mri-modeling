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
- [x] Part 2: Eigenfaces, compactness plot, and random-forest classification
- [x] Part 3.1: Two-layer CNN classifier for LFW, trained and evaluated
- [x] Part 3.2: CIFAR-10 ResNet-18, trained with mixed precision on Rangpur A100
- [ ] Part 4: Selected OASIS recognition task
- [ ] Add result plots and demonstration notes
- [ ] Document AI assistance and external sources

## Part 3 commands

Quickly verify the LFW CNN pipeline, then run the full 20 epochs:

```bash
python pattern_recognition_demo2/part3_lfw_cnn.py --smoke-test
python pattern_recognition_demo2/part3_lfw_cnn.py --epochs 20
```

Train CIFAR-10 ResNet-18 on Rangpur using the reproducible Slurm script:

```bash
sbatch train_cifar_a100.sh
```

During the demonstration, load the best checkpoint, run inference, and train one
additional epoch:

```bash
python pattern_recognition_demo2/part3_cifar10_resnet18.py --demo --batch-size 256 --workers 4
```

## Part 3 results

Both LFW methods used the same 75/25 split with `random_state=42`.

| Task and model | Test accuracy | Notes |
| --- | ---: | --- |
| LFW PCA + Random Forest | 60.87% | 150 components retained 94.72% variance |
| LFW two-layer CNN | 87.27% | 20 epochs on Apple M2 MPS |
| CIFAR-10 ResNet-18 | **94.34%** | 55 epochs, mixed precision, NVIDIA A100 |

The CIFAR-10 training loop took **341.5 seconds**. The complete Slurm job took
**348 seconds (5:48)**, including Python startup and data loading. This satisfies
the challenge target of at least 94% accuracy in approximately 360 seconds or
faster. The best result occurred at epoch 53; the final epoch accuracy was
94.20%. Model checkpoints and raw cluster logs are kept locally under
`results/` and are excluded from Git because of their size.

### Key implementation decisions

- The ResNet-18 is implemented directly in PyTorch rather than loaded as a
  pre-built or pre-trained model.
- The 7x7 ImageNet stem is replaced by a 3x3, stride-1 stem for 32x32 CIFAR-10
  images.
- Random crop and horizontal flip provide training-time data augmentation.
- CUDA automatic mixed precision reduces training time and GPU memory use.
- SGD with momentum, weight decay, label smoothing, and cosine learning-rate
  decay provide stable convergence.
