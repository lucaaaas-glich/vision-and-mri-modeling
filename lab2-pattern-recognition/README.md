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
- [x] Part 4 Task 1: OASIS convolutional VAE implementation
- [x] Part 4 Task 1: Train on Rangpur and record VAE results
- [x] Part 4 Task 2: OASIS categorical U-Net implementation (medium route)
- [x] Part 4 Task 2: Train and verify DSC >0.9 for every label
- [ ] Add result plots and demonstration notes
- [x] Document AI assistance and external sources

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

For a Chinese revision guide covering tensor shapes, parameter calculations,
training logic, ResNet-18, mixed precision, results, and demonstration questions,
see [`P3_REVIEW_ZH.md`](P3_REVIEW_ZH.md).

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

## Part 4 Task 1: OASIS VAE

The first recognition task is a convolutional variational autoencoder for
preprocessed OASIS brain MR images.  Before training, inspect the cluster data
so that the chosen directory and array layout are documented:

```bash
python pattern_recognition_demo2/part4_inspect_oasis.py /home/groups/comp3710
```

Run a one-epoch synthetic smoke test (this verifies code only and is not an
assignment result):

```bash
python pattern_recognition_demo2/part4_oasis_vae.py --smoke-test
```

After identifying the exact OASIS subdirectory, submit the real A100 job:

```bash
sbatch train_oasis_vae_a100.sh /home/groups/comp3710/OASIS
```

The VAE uses a convolutional encoder, separate `mu` and `logvar` layers, the
reparameterisation trick and a convolutional decoder.  Its objective combines
pixel reconstruction MSE with beta-weighted KL divergence.  The program saves
the best validation checkpoint, learning curves, original/reconstructed image
pairs, a PCA projection of latent means, a sampled two-dimensional manifold,
and machine-readable metrics under `results/part4_vae/`.

### VAE result

The model was trained on 9,664 official training slices and evaluated on 1,120
official validation slices. At 128x128 resolution, 20 epochs took **156.0
seconds** on a Rangpur A100. The best validation loss was **0.007456** at epoch
20 (reconstruction MSE 0.006039 and KL divergence 1.416143). The reconstruction
and sampled-manifold figures show preserved brain shape and continuous changes
through latent space.

| VAE evidence | File |
| --- | --- |
| Original and reconstructed MR slices | [`reconstructions.png`](results/part4_vae/reconstructions.png) |
| Sampled 2-D manifold slice | [`sampled_manifold.png`](results/part4_vae/sampled_manifold.png) |
| PCA projection of latent means | [`latent_projection.png`](results/part4_vae/latent_projection.png) |
| Learning curves | [`learning_curves.png`](results/part4_vae/learning_curves.png) |
| Reproducible configuration and history | [`metrics.json`](results/part4_vae/metrics.json) |

## Part 4 Task 2: OASIS U-Net

The custom U-Net pairs `case_*.nii.png` images with `seg_*.nii.png` masks. Mask
values 0/85/170/255 are mapped to four classes and represented as categorical
one-hot tensors. Cross-entropy plus soft Dice loss handles categorical accuracy
and class overlap, while reported DSC is accumulated separately for every label
over the complete validation/test set.

```bash
python pattern_recognition_demo2/part4_oasis_unet.py --smoke-test --image-size 64
sbatch train_oasis_unet_a100.sh
python pattern_recognition_demo2/part4_oasis_unet.py --demo
```

The best checkpoint is selected by the *lowest* validation-label DSC, because
the requirement is greater than 0.9 for every label rather than merely a high
mean score. Outputs are saved under `results/part4_unet/`.

### U-Net result

The U-Net used all 9,664 training, 1,120 validation and 544 test slices. Five
epochs at 128x128 resolution took **90.6 seconds** on a Rangpur A100. Every test
label exceeded the required DSC of 0.9:

| Label value | Class index | Validation DSC | Test DSC |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0.9990 | **0.9990** |
| 85 | 1 | 0.9334 | **0.9405** |
| 170 | 2 | 0.9453 | **0.9444** |
| 255 | 3 | 0.9686 | **0.9685** |

The separate demo inference reproduced the same scores with test loss 0.097913.
See [`test_predictions.png`](results/part4_unet/test_predictions.png) for MRI,
ground-truth and predicted masks, [`learning_curves.png`](results/part4_unet/learning_curves.png)
for per-label DSC, and [`metrics.json`](results/part4_unet/metrics.json) for the
complete configuration and history. Raw Rangpur output is retained under
[`results/part4_logs/`](results/part4_logs/).

## Part 4 references

- D. P. Kingma and M. Welling, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
- O. Ronneberger, P. Fischer and T. Brox, [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).
- D. S. Marcus et al., [Open Access Series of Imaging Studies (OASIS)](https://doi.org/10.1162/jocn.2007.19.9.1498).

For Chinese demonstration and revision notes, see [`P4_REVIEW_ZH.md`](P4_REVIEW_ZH.md).

## AI assistance statement

OpenAI ChatGPT/Codex was used as a learning and development assistant for code
structure, Chinese explanatory comments, debugging PyTorch/Slurm compatibility,
and documentation. The implementation was executed on the specified datasets,
and all reported metrics and figures were produced and checked from the actual
Rangpur runs. The student is responsible for understanding and explaining the
models, losses, evaluation and results during demonstration.
