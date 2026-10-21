# Image Classification and Brain MRI Modeling

Python/PyTorch experiments spanning four connected areas:

1. discrete Fourier transforms and PyTorch GPU timing;
2. PCA eigenfaces and random-forest classification;
3. an LFW CNN classifier and a CIFAR-10 ResNet-18 challenge; and
4. an advanced recognition task using the OASIS brain MRI dataset.

## Repository structure

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
python pattern_recognition_demo2/part3_cifar10_resnet18.py \
  --demo \
  --checkpoint results/part3_cifar10_resnet18_94_34.pt \
  --batch-size 256 \
  --workers 4
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

## Parts 1-3 reproducible evidence

The lightweight plots and metrics below are committed so the demonstration does
not depend on terminal scrollback. Large datasets and model checkpoints remain
excluded from Git.

| Part | Evidence | Reproduce in PyCharm or a terminal |
| --- | --- | --- |
| 1 | [`benchmark.json`](results/part1/benchmark.json), [`fourier_reconstruction.png`](results/part1/fourier_reconstruction.png), [`dft_spectrum.png`](results/part1/dft_spectrum.png) | `python pattern_recognition_demo2/part1_fourier.py` and `python pattern_recognition_demo2/part1_benchmark.py` |
| 2 | [`metrics.json`](results/part2/metrics.json), [`eigenfaces.png`](results/part2/eigenfaces.png), [`compactness.png`](results/part2/compactness.png), [`confusion_matrix.csv`](results/part2/confusion_matrix.csv) | `python pattern_recognition_demo2/part2_eigenfaces.py` |
| 3.1 | [`metrics.json`](results/part3_lfw_cnn/metrics.json), [`learning_curves.png`](results/part3_lfw_cnn/learning_curves.png), [`confusion_matrix.csv`](results/part3_lfw_cnn/confusion_matrix.csv) | `python pattern_recognition_demo2/part3_lfw_cnn.py --epochs 20` |
| 3.2 | [`metrics.json`](results/part3_cifar10/metrics.json) | Use `sbatch train_cifar_a100.sh`; the checkpoint is deliberately not committed. |

The committed Part 1 benchmark was generated on CPU and records its device in
the JSON file. Run it again on a CUDA or MPS machine when demonstrating GPU
timing; do not describe the CPU tensor column as a GPU result. The committed LFW
CNN evidence is a seeded 20-epoch CPU rerun (86.65%); the earlier M2 MPS run in
the summary table reached 87.27%. Small differences across PyTorch backends are
expected, so the metrics file records the device and full learning history.

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

The run was initially named `part4_unet_pilot`, but `max_samples` was unset and
the complete official splits were used. It became the final selected model once
every validation and test label passed the required threshold.

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

## Part 4 Task 3: OASIS WGAN-GP

Task 3 uses a custom Wasserstein GAN with gradient penalty. The generator maps
128-D Gaussian noise to a 128x128 grayscale MR slice. The critic outputs an
unbounded Wasserstein score rather than a probability. Gradient penalty replaces
weight clipping and encourages the critic gradient norm to remain near one.

```bash
python pattern_recognition_demo2/part4_oasis_wgan_gp.py \
  --smoke-test --epochs 1 --image-size 64 --base-channels 16 --batch-size 16
sbatch train_oasis_wgan_a100.sh
python pattern_recognition_demo2/part4_oasis_wgan_gp.py --demo
```

If a short Slurm job ends after saving a checkpoint, continue to a larger total
epoch number without restarting:

```bash
python pattern_recognition_demo2/part4_oasis_wgan_gp.py \
  --resume --epochs 30 --output-dir results/part4_wgan_gp
```

The program saves fixed-noise samples during training, fresh final generations,
loss/critic-diagnostic curves and simple pairwise-diversity measurements. These
diagnostics help identify mode collapse but do not replace visual assessment of
brain realism.

### WGAN-GP result

The model used all **9,664** OASIS training slices at 128x128 resolution. It was
trained for 30 epochs in two checkpointed A100 jobs (epochs 1--8 took 84.1
seconds and epochs 9--30 took 213.4 seconds). At epoch 30 the Wasserstein
distance was **14.4555** and the gradient penalty was **0.2824**. The final set
of newly sampled images had pixel standard deviation **0.1995** and mean
pairwise L1 distance **0.0501**.

The fixed-noise grids show the same latent samples progressing from unstructured
blobs at epoch 1 to recognisable axial brain-like MR slices by epochs 14--30.
The final grid contains variation in outline, internal dark regions and tissue
texture, with no obvious exact duplicates. These numerical diversity measures
and visual grids provide evidence against obvious mode collapse, although they
are not clinical measures of anatomical realism. A temporary jump in all critic
diagnostics at epoch 9 occurred at the boundary between the two Slurm jobs; the
metrics subsequently stabilised, so the selected generator is the epoch-30
checkpoint rather than a value chosen from a single loss alone.

| WGAN-GP evidence | File |
| --- | --- |
| Fresh final random generations | [`final_generated_brains.png`](results/part4_wgan_gp/final_generated_brains.png) |
| Fixed-noise samples at epoch 1 | [`samples_epoch_001.png`](results/part4_wgan_gp/samples_epoch_001.png) |
| Fixed-noise samples at epoch 14 | [`samples_epoch_014.png`](results/part4_wgan_gp/samples_epoch_014.png) |
| Fixed-noise samples at epoch 22 | [`samples_epoch_022.png`](results/part4_wgan_gp/samples_epoch_022.png) |
| Fixed-noise samples at epoch 30 | [`samples_epoch_030.png`](results/part4_wgan_gp/samples_epoch_030.png) |
| Loss and critic diagnostics | [`learning_curves.png`](results/part4_wgan_gp/learning_curves.png) |
| Configuration and complete history | [`metrics.json`](results/part4_wgan_gp/metrics.json) |

The large generator/critic checkpoint is retained on Rangpur and excluded from
Git. Raw outputs for both training jobs are retained under
[`results/part4_logs/`](results/part4_logs/).

## Part 4 references

- D. P. Kingma and M. Welling, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).
- O. Ronneberger, P. Fischer and T. Brox, [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).
- M. Arjovsky, S. Chintala and L. Bottou, [Wasserstein GAN](https://arxiv.org/abs/1701.07875).
- I. Gulrajani et al., [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028).
- D. S. Marcus et al., [Open Access Series of Imaging Studies (OASIS)](https://doi.org/10.1162/jocn.2007.19.9.1498).

For Chinese demonstration and revision notes, see [`P4_REVIEW_ZH.md`](P4_REVIEW_ZH.md).

## AI assistance statement

OpenAI ChatGPT/Codex was used as a learning and development assistant for code
structure, Chinese explanatory comments, debugging PyTorch/Slurm compatibility,
and documentation. The implementation was executed on the specified datasets,
and all reported metrics and figures were produced and checked from the actual
Rangpur runs. The student is responsible for understanding and explaining the
models, losses, evaluation and results during demonstration.
