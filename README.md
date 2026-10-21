# Vision and MRI Modeling

Computer vision experiments covering image classification, brain MRI segmentation,
reconstruction and generation, with reproducible Python/PyTorch code and recorded results.

## Highlights

- **Image classification:** compare PCA + Random Forest with a two-layer CNN on
  the same LFW split; recorded CPU test accuracy is 60.87% and 86.65%, respectively.
- **Brain MRI segmentation:** train U-Net on 9,664 OASIS slices and evaluate on
  544 test slices; Dice exceeds 0.94 for all four classes, including background.
- **Generative modeling:** explore VAE reconstruction and WGAN-GP generation.
- **Numerical foundations:** Fourier/DFT benchmarks and PyTorch fractal analysis.

## Explore the project

| Component | Code and explanation |
| --- | --- |
| Classification, MRI and Fourier experiments | [Technical guide](lab2-pattern-recognition/README.md) |
| Saved metrics and visualizations | [Results](lab2-pattern-recognition/results/) |
| Fractal modeling and analysis | [Barnsley fern](lab1-barnsley-fern/README.md) |

Start with the component README for environment setup and commands. The directory
names are kept stable so existing scripts and result links continue to work.
Training used PyTorch on CPU, Apple MPS and an NVIDIA A100/Slurm environment.
Large datasets and model checkpoints are not distributed in this repository.

## Interpreting the results

Metrics describe the recorded datasets and splits, not clinical validation or
general performance guarantees. Each component documents the exact experiment,
hardware and limitations; experiments with different evaluation protocols should
not be compared as controlled improvements.

## Development and attribution

Development history is retained in Git. AI assistance is documented in the
component notes and [AI usage record](lab1-barnsley-fern/AI_USAGE.md).
Upstream libraries and datasets retain their own terms and attribution.
