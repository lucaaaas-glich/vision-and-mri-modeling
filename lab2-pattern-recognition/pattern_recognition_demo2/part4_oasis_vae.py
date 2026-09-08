"""COMP3710 Lab 2 Part 4 Task 1: convolutional VAE for OASIS brain MR images.

The program trains a VAE, saves the best validation checkpoint and produces:
training curves, reconstructions, a 2-D latent projection and a sampled
manifold.  Chinese comments explain the concepts likely to be asked in the
demonstration.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Rangpur batch jobs do not have a graphical display.
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
ARRAY_SUFFIXES = {".npy", ".npz"}
EXCLUDED_NAME_PARTS = ("label", "mask", "seg", "target", "truth", "atlas")
IMAGE_KEYS = ("images", "image", "x", "data", "mri", "scan", "volume", "vol")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalise_image(array: np.ndarray) -> np.ndarray:
    """Convert one MR slice to finite float32 values in [0, 1]."""
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if array.ndim == 3:
        # Accept HWC/CHW colour-like arrays, although MR data is normally grayscale.
        channel_axis = next((i for i, size in enumerate(array.shape) if size <= 4), None)
        if channel_axis is not None:
            array = array.mean(axis=channel_axis)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D slice, got shape {array.shape}")

    low, high = np.percentile(array, (0.5, 99.5))
    if high <= low:
        low, high = float(array.min()), float(array.max())
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def resize_tensor(image: np.ndarray, image_size: int) -> torch.Tensor:
    tensor = torch.from_numpy(normalise_image(image))[None, None]
    tensor = F.interpolate(tensor, size=(image_size, image_size),
                           mode="bilinear", align_corners=False)
    return tensor[0]


def select_npz_array(archive: np.lib.npyio.NpzFile) -> np.ndarray:
    for key in IMAGE_KEYS:
        if key in archive.files:
            return archive[key]
    candidates = [archive[key] for key in archive.files
                  if np.issubdtype(archive[key].dtype, np.number)]
    if not candidates:
        raise ValueError(f"NPZ contains no numeric arrays; keys={archive.files}")
    # The image array is normally the largest numeric array in the archive.
    return max(candidates, key=lambda item: item.size)


def slice_from_array(array: np.ndarray, sample_index: int, slice_axis: int,
                     slices_per_volume: int) -> np.ndarray:
    """Extract one 2-D image from a 2-D image, image batch, or 3-D volume."""
    array = np.asarray(array)
    while array.ndim > 3 and 1 in array.shape:
        array = np.squeeze(array)
    if array.ndim == 2:
        return array
    if array.ndim != 3:
        raise ValueError(f"Unsupported array shape {array.shape}; expected 2-D or 3-D")

    # A small first/last dimension is most likely a colour channel.
    small_axis = next((i for i, size in enumerate(array.shape) if size <= 4), None)
    if small_axis is not None:
        return array.mean(axis=small_axis)

    axis = slice_axis % 3
    count = array.shape[axis]
    # Select evenly-spaced central slices; outer slices are often almost all background.
    margin = max(1, count // 5)
    positions = np.linspace(margin, count - margin - 1,
                            max(1, slices_per_volume)).round().astype(int)
    position = int(positions[sample_index % len(positions)])
    return np.take(array, position, axis=axis)


class OASISSliceDataset(Dataset):
    """Lazy dataset supporting common preprocessed OASIS image/NumPy layouts."""

    def __init__(self, root: Path, image_size: int = 128, slice_axis: int = 0,
                 slices_per_volume: int = 8, max_files: int | None = None,
                 array_layout: str = "auto"):
        self.root = root
        self.image_size = image_size
        self.slice_axis = slice_axis
        self.slices_per_volume = slices_per_volume
        self.array_layout = array_layout
        candidates = sorted(
            path for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES | ARRAY_SUFFIXES
            and not any(part in path.name.lower() for part in EXCLUDED_NAME_PARTS)
        )
        self.files = candidates[:max_files] if max_files else candidates
        if not self.files:
            raise FileNotFoundError(
                f"No .npy/.npz/image files found under {root}. "
                "Run part4_inspect_oasis.py and check the dataset format."
            )

        self.references: list[tuple[Path, int]] = []
        for path in self.files:
            if path.suffix.lower() in IMAGE_SUFFIXES:
                self.references.append((path, 0))
                continue
            try:
                loaded = np.load(path, mmap_mode="r", allow_pickle=False)
                if isinstance(loaded, np.lib.npyio.NpzFile):
                    array = select_npz_array(loaded)
                    shape = array.shape
                    loaded.close()
                else:
                    shape = loaded.shape
                if self._is_batch(path, shape):
                    repeats = shape[0]
                else:
                    repeats = slices_per_volume if len(shape) == 3 and min(shape) > 4 else 1
                self.references.extend((path, index) for index in range(repeats))
            except Exception as error:
                print(f"Warning: skipping {path}: {error}")

        if not self.references:
            raise RuntimeError("Files were found, but none had a supported numeric image layout")

    def __len__(self) -> int:
        return len(self.references)

    def _is_batch(self, path: Path, shape: tuple[int, ...]) -> bool:
        """Decide whether axis 0 contains independent 2-D images."""
        if self.array_layout == "batch":
            return len(shape) in {3, 4}
        if self.array_layout == "volume":
            return False
        # Common saved batches are [N,H,W] or [N,H,W,1]/[N,1,H,W].
        if len(shape) == 4 and (shape[1] <= 4 or shape[-1] <= 4):
            return True
        batch_words = ("image", "images", "train", "test", "data", "slice")
        return (len(shape) == 3 and shape[1] == shape[2]
                and any(word in path.stem.lower() for word in batch_words))

    def __getitem__(self, index: int) -> torch.Tensor:
        path, sample_index = self.references[index]
        if path.suffix.lower() in IMAGE_SUFFIXES:
            with Image.open(path) as image:
                array = np.asarray(image.convert("F"), dtype=np.float32)
        else:
            loaded = np.load(path, mmap_mode="r", allow_pickle=False)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                array = select_npz_array(loaded)
                if self._is_batch(path, array.shape):
                    array = array[sample_index]
                else:
                    array = slice_from_array(array, sample_index, self.slice_axis,
                                             self.slices_per_volume)
                loaded.close()
            else:
                if self._is_batch(path, loaded.shape):
                    array = loaded[sample_index]
                else:
                    array = slice_from_array(loaded, sample_index, self.slice_axis,
                                             self.slices_per_volume)
        return resize_tensor(array, self.image_size)


class SyntheticBrainDataset(Dataset):
    """Tiny fake dataset used only to verify the code path, never as a result."""

    def __init__(self, size: int = 64, image_size: int = 128, seed: int = 42):
        generator = np.random.default_rng(seed)
        yy, xx = np.mgrid[-1:1:complex(image_size), -1:1:complex(image_size)]
        self.images = []
        for _ in range(size):
            width, height = generator.uniform(0.62, 0.88, 2)
            brain = ((xx / width) ** 2 + (yy / height) ** 2 < 1).astype(np.float32)
            ventricle = ((xx / 0.15) ** 2 + ((yy + 0.03) / 0.22) ** 2 < 1)
            brain[ventricle] *= generator.uniform(0.1, 0.35)
            brain *= np.clip(1.0 - 0.25 * np.sqrt(xx**2 + yy**2), 0, 1)
            brain += generator.normal(0, 0.025, brain.shape)
            self.images.append(torch.from_numpy(np.clip(brain, 0, 1))[None].float())

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.images[index]


class ConvVAE(nn.Module):
    """Convolutional VAE: encoder -> (mu, logvar) -> z -> decoder."""

    def __init__(self, image_size: int = 128, latent_dim: int = 16):
        super().__init__()
        if image_size < 32 or image_size % 16 != 0:
            raise ValueError("image_size must be >=32 and divisible by 16")
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.feature_size = image_size // 16
        self.encoder = nn.Sequential(
            self._encoder_block(1, 32),
            self._encoder_block(32, 64),
            self._encoder_block(64, 128),
            self._encoder_block(128, 256),
        )
        flattened = 256 * self.feature_size * self.feature_size
        self.fc_mu = nn.Linear(flattened, latent_dim)
        self.fc_logvar = nn.Linear(flattened, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, flattened)
        self.decoder = nn.Sequential(
            self._decoder_block(256, 128),
            self._decoder_block(128, 64),
            self._decoder_block(64, 32),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _encoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    @staticmethod
    def _decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2,
                               padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x).flatten(1)
        return self.fc_mu(features), self.fc_logvar(features)

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # z = mu + sigma*epsilon makes random sampling differentiable.
        # This reparameterisation trick lets gradients reach mu and log-variance.
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_input(z)
        x = x.view(-1, 256, self.feature_size, self.feature_size)
        return self.decoder(x)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(reconstruction: torch.Tensor, image: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor, beta: float):
    # Reconstruction loss asks the output to resemble the input.
    reconstruction_loss = F.mse_loss(reconstruction, image, reduction="mean")
    # KL forces q(z|x) towards N(0,I), producing a continuous sampleable manifold.
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss


@dataclass
class EpochMetrics:
    total: float
    reconstruction: float
    kl: float


def run_epoch(model: ConvVAE, loader: DataLoader, device: torch.device,
              beta: float, optimizer=None, scaler=None) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(3, dtype=np.float64)
    sample_count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images in loader:
            images = images.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                reconstruction, mu, logvar = model(images)
                losses = vae_loss(reconstruction, images, mu, logvar, beta)
            if training:
                scaler.scale(losses[0]).backward()
                scaler.step(optimizer)
                scaler.update()
            batch_size = images.size(0)
            totals += np.array([loss.item() for loss in losses]) * batch_size
            sample_count += batch_size
    # Convert NumPy scalars to plain Python floats so the checkpoint remains
    # compatible with PyTorch's safer ``weights_only=True`` loader.
    return EpochMetrics(*(float(value) for value in totals / sample_count))


def save_reconstructions(model: ConvVAE, loader: DataLoader, device: torch.device,
                         output: Path, count: int = 8) -> None:
    model.eval()
    images = next(iter(loader))[:count].to(device)
    with torch.inference_mode():
        mu, _ = model.encode(images)
        reconstructions = model.decode(mu)
    images = images.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()
    fig, axes = plt.subplots(2, len(images), figsize=(2 * len(images), 4))
    for column in range(len(images)):
        axes[0, column].imshow(images[column, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, column].imshow(reconstructions[column, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstructed")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def collect_latents(model: ConvVAE, loader: DataLoader, device: torch.device,
                    maximum: int = 3000) -> np.ndarray:
    model.eval()
    batches = []
    with torch.inference_mode():
        for images in loader:
            mu, _ = model.encode(images.to(device, non_blocking=True))
            batches.append(mu.cpu())
            if sum(len(batch) for batch in batches) >= maximum:
                break
    return torch.cat(batches).numpy()[:maximum]


def save_latent_projection(latents: np.ndarray, output: Path) -> None:
    centred = latents - latents.mean(axis=0, keepdims=True)
    if centred.shape[1] > 2:
        # PCA is used only for visualisation; VAE training itself stays nonlinear.
        _, _, vectors = np.linalg.svd(centred, full_matrices=False)
        points = centred @ vectors[:2].T
    else:
        points = centred[:, :2]
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(points[:, 0], points[:, 1], s=8, alpha=0.45)
    axis.set(title="OASIS VAE latent-space projection", xlabel="component 1",
             ylabel="component 2")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_sampled_manifold(model: ConvVAE, device: torch.device, output: Path,
                          grid_size: int = 10, span: float = 2.5) -> None:
    """Vary the first two latent coordinates over a grid and decode each point."""
    values = torch.linspace(-span, span, grid_size, device=device)
    latent = torch.zeros(grid_size * grid_size, model.latent_dim, device=device)
    row = 0
    for y in values.flip(0):
        for x in values:
            latent[row, 0] = x
            if model.latent_dim > 1:
                latent[row, 1] = y
            row += 1
    model.eval()
    with torch.inference_mode():
        generated = model.decode(latent).cpu()
    canvas = torch.zeros(grid_size * model.image_size, grid_size * model.image_size)
    for index, image in enumerate(generated):
        y, x = divmod(index, grid_size)
        canvas[y * model.image_size:(y + 1) * model.image_size,
               x * model.image_size:(x + 1) * model.image_size] = image[0]
    fig, axis = plt.subplots(figsize=(10, 10))
    axis.imshow(canvas.numpy(), cmap="gray", vmin=0, vmax=1)
    axis.set_title("Sampled 2-D slice through the VAE manifold")
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_history(history: dict[str, list[float]], output: Path) -> None:
    epochs = np.arange(1, len(history["train_total"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["train_total"], label="train")
    axes[0].plot(epochs, history["val_total"], label="validation")
    axes[0].set(title="VAE total loss", xlabel="epoch", ylabel="loss")
    axes[0].legend()
    axes[1].plot(epochs, history["val_reconstruction"], label="reconstruction")
    axes[1].plot(epochs, history["val_kl"], label="KL divergence")
    axes[1].set(title="Validation loss components", xlabel="epoch", ylabel="loss")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_loaders(dataset: Dataset, batch_size: int, workers: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    if isinstance(dataset, OASISSliceDataset):
        # Keep every slice from one volume/file in the same split.  Otherwise the
        # validation set could contain another slice of a training subject.
        unique_files = sorted({path for path, _ in dataset.references})
        if len(unique_files) > 1:
            shuffled = unique_files.copy()
            random.Random(seed).shuffle(shuffled)
            validation_file_count = max(1, round(0.15 * len(shuffled)))
            validation_files = set(shuffled[:validation_file_count])
            train_indices = [i for i, (path, _) in enumerate(dataset.references)
                             if path not in validation_files]
            validation_indices = [i for i, (path, _) in enumerate(dataset.references)
                                  if path in validation_files]
            train_set = Subset(dataset, train_indices)
            val_set = Subset(dataset, validation_indices)
            print(f"File-level split: {len(train_indices)} train / "
                  f"{len(validation_indices)} validation slices")
        else:
            print("Warning: only one data file; using a sample-level split. "
                  "Prefer separate subject files to prevent leakage.")
            validation_size = max(1, round(0.15 * len(dataset)))
            train_set, val_set = random_split(
                dataset, [len(dataset) - validation_size, validation_size],
                generator=generator
            )
    else:
        validation_size = max(1, round(0.15 * len(dataset)))
        train_set, val_set = random_split(
            dataset, [len(dataset) - validation_size, validation_size],
            generator=generator
        )
    if len(train_set) < 1 or len(val_set) < 1:
        raise ValueError("At least two images/slices are required")
    common = dict(batch_size=batch_size, num_workers=workers,
                  pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    return (DataLoader(train_set, shuffle=True, **common),
            DataLoader(val_set, shuffle=False, **common))


def load_checkpoint(model: ConvVAE, path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an OASIS convolutional VAE")
    parser.add_argument("--data-dir", type=Path, default=Path("/home/groups/comp3710"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/part4_vae"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=1e-3,
                        help="weight of KL divergence in beta-VAE loss")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--slice-axis", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--slices-per-volume", type=int, default=8)
    parser.add_argument("--array-layout", choices=("auto", "volume", "batch"),
                        default="auto", help="interpretation of 3-D NumPy arrays")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true",
                        help="test the pipeline on synthetic images; not an assignment result")
    parser.add_argument("--demo", action="store_true",
                        help="load the best checkpoint and regenerate evaluation figures")
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_vae.pt"
    print("Device:", device)

    if args.smoke_test:
        dataset: Dataset = SyntheticBrainDataset(64, args.image_size, args.seed)
        args.epochs = 1
        args.workers = 0
        print("SMOKE TEST: using synthetic images only")
    else:
        dataset = OASISSliceDataset(
            args.data_dir, args.image_size, args.slice_axis,
            args.slices_per_volume, args.max_files, args.array_layout
        )
    print(f"Dataset samples/slices: {len(dataset)}")
    train_loader, val_loader = make_loaders(
        dataset, args.batch_size, args.workers, args.seed
    )
    sample = dataset[0]
    print(f"Input tensor: shape={tuple(sample.shape)}, "
          f"range=[{sample.min():.3f}, {sample.max():.3f}]")

    model = ConvVAE(args.image_size, args.latent_dim).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters()
                          if parameter.requires_grad)
    print(model)
    print(f"Trainable parameters: {parameter_count:,}")

    if args.demo:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = load_checkpoint(model, checkpoint_path, device)
        metrics = run_epoch(model, val_loader, device, args.beta)
        print(f"Loaded epoch {checkpoint['epoch']} | validation loss {metrics.total:.6f} "
              f"reconstruction {metrics.reconstruction:.6f} KL {metrics.kl:.6f}")
        save_reconstructions(model, val_loader, device,
                             args.output_dir / "reconstructions_demo.png")
        save_sampled_manifold(model, device, args.output_dir / "manifold_demo.png")
        print("Demo figures saved to:", args.output_dir)
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = {key: [] for key in (
        "train_total", "train_reconstruction", "train_kl",
        "val_total", "val_reconstruction", "val_kl"
    )}
    best_validation = math.inf
    training_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, device, args.beta,
                                  optimizer, scaler)
        validation_metrics = run_epoch(model, val_loader, device, args.beta)
        for split, metrics in (("train", train_metrics), ("val", validation_metrics)):
            history[f"{split}_total"].append(metrics.total)
            history[f"{split}_reconstruction"].append(metrics.reconstruction)
            history[f"{split}_kl"].append(metrics.kl)
        if validation_metrics.total < best_validation:
            best_validation = validation_metrics.total
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "validation_loss": best_validation,
                "image_size": args.image_size,
                "latent_dim": args.latent_dim,
                "beta": args.beta,
            }, checkpoint_path)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train {train_metrics.total:.6f} "
            f"(recon {train_metrics.reconstruction:.6f}, KL {train_metrics.kl:.6f}) | "
            f"val {validation_metrics.total:.6f} "
            f"(recon {validation_metrics.reconstruction:.6f}, KL {validation_metrics.kl:.6f}) | "
            f"time {time.perf_counter() - epoch_start:.1f}s"
        )

    elapsed = time.perf_counter() - training_start
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    save_history(history, args.output_dir / "learning_curves.png")
    save_reconstructions(model, val_loader, device,
                         args.output_dir / "reconstructions.png")
    latents = collect_latents(model, val_loader, device)
    save_latent_projection(latents, args.output_dir / "latent_projection.png")
    save_sampled_manifold(model, device, args.output_dir / "sampled_manifold.png")
    result = {
        "best_epoch": checkpoint["epoch"],
        "best_validation_loss": checkpoint["validation_loss"],
        "training_seconds": elapsed,
        "dataset_samples": len(dataset),
        "trainable_parameters": parameter_count,
        "configuration": {
            "data_dir": str(args.data_dir),
            **{key: value for key, value in vars(args).items()
               if key not in {"data_dir", "output_dir"}},
        },
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"Best validation loss: {best_validation:.6f} at epoch {checkpoint['epoch']}")
    print(f"Total training time: {elapsed:.1f}s")
    print("Checkpoint and figures saved to:", args.output_dir)


if __name__ == "__main__":
    main()
