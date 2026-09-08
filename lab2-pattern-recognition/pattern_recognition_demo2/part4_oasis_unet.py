"""COMP3710 Lab 2 Part 4 Task 2: categorical U-Net segmentation of OASIS MRI.

The four mask intensities 0/85/170/255 are mapped to class indices 0/1/2/3.
Targets and predictions are represented as four-channel one-hot tensors, as
required by the assignment.  Evaluation reports dataset-level DSC per label.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


NUM_CLASSES = 4
MASK_STEP = 85.0
PALETTE = np.array([
    [0, 0, 0], [230, 80, 70], [70, 180, 110], [70, 120, 230]
], dtype=np.uint8)


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


class OASISSegmentationDataset(Dataset):
    """Pair `case_...png` MRI slices with `seg_...png` categorical masks."""

    def __init__(self, image_dir: Path, mask_dir: Path, image_size: int = 256,
                 augment: bool = False, max_samples: int | None = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment
        images = sorted(image_dir.glob("*.png"))
        if max_samples:
            images = images[:max_samples]
        self.pairs: list[tuple[Path, Path]] = []
        for image_path in images:
            mask_name = image_path.name.replace("case_", "seg_", 1)
            mask_path = mask_dir / mask_name
            if mask_path.exists():
                self.pairs.append((image_path, mask_path))
        if not self.pairs:
            raise FileNotFoundError(
                f"No image/mask pairs found in {image_dir} and {mask_dir}"
            )
        if len(self.pairs) != len(images):
            print(f"Warning: paired {len(self.pairs)} of {len(images)} images")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as source:
            image = torch.from_numpy(
                np.asarray(source.convert("L"), dtype=np.float32).copy()
            )[None] / 255.0
        with Image.open(mask_path) as source:
            mask_values = torch.from_numpy(
                np.asarray(source.convert("L"), dtype=np.float32).copy()
            )[None, None]

        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(image[None], size=(self.image_size, self.image_size),
                                  mode="bilinear", align_corners=False)[0]
        if mask_values.shape[-2:] != (self.image_size, self.image_size):
            mask_values = F.interpolate(mask_values, size=(self.image_size, self.image_size),
                                        mode="nearest")
        mask_indices = torch.round(mask_values[0, 0] / MASK_STEP).long().clamp(0, 3)

        # Spatial augmentation must be applied identically to image and mask.
        if self.augment and torch.rand(()) < 0.5:
            image = torch.flip(image, dims=(-1,))
            mask_indices = torch.flip(mask_indices, dims=(-1,))

        # Categorical target shape: [classes, height, width].
        target_one_hot = F.one_hot(mask_indices, NUM_CLASSES).permute(2, 0, 1).float()
        return image, target_one_hot, image_path.name


class SyntheticSegmentationDataset(Dataset):
    """Small four-class dataset for a pipeline smoke test only."""

    def __init__(self, count: int = 32, image_size: int = 64, seed: int = 42):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[-1:1:complex(image_size), -1:1:complex(image_size)]
        self.samples = []
        for _ in range(count):
            scale = rng.uniform(0.9, 1.05)
            radius = np.sqrt((xx / scale) ** 2 + yy**2)
            mask = np.zeros_like(radius, dtype=np.int64)
            mask[radius < 0.82] = 1
            mask[radius < 0.58] = 2
            mask[((xx / 0.18) ** 2 + (yy / 0.28) ** 2) < 1] = 3
            image = mask.astype(np.float32) / 3 + rng.normal(0, 0.03, mask.shape)
            one_hot = F.one_hot(torch.from_numpy(mask), NUM_CLASSES).permute(2, 0, 1).float()
            self.samples.append((torch.from_numpy(np.clip(image, 0, 1))[None].float(),
                                 one_hot, "synthetic.png"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


class DoubleConv(nn.Module):
    """Two 3x3 convolutions; GroupNorm remains stable with small MRI batches."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample and concatenate the matching encoder skip connection."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        return self.conv(torch.cat((skip, x), dim=1))


class OASISUNet(nn.Module):
    """U-Net built directly in PyTorch; output is four categorical logits."""

    def __init__(self, num_classes: int = NUM_CLASSES, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.enc1 = DoubleConv(1, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c * 8, c * 16)
        self.up4 = UpBlock(c * 16, c * 8, c * 8)
        self.up3 = UpBlock(c * 8, c * 4, c * 4)
        self.up2 = UpBlock(c * 4, c * 2, c * 2)
        self.up1 = UpBlock(c * 2, c, c)
        self.classifier = nn.Conv2d(c, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        x = self.bottleneck(self.pool(e4))
        x = self.up4(x, e4)
        x = self.up3(x, e3)
        x = self.up2(x, e2)
        x = self.up1(x, e1)
        return self.classifier(x)


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor,
                      class_weights: torch.Tensor, dice_weight: float):
    # Cross entropy accepts the categorical one-hot target directly.
    cross_entropy = F.cross_entropy(logits, target, weight=class_weights)
    probabilities = logits.softmax(dim=1)
    intersection = (probabilities * target).sum(dim=(0, 2, 3))
    denominator = (probabilities + target).sum(dim=(0, 2, 3))
    soft_dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    dice_loss = 1 - soft_dice.mean()
    return cross_entropy + dice_weight * dice_loss, cross_entropy, dice_loss


def estimate_class_weights(loader: DataLoader, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    for _, targets, _ in loader:
        counts += targets.sum(dim=(0, 2, 3), dtype=torch.float64)
    frequencies = counts / counts.sum()
    # Inverse square-root weighting helps rare tissues without destabilising CE.
    weights = (frequencies.mean() / frequencies.clamp_min(1e-12)).sqrt()
    weights = (weights / weights.mean()).clamp(0.25, 4.0).float().to(device)
    print("Class pixel frequencies:", [f"{value:.4f}" for value in frequencies.tolist()])
    print("Cross-entropy weights:", [f"{value:.3f}" for value in weights.tolist()])
    return weights


def run_epoch(model, loader, device, class_weights, dice_weight,
              optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = sample_count = 0
    intersections = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    denominators = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(images)
                loss, _, _ = segmentation_loss(
                    logits, targets, class_weights, dice_weight
                )
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            predictions = F.one_hot(logits.argmax(1), NUM_CLASSES).permute(0, 3, 1, 2)
            # MPS cannot create float64 tensors, so move to CPU before casting.
            intersections += (predictions * targets).sum((0, 2, 3)).cpu().double()
            denominators += (predictions + targets).sum((0, 2, 3)).cpu().double()
            loss_sum += loss.item() * images.size(0)
            sample_count += images.size(0)
    dice = (2 * intersections + 1e-6) / (denominators + 1e-6)
    return loss_sum / sample_count, dice.numpy()


def mask_rgb(mask: np.ndarray) -> np.ndarray:
    return PALETTE[mask.astype(np.int64)]


def save_predictions(model, loader, device, output: Path, maximum: int = 6) -> None:
    model.eval()
    images, targets, _ = next(iter(loader))
    images = images[:maximum].to(device)
    with torch.inference_mode():
        prediction = model(images).argmax(1).cpu().numpy()
    truth = targets[:maximum].argmax(1).numpy()
    images = images.cpu().numpy()
    fig, axes = plt.subplots(len(images), 3, figsize=(9, 3 * len(images)), squeeze=False)
    for row in range(len(images)):
        axes[row, 0].imshow(images[row, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(mask_rgb(truth[row]))
        axes[row, 2].imshow(mask_rgb(prediction[row]))
        for axis in axes[row]:
            axis.axis("off")
    axes[0, 0].set_title("MRI")
    axes[0, 1].set_title("Ground truth")
    axes[0, 2].set_title("U-Net prediction")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_real_datasets(root: Path, image_size: int, max_samples: int | None):
    def dataset(split: str, augment: bool):
        return OASISSegmentationDataset(
            root / f"keras_png_slices_{split}",
            root / f"keras_png_slices_seg_{split}",
            image_size=image_size, augment=augment, max_samples=max_samples
        )
    return dataset("train", True), dataset("validate", False), dataset("test", False)


def make_loader(dataset, batch_size, workers, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OASIS four-class U-Net")
    parser.add_argument("--data-root", type=Path,
                        default=Path("/home/groups/comp3710/OASIS"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/part4_unet"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--demo", action="store_true",
                        help="load checkpoint, run test inference and save predictions")
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_unet.pt"
    print("Device:", device)

    if args.smoke_test:
        args.epochs, args.workers = 1, 0
        train_set = SyntheticSegmentationDataset(32, args.image_size, args.seed)
        val_set = SyntheticSegmentationDataset(8, args.image_size, args.seed + 1)
        test_set = SyntheticSegmentationDataset(8, args.image_size, args.seed + 2)
        print("SMOKE TEST: synthetic segmentations only")
    else:
        train_set, val_set, test_set = build_real_datasets(
            args.data_root, args.image_size, args.max_samples
        )
    print(f"Samples: train={len(train_set)}, validation={len(val_set)}, test={len(test_set)}")
    train_loader = make_loader(train_set, args.batch_size, args.workers, True)
    val_loader = make_loader(val_set, args.batch_size, args.workers, False)
    test_loader = make_loader(test_set, args.batch_size, args.workers, False)

    model = OASISUNet(NUM_CLASSES, args.base_channels).to(device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    class_weights = estimate_class_weights(train_loader, device)

    if args.demo:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        test_loss, test_dice = run_epoch(
            model, test_loader, device, class_weights, args.dice_weight
        )
        print(f"Test inference | loss {test_loss:.6f} | "
              + " | ".join(f"label {i} DSC {score:.4f}"
                             for i, score in enumerate(test_dice)))
        save_predictions(model, test_loader, device,
                         args.output_dir / "demo_predictions.png")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = {"train_loss": [], "val_loss": [],
               **{f"val_dice_{i}": [] for i in range(NUM_CLASSES)}}
    best_min_dice = -1.0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_dice = run_epoch(
            model, train_loader, device, class_weights, args.dice_weight,
            optimizer, scaler
        )
        val_loss, val_dice = run_epoch(
            model, val_loader, device, class_weights, args.dice_weight
        )
        scheduler.step()
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        for label, score in enumerate(val_dice):
            history[f"val_dice_{label}"].append(float(score))
        # Assignment requires every label >0.9, so select by the weakest label.
        minimum_dice = float(val_dice.min())
        if minimum_dice > best_min_dice:
            best_min_dice = minimum_dice
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "validation_dice": val_dice.tolist(),
                        "image_size": args.image_size,
                        "base_channels": args.base_channels}, checkpoint_path)
        train_dice_text = [round(float(x), 4) for x in train_dice]
        val_dice_text = [round(float(x), 4) for x in val_dice]
        print(f"Epoch {epoch:03d}/{args.epochs} | train loss {train_loss:.5f} "
              f"DSC {train_dice_text} | val loss {val_loss:.5f} "
              f"DSC {val_dice_text} | "
              f"time {time.perf_counter() - epoch_start:.1f}s")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_dice = run_epoch(
        model, test_loader, device, class_weights, args.dice_weight
    )
    save_predictions(model, test_loader, device, args.output_dir / "test_predictions.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = np.arange(1, args.epochs + 1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set(title="U-Net loss", xlabel="epoch", ylabel="loss")
    axes[0].legend()
    for label in range(NUM_CLASSES):
        axes[1].plot(epochs, history[f"val_dice_{label}"], label=f"label {label}")
    axes[1].axhline(0.9, color="black", linestyle="--", label="required 0.9")
    axes[1].set(title="Validation DSC by label", xlabel="epoch", ylabel="DSC",
                ylim=(0, 1.01))
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "learning_curves.png", dpi=180)
    plt.close(fig)

    result = {
        "best_epoch": checkpoint["epoch"],
        "best_validation_dice": checkpoint["validation_dice"],
        "test_loss": float(test_loss),
        "test_dice": [float(value) for value in test_dice],
        "all_test_labels_above_0.9": bool(np.all(test_dice > 0.9)),
        "training_seconds": time.perf_counter() - started,
        "configuration": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2),
                                                   encoding="utf-8")
    print("Best validation DSC:", checkpoint["validation_dice"])
    print("Test DSC:", test_dice.tolist())
    print("All test labels above 0.9:", bool(np.all(test_dice > 0.9)))
    print("Checkpoint and figures saved to:", args.output_dir)


if __name__ == "__main__":
    main()
