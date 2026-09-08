"""COMP3710 Lab 2 Part 4 Task 3: WGAN-GP generation of OASIS brain MR images.

WGAN-GP is used instead of the original GAN loss because Wasserstein scores and
the gradient penalty usually provide more stable training. The script saves
generated samples, loss curves, checkpoints and simple diversity diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image


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


class OASISGANDataset(Dataset):
    """Load official OASIS PNG slices and map [0,1] intensities to [-1,1]."""

    def __init__(self, image_dir: Path, image_size: int = 128,
                 max_samples: int | None = None):
        self.files = sorted(image_dir.glob("*.png"))
        if max_samples:
            self.files = self.files[:max_samples]
        self.image_size = image_size
        if not self.files:
            raise FileNotFoundError(f"No PNG images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.files[index]) as source:
            image = source.convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(array)[None] * 2.0 - 1.0


class SyntheticBrainDataset(Dataset):
    """Small fake dataset used only to verify the GAN training pipeline."""

    def __init__(self, count: int, image_size: int, seed: int):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[-1:1:complex(image_size), -1:1:complex(image_size)]
        self.images = []
        for _ in range(count):
            width, height = rng.uniform(0.62, 0.9, 2)
            radius = np.sqrt((xx / width) ** 2 + (yy / height) ** 2)
            brain = np.clip(1.0 - radius, 0, 1)
            ventricle = (xx / rng.uniform(0.1, 0.2)) ** 2 + (
                yy / rng.uniform(0.16, 0.28)
            ) ** 2 < 1
            brain[ventricle] *= 0.15
            brain += rng.normal(0, 0.025, brain.shape)
            image = torch.from_numpy(np.clip(brain, 0, 1))[None].float()
            self.images.append(image * 2 - 1)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.images[index]


def validate_image_size(image_size: int) -> None:
    if image_size < 32 or image_size & (image_size - 1):
        raise ValueError("image_size must be a power of two and at least 32")


class Generator(nn.Module):
    """Map a Gaussian latent vector to one grayscale MR slice."""

    def __init__(self, latent_dim: int = 128, image_size: int = 128,
                 base_channels: int = 64):
        super().__init__()
        validate_image_size(image_size)
        upsample_count = int(math.log2(image_size)) - 2  # number of 2x steps from 4
        start_channels = base_channels * 8
        self.latent_dim = latent_dim
        self.start_channels = start_channels
        self.project = nn.Sequential(
            nn.Linear(latent_dim, start_channels * 4 * 4, bias=False),
            nn.BatchNorm1d(start_channels * 4 * 4),
            nn.ReLU(inplace=True),
        )
        layers: list[nn.Module] = []
        channels = start_channels
        for _ in range(upsample_count - 1):
            next_channels = max(base_channels // 2, channels // 2)
            layers.extend([
                nn.ConvTranspose2d(channels, next_channels, 4, stride=2,
                                   padding=1, bias=False),
                nn.BatchNorm2d(next_channels),
                nn.ReLU(inplace=True),
            ])
            channels = next_channels
        layers.extend([
            nn.ConvTranspose2d(channels, 1, 4, stride=2, padding=1),
            nn.Tanh(),
        ])
        self.network = nn.Sequential(*layers)
        self.apply(initialise_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z).view(-1, self.start_channels, 4, 4)
        return self.network(x)


class Critic(nn.Module):
    """Assign an unrestricted Wasserstein score; this is not a probability."""

    def __init__(self, image_size: int = 128, base_channels: int = 64):
        super().__init__()
        validate_image_size(image_size)
        downsample_count = int(math.log2(image_size)) - 2
        layers: list[nn.Module] = [
            nn.Conv2d(1, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        channels = base_channels
        for _ in range(downsample_count - 1):
            next_channels = min(base_channels * 8, channels * 2)
            layers.extend([
                nn.Conv2d(channels, next_channels, 4, stride=2,
                          padding=1, bias=False),
                # BatchNorm is avoided in a WGAN-GP critic because samples in
                # one batch should not affect each other's gradient penalty.
                nn.GroupNorm(min(8, next_channels), next_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            channels = next_channels
        self.features = nn.Sequential(*layers)
        self.score = nn.Linear(channels * 4 * 4, 1)
        self.apply(initialise_weights)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.score(self.features(image).flatten(1)).flatten()


def initialise_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def gradient_penalty(critic: Critic, real: torch.Tensor,
                     fake: torch.Tensor) -> torch.Tensor:
    """Penalise gradient norms away from 1 to enforce the Lipschitz constraint."""
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=real.device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)
    scores = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.flatten(1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def save_samples(generator: Generator, fixed_noise: torch.Tensor,
                 output: Path) -> None:
    generator.eval()
    with torch.inference_mode():
        samples = (generator(fixed_noise).cpu() + 1) / 2
    save_image(samples, output, nrow=8, padding=2)


def diversity_metrics(images: torch.Tensor) -> dict[str, float]:
    """Simple diagnostics: near-zero values would warn about mode collapse."""
    images = (images.detach().float().cpu() + 1) / 2
    flattened = images.flatten(1)
    pairwise_l1 = torch.pdist(flattened, p=1).mean() / flattened.size(1)
    return {
        "generated_pixel_standard_deviation": float(images.std()),
        "mean_pairwise_l1_distance": float(pairwise_l1),
    }


def save_learning_curves(history: dict[str, list[float]], output: Path) -> None:
    epochs = np.arange(1, len(history["critic_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["critic_loss"], label="critic")
    axes[0].plot(epochs, history["generator_loss"], label="generator")
    axes[0].set(title="WGAN-GP losses", xlabel="epoch", ylabel="loss")
    axes[0].legend()
    axes[1].plot(epochs, history["wasserstein_distance"], label="E[D(real)]-E[D(fake)]")
    axes[1].plot(epochs, history["gradient_penalty"], label="gradient penalty")
    axes[1].set(title="Critic diagnostics", xlabel="epoch", ylabel="value")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OASIS WGAN-GP")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/home/groups/comp3710/OASIS/keras_png_slices_train"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/part4_wgan_gp"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--critic-steps", type=int, default=5)
    parser.add_argument("--gp-weight", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--demo", action="store_true",
                        help="load best generator and create a fresh sample grid")
    parser.add_argument("--resume", action="store_true",
                        help="resume model, optimizers and history from latest checkpoint")
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "wgan_gp_latest.pt"
    print("Device:", device)

    if args.smoke_test:
        if not args.resume:
            args.epochs = 1
        args.workers = 0
        dataset: Dataset = SyntheticBrainDataset(64, args.image_size, args.seed)
        print("SMOKE TEST: synthetic images only")
    else:
        dataset = OASISGANDataset(args.data_dir, args.image_size, args.max_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=args.workers,
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=args.workers > 0)
    print(f"Training images: {len(dataset)}")

    generator = Generator(args.latent_dim, args.image_size, args.base_channels).to(device)
    critic = Critic(args.image_size, args.base_channels).to(device)
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")
    fixed_noise = torch.randn(64, args.latent_dim, device=device)

    if args.demo:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        generator.load_state_dict(checkpoint["generator_state"])
        fresh_noise = torch.randn(64, args.latent_dim, device=device)
        save_samples(generator, fresh_noise, args.output_dir / "demo_generated_brains.png")
        with torch.inference_mode():
            metrics = diversity_metrics(generator(fresh_noise))
        print(f"Loaded epoch {checkpoint['epoch']} | diversity {metrics}")
        print("Fresh generated samples saved to:", args.output_dir)
        return

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.learning_rate,
                                            betas=(0.0, 0.9))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.learning_rate,
                                        betas=(0.0, 0.9))
    history = {key: [] for key in (
        "critic_loss", "generator_loss", "wasserstein_distance", "gradient_penalty"
    )}
    start_epoch = 1
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot resume; checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        generator.load_state_dict(checkpoint["generator_state"])
        critic.load_state_dict(checkpoint["critic_state"])
        generator_optimizer.load_state_dict(checkpoint["generator_optimizer_state"])
        critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state"])
        history = checkpoint.get("history", history)
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming after epoch {checkpoint['epoch']} toward epoch {args.epochs}")
    started = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        critic_values, generator_values, distances, penalties = [], [], [], []
        for step, real in enumerate(loader):
            real = real.to(device, non_blocking=True)
            batch_size = real.size(0)

            critic_optimizer.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, args.latent_dim, device=device)
            fake = generator(noise).detach()
            real_score = critic(real)
            fake_score = critic(fake)
            penalty = gradient_penalty(critic, real, fake)
            critic_loss = fake_score.mean() - real_score.mean() + args.gp_weight * penalty
            critic_loss.backward()
            critic_optimizer.step()
            critic_values.append(float(critic_loss.item()))
            distances.append(float((real_score.mean() - fake_score.mean()).item()))
            penalties.append(float(penalty.item()))

            if step % args.critic_steps == 0:
                generator_optimizer.zero_grad(set_to_none=True)
                noise = torch.randn(batch_size, args.latent_dim, device=device)
                generated = generator(noise)
                generator_loss = -critic(generated).mean()
                generator_loss.backward()
                generator_optimizer.step()
                generator_values.append(float(generator_loss.item()))

        values = {
            "critic_loss": float(np.mean(critic_values)),
            "generator_loss": float(np.mean(generator_values)),
            "wasserstein_distance": float(np.mean(distances)),
            "gradient_penalty": float(np.mean(penalties)),
        }
        for key, value in values.items():
            history[key].append(value)
        # GAN losses are relative because the critic changes simultaneously;
        # a numerically smaller generator loss does not reliably mean a better
        # image. Save the latest model and judge checkpoints using fixed-noise
        # sample grids together with diversity diagnostics.
        torch.save({
            "generator_state": generator.state_dict(),
            "critic_state": critic.state_dict(),
            "generator_optimizer_state": generator_optimizer.state_dict(),
            "critic_optimizer_state": critic_optimizer.state_dict(),
            "history": history,
            "epoch": epoch,
            "image_size": args.image_size,
            "latent_dim": args.latent_dim,
            "base_channels": args.base_channels,
        }, checkpoint_path)
        if epoch == 1 or epoch % args.sample_every == 0 or epoch == args.epochs:
            save_samples(generator, fixed_noise,
                         args.output_dir / f"samples_epoch_{epoch:03d}.png")
        print(f"Epoch {epoch:03d}/{args.epochs} | critic {values['critic_loss']:.4f} | "
              f"generator {values['generator_loss']:.4f} | "
              f"W-distance {values['wasserstein_distance']:.4f} | "
              f"GP {values['gradient_penalty']:.4f} | "
              f"time {time.perf_counter() - epoch_started:.1f}s")

    elapsed = time.perf_counter() - started
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    generator.load_state_dict(checkpoint["generator_state"])
    with torch.inference_mode():
        final_images = generator(torch.randn(64, args.latent_dim, device=device))
    diagnostics = diversity_metrics(final_images)
    save_image((final_images.cpu() + 1) / 2,
               args.output_dir / "final_generated_brains.png", nrow=8, padding=2)
    save_learning_curves(history, args.output_dir / "learning_curves.png")
    result = {
        "selected_epoch": checkpoint["epoch"],
        "training_seconds_this_run": elapsed,
        "training_images": len(dataset),
        "diversity_diagnostics": diagnostics,
        "configuration": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2),
                                                   encoding="utf-8")
    print("Selected generator epoch:", checkpoint["epoch"])
    print("Diversity diagnostics:", diagnostics)
    print(f"Total training time: {elapsed:.1f}s")
    print("Checkpoint and evidence saved to:", args.output_dir)


if __name__ == "__main__":
    main()
