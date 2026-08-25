import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def select_device():
    """Use Apple MPS when available and otherwise fall back to the CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def generate_fern(
    num_walkers=20_000,
    num_steps=150,
    burn_in=20,
    device=None,
    seed=3710,
):
    """Generate Barnsley Fern points with parallel PyTorch walkers.

    Each walker independently selects one of the four affine transformations at
    every step. Points remain on the selected device during generation and are
    transferred to the CPU only once at the end.
    """
    if num_walkers <= 0:
        raise ValueError("num_walkers must be positive")
    if num_steps <= burn_in:
        raise ValueError("num_steps must be greater than burn_in")

    if device is None:
        device = select_device()

    torch.manual_seed(seed)

    x = torch.zeros(num_walkers, device=device)
    y = torch.zeros(num_walkers, device=device)
    saved_x = []
    saved_y = []

    for step in range(num_steps):
        probabilities = torch.rand(num_walkers, device=device)

        new_x = torch.empty_like(x)
        new_y = torch.empty_like(y)

        stem = probabilities < 0.01
        main_leaflets = (probabilities >= 0.01) & (probabilities < 0.86)
        left_leaflet = (probabilities >= 0.86) & (probabilities < 0.93)
        right_leaflet = probabilities >= 0.93

        new_x[stem] = 0.0
        new_y[stem] = 0.16 * y[stem]

        new_x[main_leaflets] = (
            0.85 * x[main_leaflets] + 0.04 * y[main_leaflets]
        )
        new_y[main_leaflets] = (
            -0.04 * x[main_leaflets] + 0.85 * y[main_leaflets] + 1.6
        )

        new_x[left_leaflet] = (
            0.20 * x[left_leaflet] - 0.26 * y[left_leaflet]
        )
        new_y[left_leaflet] = (
            0.23 * x[left_leaflet] + 0.22 * y[left_leaflet] + 1.6
        )

        new_x[right_leaflet] = (
            -0.15 * x[right_leaflet] + 0.28 * y[right_leaflet]
        )
        new_y[right_leaflet] = (
            0.26 * x[right_leaflet] + 0.24 * y[right_leaflet] + 0.44
        )

        x, y = new_x, new_y

        if step >= burn_in:
            saved_x.append(x)
            saved_y.append(y)

    all_x = torch.cat(saved_x).cpu().numpy()
    all_y = torch.cat(saved_y).cpu().numpy()
    return all_x, all_y


def plot_fern(x, y, output_path, cmap="viridis", show=True):
    """Plot and save generated fern points."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 10))
    scatter = axis.scatter(x, y, c=y, s=0.05, cmap=cmap, linewidths=0)
    axis.set_title("Barnsley Fern")
    axis.set_aspect("equal")
    axis.axis("off")
    figure.colorbar(scatter, ax=axis, label="Vertical position", fraction=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(figure)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate a PyTorch Barnsley Fern")
    parser.add_argument("--walkers", type=int, default=20_000)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--burn-in", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument(
        "--output", default="images/barnsley_fern.png", help="output image path"
    )
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    device = select_device()
    print("Using device:", device)

    x, y = generate_fern(
        num_walkers=args.walkers,
        num_steps=args.steps,
        burn_in=args.burn_in,
        device=device,
        seed=args.seed,
    )
    print(f"Generated {len(x):,} points")

    plot_fern(x, y, args.output, cmap=args.cmap, show=not args.no_show)
    print("Saved image to:", args.output)


if __name__ == "__main__":
    main()
