from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from barnsley_fern import generate_fern, select_device


def normalise_points(x, y):
    """Place the fern in a square without changing its aspect ratio."""
    x_range = float(x.max() - x.min())
    y_range = float(y.max() - y.min())
    scale = max(x_range, y_range)
    if scale == 0:
        raise ValueError("Cannot analyse points with zero spatial range")

    normalised_x = (x - x.min()) / scale
    normalised_y = (y - y.min()) / scale
    return normalised_x, normalised_y


def box_counts(x, y, grid_sizes):
    """Count occupied boxes for several square grid resolutions."""
    x, y = normalise_points(x, y)
    counts = []

    for grid_size in grid_sizes:
        x_index = np.minimum((x * grid_size).astype(np.int64), grid_size - 1)
        y_index = np.minimum((y * grid_size).astype(np.int64), grid_size - 1)
        box_ids = x_index + grid_size * y_index
        counts.append(np.unique(box_ids).size)

    return np.asarray(counts, dtype=np.float64)


def estimate_box_counting_dimension(x, y, grid_sizes):
    """Estimate dimension from the slope of log N(epsilon) vs log(1/epsilon)."""
    counts = box_counts(x, y, grid_sizes)
    log_inverse_epsilon = np.log(np.asarray(grid_sizes, dtype=np.float64))
    log_counts = np.log(counts)

    slope, intercept = np.polyfit(log_inverse_epsilon, log_counts, 1)
    fitted = slope * log_inverse_epsilon + intercept

    residual_sum = np.sum((log_counts - fitted) ** 2)
    total_sum = np.sum((log_counts - log_counts.mean()) ** 2)
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else 1.0
    return slope, intercept, r_squared, counts


def main():
    device = select_device()
    print("Using device:", device)

    x, y = generate_fern(
        num_walkers=10_000,
        num_steps=120,
        burn_in=20,
        device=device,
        seed=3710,
    )
    print(f"Generated {len(x):,} points for analysis")

    grid_sizes = np.asarray([16, 32, 64, 128, 256, 512])
    dimension, intercept, r_squared, counts = estimate_box_counting_dimension(
        x, y, grid_sizes
    )

    sample_sizes = np.asarray([20_000, 50_000, 100_000, 250_000, 500_000, len(x)])
    sample_sizes = np.unique(np.minimum(sample_sizes, len(x)))
    dimension_by_sample = []
    for sample_size in sample_sizes:
        estimate, _, _, _ = estimate_box_counting_dimension(
            x[:sample_size], y[:sample_size], grid_sizes
        )
        dimension_by_sample.append(estimate)

    print(f"Estimated box-counting dimension: {dimension:.4f}")
    print(f"Linear-fit R^2: {r_squared:.4f}")
    for grid_size, count in zip(grid_sizes, counts):
        print(f"Grid {grid_size:>3} x {grid_size:<3}: {int(count):>6} occupied boxes")

    figure, axes = plt.subplots(1, 3, figsize=(17, 6))

    axes[0].scatter(x, y, c=y, s=0.08, cmap="viridis", linewidths=0)
    axes[0].set_title("Generated Barnsley Fern")
    axes[0].set_aspect("equal")
    axes[0].axis("off")

    log_grid_sizes = np.log(grid_sizes)
    log_counts = np.log(counts)
    axes[1].scatter(log_grid_sizes, log_counts, color="tab:blue", label="Measured")
    axes[1].plot(
        log_grid_sizes,
        dimension * log_grid_sizes + intercept,
        color="tab:red",
        label=f"Fit: D = {dimension:.3f}, $R^2$ = {r_squared:.3f}",
    )
    axes[1].set_xlabel("log(1 / box size)")
    axes[1].set_ylabel("log(occupied boxes)")
    axes[1].set_title("Box-counting dimension")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        sample_sizes,
        dimension_by_sample,
        marker="o",
        color="tab:green",
    )
    axes[2].axhline(dimension, color="tab:red", linestyle="--", alpha=0.7)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Number of sampled points (log scale)")
    axes[2].set_ylabel("Estimated dimension")
    axes[2].set_title("Estimate stability as data increases")
    axes[2].grid(alpha=0.25)

    figure.suptitle("Barnsley Fern: Structure and Fractal Analysis", fontsize=15)
    figure.tight_layout()

    output_path = Path("images/fractal_analysis.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print("Saved analysis figure to:", output_path)


if __name__ == "__main__":
    main()
