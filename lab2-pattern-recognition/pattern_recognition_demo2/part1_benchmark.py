import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch


def square_wave_fourier_numpy(t, f0, n_harmonics):
    result = np.zeros_like(t)

    for k in range(n_harmonics):
        n = 2 * k + 1
        result += np.sin(
            2 * np.pi * n * f0 * t
        ) / n

    return (4 / np.pi) * result


def naive_dft_numpy(x):
    # 两层循环直接实现 DFT 定义，时间复杂度 O(N^2)。
    N = len(x)

    X = np.zeros(
        N,
        dtype=np.complex128
    )

    for k in range(N):
        for n in range(N):
            angle = -2j * np.pi * k * n / N
            X[k] += x[n] * np.exp(angle)

    return X


def naive_dft_torch(x):
    # 数学上仍是 O(N^2)，速度提升主要来自 tensor 的向量化和 GPU 并行。
    # 代价是要保存 N×N 矩阵，所以 N 增大时显存需求也按 N^2 增长。
    N = x.shape[0]

    k = torch.arange(
        N,
        device=x.device,
        dtype=torch.float32
    ).reshape(N, 1)

    n = torch.arange(
        N,
        device=x.device,
        dtype=torch.float32
    ).reshape(1, N)

    angle = -2 * torch.pi * k * n / N

    fourier_matrix = torch.exp(
        1j * angle
    )

    X = fourier_matrix @ x.to(torch.complex64)

    return X


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare direct DFT and FFT timings")
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--output-dir", type=Path, default=Path("results/part1"))
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)
    f0, duration, n_harmonics = 1, 1.0, 50
    rows = []

    print(f"\n{'N':<8}{'Naive CPU':<18}{'NumPy FFT':<18}{'Tensor DFT':<18}{'Max error':<14}")
    print("-" * 76)

    for N in args.sizes:

        # -----------------------
        # Generate same signal
        # -----------------------

        t_numpy = np.linspace(
            0.0,
            duration,
            N,
            endpoint=False
        )

        signal_numpy = square_wave_fourier_numpy(
            t_numpy,
            f0,
            n_harmonics
        )

        signal_torch = torch.tensor(
            signal_numpy,
            dtype=torch.float32,
            device=device
        )

        # -----------------------
        # 1. NumPy naive DFT
        # -----------------------

        start = time.perf_counter()

        dft_numpy = naive_dft_numpy(
            signal_numpy
        )

        naive_time = (
            time.perf_counter() - start
        )

        # -----------------------
        # 2. NumPy FFT
        # -----------------------

        # FFT 为 O(N log N)。因为单次太快，重复 100 次再取平均可降低计时误差。
        repetitions = 100

        start = time.perf_counter()

        for _ in range(repetitions):
            fft_numpy = np.fft.fft(
                signal_numpy
            )

        fft_time = (
            time.perf_counter() - start
        ) / repetitions

        # -----------------------
        # 3. GPU tensor DFT
        # -----------------------

        if device.type == "mps":

            # Warm-up
            _ = naive_dft_torch(
                signal_torch
            )

            torch.mps.synchronize()

            start = time.perf_counter()

            dft_gpu = naive_dft_torch(
                signal_torch
            )

            torch.mps.synchronize()

            gpu_time = (
                time.perf_counter() - start
            )

        else:
            start = time.perf_counter()

            dft_gpu = naive_dft_torch(
                signal_torch
            )

            gpu_time = (
                time.perf_counter() - start
            )

        # -----------------------
        # Print results
        # -----------------------

            max_error = float(np.max(np.abs(dft_numpy - dft_gpu.detach().cpu().numpy())))
            close = bool(np.allclose(dft_numpy, dft_gpu.detach().cpu().numpy(), atol=1e-2, rtol=1e-3))
            rows.append({
                "n": N, "naive_numpy_seconds": naive_time,
                "numpy_fft_seconds": fft_time, "tensor_dft_seconds": gpu_time,
                "tensor_device": str(device), "max_absolute_error": max_error,
                "results_close": close,
            })
            print(f"{N:<8}{naive_time:<18.6f}{fft_time:<18.8f}{gpu_time:<18.6f}{max_error:<14.6g}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "benchmark.json"
    output.write_text(json.dumps({"experiment": "direct DFT versus FFT", "rows": rows}, indent=2))
    print("Saved reproducible benchmark to:", output)


if __name__ == "__main__":
    main()
