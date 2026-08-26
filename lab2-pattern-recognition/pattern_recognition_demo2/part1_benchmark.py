import time
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


device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)

sizes = [
    256,
    512,
    1024,
    2048
]

f0 = 1
T = 1.0
n_harmonics = 50

print()
print(
    f"{'N':<8}"
    f"{'Naive CPU':<18}"
    f"{'NumPy FFT':<18}"
    f"{'GPU DFT':<18}"
)

print("-" * 62)


for N in sizes:

    # -----------------------
    # Generate same signal
    # -----------------------

    t_numpy = np.linspace(
        0.0,
        T,
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

    # Run repeatedly because FFT is extremely fast
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

    print(
        f"{N:<8}"
        f"{naive_time:<18.6f}"
        f"{fft_time:<18.8f}"
        f"{gpu_time:<18.6f}"
    )