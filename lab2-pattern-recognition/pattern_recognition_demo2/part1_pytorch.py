import torch
import numpy as np

N = 2048
T = 1.0
f0 = 1

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)

t = torch.linspace(
    0.0,
    T,
    N + 1,
    device=device
)[:-1]


def square_wave_torch(t, f0):
    return torch.sign(
        torch.sin(
            2.0 * torch.pi * f0 * t
        )
    )


square = square_wave_torch(t, f0)

print(square.shape)
print(square[:10])

def square_wave_fourier_torch(t, f0, n_harmonics):
    result = torch.zeros_like(t)

    for k in range(n_harmonics):
        n = 2 * k + 1

        result += (
            torch.sin(
                2 * torch.pi * n * f0 * t
            ) / n
        )

    return (4 / torch.pi) * result

signal = square_wave_fourier_torch(
    t,
    f0,
    50
)

print("Signal device:", signal.device)
print("Signal shape:", signal.shape)
print("Signal min:", signal.min().item())
print("Signal max:", signal.max().item())


import time


def naive_dft_torch(x):
    N = x.shape[0]

    # k = frequency indices
    k = torch.arange(
        N,
        device=x.device,
        dtype=torch.float32
    ).reshape(N, 1)

    # n = sample indices
    n = torch.arange(
        N,
        device=x.device,
        dtype=torch.float32
    ).reshape(1, N)

    # DFT angle matrix
    angle = -2 * torch.pi * k * n / N

    # Build complex Fourier basis
    fourier_matrix = torch.exp(
        1j * angle
    )

    # Matrix-vector multiplication
    X = fourier_matrix @ x.to(torch.complex64)

    return X

# Warm-up
_ = naive_dft_torch(signal)
torch.mps.synchronize()

start = time.time()

gpu_dft_result = naive_dft_torch(signal)

torch.mps.synchronize()
gpu_time = time.time() - start

print("GPU DFT time:", gpu_time)

torch_fft_result = torch.fft.fft(
    signal.to(torch.complex64)
)

print(
    "GPU DFT close to Torch FFT:",
    torch.allclose(
        gpu_dft_result,
        torch_fft_result,
        atol=1e-2,
        rtol=1e-3
    )
)

difference = torch.abs(
    gpu_dft_result - torch_fft_result
)

print("Max absolute difference:", difference.max().item())
print("Mean absolute difference:", difference.mean().item())

max_index = torch.argmax(difference)

print("Largest difference at index:", max_index.item())
print("GPU DFT value:", gpu_dft_result[max_index])
print("Torch FFT value:", torch_fft_result[max_index])

print(
    "Close atol=1e-2:",
    torch.allclose(
        gpu_dft_result,
        torch_fft_result,
        atol=1e-2,
        rtol=1e-3
    )
)

print(
    "Close atol=1e-1:",
    torch.allclose(
        gpu_dft_result,
        torch_fft_result,
        atol=1e-1,
        rtol=1e-2
    )
)