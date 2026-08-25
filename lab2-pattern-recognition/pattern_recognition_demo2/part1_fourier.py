import numpy as np
import matplotlib.pyplot as plt

# Number of samples
N = 2048

# Signal duration: 1 second
T = 1.0

# Fundamental frequency: 1 Hz
f0 = 1

# Number of harmonics to test
harmonics = [1, 5, 20, 50]


def square_wave(t):
    return np.sign(np.sin(2.0 * np.pi * f0 * t))


def square_wave_fourier(t, f0, N_harmonics):
    result = np.zeros_like(t)

    for k in range(N_harmonics):
        # Square wave only contains odd harmonics:
        # 1, 3, 5, 7, ...
        n = 2 * k + 1

        result += np.sin(
            2 * np.pi * n * f0 * t
        ) / n

    return (4 / np.pi) * result


# Time samples
t = np.linspace(
    0.0,
    T,
    N,
    endpoint=False
)

# Original square wave
square = square_wave(t)

plt.figure(figsize=(12, 8))

# Original square wave
plt.subplot(2, 3, 1)
plt.plot(t, square, "k", label="Square wave")
plt.title("Original Square Wave")
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.legend()


# Fourier approximations
for i, Nh in enumerate(harmonics, start=2):

    plt.subplot(2, 3, i)

    y = square_wave_fourier(
        t,
        f0,
        Nh
    )

    plt.plot(
        t,
        y,
        label=f"N={Nh} harmonics"
    )

    plt.plot(
        t,
        square,
        "k--",
        alpha=0.5,
        label="Square wave"
    )

    plt.title(
        f"Fourier Approximation with N={Nh}"
    )

    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.legend()


plt.tight_layout()
plt.show()

import time


def naive_dft(x):
    N = len(x)

    X = np.zeros(
        N,
        dtype=np.complex128
    )

    for k in range(N):

        for n in range(N):

            angle = -2j * np.pi * k * n / N

            X[k] += (
                x[n] *
                np.exp(angle)
            )

    return X


# Use 50 harmonics to build the square wave
signal = square_wave_fourier(
    t,
    f0,
    50
)


# Naive DFT
start = time.time()

dft_result = naive_dft(signal)

end = time.time()

naive_time = end - start


# NumPy FFT
start = time.time()

fft_result = np.fft.fft(signal)

end = time.time()

fft_time = end - start


print("Naive DFT time:", naive_time)
print("FFT time:", fft_time)

print(
    "Results close:",
    np.allclose(
        dft_result,
        fft_result
    )
)

# Frequency axis: only positive frequencies
xf = np.fft.fftfreq(N, d=T / N)[:N // 2]

# Convert complex DFT coefficients to magnitude
magnitude = 2.0 / N * np.abs(dft_result[:N // 2])

plt.figure(figsize=(12, 5))

plt.stem(xf, magnitude, basefmt=" ")

plt.title("DFT Magnitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

# Only show low frequencies clearly
plt.xlim(0, 50)

plt.grid(True)
plt.show()