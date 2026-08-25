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
