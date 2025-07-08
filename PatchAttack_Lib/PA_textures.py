import os
import torch
import numpy as np
import torch.nn.functional as F

# Helper Functions
def fade(t):
    return t * t * t * (t(t * 6 - 15) + 10)

def lerp(t, a, b):
    return a + t * (b - a)

def gradientText(hash, x, y):
    h = hash & 3
    if h < 2:
        u = x
        v = y
    else:
        u = y
        v = x
    return (u if h & 1 == 0 else -u) + (v if h & 2 == 0 else -v)

# Perlin Noise Generator
def perlin(size, scale=4):
    texture = np.zeros((size, size))
    perm = np.random.permutation(256)
    perm = np.tile(perm, 2)

    for i in range(size):
        for j in range(size):
            x = i / scale
            y = j / scale

            xi = int(x) & 255
            yi = int(y) & 255

            xf = x - int(x)
            yf = y - int(y)

            u = fade(xf)
            v = fade(yf)

            aa = perm[perm[xi] + yi]
            ab = perm[perm[xi] + yi + 1]
            ba = perm[perm[xi + 1] + yi]
            bb = perm[perm[xi + 1] + yi + 1]

            x1 = lerp(u, gradientText(aa, xf, yf), gradientText(ba, xf - 1, yf))
            x2 = lerp(u, gradientText(ab, xf, yf - 1), gradientText(bb, xf - 1, yf - 1))

            texture[i, j] = lerp(v, x1, x2)

            # Normalize
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
        return torch.FloatTensor(texture).unsqueeze(0)

# Gabor filter texture
def gabor(size, freq=0.1, theta=0, sigma=10, gamma=0.5):
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, y)

    # Rotation
    X_theta = X * np.cos(theta) + Y * np.sin(theta)
    Y_theta = -X * np.sin(theta) + Y * np.cos(theta)

    # Gabor function
    exponential = np.exp(-0.5 * (X_theta ** 2 + gamma ** 2 * Y_theta ** 2) / sigma ** 2)
    sinusoidal = np.cos(2 * np.pi * freq * X_theta)

    texture = exponential * sinusoidal

    # Add multiple orientations
    for angle in [np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        X_theta = X * np.cos(angle) + Y * np.sin(angle)
        Y_theta = -X * np.sin(angle) + Y * np.cos(angle)
        exponential = np.exp(-0.5 * (X_theta ** 2 + gamma ** 2 * Y_theta ** 2) / sigma ** 2)
        sinusoidal = np.cos(2 * np.pi * freq * X_theta)
        texture += 0.5 * exponential * sinusoidal

    # Normalize
    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
    return torch.FloatTensor(texture).unsqueeze(0)

# Fourier Texture
def fourier(size, freq_range=(2, 10)):
    texture = np.zeros((size, size))

    # Random frequencies and phases
    n_components = np.random.randint(3, 8)

    for i in range(n_components):
        freq_x = np.random.uniform(freq_range[0], freq_range[1])
        freq_y = np.random.uniform(freq_range[0], freq_range[1])
        phase = np.random.uniform(0, 2*np.pi)
        amplitude = np.random.uniform(0.1, 0.3)

        x = np.linspace(0, 2*np.pi, size)
        y = np.linspace(0, 2*np.pi, size)
        X, Y = np.meshgrid(x, y)

        component = amplitude * np.sin(freq_x * X + freq_y * Y + phase)
        texture += component

    # Normalize to [0, 1]
    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
    return torch.FloatTensor(texture).unsqueeze(0)

# Cellular automata texture
def cellular_texture(size, rule=30, generations=None):
    if generations is None:
        generations = size
    # Initialize random first row
    texture = np.zeros((generations, size))
    texture[0, :] = np.random.randint(0, 2, size)

    # Apply cellular automaton rule
    for i in range(1, generations):
        for j in range(size):
            left = texture[i-1, (j-1) % size]
            right = texture[i-1, (j+1) % size]
            middle = texture[i-1, j]
            # Convert area to binary number
            area = int(4 * left + 2 * middle + right)
            # Apply the rule
            texture [i, j] = (rule >> area) & 1

    if generations != size:
        texture = np.resize(texture, (size, size))

    # Convert to continuous values with smoothing
    texture = torch.FloatTensor(texture).unsqueeze(0).unsqueeze(0)
    texture = F.avg_pool2d(texture, kernel_size=3, stride=1, padding=1)
    texture = texture.squeeze()

    return texture


def gaussian_mix(size, distribution=0):
    if distribution == 0:  # Gaussian mixture
        texture = np.zeros((size, size))
        n = np.random.randint(3, 8)

        for _ in range(n):
            cx = np.random.randint(0, size)
            cy = np.random.randint(0, size)
            sigma = np.random.uniform(size / 10, size / 4)
            amplitude = np.random.uniform(0.3, 0.7)

            x, y = np.meshgrid(range(size), range(size))
            gaussian = amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
            texture += gaussian

    elif distribution == 1:  # Voronoi
        n_points = np.random.randint(10, 20)
        points = np.random.rand(n_points, 2) * size

        texture = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                distances = np.sqrt((points[:, 0] - i) ** 2 + (points[:, 1] - j) ** 2)
                texture[i, j] = np.min(distances) / size

    elif distribution == 2:  # Fractal
        texture = np.zeros((size, size))
        for octave in range(4):
            freq = 2 ** octave
            amplitude = 0.5 ** octave

            x = np.linspace(0, freq * np.pi, size)
            y = np.linspace(0, freq * np.pi, size)
            X, Y = np.meshgrid(x, y)

            texture += amplitude * (np.sin(X) * np.cos(Y))

    # Normalize
    texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
    return torch.FloatTensor(texture).unsqueeze(0)
