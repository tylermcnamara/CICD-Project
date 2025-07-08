import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PatchAttack_Lib.PA_config import configure_PA, PA_cfg

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


'''
Generates a library of grayscale textures using procedural texture functions
'''
def generateTextureLib(size=64, textures_per_type=3):
    textures = []


