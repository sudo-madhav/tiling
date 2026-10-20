"""A weight-free encoder: fixed multi-scale filters, no training, no files.

Two jobs. It is the CPU test double for every learned encoder, and it is the
"untrained features" control arm of the ablation: if a learned encoder does not
beat this one, the gain came from multi-scale filtering and pooling, not from
what the network learned.

Channels per level: complex-cell Gabor energy (quadrature modulus, phase-invariant)
at 4 orientations on luminance, plus two opponent-colour means. Levels at strides
2, 4 and 8, average-pooled. A pooled "semantic" map at stride 16 concatenates the
coarse levels.
"""

import math

import numpy as np
from scipy import fft as sfft

from .base import Encoder, FeatureMap

_LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def _pool(x, s):
    h, w = x.shape[0] // s, x.shape[1] // s
    return x[:h * s, :w * s].reshape(h, s, w, s, -1).mean((1, 3))


class GaborEncoder(Encoder):
    name = "gabor"
    kind = "filters"
    multiple = 16

    def __init__(self, n_orient=4, wavelengths=(4.0, 8.0, 16.0)):
        self.n_orient = n_orient
        self.wavelengths = wavelengths
        self._cache = {}

    def layers(self):
        return {"g1": 2, "g2": 4, "g3": 8, "sem": 16}

    def _filters(self, h, w):
        key = (h, w)
        if key not in self._cache:
            ky = sfft.fftfreq(h)[:, None]
            kx = sfft.fftfreq(w)[None, :]
            bank = []
            for lam in self.wavelengths:
                f0 = 1.0 / lam
                for k in range(self.n_orient):
                    th = math.pi * k / self.n_orient
                    cx, cy = f0 * math.cos(th), f0 * math.sin(th)
                    sig = 0.35 * f0
                    g = np.exp(-((kx - cx) ** 2 + (ky - cy) ** 2) / (2 * sig ** 2))
                    bank.append(g.astype(np.float32))
            self._cache[key] = np.stack(bank).reshape(len(self.wavelengths), self.n_orient, h, w)
        return self._cache[key]

    def _forward(self, rgb):
        h, w = rgb.shape[:2]
        lum = rgb @ _LUMA
        opp = np.stack([rgb[..., 0] - rgb[..., 1], 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]], -1)
        spec = sfft.fft2(lum - lum.mean())
        bank = self._filters(h, w)
        out = {}
        coarse = []
        for level, (name, stride) in enumerate((("g1", 2), ("g2", 4), ("g3", 8))):
            chans = []
            for k in range(self.n_orient):
                # one-sided filter: real part is the even response, imaginary the odd
                chans.append(np.abs(sfft.ifft2(spec * bank[level, k])))
            stack = np.concatenate([np.stack(chans, -1), opp], -1).astype(np.float32)
            pooled = _pool(stack, stride)
            out[name] = FeatureMap(pooled, stride, "texture")
            coarse.append(_pool(stack, 16))
        out["sem"] = FeatureMap(np.concatenate(coarse, -1), 16, "semantic")
        return out
