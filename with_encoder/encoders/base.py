"""Encoder interface.

An encoder maps one RGB tile (H x W x 3, float32 in [0, 1]) to named feature maps,
each (h, w, C) with a known stride in input pixels. Every encoder is frozen and
deterministic. Inputs are reflect-padded to the encoder's multiple and the
feature grid is cropped back; tiles are never resized to fit an architecture.
"""

import math
from dataclasses import dataclass

import numpy as np


class EncoderUnavailable(RuntimeError):
    """Weights or packages are not on this machine. Nothing is downloaded to fix it."""


@dataclass
class FeatureMap:
    data: np.ndarray      # (h, w, C) float32
    stride: int           # input pixels per feature pixel
    role: str             # "texture", "semantic" or "latent"


class Encoder:
    name = "base"
    kind = "base"          # "cnn", "vit", "vae" or "filters"
    multiple = 1
    # Autoencoders this encoder must not judge, because a method decodes through them.
    # See configs/encoders.example.json "conflicts".
    family = None

    def layers(self):
        raise NotImplementedError

    def _forward(self, rgb):
        """rgb: (H, W, 3) float32 in [0, 1], H and W multiples of self.multiple."""
        raise NotImplementedError

    def encode(self, rgb):
        rgb = np.asarray(rgb, np.float32)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected an H x W x 3 tile, got {rgb.shape}")
        H, W = rgb.shape[:2]
        m = self.multiple
        ph, pw = (-H) % m, (-W) % m
        x = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect") if ph or pw else rgb
        out = {}
        for name, fmap in self._forward(x).items():
            h = math.ceil(H / fmap.stride)
            w = math.ceil(W / fmap.stride)
            out[name] = FeatureMap(np.ascontiguousarray(fmap.data[:h, :w], np.float32), fmap.stride, fmap.role)
        return out
