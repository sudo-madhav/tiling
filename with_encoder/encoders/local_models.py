"""Adapters for learned frozen encoders. Local files only: nothing is ever downloaded.

Each adapter takes a path to weights that are already on disk (a file or a
Hugging Face style directory) and raises EncoderUnavailable when the path or
the package is missing. The Hugging Face offline flags are forced on before any
import, so a wrong path fails instead of fetching.

Not exercised by the CPU tests (no torch there). Check each adapter once on the
GPU machine with `python with_encoder/encoders/registry.py --check CONFIG`.
"""

import os

import numpy as np

from .base import Encoder, EncoderUnavailable, FeatureMap

for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "DIFFUSERS_OFFLINE"):
    os.environ[_k] = "1"

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _require(*names):
    mods = []
    for n in names:
        try:
            mods.append(__import__(n))
        except ImportError as e:
            raise EncoderUnavailable(f"package {n!r} is not installed ({e})") from e
    return mods


def _need_path(path):
    if not path or not os.path.exists(path):
        raise EncoderUnavailable(f"weights not found at {path!r}; nothing is downloaded, put them there first")
    return path


class _Torch(Encoder):
    def __init__(self, device="cuda", dtype="float16"):
        (torch,) = _require("torch")
        self.torch = torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype) if self.device.type == "cuda" else torch.float32

    def _tensor(self, rgb, mean=_IMAGENET_MEAN, std=_IMAGENET_STD):
        x = (rgb - mean) / std
        return self.torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None])).to(self.device, self.dtype)

    def _np(self, t):
        return t[0].float().permute(1, 2, 0).cpu().numpy()


class Dinov2Encoder(_Torch):
    """DINOv2 patch tokens (prefer the "with registers" checkpoints: fewer high-norm artefact tokens).

    path: a local Hugging Face directory, e.g. $WITH_ENCODER_WEIGHTS/dinov2-with-registers-large.
    hidden_layers: transformer blocks to tap (negative indices from the end); each becomes a
    "semantic" map named dino_L<i> at stride 14.
    """
    name = "dinov2"
    kind = "vit"
    multiple = 14

    def __init__(self, path, hidden_layers=(-1,), device="cuda", dtype="float16", family=None):
        super().__init__(device, dtype)
        (transformers,) = _require("transformers")
        self.model = transformers.AutoModel.from_pretrained(_need_path(path), local_files_only=True)
        self.model.to(self.device, self.dtype).eval()
        cfg = self.model.config
        self.patch = int(cfg.patch_size)
        self.multiple = self.patch
        self.n_reg = int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.hidden_layers = tuple(hidden_layers)
        self.family = family

    def layers(self):
        return {f"dino_L{i}": self.patch for i in self.hidden_layers}

    def _forward(self, rgb):
        H, W = rgb.shape[:2]
        h, w = H // self.patch, W // self.patch
        with self.torch.inference_mode():
            out = self.model(pixel_values=self._tensor(rgb), output_hidden_states=True)
        res = {}
        for i in self.hidden_layers:
            hs = out.last_hidden_state if i == -1 else out.hidden_states[i]
            tok = hs[0, 1 + self.n_reg:].float().reshape(h, w, -1).cpu().numpy()
            res[f"dino_L{i}"] = FeatureMap(tok, self.patch, "semantic")
        return res


class VGG19Encoder(_Torch):
    """VGG-19 ReLU activations, the classic texture-statistics network (Gatys et al.).

    path: torchvision state dict, e.g. $WITH_ENCODER_WEIGHTS/vgg19-dcbb9e9d.pth.
    """
    name = "vgg19"
    kind = "cnn"
    multiple = 8
    TAPS = {"relu1_2": (3, 1), "relu2_2": (8, 2), "relu3_4": (17, 4), "relu4_4": (26, 8), "relu5_4": (35, 16)}

    def __init__(self, path, layers=("relu2_2", "relu3_4", "relu4_4"), device="cuda", dtype="float16", family=None):
        super().__init__(device, dtype)
        (torchvision,) = _require("torchvision")
        net = torchvision.models.vgg19(weights=None)
        net.load_state_dict(self.torch.load(_need_path(path), map_location="cpu"))
        self.taps = {n: self.TAPS[n] for n in layers}
        last = max(i for i, _ in self.taps.values())
        self.features = net.features[:last + 1].to(self.device, self.dtype).eval()
        self.multiple = max(s for _, s in self.taps.values())
        self.family = family

    def layers(self):
        return {n: s for n, (_, s) in self.taps.items()}

    def _forward(self, rgb):
        want = {i: (n, s) for n, (i, s) in self.taps.items()}
        res = {}
        x = self._tensor(rgb)
        with self.torch.inference_mode():
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i in want:
                    n, s = want[i]
                    res[n] = FeatureMap(self._np(x), s, "texture")
        return res


class ConvNeXtEncoder(_Torch):
    """torchvision ConvNeXt stage outputs (strides 4, 8, 16, 32). Ablation alternative to VGG-19.

    path: torchvision state dict, e.g. $WITH_ENCODER_WEIGHTS/convnext_large-ea097f82.pth.
    """
    name = "convnext"
    kind = "cnn"
    STAGES = {"stage1": (1, 4), "stage2": (3, 8), "stage3": (5, 16), "stage4": (7, 32)}

    def __init__(self, path, arch="convnext_large", layers=("stage1", "stage2"), device="cuda",
                 dtype="float16", family=None):
        super().__init__(device, dtype)
        (torchvision,) = _require("torchvision")
        net = getattr(torchvision.models, arch)(weights=None)
        net.load_state_dict(self.torch.load(_need_path(path), map_location="cpu"))
        self.taps = {n: self.STAGES[n] for n in layers}
        last = max(i for i, _ in self.taps.values())
        self.features = net.features[:last + 1].to(self.device, self.dtype).eval()
        self.multiple = max(s for _, s in self.taps.values())
        self.family = family

    def layers(self):
        return {n: s for n, (_, s) in self.taps.items()}

    def _forward(self, rgb):
        want = {i: (n, s) for n, (i, s) in self.taps.items()}
        res = {}
        x = self._tensor(rgb)
        with self.torch.inference_mode():
            for i, block in enumerate(self.features):
                x = block(x)
                if i in want:
                    n, s = want[i]
                    res[n] = FeatureMap(self._np(x), s, "texture")
        return res


class VAEEncoder(_Torch):
    """Posterior mean of a diffusers AutoencoderKL (e.g. the 16-channel FLUX.1 or SD3.5 VAE), stride 8.

    path: a local diffusers directory; subfolder: "vae" for a full pipeline directory.
    family: a name for this autoencoder; a method that decodes through the same family is
    flagged as a conflict and its latent terms are not scored (see DESIGN.md, decision D6).
    """
    name = "vae"
    kind = "vae"
    multiple = 8

    def __init__(self, path, subfolder=None, device="cuda", dtype="float16", family="unknown-vae"):
        super().__init__(device, dtype)
        (diffusers,) = _require("diffusers")
        kw = dict(local_files_only=True)
        if subfolder:
            kw["subfolder"] = subfolder
        self.vae = diffusers.AutoencoderKL.from_pretrained(_need_path(path), **kw).to(self.device, self.dtype).eval()
        cfg = self.vae.config
        self.scale = float(getattr(cfg, "scaling_factor", 1.0) or 1.0)
        self.shift = float(getattr(cfg, "shift_factor", 0.0) or 0.0)
        self.family = family

    def layers(self):
        return {"latent": 8}

    def _forward(self, rgb):
        x = self._tensor(rgb, mean=np.float32(0.5), std=np.float32(0.5))
        with self.torch.inference_mode():
            z = self.vae.encode(x).latent_dist.mean
        z = (z - self.shift) * self.scale
        return {"latent": FeatureMap(self._np(z), 8, "latent")}
