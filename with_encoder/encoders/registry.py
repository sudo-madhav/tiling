"""Build encoders and the encoder stack from a JSON config. Nothing is downloaded.

    python with_encoder/encoders/registry.py --check with_encoder/configs/encoders.example.json

A stack assigns one encoder to each role:
  texture   mid-level texture statistics (VGG-19 / ConvNeXt; or the weight-free Gabor control)
  semantic  material identity and the tile graph (DINOv2)
  latent    optional autoencoder latent (FLUX.1 / SD3.5 VAE); never the method's own autoencoder
"""

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from encoders.base import EncoderUnavailable, FeatureMap  # noqa: E402
    from encoders.gabor import GaborEncoder  # noqa: E402
else:
    from .base import EncoderUnavailable, FeatureMap
    from .gabor import GaborEncoder

ROLES = ("texture", "semantic", "latent")


def _expand(v):
    return os.path.expandvars(os.path.expanduser(v)) if isinstance(v, str) else v


def build_encoder(spec, device="cuda", dtype="float16"):
    spec = {k: _expand(v) for k, v in spec.items() if not k.startswith("_")}
    kind = spec.pop("type")
    spec.setdefault("device", device)
    spec.setdefault("dtype", dtype)
    if kind == "gabor":
        g = GaborEncoder()
        g.family = spec.get("family")
        return g
    if __package__ in (None, ""):
        from encoders import local_models
    else:
        from . import local_models
    cls = dict(dinov2=local_models.Dinov2Encoder, vgg19=local_models.VGG19Encoder,
               convnext=local_models.ConvNeXtEncoder, vae=local_models.VAEEncoder).get(kind)
    if cls is None:
        raise ValueError(f"unknown encoder type {kind!r}")
    if "layers" in spec:
        spec["layers"] = tuple(spec["layers"])
    if "hidden_layers" in spec:
        spec["hidden_layers"] = tuple(spec["hidden_layers"])
    return cls(**spec)


class Stack:
    """Encoders by role. One encoder may serve several roles; it then runs once per tile."""

    def __init__(self, encoders):
        self.encoders = {r: e for r, e in encoders.items() if e is not None}

    def has(self, role):
        return role in self.encoders

    def family(self, role):
        e = self.encoders.get(role)
        return getattr(e, "family", None) if e is not None else None

    def describe(self):
        return {r: dict(name=e.name, kind=e.kind, family=getattr(e, "family", None), layers=e.layers())
                for r, e in self.encoders.items()}

    def encode(self, rgb, roles):
        """{role: {layer: FeatureMap}} for the requested roles that this stack has.

        An encoder placed in a role it has no maps for (e.g. a deep ConvNeXt stage as a
        non-DINOv2 semantic cross-check) has all its maps relabelled to that role."""
        out, done = {}, {}
        for r in roles:
            e = self.encoders.get(r)
            if e is None:
                continue
            if id(e) not in done:
                done[id(e)] = e.encode(rgb)
            maps = {n: f for n, f in done[id(e)].items() if f.role == r}
            out[r] = maps or {n: FeatureMap(f.data, f.stride, r) for n, f in done[id(e)].items()}
        return out


def build_stack(cfg):
    """cfg: {"stack": {role: spec or null}, "device": ..., "dtype": ...} or the string "gabor"."""
    if cfg == "gabor" or cfg is None:
        g = GaborEncoder()
        return Stack(dict(texture=g, semantic=g))
    if isinstance(cfg, (str, Path)):
        cfg = json.loads(Path(cfg).read_text())
    device, dtype = cfg.get("device", "cuda"), cfg.get("dtype", "float16")
    built = {}
    for role in ROLES:
        spec = cfg.get("stack", {}).get(role)
        if spec:
            built[role] = build_encoder(dict(spec), device, dtype)
    return Stack(built)


def conflicts(cfg, method):
    """Autoencoder families the given method decodes through (from cfg["conflicts"]).

    A key matches the method name exactly or as a prefix before "_": "vosr" covers "vosr_tile512"."""
    if isinstance(cfg, (str, Path)):
        cfg = json.loads(Path(cfg).read_text())
    fam = set()
    for key, families in ((cfg or {}).get("conflicts", {}) or {}).items():
        if key.startswith("_"):
            continue
        if method == key or method.startswith(key + "_"):
            fam |= set(families)
    return fam


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check that every encoder in a config loads from local files.")
    ap.add_argument("--check", required=True, help="encoder config JSON")
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.check).read_text())
    ok = True
    for role in ROLES:
        spec = cfg.get("stack", {}).get(role)
        if not spec:
            print(f"{role:9s} (none)")
            continue
        try:
            e = build_encoder(dict(spec), cfg.get("device", "cuda"), cfg.get("dtype", "float16"))
            import numpy as np
            out = e.encode(np.random.default_rng(0).random((224, 224, 3), dtype=np.float32))
            shapes = {n: (f.data.shape, f.stride, f.role) for n, f in out.items()}
            print(f"{role:9s} {e.name}: {shapes}")
        except EncoderUnavailable as err:
            ok = False
            print(f"{role:9s} UNAVAILABLE: {err}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
