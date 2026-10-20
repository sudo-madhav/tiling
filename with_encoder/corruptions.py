#!/usr/bin/env python3
"""Controlled tiled-SR failure modes made from ground-truth images (no model, no download).

Each corruption isolates one failure so every ablation arm can be checked against
what it should and should not see (ABLATION.md, table 1). All operate on the
ground truth at full resolution, on the tile layout of the analysis grid.

  clean                   the ground truth itself
  F1_texture_resample     high band shifted 3 px where the GT is not phase-locked: valid
                          generative texture, should NOT be penalised (much)
  F2_colour_drift         smooth b* ramp from +d to -d across the image, no seams
  F3_seam_step            odd tile columns pasted with a 2 px horizontal misregistration
  F4_texture_clash        left half keeps the high band, right half keeps 30% of it
  F5_oversmooth           bicubic upsampling of the LR
  F6_coherent_sharpen     high band x1.4 everywhere: a consistent (allometric) departure
  F7_tile_colour_jitter   per-tile constant b* offset of +-d/2 (checkerboard): seams + low band

    python with_encoder/corruptions.py --config genet_ablation/config.json --scale 8 \
        --tile-sr 512 --out /path/to/with_encoder_synth --limit 100
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import fft as sfft
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "genet_ablation"))
import featstats as fs  # noqa: E402

gmci, wcsi = fs.gmci, fs.wcsi

KINDS = ("clean", "F1_texture_resample", "F2_colour_drift", "F3_seam_step", "F4_texture_clash",
         "F5_oversmooth", "F6_coherent_sharpen", "F7_tile_colour_jitter")

_M_XYZ_RGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                       [-0.9692660, 1.8760108, 0.0415560],
                       [0.0556434, -0.2040259, 1.0572252]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def lab_to_srgb(lab):
    """CIELAB (D65) -> sRGB in [0, 1], the inverse of wcsi.srgb_to_lab."""
    lab = np.asarray(lab, np.float64)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    d = 6 / 29

    def finv(t):
        return np.where(t > d, t ** 3, 3 * d * d * (t - 4 / 29))

    xyz = np.stack([finv(fx), finv(fy), finv(fz)], -1) * _WHITE
    rgb = xyz @ _M_XYZ_RGB.T
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.clip(rgb, 0, None) ** (1 / 2.4) - 0.055)
    return np.clip(rgb, 0, 1).astype(np.float32)


def layout(h0, w0, tile_lr, overlap_lr):
    return gmci._rects_for(h0, w0, tile_lr, overlap_lr)


def rigidity_full(gt01, rects, s):
    """Rigidity of the ground truth, computed tile by tile (bounded memory) and max-merged."""
    lab = wcsi.srgb_to_lab(gt01, 1.0)
    g = np.zeros(gt01.shape[:2], np.float32)
    for y0, x0, h, w in rects:
        ys, xs = slice(s * y0, s * (y0 + h)), slice(s * x0, s * (x0 + w))
        g[ys, xs] = np.maximum(g[ys, xs], gmci.rigidity_map(np.ascontiguousarray(lab[ys, xs, 0])))
    return g


def split(img, s, width=0.15):
    """(low, high): high is the content above the LR Nyquist frequency 1/(2s), the band an SR
    model is free to invent. Raised-cosine transition of +-width around the cutoff."""
    img = np.asarray(img, np.float32)
    h, w = img.shape[:2]
    fy = sfft.fftfreq(h)[:, None]
    fx = sfft.rfftfreq(w)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2) * (2 * s)
    lo, hi = 1 - width, 1 + width
    gain = np.clip((f - lo) / (hi - lo), 0, 1)
    gain = (0.5 - 0.5 * np.cos(np.pi * gain)).astype(np.float32)
    high = np.empty_like(img)
    for c in range(img.shape[2]):
        high[..., c] = sfft.irfft2(sfft.rfft2(img[..., c], workers=-1) * gain, s=(h, w), workers=-1)
    return img - high, high


def corrupt(kind, gt01, lr01, s, rects, delta_e=3.5, gamma=None):
    H, W = gt01.shape[:2]
    if kind == "clean":
        return gt01.copy()
    if kind == "F5_oversmooth":
        return np.clip(wcsi.upsample_bicubic(lr01, s), 0, 1)[:H, :W]
    low, R = split(gt01, s)
    if kind == "F1_texture_resample":
        # Swap whole regions, not a soft mix: where 0 < m < 1, R and its shifted copy partly
        # cancel and the corruption would remove high-frequency energy instead of moving it.
        g = gamma if gamma is not None else rigidity_full(gt01, rects, s)
        keep = ndimage.binary_dilation(ndimage.gaussian_filter(g, 1.0) >= 0.3, iterations=3)
        m = ndimage.gaussian_filter(keep.astype(np.float32), 0.7)[..., None]
        mix = m * R + (1 - m) * np.roll(R, (3, 3), (0, 1))
        return np.clip(low + split(mix, s)[1], 0, 1)  # the mask edge must not leak below Nyquist
    if kind == "F4_texture_clash":
        out = gt01.copy()
        for y0, x0, h, w in rects:
            if s * (x0 + w / 2) >= W / 2:
                ys, xs = slice(s * y0, s * (y0 + h)), slice(s * x0, s * (x0 + w))
                out[ys, xs] = low[ys, xs] + 0.3 * R[ys, xs]
        return np.clip(out, 0, 1)
    if kind == "F6_coherent_sharpen":
        return np.clip(low + 1.4 * R, 0, 1)
    if kind == "F2_colour_drift":
        lab = wcsi.srgb_to_lab(gt01, 1.0).astype(np.float64)
        lab[..., 2] += np.linspace(delta_e, -delta_e, W)[None, :]
        return lab_to_srgb(lab)
    if kind == "F7_tile_colour_jitter":
        lab = wcsi.srgb_to_lab(gt01, 1.0).astype(np.float64)
        out = lab.copy()
        pos = wcsi._grid_positions(rects)
        for k, (y0, x0, h, w) in enumerate(rects):
            r, c = pos[k] if pos is not None else (y0, x0)
            sign = 1.0 if (r + c) % 2 == 0 else -1.0
            ys, xs = slice(s * y0, s * (y0 + h)), slice(s * x0, s * (x0 + w))
            out[ys, xs, 2] = lab[ys, xs, 2] + sign * delta_e / 2
        return lab_to_srgb(out)
    if kind == "F3_seam_step":
        out = gt01.copy()
        pos = wcsi._grid_positions(rects)
        for k, (y0, x0, h, w) in enumerate(rects):
            col = pos[k][1] if pos is not None else k
            ys, xs = slice(s * y0, s * (y0 + h)), slice(s * x0, s * (x0 + w))
            region = gt01[ys, xs]
            if col % 2 == 1:
                region = np.concatenate([region[:, :2], region[:, :-2]], 1)
            out[ys, xs] = region
        return out
    raise ValueError(f"unknown corruption {kind!r}")


def _save(path, img01):
    from PIL import Image
    Image.fromarray((np.clip(img01, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def _tiles_jsonl(path, rects):
    with open(path, "w") as fh:
        for k, (y, x, h, w) in enumerate(rects):
            fh.write(json.dumps(dict(id=k, x=x, y=y, w=w, h=h)) + "\n")


def main(argv=None):
    import common  # genet_ablation/common.py
    ap = argparse.ArgumentParser(description="Write controlled tiled-SR corruptions of ground-truth images.")
    ap.add_argument("--config", required=True, help="genet_ablation config (gt and lr directories)")
    ap.add_argument("--scale", type=int, required=True)
    ap.add_argument("--tile-sr", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--delta-e", type=float, default=3.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard")
    args = ap.parse_args(argv)
    cfg = common.load_config(args.config)
    s = args.scale
    gt_idx = common.index_images(cfg["gt"])
    lr_idx = common.index_images(cfg["scales"][str(s)]["lr"])
    stems = common.select_stems(sorted(set(gt_idx) & set(lr_idx)), args.limit, common.parse_shard(args.shard))
    kinds = [k for k in args.kinds.split(",") if k]
    out = Path(args.out) / cfg.get("dataset", "data") / f"x{s}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / ("manifest.jsonl" if not args.shard else f"manifest.shard{args.shard.replace('/', 'of')}.jsonl")
    tile_lr = common.sr_pixels_to_lr(args.tile_sr, s, "tile_sr")
    overlap_lr = tile_lr // 16
    with manifest.open("a") as man:
        for stem in stems:
            s_, sr, gt, lr, _, gt_w, lr_w, notes = gmci.align_triplet(gt_idx[stem], gt_idx[stem], lr_idx[stem], s)
            gt01 = np.clip(gt.astype(np.float32) / gt_w, 0, 1)
            lr01 = np.clip(lr.astype(np.float32) / lr_w, 0, 1)
            rects = layout(lr01.shape[0], lr01.shape[1], tile_lr, overlap_lr)
            d = out / stem
            d.mkdir(exist_ok=True)
            _tiles_jsonl(d / "tiles.jsonl", rects)
            _save(d / "gt.png", gt01)
            _save(d / "lr.png", lr01)
            gamma = rigidity_full(gt01, rects, s) if "F1_texture_resample" in kinds else None
            for kind in kinds:
                img = corrupt(kind, gt01, lr01, s, rects, args.delta_e, gamma)
                _save(d / f"{kind}.png", img)
                man.write(json.dumps(dict(stem=stem, kind=kind, sr=str(d / f"{kind}.png"), gt=str(d / "gt.png"),
                                          lr=str(d / "lr.png"), tiles=str(d / "tiles.jsonl"), scale=s,
                                          tile_lr=tile_lr, overlap_lr=overlap_lr, delta_e=args.delta_e,
                                          notes=notes)) + "\n")
            print(f"{stem}: {len(kinds)} corruptions", flush=True)
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
