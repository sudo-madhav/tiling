#!/usr/bin/env python3
"""Does a score put the ground truth first? Variants of held-out ground-truth images, scored.

Builds, for every (GT, LR) pair, the ground truth and controlled stand-ins for SR outputs on
the analysis grid, and scores each with WCSI (v1), WCSI-2 without reference and WCSI-2 with
reference (and, with --metrics ...,gmci, the legacy GMCI of with_ref_genet/). Nothing is
downloaded or trained; the stand-ins are:

  gt              the ground truth
  bicubic         bicubic of the LR: no invented detail
  bicubic_usm     bicubic + unsharp mask (amount 1.5, sigma 2 px): a sharpened feed-forward output
  resample        above-Nyquist band shifted 3 px where the GT is not phase-locked (valid texture)
  phase_random    above-Nyquist band of each tile phase-randomised: right statistics, invented positions
  tile_gain       above-Nyquist band of each tile x U(0.4, 1.6): detail that varies from tile to tile
  clash           right-half tiles keep 30% of the above-Nyquist band
  sharpen         above-Nyquist band x 1.4 everywhere: a consistent over-sharpening
  colour_jitter   per-tile b* offset of +-1.75, checkerboard
  seam            odd tile columns pasted 2 px misregistered

    python genet/natural_check.py --gt-dir GT --lr-dir LR --scale 8 --tile-sr 256 \
        --first 851 --count 50 --out check_x8.csv --workers 4 [--calib genet/calib/wcsi2_x8_t256.json]
    python genet/natural_check.py --summarize check_x8.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "with_encoder"))
sys.path.insert(0, str(_HERE.parent / "with_ref_genet"))
import wcsi  # noqa: E402
import wcsi2  # noqa: E402

VARIANTS = ("gt", "bicubic", "bicubic_usm", "resample", "phase_random", "tile_gain", "clash", "sharpen",
            "colour_jitter", "seam")
METRICS = ("wcsi", "wcsi2", "wcsi2_ref")
OPTIONAL = ("gmci",)


def _variant(kind, gt01, lr01, s, rects, seed):
    import corruptions as C
    from scipy import ndimage
    H, W = gt01.shape[:2]
    if kind == "gt":
        return gt01
    if kind == "bicubic":
        return np.clip(wcsi.upsample_bicubic(lr01, s), 0, 1)[:H, :W]
    if kind == "bicubic_usm":
        b = np.clip(wcsi.upsample_bicubic(lr01, s), 0, 1)[:H, :W]
        return np.clip(b + 1.5 * (b - ndimage.gaussian_filter(b, (2, 2, 0))), 0, 1)
    table = dict(resample="F1_texture_resample", clash="F4_texture_clash", sharpen="F6_coherent_sharpen",
                 colour_jitter="F7_tile_colour_jitter", seam="F3_seam_step")
    if kind in table:
        return C.corrupt(table[kind], gt01, lr01, s, rects)
    low, R = C.split(gt01, s)
    rng = np.random.default_rng(seed)
    out = R.copy()
    for y0, x0, h, w in rects:
        ys, xs = slice(s * y0, s * (y0 + h)), slice(s * x0, s * (x0 + w))
        if kind == "tile_gain":
            out[ys, xs] = R[ys, xs] * rng.uniform(0.4, 1.6)
        elif kind == "phase_random":
            blk = R[ys, xs]
            ph = np.angle(np.fft.fft2(rng.normal(size=blk.shape[:2])))[..., None]
            F = np.fft.fft2(blk, axes=(0, 1))
            out[ys, xs] = np.real(np.fft.ifft2(np.abs(F) * np.exp(1j * ph), axes=(0, 1)))
        else:
            raise ValueError(kind)
    return np.clip(low + C.split(out, s)[1], 0, 1)


def _job(args):
    gt_path, lr_path, s, tile_sr, variants, metrics, calib = args
    gt, gw = wcsi.load_rgb(gt_path)
    lr, lw = wcsi.load_rgb(lr_path)
    s, gt, lr, _ = wcsi._match_scale(gt, lr, s)
    gt01 = gt.astype(np.float32) / np.float32(gw)
    lr01 = lr.astype(np.float32) / np.float32(lw)
    tile = tile_sr // s
    rects = wcsi2._rects(lr.shape[0], lr.shape[1], s, tile, None, None, "lr")[0]
    rows = []
    for k, kind in enumerate(variants):
        img = _variant(kind, gt01, lr01, s, rects, seed=k)
        row = dict(image=Path(gt_path).stem, variant=kind)
        for m in metrics:
            t0 = time.perf_counter()
            if m == "wcsi":
                d = wcsi.wcsi_score(img, lr01, scale=s, tile=tile, return_details=True)
                row[m] = d["wcsi"]
                row.update({f"{m}:{t}": v for t, v in d["terms"].items()})
            elif m == "wcsi2":
                d = wcsi2.wcsi2_score(img, lr01, scale=s, tile=tile, calib=calib, return_details=True)
                row[m] = d["wcsi2"]
                row.update({f"{m}:{t}": v for t, v in d["terms"].items()})
            elif m == "gmci":
                import gmci
                d = gmci.gmci_score(img, gt01, lr01, scale=s, tile=tile, return_details=True)
                row[m] = d["gmci"]
                row[f"{m}:rho"] = d["rho_coherence"]
            else:
                d = wcsi2.wcsi2_ref_score(img, gt01, lr01, scale=s, tile=tile, return_details=True)
                row[m] = d["wcsi2_ref"]
                row.update({f"{m}:{t}": v for t, v in d["terms"].items()})
                row[f"{m}:rho"] = d["rho_coherence"]
            row[f"{m}:sec"] = round(time.perf_counter() - t0, 2)
        rows.append(row)
    return rows


def summarize(path):
    rows = list(csv.DictReader(open(path)))
    metrics = [m for m in METRICS + OPTIONAL if m in rows[0]]
    by = {}
    for r in rows:
        by.setdefault(r["variant"], {}).setdefault(r["image"], r)
    imgs = sorted(by["gt"])
    out = ["| variant | " + " | ".join(f"{m} mean | GT wins / ties" for m in metrics) + " |",
           "|---|" + "---|---|" * len(metrics)]
    for v in VARIANTS:
        if v not in by:
            continue
        cells = []
        for m in metrics:
            vals = [float(by[v][i][m]) for i in imgs if i in by[v]]
            diff = np.array([float(by["gt"][i][m]) - float(by[v][i][m]) for i in imgs if i in by[v]])
            cells += [f"{np.mean(vals):.1f}",
                      "-" if v == "gt" else f"{100 * np.mean(diff > 0.05):.0f}% / {100 * np.mean(np.abs(diff) <= 0.05):.0f}%"]
        out.append(f"| {v} | " + " | ".join(cells) + " |")
    secs = [f"{m} {np.mean([float(r[m + ':sec']) for r in rows]):.1f} s" for m in metrics if m + ":sec" in rows[0]]
    out.append(f"\n{len(imgs)} images; mean seconds per score: " + ", ".join(secs))
    text = "\n".join(out)
    print(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gt-dir")
    ap.add_argument("--lr-dir")
    ap.add_argument("--scale", type=int)
    ap.add_argument("--tile-sr", type=int, default=256)
    ap.add_argument("--first", type=int, default=0, help="index of the first stem (sorted)")
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--metrics", default=",".join(METRICS))
    ap.add_argument("--calib", default="auto")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--summarize")
    args = ap.parse_args(argv)
    if args.summarize:
        summarize(args.summarize)
        return 0
    pairs = wcsi2._pairs_from_dirs(args.gt_dir, args.lr_dir, 0)[args.first:]
    if args.count:
        pairs = pairs[:args.count]
    variants = [v for v in args.variants.split(",") if v]
    metrics = [m for m in args.metrics.split(",") if m]
    unknown = set(metrics) - set(METRICS + OPTIONAL)
    if unknown:
        ap.error(f"unknown metrics {sorted(unknown)}")
    calib = None if args.calib == "none" else args.calib
    jobs = [(str(g), str(lo), args.scale, args.tile_sr, variants, metrics, calib) for g, lo in pairs]
    rows = []
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.workers) as pool:
            for r in pool.map(_job, jobs):
                rows += r
                print(f"  {r[0]['image']}", flush=True)
    else:
        for j in jobs:
            rows += _job(j)
            print(f"  {rows[-1]['image']}", flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("image", "variant"), k))
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, keys)
        wr.writeheader()
        wr.writerows(rows)
    summarize(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
