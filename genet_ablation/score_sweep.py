#!/usr/bin/env python3
"""Score WCSI on gt and on each super-resolution method. Does not run any SR model.

Reads genet/wcsi.py. One JSON file per image per condition, so a stopped run resumes.
Full DIV8K is tens of thousands of scores; use --limit first. See PROTOCOL.md.

    python genet_ablation/score_sweep.py --config genet_ablation/config.json --dry-run
    python genet_ablation/score_sweep.py --config genet_ablation/config.json \
        --scales 8 --methods bicubic,vosr --targets sr,gt --tiles 512 --limit 20
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "genet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import wcsi  # noqa: E402


def scale_spec(cfg, scale):
    spec = cfg["scales"].get(str(scale))
    if spec is None:
        raise ValueError(f"scale {scale} is not in the config (have {sorted(cfg['scales'])})")
    return spec


def resolve_method(spec, name):
    if name not in spec["methods"]:
        raise ValueError(f"method {name!r} is not in this scale (have {sorted(spec['methods'])})")
    return common.method_spec(spec["methods"][name])


def build_jobs(cfg, scales, methods, targets, tiles, overlaps, offsets, limit, shard):
    """Job dicts. gt is scored once per image and analysis grid, not once per SR method."""
    dataset = cfg.get("dataset", "div8k")
    if not Path(cfg["gt"]).is_dir():
        raise FileNotFoundError(f"gt directory does not exist: {cfg['gt']}")
    gt = common.index_images(cfg["gt"])
    jobs = []
    for scale in scales:
        spec = scale_spec(cfg, scale)
        if not Path(spec["lr"]).is_dir():
            print(f"  skip scale {scale}: lr directory missing ({spec['lr']})")
            continue
        lr = common.index_images(spec["lr"])
        stems = common.select_stems(set(gt) & set(lr), limit, shard)
        available = {}
        for name in methods:
            info = resolve_method(spec, name)
            if name == "bicubic" and info["path"] == "":
                available[name] = {stem: None for stem in stems}
            elif info["path"] == "" or not Path(info["path"]).is_dir():
                print(f"  skip {name} at {scale}x: path unset or missing ({info['path']!r})")
                continue
            else:
                files = common.index_images(info["path"])
                available[name] = {stem: files.get(stem) for stem in stems}
        for tile_sr in tiles:
            for overlap_sr in overlaps[tile_sr]:
                for offset_sr in offsets[tile_sr]:
                    common.analysis_grid(scale, tile_sr, overlap_sr, offset_sr)
                    if "gt" in targets:
                        for stem in stems:
                            jobs.append(_job(cfg, dataset, scale, "gt", stem, gt[stem], lr[stem],
                                             tile_sr, overlap_sr, offset_sr, spec["lr"]))
                    if "sr" in targets:
                        for name, files in available.items():
                            for stem in stems:
                                jobs.append(_job(cfg, dataset, scale, name, stem, files[stem], lr[stem],
                                                 tile_sr, overlap_sr, offset_sr, spec["lr"]))
    return jobs


def _job(cfg, dataset, scale, method, stem, sr, lr, tile_sr, overlap_sr, offset_sr, lr_dir):
    return dict(dataset=dataset, scale=int(scale), method=method, stem=stem,
                sr=None if sr is None else str(sr), lr=str(lr),
                tile_sr=tile_sr, overlap_sr=overlap_sr, offset_sr=offset_sr,
                out=str(common.result_path(cfg["out"], dataset, scale, method, tile_sr, overlap_sr, offset_sr, stem)),
                lr_dir=str(lr_dir))


def _score(sr, lr, scale, tile_sr, overlap_sr, offset_sr):
    tile, overlap, offset = common.analysis_grid(scale, tile_sr, overlap_sr, offset_sr)
    kwargs = dict(scale=scale, return_details=True)
    if offset == 0:
        kwargs.update(tile=tile, overlap=overlap)
    else:
        lr_w, lr_h = common.header_size(lr)
        if isinstance(sr, (str, Path)):
            sr_w, sr_h = common.header_size(sr)
        else:
            sr_h, sr_w = sr.shape[:2]
        width, height = common.usable_lr_size((lr_w, lr_h), (sr_w, sr_h), scale)
        kwargs.update(tile_coords=common.lattice_coords(height, width, tile, overlap, offset), tile_space="lr")
    t0 = time.perf_counter()
    details = wcsi.wcsi_score(sr, lr, **kwargs)
    return details, time.perf_counter() - t0


def _bicubic(lr_path, scale):
    rgb, white = wcsi.load_rgb(lr_path)
    up = wcsi.upsample_bicubic(np.clip(rgb.astype(np.float32) / np.float32(white), 0, 1), scale)
    return np.clip(up, 0, 1).astype(np.float32)


def score_job(job):
    dest = Path(job["out"])
    if dest.is_file():
        existing = json.loads(dest.read_text())
        if existing.get("status") == "ok" or not job.get("retry_errors"):
            return existing
    row = dict(dataset=job["dataset"], scale=job["scale"], method=job["method"], image=job["stem"],
               tile_sr=job["tile_sr"], overlap_sr=job["overlap_sr"], offset_sr=job["offset_sr"],
               status="ok", wcsi=None, F_total=None, kappa=None, n_tiles=None,
               seconds=None, sr_w=None, sr_h=None, error=None, notes="")
    for term in common.TERMS:
        row[term] = None
    try:
        if job["sr"] is None and job["method"] != "bicubic":
            row["status"] = "missing"
            row["error"] = "no SR file for this stem"
        else:
            sr = _bicubic(job["lr"], job["scale"]) if job["method"] == "bicubic" and job["sr"] is None else job["sr"]
            if isinstance(sr, (str, Path)):
                row["sr_w"], row["sr_h"] = common.header_size(sr)
            else:
                row["sr_h"], row["sr_w"] = int(sr.shape[0]), int(sr.shape[1])
            details, seconds = _score(sr, job["lr"], job["scale"], job["tile_sr"], job["overlap_sr"], job["offset_sr"])
            row["seconds"] = round(seconds, 3)
            row["wcsi"] = details["wcsi"]
            row["F_total"] = details["F_total"]
            row["kappa"] = details["kappa"]
            row["n_tiles"] = details["layout"]["n_tiles"]
            row["notes"] = " | ".join(details["notes"])
            for term in common.TERMS:
                row[term] = details["terms"][term]
    except Exception as exc:  # one bad image should not stop the sweep
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc(limit=4)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(row))
    return row


def _format(row):
    score = "fail" if row["wcsi"] is None else f"{row['wcsi']:.2f}"
    return (f"{row['method']:16s} x{row['scale']} tile {row['tile_sr']:4d}  {row['image']}  "
            f"WCSI {score}  {row['status']}  {row['seconds'] or 0:.1f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description="WCSI sweep over methods, scales and analysis tiles.")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.json"))
    ap.add_argument("--scales", default="8,4")
    ap.add_argument("--methods", default="bicubic,vosr,casr,seedvr",
                    help="SR methods to score. gt is requested with --targets, not here")
    ap.add_argument("--targets", default="sr,gt", help="sr, gt, or both")
    ap.add_argument("--tiles", default="1024,512,256", help="analysis tile, SR pixels")
    ap.add_argument("--overlaps", default="tile/16", help="'tile/16' or comma SR-pixel overlaps")
    ap.add_argument("--offsets", default="0", help="'0', '0,half', or comma SR-pixel offsets")
    ap.add_argument("--limit", type=int, default=0, help="first N paired stems (0 = all)")
    ap.add_argument("--shard", help="i/n, score stems whose sorted index mod n equals i")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retry-errors", action="store_true",
                    help="recompute images whose JSON has status error or missing")
    args = ap.parse_args(argv)

    cfg = common.load_config(args.config)
    scales = common.parse_int_list(args.scales)
    tiles = common.parse_int_list(args.tiles)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    targets = {t.strip() for t in args.targets.split(",") if t.strip()}
    unknown = targets - {"sr", "gt"}
    if unknown:
        ap.error(f"unknown targets {sorted(unknown)}")
    overlaps = common.parse_overlaps(args.overlaps, tiles)
    offsets = common.parse_offsets(args.offsets, tiles)
    jobs = build_jobs(cfg, scales, methods, targets, tiles, overlaps, offsets,
                      args.limit, common.parse_shard(args.shard))
    for job in jobs:
        job["retry_errors"] = args.retry_errors
    done = sum(Path(job["out"]).is_file() for job in jobs)
    print(f"{len(jobs)} scores, {done} already on disk, {len(jobs) - done} to run")
    if args.dry_run:
        for job in jobs[:8]:
            print(f"  x{job['scale']} {job['method']} {job['stem']} tile {job['tile_sr']} "
                  f"ov {job['overlap_sr']} off {job['offset_sr']}")
        if len(jobs) > 8:
            print(f"  ... {len(jobs) - 8} more")
        return 0
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.workers) as pool:
            for row in pool.map(score_job, jobs):
                print(_format(row), flush=True)
    else:
        for job in jobs:
            print(_format(score_job(job)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
