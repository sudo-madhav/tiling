#!/usr/bin/env python3
"""Score real SR outputs (VOSR, CASR, SeedVR, bicubic, ...) with the hybrid scores.

Reads the genet_ablation config (gt, lr and method directories). Writes one JSON
per image with every term of both settings, so every configuration of terms.py
can be tabulated later. Resumable; shard by image for one process per GPU.

    CUDA_VISIBLE_DEVICES=0 python with_encoder/score_methods.py --config genet_ablation/config.json \
        --encoders with_encoder/configs/encoders.json --tag learned --scale 8 --tile-sr 512 \
        --methods bicubic,vosr,casr,seedvr --targets sr,gt --shard 0/8
    python with_encoder/score_methods.py --summarize --config genet_ablation/config.json --tag learned --scale 8
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "genet_ablation"))
import common  # noqa: E402
import score_sweep  # noqa: E402
import terms as T  # noqa: E402
from encoders.registry import build_stack, conflicts  # noqa: E402
from hybrid_noref import hybrid_noref  # noqa: E402
from hybrid_ref import hybrid_ref  # noqa: E402


def out_path(cfg, tag, scale, method, tile_sr, stem):
    return (Path(cfg["out"]) / "with_encoder" / tag / cfg.get("dataset", "div8k") / f"x{scale}"
            / method / f"tile{tile_sr}" / f"{stem}.json")


def jobs(cfg, scale, methods, targets, limit, shard):
    spec = score_sweep.scale_spec(cfg, scale)
    gt = common.index_images(cfg["gt"])
    lr = common.index_images(spec["lr"])
    stems = common.select_stems(set(gt) & set(lr), limit, common.parse_shard(shard))
    out = []
    if "gt" in targets:
        out += [("gt", s, gt[s], gt[s], lr[s]) for s in stems]
    if "sr" in targets:
        for name in methods:
            info = score_sweep.resolve_method(spec, name)
            if name == "bicubic" and info["path"] == "":
                out += [(name, s, None, gt[s], lr[s]) for s in stems]
            elif info["path"] and Path(info["path"]).is_dir():
                files = common.index_images(info["path"])
                out += [(name, s, files.get(s), gt[s], lr[s]) for s in stems]
            else:
                print(f"skip {name}: path unset or missing ({info['path']!r})")
    return out


def score(args):
    cfg = common.load_config(args.config)
    enc_cfg = None if args.encoders == "gabor" else json.loads(Path(args.encoders).read_text())
    stack = build_stack(enc_cfg or "gabor")
    s = args.scale
    tile = common.sr_pixels_to_lr(args.tile_sr, s, "tile_sr")
    methods = [m for m in args.methods.split(",") if m]
    targets = set(args.targets.split(","))
    for method, stem, sr, gt, lr in jobs(cfg, s, methods, targets, args.limit, args.shard):
        dest = out_path(cfg, args.tag, s, method, args.tile_sr, stem)
        if dest.is_file() and not (args.retry_errors and "error" in json.loads(dest.read_text())):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        rec = dict(method=method, stem=stem, scale=s, tile_sr=args.tile_sr, tag=args.tag)
        t0 = time.time()
        try:
            if sr is None and method != "bicubic":
                raise FileNotFoundError(f"no {method} output for {stem}")
            img = score_sweep._bicubic(lr, s) if sr is None else str(sr)
            fam = conflicts(enc_cfg, method) if enc_cfg else set()
            ref = hybrid_ref(img, str(gt), str(lr), stack=stack, scale=s, tile=tile, method_families=fam,
                             gate=args.gate, device=args.device, return_details=True)
            nr = hybrid_noref(img, str(lr), stack=stack, scale=s, tile=tile, calib=args.calib,
                              feat_calib=args.feat_calib, device=args.device, return_details=True)
            rec.update(ref=ref, noref=nr, encoder_conflict=ref["encoder_conflict"])
        except Exception as e:
            rec.update(error=f"{type(e).__name__}: {e}", trace=traceback.format_exc(limit=3))
        rec["seconds"] = round(time.time() - t0, 2)
        dest.write_text(json.dumps(rec, default=float))
        print(f"{method} {stem} {'ERROR ' + rec['error'] if 'error' in rec else 'ok'} {rec['seconds']}s", flush=True)
    return 0


def summarize(args):
    cfg = common.load_config(args.config)
    root = Path(cfg["out"]) / "with_encoder" / args.tag / cfg.get("dataset", "div8k") / f"x{args.scale}"
    rows = []
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        recs = [json.loads(f.read_text()) for f in sorted((mdir / f"tile{args.tile_sr}").glob("*.json"))]
        recs = [r for r in recs if "error" not in r]
        if not recs:
            continue
        row = dict(method=mdir.name, n=len(recs))
        for side, key in (("ref", "ref"), ("noref", "noref")):
            for name in (T.CONFIGS_REF if side == "ref" else T.CONFIGS_NOREF):
                cs = [r[key]["configs"][name] for r in recs]
                if args.quantity == "score":
                    sc = [c["score"] for c in cs if c["score"] is not None]
                    row[f"{side}:{name}"] = float(np.mean(sc)) if sc else None
                else:
                    e = [c["F_norm"] for c in cs if c.get("F_norm") is not None]
                    row[f"{side}:{name}"] = float(np.median(e)) if e else None
            rho = [r[key]["rho_feature"] for r in recs if r[key]["rho_feature"] is not None]
            row[f"{side}:rho_feature"] = float(np.median(rho)) if rho else None
        rows.append(row)
    cols = [c for c in rows[0] if c not in ("method", "n") and any(r[c] is not None for r in rows)] if rows else []
    what = ("mean score (higher is better)" if args.quantity == "score"
            else "median energy in JND (lower is better; does not saturate)")
    lines = [f"Cells: {what}; rho_feature is the median.", "",
             "| method | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for r in rows:
        lines.append(f"| {r['method']} | {r['n']} | "
                     + " | ".join("n/a" if r[c] is None else f"{r[c]:.2f}" for c in cols) + " |")
    out = root / f"summary_tile{args.tile_sr}_{args.quantity}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Hybrid scores of real SR outputs.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--encoders", default="gabor")
    ap.add_argument("--tag", default="gabor")
    ap.add_argument("--scale", type=int, required=True)
    ap.add_argument("--tile-sr", type=int, default=512)
    ap.add_argument("--methods", default="bicubic,vosr,casr,seedvr")
    ap.add_argument("--targets", default="sr,gt")
    ap.add_argument("--gate", default="pix", choices=("pix", "feat", "none", "all"))
    ap.add_argument("--calib", default="auto", help="WCSI-2 pixel calibration ('auto' or a JSON)")
    ap.add_argument("--feat-calib", help="feature calibration JSON from `hybrid_noref.py calibrate`")
    ap.add_argument("--device", help="WCSI-2 kernels: numpy (default), cuda:k")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard")
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--quantity", choices=("score", "energy"), default="score",
                    help="summary cells: mean score, or median energy (use when scores saturate)")
    args = ap.parse_args(argv)
    return summarize(args) if args.summarize else score(args)


if __name__ == "__main__":
    sys.exit(main())
