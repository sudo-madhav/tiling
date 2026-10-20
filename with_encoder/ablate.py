#!/usr/bin/env python3
"""Run every ablation arm on the synthetic corruption manifest, then summarise.

    python with_encoder/ablate.py run --manifest OUT/div8k/x8/manifest*.jsonl \
        --encoders with_encoder/configs/encoders.json --tag learned --gates pix,feat,none,all \
        --results OUT/ablation_learned.jsonl [--shard 0/8]
    python with_encoder/ablate.py run ... --encoders gabor --tag gabor --results OUT/ablation_gabor.jsonl
    python with_encoder/ablate.py summarize OUT/ablation_*.jsonl --out OUT/ablation_summary

One row per (image, corruption, gate, encoder tag) with every term's energy, so all
configurations in terms.py come from one pass. Baselines: MSE and 1-SSIM always;
LPIPS / DISTS only with --external-baselines and only if the packages and their
weights are already on disk (they are never fetched).
"""

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "genet_ablation"))
import terms as T  # noqa: E402
from encoders.registry import build_stack  # noqa: E402
from hybrid_noref import hybrid_noref  # noqa: E402
from hybrid_ref import hybrid_ref  # noqa: E402
import featstats as fs  # noqa: E402

wcsi = fs.wcsi
CATCH = ("F2_colour_drift", "F3_seam_step", "F4_texture_clash", "F5_oversmooth", "F7_tile_colour_jitter")
PASS = "F1_texture_resample"
ALLOMETRIC = "F6_coherent_sharpen"


def _lum01(path):
    a, white = wcsi.load_rgb(path)
    a = np.clip(a.astype(np.float32) / white, 0, 1)
    return a @ np.array([0.299, 0.587, 0.114], np.float32), a


def ssim(a, b):
    f = lambda x: ndimage.gaussian_filter(x, 1.5)  # noqa: E731
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ma, mb = f(a), f(b)
    saa, sbb, sab = f(a * a) - ma * ma, f(b * b) - mb * mb, f(a * b) - ma * mb
    m = ((2 * ma * mb + c1) * (2 * sab + c2)) / ((ma * ma + mb * mb + c1) * (saa + sbb + c2))
    return float(m.mean())


def baselines(sr_path, gt_path, external):
    ls, rs = _lum01(sr_path)
    lg, rg = _lum01(gt_path)
    h, w = min(ls.shape[0], lg.shape[0]), min(ls.shape[1], lg.shape[1])
    out = dict(A_mse=float(np.mean((rs[:h, :w] - rg[:h, :w]) ** 2)) * 1e3,
               A_one_minus_ssim=1.0 - ssim(ls[:h, :w], lg[:h, :w]))
    if external:
        out.update(_external(rs[:h, :w], rg[:h, :w]))
    return out


def _external(a, b):
    """LPIPS / DISTS if importable with cached weights. Errors become None, never a download."""
    res = {}
    try:
        import torch
        # Both wrap torchvision's VGG-16, which torchvision fetches on first use. Refuse unless cached.
        if not (Path(torch.hub.get_dir()) / "checkpoints" / "vgg16-397923af.pth").exists():
            return dict(B_lpips=None, B_dists=None)
        x = torch.from_numpy(a.transpose(2, 0, 1)[None] * 2 - 1).float()
        y = torch.from_numpy(b.transpose(2, 0, 1)[None] * 2 - 1).float()
    except Exception:
        return dict(B_lpips=None, B_dists=None)
    try:
        import lpips
        res["B_lpips"] = float(lpips.LPIPS(net="vgg", verbose=False)(x, y).item())
    except Exception:
        res["B_lpips"] = None
    try:
        from DISTS_pytorch import DISTS
        res["B_dists"] = float(DISTS()((x + 1) / 2, (y + 1) / 2).item())
    except Exception:
        res["B_dists"] = None
    return res


def run(args):
    rows = [json.loads(line) for m in args.manifest for line in Path(m).read_text().splitlines() if line.strip()]
    if args.shard:
        k, n = (int(v) for v in args.shard.split("/"))
        stems = sorted({r["stem"] for r in rows})
        keep = set(stems[k::n])
        rows = [r for r in rows if r["stem"] in keep]
    cfg = None if args.encoders == "gabor" else json.loads(Path(args.encoders).read_text())
    stack = build_stack(cfg or "gabor")
    gates = [g for g in args.gates.split(",") if g]
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                done.add((r["stem"], r["kind"], r["gate"]))
    with out.open("a") as fh:
        for r in rows:
            base = None
            for gate in gates:
                if (r["stem"], r["kind"], gate) in done:
                    continue
                t0 = time.time()
                rec = dict(stem=r["stem"], kind=r["kind"], gate=gate, tag=args.tag, scale=r["scale"])
                try:
                    ref = hybrid_ref(r["sr"], r["gt"], r["lr"], stack=stack, scale=r["scale"], tile=r["tile_lr"],
                                     overlap=r["overlap_lr"], gate=gate, with_gmci=base is None)
                    if base is None:
                        base = dict(pix={k: ref["energies"][k] for k in
                                         ("pix_lr", "pix_seam", "pix_master", "pix_frust", "pix_emerge")},
                                    noref=hybrid_noref(r["sr"], r["lr"], stack=stack, scale=r["scale"],
                                                       tile_coords=r["tiles"]),
                                    base=baselines(r["sr"], r["gt"], args.external_baselines))
                    ref["energies"].update(base["pix"])
                    rec.update(energies_ref=ref["energies"], rho_ref=ref["rho_feature"],
                               energies_noref=base["noref"]["energies"], rho_noref=base["noref"]["rho_feature"],
                               baselines=base["base"], seconds=round(time.time() - t0, 2))
                except Exception as e:  # keep the sweep going; --retry by rerunning
                    rec.update(error=f"{type(e).__name__}: {e}", trace=traceback.format_exc(limit=3))
                fh.write(json.dumps(rec, default=float) + "\n")
                fh.flush()
                print(f"{r['stem']} {r['kind']} gate={gate} {'ERROR' if 'error' in rec else 'ok'}", flush=True)
    return 0


def _arm_values(rec):
    """{arm: energy-like value, lower is better} for one result row."""
    v = {}
    for name, c in T.score_all(rec["energies_ref"], "ref").items():
        v[f"ref:{name}"] = c["F_norm"] if c["used"] else None
    for name, c in T.score_all(rec["energies_noref"], "noref").items():
        v[f"noref:{name}"] = c["F_norm"] if c["used"] else None
    for k, x in rec["baselines"].items():
        v[f"base:{k}"] = x
    return v


def summarize(args):
    recs = []
    for p in args.results:
        recs += [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
    recs = [r for r in recs if "error" not in r]
    by = {}
    for r in recs:
        by.setdefault((r["tag"], r["gate"], r["stem"]), {})[r["kind"]] = r
    deltas = {}   # (tag, gate, arm) -> kind -> [paired delta per image]
    rhos = {}
    for (tag, gate, stem), kinds in by.items():
        if "clean" not in kinds:
            continue
        v0 = _arm_values(kinds["clean"])
        for kind, rec in kinds.items():
            if kind == "clean":
                continue
            v = _arm_values(rec)
            for arm, x in v.items():
                if x is None or v0.get(arm) is None:
                    continue
                deltas.setdefault((tag, gate, arm), {}).setdefault(kind, []).append(x - v0[arm])
            if kind == ALLOMETRIC:
                for side in ("ref", "noref"):
                    if rec.get(f"rho_{side}") is not None:
                        rhos.setdefault((tag, gate, side), []).append(rec[f"rho_{side}"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kinds = (PASS,) + CATCH + (ALLOMETRIC,)
    lines = ["| encoders | gate | arm | ISR (median) | " + " | ".join(k.split("_", 1)[0] for k in kinds) + " |",
             "|---|---|---|---|" + "---|" * len(kinds)]
    with (out / "ablation.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["tag", "gate", "arm", "isr_median", "n_images"] + [f"{k}_median_delta" for k in kinds]
                    + [f"{k}_ratio" for k in kinds])
        for (tag, gate, arm), dk in sorted(deltas.items()):
            med = {k: (float(np.median(dk[k])) if k in dk else None) for k in kinds}
            catch = [abs(med[k]) for k in CATCH if med[k] is not None]
            scale_ = float(np.median(catch)) if catch else None
            ratio = {k: (None if med[k] is None or not scale_ else abs(med[k]) / scale_) for k in kinds}
            isr = []
            n_img = len(dk.get(PASS, []))
            for i in range(n_img):
                c = [abs(dk[k][i]) for k in CATCH if k in dk and i < len(dk[k])]
                if c:
                    eps = 1e-3 * (scale_ or 1.0)
                    isr.append(float(np.mean(c)) / (abs(dk[PASS][i]) + eps))
            isr_med = float(np.median(isr)) if isr else None
            wr.writerow([tag, gate, arm, isr_med, n_img] + [med[k] for k in kinds] + [ratio[k] for k in kinds])

            def mark(k):
                r = ratio[k]
                if r is None:
                    return "n/a"
                ok = (r <= args.pass_ratio) if k == PASS else (r >= args.catch_ratio)
                return f"{r:.2f}{'' if k == ALLOMETRIC else (' ok' if ok else ' FAIL')}"
            lines.append(f"| {tag} | {gate} | {arm} | {'n/a' if isr_med is None else f'{isr_med:.1f}'} | "
                         + " | ".join(mark(k) for k in kinds) + " |")
    rho_lines = [f"| {tag} | {gate} | {side} | {np.median(v):.3f} | {len(v)} |" for (tag, gate, side), v in sorted(rhos.items())]
    md = ["# with_encoder ablation on controlled corruptions", "",
          "Cells: median paired change from the clean image, divided by the arm's median change over the "
          f"catch set ({', '.join(CATCH)}). F1 should stay below {args.pass_ratio}, the catch set above "
          f"{args.catch_ratio}. ISR = mean catch change / F1 change, per image, median.", ""] + lines
    if rho_lines:
        md += ["", "## Coherence on F6 (consistent sharpening; should be high)", "",
               "| encoders | gate | setting | median rho | n |", "|---|---|---|---|---|"] + rho_lines
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--manifest", required=True, nargs="+", help="one or more manifest JSONL files")
    r.add_argument("--encoders", default="gabor")
    r.add_argument("--tag", default="gabor")
    r.add_argument("--gates", default="pix")
    r.add_argument("--results", required=True)
    r.add_argument("--shard")
    r.add_argument("--external-baselines", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("results", nargs="+")
    s.add_argument("--out", required=True)
    s.add_argument("--pass-ratio", type=float, default=0.25)
    s.add_argument("--catch-ratio", type=float, default=0.25)
    args = ap.parse_args(argv)
    return run(args) if args.cmd == "run" else summarize(args)


if __name__ == "__main__":
    sys.exit(main())
