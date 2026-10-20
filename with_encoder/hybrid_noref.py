#!/usr/bin/env python3
"""Hybrid pixel + frozen-feature score WITHOUT a reference (SR and LR only).

Pixel space keeps the WCSI terms (renormalisation against the LR, seams, stroke
angles, distant texture, 14-D sheaf) and adds an aliasing-peak probe. Frozen
features add two things WCSI cannot do in pixels:

  - the material graph: which tiles are the same stuff, from DINOv2 descriptors
    of the bicubic-upsampled LR (never of the SR, so a method cannot move its own
    judging graph);
  - the feature sheaf: the upgrade each tile received, psi(SR) - psi(U(LR)) in a
    frozen feature basis, must agree between same-material tiles after the
    Gaussian transport between their bicubic anchors.

    python with_encoder/hybrid_noref.py SR.png LR.png --scale 8 --tile 64 \
        --encoders with_encoder/configs/encoders.json --json out.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine  # noqa: E402
import featstats as fs  # noqa: E402
import terms as T  # noqa: E402
from encoders.registry import Stack, build_stack  # noqa: E402

wcsi = fs.wcsi
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "genet"))
import wcsi_viz  # noqa: E402

DEFAULTS = dict(
    k_pca=16,
    k_rho=4,
    neff_frac=0.25,
    basis_tiles=16,
    max_basis_tokens=200_000,
    gate="none",              # no ground truth, so no rigidity gate
    jnd_feat=0.2,
    jnd_angle_deg=4.0,
    jnd_freq=0.05,
    jnd_peak=math.log(2.0),
    nyq_floor=math.log(4.0),  # aliasing peak 4x the annulus median before it counts (nominal)
    n_q=8,
    n_proj_spectrum=12,
    tau_sem=0.3,
    tau_anchor=1.0,           # RMS JND per coordinate between bicubic anchors; sweep in ABLATION.md
    far_top=8,
    prior_pairs=1.0,
    harmonic_frac=0.05,
    harmonic_cap=4,
    rho_max_n=6000,
    drift_z=3.0,
)


def _robust_outliers(drift, w, z_thr):
    """Tiles whose semantic drift (SR vs U(LR)) is far above that of their same-material tiles."""
    N = len(drift)
    out = []
    for i in range(N):
        nb = np.flatnonzero(w[i] >= 0.5)
        nb = nb[nb != i]
        if len(nb) < 3:
            continue
        med = float(np.median(drift[nb]))
        mad = 1.4826 * float(np.median(np.abs(drift[nb] - med))) + 1e-6
        z = (drift[i] - med) / mad
        if z > z_thr:
            out.append(dict(index=i, drift=float(drift[i]), z=float(z), n_material=int(len(nb))))
    return sorted(out, key=lambda r: -r["z"])


def hybrid_noref(sr, lr, stack=None, scale=None, tile=None, overlap=None, tile_coords=None,
                 tile_space="lr", return_details=False, **params):
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown parameters: {sorted(unknown)}")
    p = {**DEFAULTS, **params}
    stack = stack if isinstance(stack, Stack) else build_stack(stack)
    D = wcsi_viz.decompose(sr, lr, scale=scale, tile=tile, overlap=overlap, tile_coords=tile_coords,
                           tile_space=tile_space)
    det, s, rects = D["det"], D["s"], D["rects"]
    notes = list(det["notes"])
    N = len(rects)
    tv = det["terms"]
    E = {k: None for k, v in T.TERMS.items() if v["setting"] == "noref"}
    E.update(nr_rg=tv["E_RG"], nr_seam=tv["E_GJ"], nr_pcp=tv["E_PCP"], nr_cyto=tv["E_cyto"], nr_sheaf=tv["E_sheaf"])

    sr01 = np.clip(D["sr_rgb"].astype(np.float32) / D["sr_white"], 0, 1)
    lr01 = np.clip(D["lr_rgb"].astype(np.float32) / D["lr_white"], 0, 1)
    up01 = np.clip(wcsi.upsample_bicubic(lr01, s), 0, 1)[:sr01.shape[0], :sr01.shape[1]]
    if D["canv"] is not None:
        notes.append("feature terms use the stitched SR image, not the separate tile canvases")

    peaks = []
    for r in rects:
        _, k, _ = fs.nyquist_stats(engine.crop(D["sr_lab"], r, s)[..., 0])
        peaks.append((max(k - p["nyq_floor"], 0.0) / p["jnd_peak"]) ** 2)
    E["nr_nyq"] = float(np.mean(peaks))

    roles = [r for r in ("texture", "latent", "semantic") if stack.has(r)]
    bases, gate = engine.fit_bases(stack, sr01, up01, rects, s, roles, p)
    per = [engine.tile_stats(stack, sr01, up01, r, s, bases, gate, p) for r in rects]

    I, J, near, far = D["I"], D["J"], D["near"], D["far"]
    rho, graph, outliers = None, {}, []
    frust_t = np.zeros(N)
    dk = engine.semantic_key(per)
    if N > 1 and dk is not None:
        cos = fs.centered_cosine(engine.descriptors(per, dk, "anchor"))
        aff = fs.affinity(cos[I, J], p["tau_sem"])
        keys = engine.sheaf_keys(per)
        if keys:
            # Same material is not enough without a reference: a tile holding a sharp contour
            # legitimately gains more from SR than a plain tile of the same material. Edges also
            # need similar bicubic-anchor statistics (WCSI's colour/band kernel, in feature space).
            pa = np.stack([np.concatenate([t["layers"][k]["psi_a"] for k in keys]) for t in per])
            aff = aff * np.exp(-np.mean((pa[I] - pa[J]) ** 2, 1) / (2 * p["tau_anchor"] ** 2))
        w = np.where(near, aff, fs.keep_top(I, J, aff, far, N, p["far_top"]))
        if far.any():
            fk = np.flatnonzero(far)
            wf = aff[fk]
            E["nr_cyto_dino"] = float(np.sum(wf * D["cyto_div"][fk].sum(1)) / (wf.sum() + p["prior_pairs"]))
        if keys:
            delta = engine.deltas(per, keys)
            edges = engine.build_edges(per, keys, I, J, w, p)
            E["nr_feat_sheaf"], frust_t = fs.frustration(delta, edges, p["prior_pairs"])
            rho = fs.coherence(delta, edges, p["harmonic_frac"], p["harmonic_cap"], p["rho_max_n"])
        Wm = np.eye(N)
        Wm[I, J] = Wm[J, I] = aff
        dt, da = engine.descriptors(per, dk, "test"), engine.descriptors(per, dk, "anchor")
        mean_a = da.mean(0)
        a_, t_ = da - mean_a, dt - mean_a
        drift = 1 - np.sum(a_ * t_, 1) / (np.linalg.norm(a_, axis=1) * np.linalg.norm(t_, axis=1) + 1e-12)
        outliers = _robust_outliers(drift, Wm, p["drift_z"])
        graph = dict(descriptor_layer=list(dk), n_near=int(near.sum()),
                     n_far_kept=int((w[far] > 0).sum()) if far.any() else 0,
                     mean_affinity_far=float(aff[far].mean()) if far.any() else None,
                     mean_affinity_far_pixel=float(D["w"][far].mean()) if far.any() else None)

    scores = T.score_all(E, "noref")
    out = dict(score=scores["N3_hybrid"]["score"], wcsi=det["wcsi"], energies=E, configs=scores, rho_feature=rho)
    if not return_details:
        return out
    out.update(scale=s, n_tiles=N, encoders=stack.describe(), graph=graph,
               semantic_drift_outliers=outliers,
               tiles=[dict(index=i, id=D["ids"][i], y0=r[0], x0=r[1], h=r[2], w=r[3],
                           frust=float(frust_t[i]), nyq=peaks[i]) for i, r in enumerate(rects)],
               notes=notes)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Hybrid pixel + frozen-feature score without a reference.")
    ap.add_argument("sr")
    ap.add_argument("lr")
    ap.add_argument("--scale", type=int)
    ap.add_argument("--tile", type=int, help="analysis tile in LR pixels")
    ap.add_argument("--overlap", type=int)
    ap.add_argument("--tiles-jsonl", help="tile rectangles, as in genet/wcsi.py")
    ap.add_argument("--tile-space", choices=("lr", "sr"), default="lr")
    ap.add_argument("--encoders", default="gabor", help="encoder config JSON, or 'gabor'")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    cfg = None if args.encoders == "gabor" else json.loads(Path(args.encoders).read_text())
    d = hybrid_noref(args.sr, args.lr, stack=cfg or "gabor", scale=args.scale, tile=args.tile,
                     overlap=args.overlap, tile_coords=args.tiles_jsonl, tile_space=args.tile_space,
                     return_details=True)
    rho = "n/a" if d["rho_feature"] is None else f"{d['rho_feature']:.3f}"
    print(f"hybrid (N3) {d['score']:.2f}   WCSI {d['wcsi']:.2f}   rho_feature {rho}   {d['n_tiles']} tiles")
    for k, v in d["energies"].items():
        print(f"  {k:14s} {'n/a' if v is None else f'{v:9.4f}'}   {T.TERMS[k]['source']}")
    for name, c in d["configs"].items():
        print(f"  {name:20s} {c['score']:7.2f}" + (f"   missing {c['missing']}" if c["missing"] else ""))
    for o in d["semantic_drift_outliers"][:5]:
        print(f"  semantic drift outlier: tile {o['index']} (z {o['z']:.1f})")
    for n in d["notes"]:
        print(f"  note: {n}")
    if args.json:
        Path(args.json).write_text(json.dumps(d, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
