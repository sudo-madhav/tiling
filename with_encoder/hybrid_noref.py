#!/usr/bin/env python3
"""Hybrid pixel + frozen-feature score WITHOUT a reference (SR and LR only).

Pixel space: the WCSI-2 terms (genet/wcsi2.py: coarse colour, seams, per-tile invented
detail and its consistency, each beyond what natural images score) and an aliasing-peak
probe. Frozen features add:

  - the material graph: which tiles are the same stuff, from DINOv2 descriptors
    of the bicubic-upsampled LR (never of the SR, so a method cannot move its own
    judging graph);
  - feature detail: each tile's feature energy per layer against the natural upgrade
    predicted from its bicubic anchor (a model fitted on ground-truth / LR pairs), so
    missing detail costs as much as invented excess detail;
  - feature consistency: the upgrade psi(SR) - psi(U(LR)) must agree between same-material
    tiles after the Gaussian transport between their anchors, beyond the disagreement
    natural images show.

Without calibration bicubic is the optimum of a pure consistency term (no upgrade is a
consistent upgrade), which is the flaw WCSI v1 had; calibrate once per encoder stack, scale
and tile:

    python with_encoder/hybrid_noref.py calibrate --gt GT_DIR --lr LR_DIR --scale 8 --tile-sr 512 \\
        --encoders with_encoder/configs/encoders.json --out feat_calib_x8_t512.json
    python with_encoder/hybrid_noref.py SR.png LR.png --scale 8 --tile-sr 512 \\
        --encoders with_encoder/configs/encoders.json --feat-calib feat_calib_x8_t512.json --json out.json
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
import wcsi2  # noqa: E402

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
    nyq_floor=None,           # aliasing-peak floor; None = calibrated, else log 4 (nominal)
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
    sat_loge=4.0,             # feature log-energy residuals saturate at this many nats
    calib="auto",             # WCSI-2 pixel natural model ("auto", path, dict, None)
    feat_calib=None,          # feature natural model (path or dict from `calibrate`), None = uncalibrated
    device=None,              # WCSI-2 kernels: None (numpy), "cuda:k", "torch:cpu"
    with_v1=False,            # also compute the legacy WCSI v1 terms (wcsi_*)
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


def _params(params):
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown parameters: {sorted(unknown)}")
    return {**DEFAULTS, **params}


def layer_name(key):
    return f"{key[0]}/{key[1]}/{key[2]}"


def _prepare(sr, lr, scale, tile, tile_sr, overlap, tile_coords, tile_space):
    sr_rgb, sr_w = wcsi.load_rgb(sr)
    lr_rgb, lr_w = wcsi.load_rgb(lr)
    s, sr_rgb, lr_rgb, notes = wcsi._match_scale(sr_rgb, lr_rgb, scale)
    t_lr = wcsi2._tile_arg(s, tile, tile_sr)
    rects, ids, _, layout, rnotes = wcsi2._rects(lr_rgb.shape[0], lr_rgb.shape[1], s, t_lr, overlap,
                                                 tile_coords, tile_space)
    pos = wcsi._grid_positions(rects)
    I, J, near, far, _ = wcsi._tile_graph(rects, pos)
    sr01 = np.clip(sr_rgb.astype(np.float32) / np.float32(sr_w), 0, 1)
    lr01 = np.clip(lr_rgb.astype(np.float32) / np.float32(lr_w), 0, 1)
    up01 = np.clip(wcsi.upsample_bicubic(lr01, s), 0, 1)[:sr01.shape[0], :sr01.shape[1]]
    return dict(s=s, sr=sr_rgb, lr=lr_rgb, sr01=sr01, up01=up01, rects=rects, ids=ids, layout=layout,
                t_lr=t_lr, I=I, J=J, near=near, far=far, notes=notes + rnotes)


def feature_pass(stack, sr01, up01, rects, s, I, J, near, far, p):
    """Per-tile feature statistics of sr01 against its bicubic anchor up01 and everything the
    feature terms (and their calibration) need."""
    N = len(rects)
    roles = [r for r in ("texture", "latent", "semantic") if stack.has(r)]
    bases, gate = engine.fit_bases(stack, sr01, up01, rects, s, roles, p)
    per = [engine.tile_stats(stack, sr01, up01, r, s, bases, gate, p) for r in rects]
    keys = engine.sheaf_keys(per)
    out = dict(per=per, keys=keys, dk=engine.semantic_key(per), aff=None, w=None, edges=[], delta=None,
               layer_r2=np.zeros((0, len(keys))), edge_w=np.zeros(0))
    out["X"] = np.array([[t["layers"][k]["loge_a"] for k in keys] for t in per]).reshape(N, len(keys))
    out["Y"] = np.array([[t["layers"][k]["loge_t"] for k in keys] for t in per]).reshape(N, len(keys))
    if N < 2 or out["dk"] is None:
        return out
    cos = fs.centered_cosine(engine.descriptors(per, out["dk"], "anchor"))
    aff = fs.affinity(cos[I, J], p["tau_sem"])
    if keys:
        # Same material is not enough without a reference: a tile holding a sharp contour
        # legitimately gains more from SR than a plain tile of the same material. Edges also
        # need similar bicubic-anchor statistics (WCSI's colour/band kernel, in feature space).
        pa = np.stack([np.concatenate([t["layers"][k]["psi_a"] for k in keys]) for t in per])
        aff = aff * np.exp(-np.mean((pa[I] - pa[J]) ** 2, 1) / (2 * p["tau_anchor"] ** 2))
    w = np.where(near, aff, fs.keep_top(I, J, aff, far, N, p["far_top"]))
    out.update(aff=aff, w=w)
    if keys:
        delta = engine.deltas(per, keys)
        edges = engine.build_edges(per, keys, I, J, w, p)
        sizes = [len(per[0]["layers"][k]["psi_t"]) for k in keys]
        bounds = np.cumsum([0] + sizes)
        r2 = []
        for i, j, A, S, _ in edges:
            r = (S * (A @ delta[i] - delta[j])) ** 2
            r2.append([float(r[bounds[k]:bounds[k + 1]].mean()) for k in range(len(keys))])
        out.update(delta=delta, edges=edges, layer_r2=np.array(r2).reshape(len(edges), len(keys)),
                   edge_w=np.array([e[4] for e in edges]))
    return out


def aliasing_peaks(lab, rects, s):
    return np.array([fs.nyquist_stats(engine.crop(lab, r, s)[..., 0])[1] for r in rects])


# ---------------------------------------------------------------- natural model

def load_feat_calibration(calib, keys, s, notes):
    if calib is None:
        return None
    cal = calib if isinstance(calib, dict) else json.loads(Path(calib).read_text())
    if int(cal["scale"]) != int(s):
        raise ValueError(f"feature calibration is for scale {cal['scale']}, scoring at {s}")
    names = [layer_name(k) for k in keys]
    missing = [n for n in names if n not in cal["layers"]]
    if missing:
        raise ValueError(f"feature calibration has no layers {missing}: fitted with another encoder stack")
    return cal


def calibrate(pairs, stack=None, scale=None, tile=None, tile_sr=None, overlap=None, out=None, note="",
              progress=None, **params):
    """Fit the feature natural model on ground-truth / LR pairs (GT plays the SR).

    Per sheaf layer: the natural feature energy of a tile as a linear function of its bicubic
    anchor's (a, b, residual variance sigma2), and the natural sheaf residual level nu_pair
    (mean squared transported upgrade difference between same-material tiles). Also the
    aliasing-peak floor (95th percentile of ground-truth tiles). Use the encoder stack,
    scale and analysis tile that will be scored.
    """
    p = _params(params)
    stack = stack if isinstance(stack, Stack) else build_stack(stack)
    X, Y, R2, W, peaks = [], [], [], [], []
    keys, n_img, layout, s = None, 0, None, scale
    for k, (gt, lr) in enumerate(pairs):
        d = _prepare(gt, lr, scale, tile, tile_sr, overlap, None, "lr")
        s, layout = d["s"], d["layout"]
        f = feature_pass(stack, d["sr01"], d["up01"], d["rects"], s, d["I"], d["J"], d["near"], d["far"], p)
        if keys is None:
            keys = f["keys"]
        elif f["keys"] != keys:
            raise ValueError("the encoder stack returned different layers on two images")
        X.append(f["X"]), Y.append(f["Y"]), R2.append(f["layer_r2"]), W.append(f["edge_w"])
        peaks.append(aliasing_peaks(wcsi.srgb_to_lab(d["sr01"], 1.0), d["rects"], s))
        n_img += 1
        if progress:
            progress(k)
    X, Y, R2, W, peaks = (np.concatenate(a) for a in (X, Y, R2, W, peaks))
    layers = {}
    for c, key in enumerate(keys):
        A = np.stack([np.ones(len(X)), X[:, c]], 1)
        a, b = np.linalg.lstsq(A, Y[:, c], rcond=None)[0]
        res = Y[:, c] - a - b * X[:, c]
        nu = float(np.sum(W * R2[:, c]) / max(float(W.sum()), 1e-12)) if len(W) else 1.0
        layers[layer_name(key)] = dict(a=float(a), b=float(b), sigma2=float(max(np.var(res), 1e-4)),
                                       nu_pair=max(nu, 1e-3), r2=float(1 - np.var(res) / max(np.var(Y[:, c]), 1e-12)))
    cal = dict(kind="with_encoder_noref", version=1, note=note, scale=int(s), encoders=stack.describe(),
               tile_lr=layout.get("tile", [None])[0], overlap_lr=layout.get("overlap"), n_images=n_img,
               n_tiles=int(len(X)), layers=layers, nyq_floor=float(np.percentile(peaks, 95)),
               params={k: p[k] for k in ("k_pca", "k_rho", "neff_frac", "jnd_feat", "tau_sem", "tau_anchor",
                                         "far_top")})
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(cal, indent=1, default=str))
    return cal


# ---------------------------------------------------------------- score

def hybrid_noref(sr, lr, stack=None, scale=None, tile=None, overlap=None, tile_coords=None,
                 tile_space="lr", tile_sr=None, return_details=False, **params):
    p = _params(params)
    stack = stack if isinstance(stack, Stack) else build_stack(stack)
    d = _prepare(sr, lr, scale, tile, tile_sr, overlap, tile_coords, tile_space)
    s, rects, I, J, near, far = d["s"], d["rects"], d["I"], d["J"], d["near"], d["far"]
    notes = d["notes"]
    N = len(rects)
    E = {k: None for k, v in T.TERMS.items() if v["setting"] == "noref"}

    grid = dict(tile_coords=tile_coords, tile_space=tile_space) if tile_coords is not None else \
        dict(tile=d["t_lr"], overlap=overlap)
    w2 = wcsi2.wcsi2_score(d["sr"], d["lr"], scale=s, calib=p["calib"], device=p["device"],
                           return_details=True, **grid)
    t2 = w2["terms"]
    E.update(nr_low=t2["E_low"], nr_seam=t2["E_seam"], nr_detail=t2["E_detail"], nr_consist=t2["E_consist"])
    notes += [f"WCSI-2: {n}" for n in w2["notes"]]

    f = feature_pass(stack, d["sr01"], d["up01"], rects, s, I, J, near, far, p)
    per, keys = f["per"], f["keys"]
    cal = load_feat_calibration(p["feat_calib"], keys, s, notes)
    if cal is None:
        notes.append("feature terms uncalibrated (no feat_calib): nr_feat_detail is not computed and "
                     "nr_feat_consist counts disagreement beyond sampling noise only, so bicubic is its optimum")
    elif cal.get("tile_lr") and d["layout"].get("tile") and cal["tile_lr"] != d["layout"]["tile"][0]:
        notes.append(f"feature calibration fitted at tile {cal['tile_lr']} LR px, scoring at {d['layout']['tile'][0]}")

    nyq_floor = p["nyq_floor"] if p["nyq_floor"] is not None else (cal["nyq_floor"] if cal else math.log(4.0))
    k_peak = aliasing_peaks(wcsi.srgb_to_lab(d["sr01"], 1.0), rects, s)
    peaks = (np.maximum(k_peak - nyq_floor, 0.0) / p["jnd_peak"]) ** 2
    E["nr_nyq"] = float(np.mean(peaks))

    detail_t = np.zeros(N)
    if cal is not None and keys:
        L = [cal["layers"][layer_name(k)] for k in keys]
        a, b = np.array([x["a"] for x in L]), np.array([x["b"] for x in L])
        tol2 = np.array([x["sigma2"] for x in L]) + (2 * math.log(1 + p["jnd_feat"])) ** 2
        eps = p["sat_loge"] * np.tanh((f["Y"] - a - b * f["X"]) / p["sat_loge"])
        z2 = eps ** 2 / tol2
        detail_t = z2.mean(1)
        E["nr_feat_detail"] = max(0.0, float(np.mean(detail_t - 1.0)))

    rho, graph, outliers, frust_t = None, {}, [], np.zeros(N)
    if f["edges"]:
        nu = np.array([cal["layers"][layer_name(k)]["nu_pair"] for k in keys]) if cal else np.ones(len(keys))
        v = np.mean(f["layer_r2"] / nu, 1) - 1.0
        W = f["edge_w"]
        E["nr_feat_consist"] = max(0.0, float(np.sum(W * v) / (W.sum() + p["prior_pairs"])))
        _, frust_t = fs.frustration(f["delta"], f["edges"], p["prior_pairs"])
        rho = fs.coherence(f["delta"], f["edges"], p["harmonic_frac"], p["harmonic_cap"], p["rho_max_n"])
    if f["aff"] is not None:
        aff, w = f["aff"], f["w"]
        Wm = np.eye(N)
        Wm[I, J] = Wm[J, I] = aff
        dt, da = engine.descriptors(per, f["dk"], "test"), engine.descriptors(per, f["dk"], "anchor")
        mean_a = da.mean(0)
        a_, t_ = da - mean_a, dt - mean_a
        drift = 1 - np.sum(a_ * t_, 1) / (np.linalg.norm(a_, axis=1) * np.linalg.norm(t_, axis=1) + 1e-12)
        outliers = _robust_outliers(drift, Wm, p["drift_z"])
        graph = dict(descriptor_layer=list(f["dk"]), n_near=int(near.sum()),
                     n_far_kept=int((w[far] > 0).sum()) if far.any() else 0,
                     mean_affinity_far=float(aff[far].mean()) if far.any() else None)

    v1 = None
    if p["with_v1"]:
        v1 = _legacy_v1(d, tile_coords, tile_space, overlap, f, E, notes)

    scores = T.score_all(E, "noref")
    out = dict(score=scores["N3_hybrid"]["score"], wcsi2=w2["wcsi2"], wcsi=v1, energies=E, configs=scores,
               rho_feature=rho)
    if not return_details:
        return out
    out.update(scale=s, n_tiles=N, encoders=stack.describe(), graph=graph,
               calibration=dict(pixel=w2["calibration"]["source"],
                                feature=None if cal is None else (p["feat_calib"] if not isinstance(p["feat_calib"], dict)
                                                                  else "dict")),
               semantic_drift_outliers=outliers,
               tiles=[dict(index=i, id=d["ids"][i], y0=r[0], x0=r[1], h=r[2], w=r[3],
                           frust=float(frust_t[i]), feat_detail=float(detail_t[i]), nyq=float(peaks[i]),
                           pixel_advice=w2["tiles"][i]["advice"]) for i, r in enumerate(rects)],
               notes=notes)
    return out


def _legacy_v1(d, tile_coords, tile_space, overlap, f, E, notes):
    """WCSI v1 terms (wcsi_*) and E_cyto re-weighted by the feature graph, for the N0/N1 arms."""
    import wcsi_viz
    grid = dict(tile_coords=tile_coords, tile_space=tile_space) if tile_coords is not None else \
        dict(tile=d["t_lr"], overlap=overlap)
    D = wcsi_viz.decompose(d["sr"], d["lr"], scale=d["s"], **grid)
    tv = D["det"]["terms"]
    E.update(wcsi_rg=tv["E_RG"], wcsi_seam=tv["E_GJ"], wcsi_pcp=tv["E_PCP"], wcsi_cyto=tv["E_cyto"],
             wcsi_sheaf=tv["E_sheaf"])
    if f["aff"] is not None and len(D["rects"]) == len(d["rects"]) and D["far"].any():
        fk = np.flatnonzero(D["far"])
        wf = f["aff"][fk]
        E["wcsi_cyto_dino"] = float(np.sum(wf * D["cyto_div"][fk].sum(1)) / (wf.sum() + DEFAULTS["prior_pairs"]))
    elif f["aff"] is not None:
        notes.append("wcsi_cyto_dino: WCSI v1 tiled the image differently, not computed")
    return D["det"]["wcsi"]


# ---------------------------------------------------------------- CLI

def _stack_arg(text):
    return "gabor" if text == "gabor" else json.loads(Path(text).read_text())


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "calibrate":
        ap = argparse.ArgumentParser(description="Fit the feature natural model on ground-truth / LR pairs.")
        ap.add_argument("--gt", required=True)
        ap.add_argument("--lr", required=True)
        ap.add_argument("--scale", type=int, required=True)
        ap.add_argument("--tile", type=int)
        ap.add_argument("--tile-sr", type=int)
        ap.add_argument("--overlap", type=int)
        ap.add_argument("--encoders", default="gabor")
        ap.add_argument("--limit", type=int, default=0)
        ap.add_argument("--note", default="")
        ap.add_argument("--out", required=True)
        args = ap.parse_args(argv[1:])
        pairs = wcsi2._pairs_from_dirs(args.gt, args.lr, args.limit)
        cal = calibrate(pairs, _stack_arg(args.encoders), args.scale, tile=args.tile, tile_sr=args.tile_sr,
                        overlap=args.overlap, out=args.out, note=args.note,
                        progress=lambda k: print(f"  {k + 1}/{len(pairs)}", flush=True))
        print(f"wrote {args.out}: {cal['n_images']} images, {cal['n_tiles']} tiles, {len(cal['layers'])} layers")
        return 0
    ap = argparse.ArgumentParser(description="Hybrid pixel + frozen-feature score without a reference.")
    ap.add_argument("sr")
    ap.add_argument("lr")
    ap.add_argument("--scale", type=int)
    ap.add_argument("--tile", type=int, help="analysis tile in LR pixels")
    ap.add_argument("--tile-sr", type=int, help="analysis tile in full-resolution pixels")
    ap.add_argument("--overlap", type=int)
    ap.add_argument("--tiles-jsonl", help="tile rectangles, as in genet/wcsi.py")
    ap.add_argument("--tile-space", choices=("lr", "sr"), default="lr")
    ap.add_argument("--encoders", default="gabor", help="encoder config JSON, or 'gabor'")
    ap.add_argument("--calib", default="auto", help="WCSI-2 pixel calibration ('auto', JSON, 'none')")
    ap.add_argument("--feat-calib", help="feature calibration JSON from `hybrid_noref.py calibrate`")
    ap.add_argument("--device", help="WCSI-2 kernels: numpy (default), cuda, cuda:k, torch:cpu")
    ap.add_argument("--v1", action="store_true", help="also compute the legacy WCSI v1 terms")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    d = hybrid_noref(args.sr, args.lr, stack=_stack_arg(args.encoders), scale=args.scale, tile=args.tile,
                     tile_sr=args.tile_sr, overlap=args.overlap, tile_coords=args.tiles_jsonl,
                     tile_space=args.tile_space, return_details=True,
                     calib=None if args.calib == "none" else args.calib, feat_calib=args.feat_calib,
                     device=args.device, with_v1=args.v1)
    rho = "n/a" if d["rho_feature"] is None else f"{d['rho_feature']:.3f}"
    print(f"hybrid (N3) {d['score']:.2f}   WCSI-2 {d['wcsi2']:.2f}   rho_feature {rho}   {d['n_tiles']} tiles")
    for k, v in d["energies"].items():
        if v is not None or not k.startswith("wcsi_"):
            print(f"  {k:16s} {'n/a' if v is None else f'{v:9.4f}'}   {T.TERMS[k]['source']}")
    for name, c in d["configs"].items():
        if c["score"] is not None:
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
