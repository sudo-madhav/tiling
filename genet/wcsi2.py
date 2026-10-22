#!/usr/bin/env python3
"""WCSI-2: tile-consistency scores for tiled super-resolution, without and with ground truth.

    wcsi2_score(sr, lr, ...)          no reference: SR against its LR input
    wcsi2_ref_score(sr, gt, lr, ...)  with reference: SR against the ground-truth HR image
    calibrate(pairs, scale, ...)      fit the no-reference natural model on (GT, LR) pairs

    python genet/wcsi2.py noref SR.png LR.png --scale 8 --tile-sr 512
    python genet/wcsi2.py ref SR.png GT.png LR.png --scale 8 --tile-sr 512 --device cuda:0
    python genet/wcsi2.py calibrate --gt GT_DIR --lr LR_DIR --scale 8 --tile-sr 512 --out x8.json

Both scores cut the image into the tile grid of the SR model and describe the band each tile
invents (frequencies above the LR Nyquist limit 1/(2s)) by the same per-tile vector y_i:
log energy per octave and orientation of the luminance residual, the orientation tensor Q,
log chroma energy per octave and the top-octave peak ratio (aliasing, checkerboards).
Each tile is compared with a target t_i for that vector:

  no reference   t_i = a + B x_i, x_i the same vector of the LR tile one scale down
                 (LR coarse-grained by s), (a, B) and the natural spread fitted on
                 ground-truth / LR pairs (`calibrate`)
  reference      t_i = y_i of the ground-truth tile

and the residual e_i = y_i - t_i, in units of its tolerance (just-noticeable difference,
sampling error of both estimates, and without a reference the natural spread), gives:

  E_detail   mean over tiles of |e_i|^2: missing or excess invented detail
  E_consist  affinity-weighted mean over tile pairs of |e_i - e_j|^2: tiles of one material
             that depart differently. A shared departure (uniformly too sharp) is E_detail,
             not E_consist.
  E_low      (no reference) colour of the coarse-grained SR against the LR, low-passed
  E_tone     (reference) the same against the coarse-grained ground truth
  E_colour   (reference) sliced Wasserstein distance of the tiles' CIELAB pixel clouds
  E_struct   (reference) gradient error where the ground truth is phase-locked
  E_seam     seam steps at the tile overlaps (beyond the ground truth's own, with reference)

Without a reference, every term subtracts what natural images score (E_detail and E_consist
count only the excess over one natural spread, E_low and E_seam the calibrated floor), so the
ground-truth image is the optimum in expectation. Score = 100 / (1 + F / sum of weights),
F the weighted sum of the terms: 100 when every term is zero, 50 at one unit on average.
See WCSI2.md for the derivation and for what changed from WCSI (genet/wcsi.py).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wcsi  # noqa: E402
from wcsi2_backend import ENBW, band_layout, get_backend, group_rects  # noqa: E402

CALIB_DIR = Path(__file__).resolve().parent / "calib"

DEFAULTS = dict(
    sigma_ds=0.4,          # D_s Gaussian sigma, in units of s HR px
    sigma_low=1.0,         # low-pass before the coarse colour comparison, LR px
    floor_de=0.75,         # band RMS (dE) below which a band is invisible; floor = floor_de^2
    jnd_de=1.5,            # CIEDE2000
    jnd_loge=0.25,         # log band energy (about 13% contrast)
    jnd_q=2 * math.sin(math.radians(4.0)),  # 4 degree turn of a fully oriented texture
    jnd_peak=0.5,          # log peak ratio
    jnd_struct=0.05,       # normalised gradient error (about a 0.45 px edge shift at sigma 1 px)
    jnd_seam=0.25,         # log ratio of the jump at a seam line to the jump beside it
    sat_loge=4.0,          # log-energy residuals saturate at this many nats (s tanh(e / s))
    sigma_cyto=5.0,        # Lab bandwidth of the material affinity
    tau_band=0.69,         # log band-energy bandwidth of the affinity
    prior_pairs=1.0,       # pseudo-pairs of zero excess in the pair means
    neighbour_weight=0.0,  # extra edge weight for touching tiles, whatever their material
    beta_grad=1.0,         # gradient weight in the seam step
    n_proj=16,             # sliced Wasserstein directions (reference)
    colour_samples=4096,   # pixels per tile in the sliced Wasserstein (reference)
    struct_margin=16,      # context around a tile for rigidity and gradients, HR px
    weights=None,          # {term: weight}; default 1 for every term
)

TERMS_NOREF = ("E_low", "E_seam", "E_detail", "E_consist")
TERMS_REF = ("E_tone", "E_colour", "E_struct", "E_detail", "E_consist", "E_seam")
PHENOTYPES = ("faithful", "consistent departure", "inconsistent departure", "faithful among inconsistent peers")


# ---------------------------------------------------------------- layout

def coord_names(s):
    n_oct = band_layout(16, 16, s).n_oct
    names = [f"lum_o{o}_{a}" for o in range(n_oct) for a in (0, 45, 90, 135)]
    return names + ["Q_xx", "Q_xy"] + [f"chroma_o{o}" for o in range(n_oct)] + ["peak"]


def _q_slice(s):
    n = 4 * band_layout(16, 16, s).n_oct
    return slice(n, n + 2)


def saturate(eps, s, p):
    """Residuals in log energy saturate smoothly (s tanh(e / s)): a tile at 2% of its target
    detail and one at 5% are both 'no detail', not 1 nat apart. Q is left linear."""
    out = p["sat_loge"] * np.tanh(eps / p["sat_loge"])
    qs = _q_slice(s)
    out[..., qs] = eps[..., qs]
    return out


def _rects(H0, W0, s, tile, overlap, tile_coords, tile_space):
    notes = []
    if tile_coords is not None:
        if tile is not None or overlap is not None:
            raise ValueError("pass either tile/overlap or tile_coords, not both")
        rects, ids, paths, cnotes = wcsi.read_tile_coords(tile_coords, s, tile_space)
        notes += cnotes
        for k, (y0, x0, h, w) in enumerate(rects):
            if y0 < 0 or x0 < 0 or y0 + h > H0 or x0 + w > W0 or min(h, w) < 8:
                raise ValueError(f"tile {ids[k]}: LR rect x={x0} y={y0} w={w} h={h} must lie inside "
                                 f"the LR image {W0}x{H0} and be at least 8 px on each side")
        return rects, ids, paths, dict(source="coords"), notes
    if tile is None:
        tile = max(8, min(1024 // s, min(H0, W0) // 3))
        notes.append(f"nominal grid (tile {tile} LR px); pass tile/overlap to match the SR model")
    if overlap is None:
        overlap = tile // 16
    th, tw = min(tile, H0), min(tile, W0)
    if not 0 <= overlap < min(th, tw):
        raise ValueError(f"overlap {overlap} must be in [0, {min(th, tw)})")
    rects = [(y0, x0, th, tw) for y0 in wcsi.tile_starts(H0, th, overlap) for x0 in wcsi.tile_starts(W0, tw, overlap)]
    return rects, list(range(len(rects))), None, dict(source="grid", tile=[th, tw], overlap=overlap), notes


def _tile_arg(s, tile, tile_sr):
    if tile_sr is None:
        return tile
    if tile is not None:
        raise ValueError("pass tile (LR px) or tile_sr (HR px), not both")
    if tile_sr % s:
        raise ValueError(f"tile_sr {tile_sr} is not divisible by the scale {s}")
    return tile_sr // s


# ---------------------------------------------------------------- per-tile statistics

def texture_table(be, R, rects, px, s, p):
    """Per-tile vector y (n, d), its sampling standard error (n, d) and visibility (n,).

    R is a residual image (device array), rects are LR rectangles, px the pixels per LR pixel
    of R (s for an HR image, 1 for the LR) and s the band's scale.
    """
    N = len(rects)
    names = coord_names(s)
    y, se, vis = np.zeros((N, len(names))), np.zeros((N, len(names))), np.zeros(N)
    floor = p["floor_de"] ** 2
    for (h, w), idx in group_rects(rects).items():
        lay = band_layout(px * h, px * w, s)
        nb = lay.n_lum
        for b0 in range(0, len(idx), be.batch):
            part = idx[b0:b0 + be.batch]
            sp = be.spectra(be.crop(R, rects, px, part), lay)
            El, J = sp["lum"][:, :nb], sp["lum"][:, nb:nb + 3]
            Ec = sp["chroma"]
            k_l = El ** 2 / np.maximum(sp["lum_sq"], 1e-30) / 2 / ENBW ** 2
            k_c = Ec ** 2 / np.maximum(sp["chroma_sq"], 1e-30) / 2 / ENBW ** 2
            jxx, jxy, jyy = J[:, 0], J[:, 1], J[:, 2]
            tr = jxx + jyy
            hyp = np.hypot(jxx - jyy, 2 * jxy)
            fac = np.where(tr > 1e-20, hyp / np.maximum(tr, 1e-20) ** 2, 0.0)
            Q = -fac[:, None] * np.stack([jxx - jyy, 2 * jxy], 1)
            y[part] = np.concatenate([np.log(El + floor), Q, np.log(Ec + floor),
                                      np.log(np.maximum(sp["peak"], 1.0))[:, None]], 1)
            se[part] = np.concatenate([El / (El + floor) / np.sqrt(np.maximum(k_l, 1.0)), np.zeros((len(part), 2)),
                                       Ec / (Ec + floor) / np.sqrt(np.maximum(k_c, 1.0)),
                                       np.zeros((len(part), 1))], 1)
            vis[part] = np.minimum(1.0, El.sum(1) / p["jnd_de"] ** 2)
    return y, se, vis


def tile_moments(be, lab, rects, px):
    N = len(rects)
    mu, cov = np.zeros((N, 3)), np.zeros((N, 3, 3))
    for _, idx in group_rects(rects).items():
        for b0 in range(0, len(idx), be.batch):
            part = idx[b0:b0 + be.batch]
            mu[part], cov[part] = be.moments(be.crop(lab, rects, px, part))
    return mu, cov


def jnd_vector(s, p):
    n_oct = band_layout(16, 16, s).n_oct
    return np.array([p["jnd_loge"]] * (4 * n_oct) + [p["jnd_q"]] * 2 + [p["jnd_loge"]] * n_oct + [p["jnd_peak"]])


def affinity(mu, cov, loge, I, J, p):
    """Material affinity of tile pairs: Bures distance of the colour Gaussians and the band energies."""
    w = np.empty(len(I))
    for k0 in range(0, len(I), wcsi._CHUNK):
        a, b = I[k0:k0 + wcsi._CHUNK], J[k0:k0 + wcsi._CHUNK]
        w[k0:k0 + wcsi._CHUNK] = np.exp(-wcsi.bures_w2(mu[a], cov[a], mu[b], cov[b]) / (2 * p["sigma_cyto"] ** 2)
                                        - np.mean((loge[a] - loge[b]) ** 2, 1) / (2 * p["tau_band"] ** 2))
    return w


def low_band_de2(M, target, rects, sigma):
    """Per-tile mean CIEDE2000^2 between low-passed M and low-passed target (both LR size)."""
    a = ndimage.gaussian_filter(M, (sigma, sigma, 0), mode="reflect")
    b = ndimage.gaussian_filter(target, (sigma, sigma, 0), mode="reflect")
    de2 = wcsi.ciede2000(a, b) ** 2
    out = np.array([float(de2[y0:y0 + h, x0:x0 + w].mean()) for y0, x0, h, w in rects])
    shift = np.array([(a[y0:y0 + h, x0:x0 + w] - b[y0:y0 + h, x0:x0 + w]).reshape(-1, 3).mean(0)
                      for y0, x0, h, w in rects])
    return out, shift


def _load(be, src):
    rgb, white = wcsi.load_rgb(src)
    return rgb, white


# ---------------------------------------------------------------- calibration model

def find_calibration(s, tile_lr=None, directory=CALIB_DIR):
    """The calibration file for scale s whose tile (HR px) is closest to s * tile_lr, or None."""
    found = []
    for f in Path(directory).glob(f"wcsi2_x{s}_t*.json"):
        try:
            found.append((int(f.stem.rsplit("_t", 1)[1]), f))
        except ValueError:
            continue
    if not found:
        return None
    if not tile_lr:
        return max(found)[1]
    return min(found, key=lambda tf: (abs(math.log(tf[0] / (s * tile_lr))), -tf[0]))[1]


def load_calibration(calib, s, tile_lr=None):
    """The no-reference natural model for scale s: dict with a, B, sigma_nat2, sigma_pair2, floors.

    calib: "auto" (the file in genet/calib for this scale whose tile is closest), a path, a
    dict, or None (scale invariance t = x, nominal spreads, no floors).
    """
    names = coord_names(s)
    d = len(names)
    src, cal = None, None
    if isinstance(calib, dict):
        cal, src = calib, "dict"
    elif calib == "auto":
        path = find_calibration(s, tile_lr)
        if path is not None:
            cal, src = json.loads(path.read_text()), str(path)
    elif isinstance(calib, (str, Path)):
        if not Path(calib).is_file():
            raise FileNotFoundError(calib)
        cal, src = json.loads(Path(calib).read_text()), str(calib)
    elif calib is not None:
        raise TypeError(f"calib must be 'auto', a path, a dict or None, got {type(calib).__name__}")
    if cal is None:
        sig2 = np.full(d, 0.25)
        sig2[_q_slice(s)] = 0.09
        return dict(a=np.zeros(d), B=np.eye(d), sigma_nat2=sig2, sigma_pair2=sig2.copy(),
                    low_floor=0.0, seam_floor=0.0, source=None, tile_lr=None, scale=s)
    if cal.get("coords") != names:
        raise ValueError(f"calibration {src} has coordinates {cal.get('coords')}, expected {names} for scale {s}")
    return dict(a=np.asarray(cal["a"], float), B=np.asarray(cal["B"], float),
                sigma_nat2=np.asarray(cal["sigma_nat2"], float), sigma_pair2=np.asarray(cal["sigma_pair2"], float),
                low_floor=float(cal.get("low_floor", 0.0)), seam_floor=float(cal.get("seam_floor", 0.0)),
                source=src, tile_lr=cal.get("tile_lr"), scale=s)


# ---------------------------------------------------------------- shared assembly

def _residual(y, target, v, s, p):
    """Saturated residual per tile; orientation weighted by the visibility v of both textures."""
    eps = saturate(y - target, s, p)
    eps[:, _q_slice(s)] *= v[:, None]
    return eps


def _pair_residual(eps, y, target, v, I, J, s):
    """Difference of the residuals of pairs; orientation weighted by the smaller visibility."""
    qs = _q_slice(s)
    de = eps[I] - eps[J]
    de[:, qs] = (y[I, qs] - target[I, qs] - y[J, qs] + target[J, qs]) * np.minimum(v[I], v[J])[:, None]
    return de


def _pair_means(w, zp2, prior, excess):
    """Affinity-weighted mean of per-pair energies; with `excess`, of the excess over one."""
    x = zp2 - 1.0 if excess else zp2
    return float(np.sum(w * x)) / (float(np.sum(w)) + prior)


def _per_tile_pairs(N, I, J, w, zp2):
    num, den = np.zeros(N), np.zeros(N)
    np.add.at(num, I, w * zp2)
    np.add.at(num, J, w * zp2)
    np.add.at(den, I, w)
    np.add.at(den, J, w)
    return np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)


def _score(values, weights):
    weights = weights or {}
    used = [(weights.get(k, 1.0), v) for k, v in values.items() if v is not None]
    F = sum(wt * v for wt, v in used)
    norm = sum(wt for wt, _ in used)
    return (100.0 / (1.0 + F / norm) if norm > 0 else 100.0), F, norm


_ORIENT = {0: "vertical strokes", 45: "45 degree strokes", 90: "horizontal strokes", 135: "135 degree strokes"}


def _advice(z, names, n_oct, low=None, shift=None):
    """One instruction from the largest signed residual (in tolerance units)."""
    k = int(np.argmax(np.abs(z)))
    cand = abs(z[k])
    if low is not None and math.sqrt(max(low, 0.0)) > cand and low >= 1:
        dL, da, db = shift
        parts = [f"lightness {dL:+.1f}"] if abs(dL) >= 0.5 else []
        parts += [f"a* {da:+.1f}"] if abs(da) >= 0.5 else []
        parts += [f"b* {db:+.1f}"] if abs(db) >= 0.5 else []
        return "coarse colour drifts from the target (" + (", ".join(parts) or "small shift, large spread") + ")"
    if cand < 1:
        return "within tolerance on every coordinate"
    name, v = names[k], z[k]
    if name.startswith("lum_o"):
        o, ang = int(name[5]), int(name.split("_")[2])
        where = "coarsest" if o == 0 else ("finest" if o == n_oct - 1 else "middle")
        return (f"{'too little' if v < 0 else 'too much'} invented detail in the {where} octave "
                f"({_ORIENT[ang]}), {abs(v):.1f} tolerances")
    if name.startswith("Q_"):
        return f"orientation of the invented texture is off ({abs(v):.1f} tolerances)"
    if name.startswith("chroma"):
        return "chroma detail missing" if v < 0 else f"false colour / chroma noise ({v:.1f} tolerances)"
    return f"periodic artefact in the finest octave (ringing, checkerboard, aliasing), {v:.1f} tolerances"


# ---------------------------------------------------------------- no reference

def wcsi2_score(sr, lr, scale=None, tile=None, overlap=None, tile_sr=None, tiles=None, tile_coords=None,
                tile_space="lr", calib="auto", device=None, return_details=False, **params):
    """No-reference WCSI-2 of `sr` against its LR input. See the module docstring.

    tile, overlap   the SR model's grid in LR px (tile_sr: the tile in HR px)
    tile_coords     a JSONL path or list of tile dicts instead of the grid (see wcsi.read_tile_coords)
    tiles           optional unblended tile outputs: the seam term then compares the overlaps
    calib           natural model ("auto", path, dict, None) - see load_calibration
    device          None (numpy), "cuda[:k]" or "torch:cpu"
    """
    p = _params(params)
    be = get_backend(device)
    sr_rgb, sr_white = _load(be, sr)
    lr_rgb, lr_white = _load(be, lr)
    s, sr_rgb, lr_rgb, notes = wcsi._match_scale(sr_rgb, lr_rgb, scale)
    H0, W0 = lr_rgb.shape[:2]
    rects, ids, paths, layout, rnotes = _rects(H0, W0, s, _tile_arg(s, tile, tile_sr), overlap, tile_coords, tile_space)
    notes += rnotes
    if tiles is None and paths is not None:
        tiles = paths
    N = len(rects)
    pos = wcsi._grid_positions(rects)
    I, J, near, far, _ = wcsi._tile_graph(rects, pos)
    layout.update(n_tiles=N, rows=1 + max(q[0] for q in pos) if pos else None,
                  cols=1 + max(q[1] for q in pos) if pos else None)

    sr_lab = be.lab(sr_rgb, sr_white)
    lr_lab = be.lab(lr_rgb, lr_white)
    del sr_rgb, lr_rgb
    sigma = p["sigma_ds"] * s
    M, R = be.split(sr_lab, s, sigma)
    _, R0 = be.split(lr_lab, s, sigma)
    y, se, vis = texture_table(be, R, rects, s, s, p)
    x0, se0, vis0 = texture_table(be, R0, rects, 1, s, p)
    del R, R0
    mu0, cov0 = tile_moments(be, lr_lab, rects, 1)
    lr_np = be.numpy(lr_lab)
    low, shift = low_band_de2(M, lr_np, rects, p["sigma_low"])

    sr_np = be.numpy(sr_lab)
    del sr_lab
    pairs = list(zip(I[near].tolist(), J[near].tolist()))
    seam = _seams_noref(sr_np, lr_np, rects, pairs, s, p, tiles, notes)

    cal = load_calibration(calib, s, rects[0][2])
    if cal["source"] is None:
        notes.append(f"uncalibrated: no natural model for scale {s} (run `wcsi2.py calibrate`); "
                     "using scale invariance t = x and nominal spreads")
    elif cal["tile_lr"] and layout.get("tile") and cal["tile_lr"] != layout["tile"][0]:
        notes.append(f"calibration fitted at tile {cal['tile_lr']} LR px, scoring at {layout['tile'][0]}; "
                     "sampling errors are rescaled, natural spreads are not")
    names = coord_names(s)
    jnd = jnd_vector(s, p)
    B2 = cal["B"] ** 2
    pred = cal["a"] + x0 @ cal["B"].T
    v = np.minimum(vis, vis0)
    eps = _residual(y, pred, v, s, p)
    tol2 = cal["sigma_nat2"] + jnd ** 2 + se ** 2 + se0 ** 2 @ B2.T
    z = eps / np.sqrt(tol2)
    detail_tile = np.mean(z ** 2, 1)
    e_detail = max(0.0, float(np.mean(detail_tile - 1.0)))

    e_consist, consist_tile, w = None, np.zeros(N), np.zeros(len(I))
    if len(I):
        w = affinity(mu0, cov0, x0[:, :4 * band_layout(16, 16, s).n_oct], I, J, p) + p["neighbour_weight"] * near
        de = _pair_residual(eps, y, pred, v, I, J, s)
        tolp2 = cal["sigma_pair2"] + jnd ** 2 + se[I] ** 2 + se[J] ** 2 + (se0[I] ** 2 + se0[J] ** 2) @ B2.T
        zp2 = np.mean(de ** 2 / tolp2, 1)
        e_consist = max(0.0, _pair_means(w, zp2, p["prior_pairs"], excess=True))
        consist_tile = _per_tile_pairs(N, I, J, w, zp2)

    e_low = max(0.0, float(np.mean(low)) / p["jnd_de"] ** 2 - cal["low_floor"])
    e_seam = max(0.0, float(np.mean(seam)) - cal["seam_floor"]) if len(seam) else None
    values = dict(E_low=e_low, E_seam=e_seam, E_detail=e_detail, E_consist=e_consist)
    score, F, norm = _score(values, p["weights"])
    for k, val in values.items():
        if val is None:
            notes.append(f"{k}: undefined for this layout ({N} tiles), left out")
    if not return_details:
        return score
    n_oct = band_layout(16, 16, s).n_oct
    return dict(
        wcsi2=score, F_total=F, F_norm=norm, terms=values, scale=s, layout=layout,
        parts=dict(E_low_raw=float(np.mean(low)) / p["jnd_de"] ** 2, E_seam_raw=float(np.mean(seam)) if len(seam) else None,
                   E_detail_raw=float(np.mean(detail_tile)),
                   E_consist_raw=(float(np.sum(w * zp2) / (np.sum(w) + p["prior_pairs"])) if len(I) else None),
                   affinity_sum=float(w.sum()), detail_mean_z=dict(zip(names, np.mean(z, 0).tolist()))),
        calibration=dict(source=cal["source"], tile_lr=cal["tile_lr"]),
        coords=names,
        tiles=[dict(id=ids[i], y0=rects[i][0], x0=rects[i][1], h=rects[i][2], w=rects[i][3],
                    row=pos[i][0] if pos else None, col=pos[i][1] if pos else None,
                    detail=float(detail_tile[i]), consist=float(consist_tile[i]),
                    low=float(low[i]) / p["jnd_de"] ** 2, z=z[i].tolist(), visibility=float(vis[i]),
                    advice=_advice(z[i], names, n_oct, float(low[i]) / p["jnd_de"] ** 2, shift[i]))
               for i in range(N)],
        notes=notes,
    )


def _params(params):
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown parameters: {sorted(unknown)}")
    return {**DEFAULTS, **params}


def _line_stat(img, c, r0, r1, beta, offsets, eps):
    """Mean over rows of log((jump at line c + eps) / (mean jump at the control lines + eps))."""
    jump = wcsi._jump(img, c, r0, r1, beta)
    ctrl = np.mean([wcsi._jump(img, c + o, r0, r1, beta) for o in offsets + tuple(-o for o in offsets)], 0)
    return float(np.mean(np.log((jump + eps) / (ctrl + eps))))


def seam_stats(img, ref, rects, pairs, s, p, offsets=(3, 6), ref_offsets=(2, 3)):
    """Per neighbour pair: how much more the image steps at the seam lines than beside them.

    At the start, middle and end of each overlap band, the per-row colour + gradient jump
    across the line is compared with the mean jump at control lines 3 and 6 px away (log ratio,
    averaged over rows), so a misregistration that shows only in textured rows still counts.
    `ref` is an LR-size image (the LR, or the coarse-grained ground truth) whose own step at the
    same line is subtracted: an edge along the line is image content. The pair's value is the
    largest candidate line. None where the band is too close to the border.
    """
    eps = p["jnd_de"] ** 2
    beta = p["beta_grad"]
    reach, reach0 = max(offsets) + 3, max(ref_offsets) + 3
    imT, refT = img.transpose(1, 0, 2), ref.transpose(1, 0, 2)

    def one(im, im0, a, b, r0, r1):
        best = None
        for c in sorted({a, (a + b) // 2, b}):
            if c - reach < 0 or c + reach > im.shape[1] or r1 - r0 < 4:
                continue
            v = _line_stat(im, c, r0, r1, beta, offsets, eps)
            c0 = int(round(c / s))
            if reach0 <= c0 <= im0.shape[1] - reach0:
                v -= max(0.0, _line_stat(im0, c0, r0 // s, max(r1 // s, r0 // s + 4), beta, ref_offsets, eps))
            best = v if best is None else max(best, v)
        return best

    out = []
    for i, j in pairs:
        y0, y1, x0, x1 = wcsi._shared(rects[i], rects[j])
        ri, rj = rects[i], rects[j]
        if abs(ri[1] - rj[1]) / min(ri[3], rj[3]) >= abs(ri[0] - rj[0]) / min(ri[2], rj[2]):
            out.append(one(img, ref, s * x0, s * x1, s * y0, s * y1))
        else:
            out.append(one(imT, refT, s * y0, s * y1, s * x0, s * x1))
    return out


def _seam_energy(vals, p):
    v = np.array([0.0 if x is None else x for x in vals], float)
    return (np.maximum(v, 0.0) / p["jnd_seam"]) ** 2


def _seams_noref(sr_np, lr_np, rects, pairs, s, p, tiles, notes):
    if not pairs:
        return np.zeros(0)
    jnd2 = p["jnd_de"] ** 2
    if tiles is not None:
        tiles = list(tiles)
        if len(tiles) != len(rects):
            raise ValueError(f"got {len(tiles)} tile outputs for {len(rects)} tiles")
        canv = []
        for t, (_, _, h, w) in zip(tiles, rects):
            a, wh = wcsi.load_rgb(t)
            if a.shape[:2] != (s * h, s * w):
                raise ValueError(f"tile output {a.shape[1]}x{a.shape[0]}, expected {s * w}x{s * h}")
            canv.append(wcsi.srgb_to_lab(a, wh))
        notes.append("E_seam: overlap comparison of the tile outputs")
        return np.array([v / jnd2 for v in wcsi._gj_canvases(canv, rects, pairs, s, p["beta_grad"])])
    return _seam_energy(seam_stats(sr_np, lr_np, rects, pairs, s, p), p)


# ---------------------------------------------------------------- with reference

def _match_ref(sr, gt, lr, scale):
    sr, sr_w = wcsi.load_rgb(sr)
    gt, gt_w = wcsi.load_rgb(gt)
    notes = []
    H, W = min(sr.shape[0], gt.shape[0]), min(sr.shape[1], gt.shape[1])
    if (H, W) != tuple(sr.shape[:2]) or (H, W) != tuple(gt.shape[:2]):
        notes.append(f"cropped SR and ground truth to the common {W}x{H}")
    if lr is not None:
        lr, lr_w = wcsi.load_rgb(lr)
        s, gt_c, lr, n2 = wcsi._match_scale(gt[:H, :W], lr, scale)
        notes += n2
    else:
        if not scale:
            raise ValueError("without an LR image, pass scale")
        s, lr_w = int(scale), None
        gt_c = gt[:H // s * s, :W // s * s]
    H, W = gt_c.shape[:2]
    return s, sr[:H, :W], sr_w, gt_c, gt_w, lr, lr_w, notes


def wcsi2_ref_score(sr, gt, lr=None, scale=None, tile=None, overlap=None, tile_sr=None, tile_coords=None,
                    tile_space="lr", device=None, return_details=False, **params):
    """WCSI-2 with reference: `sr` against the ground truth `gt` (LR optional, it sets the grid).

    The same per-tile vector and tile graph as the no-reference score, with the ground-truth
    tile as target, plus three terms only a reference allows (tone, colour distribution,
    structure on phase-locked pixels). Material affinity comes from the ground truth.
    """
    p = _params(params)
    be = get_backend(device)
    s, sr_rgb, sr_white, gt_rgb, gt_white, _, _, notes = _match_ref(sr, gt, lr, scale)
    H0, W0 = gt_rgb.shape[0] // s, gt_rgb.shape[1] // s
    rects, ids, _, layout, rnotes = _rects(H0, W0, s, _tile_arg(s, tile, tile_sr), overlap, tile_coords, tile_space)
    notes += rnotes
    N = len(rects)
    pos = wcsi._grid_positions(rects)
    I, J, near, far, _ = wcsi._tile_graph(rects, pos)
    layout.update(n_tiles=N, rows=1 + max(q[0] for q in pos) if pos else None,
                  cols=1 + max(q[1] for q in pos) if pos else None)

    sr_lab = be.lab(sr_rgb, sr_white)
    gt_lab = be.lab(gt_rgb, gt_white)
    del sr_rgb, gt_rgb
    sigma = p["sigma_ds"] * s
    Ms, Rs = be.split(sr_lab, s, sigma)
    Mg, Rg = be.split(gt_lab, s, sigma)
    y, se, vis = texture_table(be, Rs, rects, s, s, p)
    ys, ses, viss = texture_table(be, Rg, rects, s, s, p)
    del Rs, Rg
    mu, cov = tile_moments(be, gt_lab, rects, s)
    tone, shift = low_band_de2(Ms, Mg, rects, p["sigma_low"])
    struct = _struct(be, sr_lab, gt_lab, rects, s, p)
    sr_np, gt_np = be.numpy(sr_lab), be.numpy(gt_lab)
    del sr_lab, gt_lab
    colour = _sliced_colour(sr_np, gt_np, rects, s, p)
    pairs = list(zip(I[near].tolist(), J[near].tolist()))
    seam = np.zeros(0)
    if pairs:
        a = seam_stats(sr_np, Mg, rects, pairs, s, p)
        b = seam_stats(gt_np, Mg, rects, pairs, s, p)
        seam = _seam_energy([None if u is None else u - v for u, v in zip(a, b)], p)

    names = coord_names(s)
    jnd = jnd_vector(s, p)
    v = np.minimum(vis, viss)
    eps = _residual(y, ys, v, s, p)
    tol2 = jnd ** 2 + se ** 2 + ses ** 2
    z = eps / np.sqrt(tol2)
    detail_tile = np.mean(z ** 2, 1)

    e_consist, consist_tile, rho, w = None, np.zeros(N), None, np.zeros(len(I))
    if len(I):
        w = affinity(mu, cov, ys[:, :4 * band_layout(16, 16, s).n_oct], I, J, p) + p["neighbour_weight"] * near
        de = _pair_residual(eps, y, ys, v, I, J, s)
        tolp2 = jnd ** 2 + se[I] ** 2 + se[J] ** 2 + ses[I] ** 2 + ses[J] ** 2
        zp2 = np.mean(de ** 2 / tolp2, 1)
        e_consist = _pair_means(w, zp2, p["prior_pairs"], excess=False)
        consist_tile = _per_tile_pairs(N, I, J, w, zp2)
        # coherence: 1 for one shared departure, 1/2 for independent ones, 0 for opposite ones
        u2 = np.sum(z ** 2, 1)
        den = 2.0 * float(np.sum(w * (u2[I] + u2[J])))
        if den > 1e-9 * max(float(w.sum()), 1e-12) * z.shape[1]:
            du2 = np.sum((z[I] - z[J]) ** 2, 1)
            rho = float(np.clip(1.0 - np.sum(w * du2) / den, 0.0, 1.0))

    jnd2 = p["jnd_de"] ** 2
    values = dict(E_tone=float(np.mean(tone)) / jnd2, E_colour=float(np.mean(colour)) / jnd2,
                  E_struct=float(np.mean(struct)) / p["jnd_struct"], E_detail=float(np.mean(detail_tile)),
                  E_consist=e_consist, E_seam=float(np.mean(seam)) if len(seam) else None)
    score, F, norm = _score(values, p["weights"])
    for k, val in values.items():
        if val is None:
            notes.append(f"{k}: undefined for this layout ({N} tiles), left out")
    master = detail_tile + tone / jnd2 + colour / jnd2 + struct / p["jnd_struct"]
    labels = [_phenotype(master[i], consist_tile[i]) for i in range(N)]
    if not return_details:
        return score
    n_oct = band_layout(16, 16, s).n_oct
    return dict(
        wcsi2_ref=score, F_total=F, F_norm=norm, terms=values, rho_coherence=rho, scale=s, layout=layout,
        parts=dict(affinity_sum=float(w.sum()), detail_mean_z=dict(zip(names, np.mean(z, 0).tolist())),
                   tile_master_mean=float(master.mean()), tile_master_std=float(master.std()),
                   phenotype_counts={k: labels.count(k) for k in PHENOTYPES}),
        coords=names,
        tiles=[dict(id=ids[i], y0=rects[i][0], x0=rects[i][1], h=rects[i][2], w=rects[i][3],
                    row=pos[i][0] if pos else None, col=pos[i][1] if pos else None,
                    master=float(master[i]), detail=float(detail_tile[i]), consist=float(consist_tile[i]),
                    tone=float(tone[i]) / jnd2, colour=float(colour[i]) / jnd2,
                    struct=float(struct[i]) / p["jnd_struct"], z=z[i].tolist(), phenotype=labels[i],
                    advice=_advice(z[i], names, n_oct, float(tone[i]) / jnd2, shift[i]))
               for i in range(N)],
        notes=notes,
    )


def _phenotype(master, consist):
    if master < 1:
        return PHENOTYPES[0] if consist < 1 else PHENOTYPES[3]
    return PHENOTYPES[1] if consist < 1 else PHENOTYPES[2]


def _struct(be, sr_lab, gt_lab, rects, s, p):
    """Per tile: sum gamma |grad SR - grad GT|^2 / sum gamma (|grad SR|^2 + |grad GT|^2 + c), gamma the
    phase congruency of the ground truth, both computed with `struct_margin` px of context."""
    m = p["struct_margin"]
    c = (p["jnd_de"] / 2) ** 2
    out = np.zeros(len(rects))
    sl, gl = be.pad(sr_lab[..., 0], m), be.pad(gt_lab[..., 0], m)
    for _, idx in group_rects(rects).items():
        step = max(1, be.batch // 2)
        for b0 in range(0, len(idx), step):
            part = idx[b0:b0 + step]
            ts, tg = be.crop(sl, rects, s, part, m), be.crop(gl, rects, s, part, m)
            gamma = be.rigidity(tg)
            sx, sy = be.gradients(ts)
            gx, gy = be.gradients(tg)
            inner = (slice(None), slice(m, -m if m else None), slice(m, -m if m else None))
            g = gamma[inner]
            num = (g * ((sx - gx) ** 2 + (sy - gy) ** 2)[inner]).reshape(len(part), -1).sum(1)
            den = (g * (sx ** 2 + sy ** 2 + gx ** 2 + gy ** 2 + c)[inner]).reshape(len(part), -1).sum(1)
            num, den = be.numpy(num).astype(np.float64), be.numpy(den).astype(np.float64)
            out[part] = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
    return out


def _sliced_colour(sr_np, gt_np, rects, s, p):
    """Per tile: sliced Wasserstein-2^2 between the SR and ground-truth CIELAB pixel clouds."""
    rng = np.random.default_rng(0)
    U = rng.normal(size=(3, p["n_proj"]))
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    out = np.zeros(len(rects))
    for k, (y0, x0, h, w) in enumerate(rects):
        st = max(1, int(math.sqrt(s * h * s * w / p["colour_samples"])))
        a = sr_np[s * y0:s * (y0 + h):st, s * x0:s * (x0 + w):st].reshape(-1, 3).astype(np.float64) @ U
        b = gt_np[s * y0:s * (y0 + h):st, s * x0:s * (x0 + w):st].reshape(-1, 3).astype(np.float64) @ U
        out[k] = float(np.mean((np.sort(a, 0) - np.sort(b, 0)) ** 2))
    return out


# ---------------------------------------------------------------- calibration

def calibrate(pairs, scale, tile=None, overlap=None, tile_sr=None, device=None, ridge=1e-3, out=None,
              progress=None, note="", **params):
    """Fit the no-reference natural model on ground-truth / LR pairs (GT plays the SR).

    pairs: iterable of (gt, lr) paths or arrays, degraded the way the evaluated LR inputs were.
    Fits t = a + B x by ridge regression over every tile, the natural residual spread per
    coordinate (minus the sampling error), the spread of residual differences over material
    pairs, and the ground truth's own E_low and E_seam (subtracted later as floors).
    """
    p = _params(params)
    be = get_backend(device)
    names = coord_names(scale)
    qs = _q_slice(scale)
    X, Y, SE, SE0, V, imgs, lows, seams = [], [], [], [], [], [], [], []
    n_img = offset = 0
    for k, (gt, lr) in enumerate(pairs):
        gt_rgb, gw = wcsi.load_rgb(gt)
        lr_rgb, lw = wcsi.load_rgb(lr)
        s, gt_rgb, lr_rgb, _ = wcsi._match_scale(gt_rgb, lr_rgb, scale)
        H0, W0 = lr_rgb.shape[:2]
        rects, _, _, layout, _ = _rects(H0, W0, s, _tile_arg(s, tile, tile_sr), overlap, None, "lr")
        I, J, near, _, _ = wcsi._tile_graph(rects, wcsi._grid_positions(rects))
        g_lab, l_lab = be.lab(gt_rgb, gw), be.lab(lr_rgb, lw)
        sigma = p["sigma_ds"] * s
        M, R = be.split(g_lab, s, sigma)
        _, R0 = be.split(l_lab, s, sigma)
        y, se, vis = texture_table(be, R, rects, s, s, p)
        x0, se0, vis0 = texture_table(be, R0, rects, 1, s, p)
        mu0, cov0 = tile_moments(be, l_lab, rects, 1)
        l_np = be.numpy(l_lab)
        low, _ = low_band_de2(M, l_np, rects, p["sigma_low"])
        pr = list(zip(I[near].tolist(), J[near].tolist()))
        seam = _seam_energy(seam_stats(be.numpy(g_lab), l_np, rects, pr, s, p), p) if pr else []
        w = affinity(mu0, cov0, x0[:, :qs.start], I, J, p) + p["neighbour_weight"] * near if len(I) else np.zeros(0)
        X.append(x0), Y.append(y), SE.append(se), SE0.append(se0), V.append(np.minimum(vis, vis0))
        imgs.append((offset, I, J, w))
        offset += len(rects)
        lows.append(float(np.mean(low)) / p["jnd_de"] ** 2)
        seams.append(float(np.mean(seam)) if len(seam) else 0.0)
        n_img += 1
        if progress:
            progress(k, layout)
    X, Y, SE, SE0, V = (np.concatenate(a) for a in (X, Y, SE, SE0, V))
    n, d = Y.shape
    xm, xs = X.mean(0), X.std(0) + 1e-9
    Z = np.concatenate([np.ones((n, 1)), (X - xm) / xs], 1)
    a, B = np.zeros(d), np.zeros((d, d))
    for c in range(d):
        wt = V ** 2 if qs.start <= c < qs.stop else np.ones(n)
        A = (Z * wt[:, None]).T @ Z + ridge * n * np.diag([0.0] + [1.0] * d)
        beta = np.linalg.solve(A, (Z * wt[:, None]).T @ Y[:, c])
        B[c] = beta[1:] / xs
        a[c] = beta[0] - B[c] @ xm
    pred = a + X @ B.T
    eps = _residual(Y, pred, V, scale, p)
    noise = SE ** 2 + SE0 ** 2 @ (B ** 2).T
    raw = np.mean(eps ** 2, 0)
    sig_nat2 = np.maximum(raw - noise.mean(0), 0.1 * raw)
    num, den, nse = np.zeros(d), 0.0, np.zeros(d)
    for off, I, J, w in imgs:
        if not len(I):
            continue
        a_, b_ = off + I, off + J
        de = _pair_residual(eps, Y, pred, V, a_, b_, scale)
        num += (w[:, None] * de ** 2).sum(0)
        nse += (w[:, None] * (noise[a_] + noise[b_])).sum(0)
        den += float(w.sum())
    rawp = num / max(den, 1e-12)
    sig_pair2 = np.maximum(rawp - nse / max(den, 1e-12), 0.1 * rawp)
    resid_var = np.var(Y, 0)
    cal = dict(note=note, version=2, scale=int(scale), coords=names, tile_lr=layout.get("tile", [None])[0],
               overlap_lr=layout.get("overlap"), n_images=n_img, n_tiles=int(n),
               a=a.tolist(), B=B.tolist(), sigma_nat2=sig_nat2.tolist(), sigma_pair2=sig_pair2.tolist(),
               low_floor=float(np.mean(lows)), seam_floor=float(np.mean(seams)),
               r2=(1 - raw / np.maximum(resid_var, 1e-12)).tolist(),
               params={k: p[k] for k in ("sigma_ds", "sigma_low", "floor_de", "jnd_de", "jnd_loge", "jnd_q",
                                          "jnd_peak", "sigma_cyto", "tau_band", "neighbour_weight")})
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(cal, indent=1))
    return cal


# ---------------------------------------------------------------- CLI

def _print_terms(d, key, terms):
    g = d["layout"]
    grid = (f"{g['rows']}x{g['cols']} tiles of {g['tile'][1]}x{g['tile'][0]} LR px, overlap {g['overlap']}"
            if g["source"] == "grid" else f"{g['n_tiles']} tiles")
    print(f"{key.upper()} {d[key]:.2f}   scale {d['scale']}x, {grid}")
    for k in terms:
        v = d["terms"][k]
        print(f"  {k:10s} {'n/a' if v is None else f'{v:9.4f}'}")
    if d.get("rho_coherence") is not None:
        print(f"  rho_coherence {d['rho_coherence']:.3f}")
    for n in d["notes"]:
        print(f"  note: {n}")


def _pairs_from_dirs(gt_dir, lr_dir, limit):
    exts = wcsi.IMAGE_EXT
    gt = {f.stem: f for f in sorted(Path(gt_dir).iterdir()) if f.suffix.lower() in exts}
    lr = {f.stem: f for f in sorted(Path(lr_dir).iterdir()) if f.suffix.lower() in exts}
    stems = sorted(set(gt) & set(lr))
    if limit:
        stems = stems[:limit]
    return [(gt[k], lr[k]) for k in stems]


def main(argv=None):
    ap = argparse.ArgumentParser(description="WCSI-2 tile-consistency scores (see WCSI2.md).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def grid(a):
        a.add_argument("--scale", type=int)
        a.add_argument("--tile", type=int, help="tile, LR px")
        a.add_argument("--tile-sr", type=int, help="tile, HR px (overrides --tile)")
        a.add_argument("--overlap", type=int, help="overlap, LR px (default tile/16)")
        a.add_argument("--device", help="numpy (default), cuda, cuda:k, torch:cpu")

    a = sub.add_parser("noref", help="SR against its LR input")
    a.add_argument("sr")
    a.add_argument("lr")
    grid(a)
    a.add_argument("--tiles-jsonl")
    a.add_argument("--tile-space", choices=("lr", "sr"), default="lr")
    a.add_argument("--calib", default="auto", help="calibration JSON, 'auto' or 'none'")
    a.add_argument("--json")
    a.add_argument("--advice", action="store_true")

    a = sub.add_parser("ref", help="SR against the ground truth")
    a.add_argument("sr")
    a.add_argument("gt")
    a.add_argument("lr", nargs="?")
    grid(a)
    a.add_argument("--json")
    a.add_argument("--advice", action="store_true")

    a = sub.add_parser("calibrate", help="fit the no-reference natural model on GT / LR pairs")
    a.add_argument("--gt", required=True)
    a.add_argument("--lr", required=True)
    grid(a)
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--note", default="", help="where the pairs came from and how the LR was made")
    a.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "calibrate":
        if not args.scale:
            ap.error("calibrate needs --scale")
        pairs = _pairs_from_dirs(args.gt, args.lr, args.limit)
        cal = calibrate(pairs, args.scale, tile=args.tile, overlap=args.overlap, tile_sr=args.tile_sr,
                        device=args.device, out=args.out, note=args.note,
                        progress=lambda k, lay: print(f"  {k + 1}/{len(pairs)}", flush=True))
        print(f"wrote {args.out}: {cal['n_images']} images, {cal['n_tiles']} tiles, "
              f"low floor {cal['low_floor']:.3f}, seam floor {cal['seam_floor']:.3f}")
        return 0
    if args.cmd == "noref":
        if args.tiles_jsonl and (args.tile is not None or args.tile_sr is not None or args.overlap is not None):
            ap.error("--tiles-jsonl replaces --tile/--overlap")
        d = wcsi2_score(args.sr, args.lr, scale=args.scale, tile=args.tile, tile_sr=args.tile_sr,
                        overlap=args.overlap, tile_coords=args.tiles_jsonl, tile_space=args.tile_space,
                        calib=None if args.calib == "none" else args.calib, device=args.device,
                        return_details=True)
        _print_terms(d, "wcsi2", TERMS_NOREF)
    else:
        d = wcsi2_ref_score(args.sr, args.gt, args.lr, scale=args.scale, tile=args.tile, tile_sr=args.tile_sr,
                            overlap=args.overlap, device=args.device, return_details=True)
        _print_terms(d, "wcsi2_ref", TERMS_REF)
    if args.advice:
        for t in d["tiles"]:
            name = f"r{t['row']}c{t['col']}" if t["row"] is not None else t["id"]
            print(f"  tile {name}: {t['advice']}")
    if args.json:
        Path(args.json).write_text(json.dumps(d, indent=1))
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
