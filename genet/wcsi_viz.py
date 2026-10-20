#!/usr/bin/env python3
"""Where a WCSI score comes from: per-tile and per-pair contributions, with figures.

    python genet/wcsi_viz.py SR.png LR.png --tile 64 --out viz/
    python genet/wcsi_viz.py SR.png LR.png --tiles-jsonl tiles.jsonl --out viz/ --tile-ids 3 17

Runs wcsi.wcsi_score (unchanged), recomputes the same quantities tile by tile and pair by
pair, checks that they add up to the reported terms, and writes to --out:

  overview.png     score, the weighted terms, and a map of each tile's share of F
  term_maps.png    one map per term: the share of that term owed to each tile
  pairs.png        LR material affinity between tiles, the neighbour graph coloured by
                   cost, distant-pair divergences against affinity, E_sheaf by coordinate
  seams.png        the worst seams: crop around the seam line and the per-row jump
                   at the line against the control lines
  far_pairs_consistent.png, far_pairs_inconsistent.png
                   distant tiles that the LR calls one material (affinity >= --min-affinity):
                   the pairs whose invented textures agree best and worst (E_cyto's
                   statistics before affinity weighting). Per pair: where they are, the SR
                   tiles with the LR inset, both residuals on one colour scale, both radial
                   spectra with slope fits, and both Gabor energy sets
  tile_<id>.png    one tile through the pipeline: SR / LR, D_s(SR) - LR colour error,
                   residual R and the LR's residual, windowed power spectrum and DCT
                   with the band, radial spectrum with the slope fit, Gabor energies,
                   psi - psi0 in JND units, and the tile's cost per term
  tiles.csv        per tile: cost per term (unweighted shares), F_i, share of F, advice
  pairs.csv        the most costly pairs: relation, affinity, cost per term
  summary.json     terms reported by wcsi_score and the sums of the shares

Attribution. F = sum_k w_k E_k, and every term is a sum of parts: E_RG of tiles (mean over
tiles), E_GJ of neighbour seams (mean over seams), E_PCP of neighbour pairs and corners,
E_cyto of distant pairs, E_sheaf of all pairs and corners. A pair's part is split equally
between its two tiles, a corner's between its four, so the per-tile shares F_i add up to F.
WCSI = 100 / (1 + F / F_JND), so a tile with share F_i alone would cost about that fraction
of the points lost.

Requires matplotlib besides wcsi.py's requirements.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import fft as sfft

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wcsi  # noqa: E402

COORDS = ("mu_L", "mu_a", "mu_b", "cov_LL", "cov_La", "cov_Lb", "cov_aa", "cov_ab", "cov_bb",
          "Q_xx", "Q_xy", "fbar", "alpha")


# ---------------------------------------------------------------- decomposition

def decompose(sr, lr, scale=None, tile=None, overlap=None, tiles=None, tile_coords=None, tile_space="lr",
              **params):
    """wcsi_score's details plus every term split into per-tile, per-pair and per-corner parts."""
    det = wcsi.wcsi_score(sr, lr, scale=scale, tile=tile, overlap=overlap, tiles=tiles, return_details=True,
                          tile_coords=tile_coords, tile_space=tile_space, **params)
    p = {**wcsi.DEFAULTS, **params}
    sr_rgb, sr_white = wcsi.load_rgb(sr)
    lr_rgb, lr_white = wcsi.load_rgb(lr)
    s, sr_rgb, lr_rgb, _ = wcsi._match_scale(sr_rgb, lr_rgb, scale)
    rects = [(t["y0"], t["x0"], t["h"], t["w"]) for t in det["tiles"]]
    ids = [t["id"] for t in det["tiles"]]
    if tiles is None and tile_coords is not None:
        tiles = wcsi.read_tile_coords(tile_coords, s, tile_space)[2]
    sr_lab, lr_lab = wcsi.srgb_to_lab(sr_rgb, sr_white), wcsi.srgb_to_lab(lr_rgb, lr_white)
    canv = canv_rgb = None
    if tiles is not None:
        loaded = [wcsi.load_rgb(t) for t in tiles]
        canv = [wcsi.srgb_to_lab(a, wh) for a, wh in loaded]
        canv_rgb = [np.clip(a.astype(np.float32) / wh, 0, 1) for a, wh in loaded]

    N = len(rects)
    sigma, f_lo = 0.4 * s, 1.0 / (2 * s)
    feats, feats0, de2 = [], [], []
    for k, (y0, x0, h, w_) in enumerate(rects):
        Hc = canv[k] if canv is not None else sr_lab[s * y0:s * (y0 + h), s * x0:s * (x0 + w_)]
        Lc = lr_lab[y0:y0 + h, x0:x0 + w_]
        M, ft = wcsi.tile_features(Hc, s, sigma, f_lo)
        de2.append(float(np.mean(wcsi.ciede2000(M, Lc) ** 2)))
        feats.append(ft)
        feats0.append(wcsi.tile_features(Lc, s, sigma, f_lo)[1])
    de2 = np.array(de2)

    def col(key, src):
        return np.array([f[key] for f in src], np.float64)

    keys = ("mu", "cov", "Q", "kbar", "fbar", "alpha", "gram", "theta", "order", "band_var", "se_f", "se_alpha")
    mu, cov, Q, kbar, fbar, alpha, gram, theta, order, band_var, se_f, se_a = (col(k, feats) for k in keys)
    mu0, cov0, Q0, kbar0, fbar0, alpha0, gram0, theta0, order0, band_var0, se_f0, se_a0 = \
        (col(k, feats0) for k in keys)
    kappa = det["kappa"]

    iu = np.triu_indices(3)
    jnd_de, jnd_a = p["jnd_de"], p["jnd_alpha"]
    jnd_q = 2 * math.sin(math.radians(p["jnd_angle_deg"]))
    floor = jnd_de ** 2 / 4
    psi = np.concatenate([mu, cov[:, iu[0], iu[1]], Q, kbar, alpha[:, None]], 1)
    psi0 = np.concatenate([mu0, kappa * cov0[:, iu[0], iu[1]], Q0, kbar0, alpha0[:, None]], 1)
    phi = np.concatenate([psi[:, :11], fbar[:, None], alpha[:, None]], 1)
    phi0 = np.concatenate([psi0[:, :11], fbar0[:, None], alpha0[:, None]], 1)
    d = phi - phi0
    var0 = kappa * np.diagonal(cov0, axis1=1, axis2=2)
    jnd_t = np.concatenate([np.full((N, 3), jnd_de),
                            p["jnd_cov"] * (np.sqrt(var0[:, iu[0]] * var0[:, iu[1]]) + jnd_de ** 2),
                            np.full((N, 2), jnd_q), p["jnd_freq"] * fbar[:, None], np.full((N, 1), jnd_a)], 1)
    se, se0 = np.zeros_like(jnd_t), np.zeros_like(jnd_t)
    se[:, 11], se[:, 12], se0[:, 11], se0[:, 12] = se_f, se_a, se_f0, se_a0
    vis, vis0 = np.minimum(1.0, band_var / jnd_de ** 2), np.minimum(1.0, band_var0 / jnd_de ** 2)
    texture = np.arange(d.shape[1]) >= 9
    G = gram + floor * np.eye(gram.shape[1])
    gnorm = np.sqrt(np.sum(G ** 2, (1, 2)))
    lr_band = np.log(np.diagonal(gram0, axis1=1, axis2=2) + floor)

    pos = wcsi._grid_positions(rects)
    I, J, near, far, corners = wcsi._tile_graph(rects, pos)
    P = len(I)
    w = np.empty(P)
    for k0 in range(0, P, wcsi._CHUNK):
        a, b = I[k0:k0 + wcsi._CHUNK], J[k0:k0 + wcsi._CHUNK]
        w[k0:k0 + wcsi._CHUNK] = np.exp(-wcsi.bures_w2(mu0[a], cov0[a], mu0[b], cov0[b]) / (2 * p["sigma_cyto"] ** 2)
                                        - np.mean((lr_band[a] - lr_band[b]) ** 2, 1) / (2 * p["tau_band"] ** 2))

    tile_part = {k: np.zeros(N) for k in wcsi.TERMS}
    pair_part = {k: np.zeros(P) for k in ("E_GJ", "E_PCP", "E_cyto", "E_sheaf")}
    corner_part = {k: np.zeros(len(corners)) for k in ("E_PCP", "E_sheaf")}
    sums = {}

    # E_RG
    w2 = wcsi.bures_w2(mu, cov, mu0, kappa * cov0)
    rg = (de2 + p["eta"] * w2) / (jnd_de ** 2 * (1 + p["eta"]))
    tile_part["E_RG"] = rg / N
    sums["E_RG"] = float(rg.mean())

    # E_GJ
    near_idx = np.flatnonzero(near)
    gj_raw = np.full(P, np.nan)
    for k in near_idx:
        pr = [(int(I[k]), int(J[k]))]
        if canv is not None:
            v = [x / jnd_de ** 2 for x in wcsi._gj_canvases(canv, rects, pr, s, p["beta_grad"])]
        else:
            v = wcsi._gj_stitched(sr_lab, lr_lab, rects, pr, s, p["beta_grad"], jnd_de ** 2)
        if v:
            gj_raw[k] = v[0]
    ok = ~np.isnan(gj_raw)
    sums["E_GJ"] = float(gj_raw[ok].mean()) if ok.any() else None
    if ok.any():
        pair_part["E_GJ"][ok] = gj_raw[ok] / ok.sum()

    # E_PCP
    sums["E_PCP"] = None
    scored = np.zeros(len(corners), bool)
    if near.any():
        a, b = I[near], J[near]
        den = float(w[near].sum()) + p["prior_pairs"]
        pair_part["E_PCP"][near] = w[near] * np.minimum(vis[a], vis[b]) * np.sum((Q[a] - Q[b]) ** 2, 1) / jnd_q ** 2 / den
        dq = np.zeros(len(corners))
        for c, cid in enumerate(corners):
            if min((order * vis)[cid].min(), (order0 * vis0)[cid].min()) >= p["defect_min_order"]:
                scored[c] = True
                dq[c] = abs(wcsi._winding(theta[cid]) - wcsi._winding(theta0[cid]))
        if scored.any():
            corner_part["E_PCP"][scored] = p["mu_defect"] * dq[scored] / 0.5 / scored.sum()
        sums["E_PCP"] = float(pair_part["E_PCP"].sum() + corner_part["E_PCP"].sum())

    # E_cyto
    sums["E_cyto"] = None
    cyto_split, cyto_div = np.zeros((P, 3)), np.zeros((P, 3))
    if far.any():
        fi = np.flatnonzero(far)
        den = float(w[far].sum()) + p["prior_pairs"]
        for k0 in range(0, len(fi), wcsi._CHUNK):
            kk = fi[k0:k0 + wcsi._CHUNK]
            x, y = I[kk], J[kk]
            jf2 = (p["jnd_freq"] * 0.5 * (fbar[x] + fbar[y])) ** 2 + se_f[x] ** 2 + se_f[y] ** 2
            ja2 = jnd_a ** 2 + se_a[x] ** 2 + se_a[y] ** 2
            jg = p["jnd_gram"] * 0.5 * (gnorm[x] + gnorm[y])
            v = np.minimum(vis[x], vis[y])
            cyto_div[kk] = np.stack([v * (fbar[x] - fbar[y]) ** 2 / jf2, v * (alpha[x] - alpha[y]) ** 2 / ja2,
                                     np.sum((G[x] - G[y]) ** 2, (1, 2)) / jg ** 2], 1)
            cyto_split[kk] = w[kk, None] * cyto_div[kk] / den
        pair_part["E_cyto"] = cyto_split.sum(1)
        sums["E_cyto"] = float(pair_part["E_cyto"].sum())

    # E_sheaf
    sums["E_sheaf"] = None
    sheaf_coord = np.zeros(d.shape[1])
    sheaf_tile_coord = np.zeros((N, d.shape[1]))
    if P:
        den = float(w.sum()) + p["prior_pairs"]
        for k0 in range(0, P, wcsi._CHUNK):
            x, y = I[k0:k0 + wcsi._CHUNK], J[k0:k0 + wcsi._CHUNK]
            wk = w[k0:k0 + wcsi._CHUNK, None] * np.where(texture, np.minimum(vis[x], vis[y])[:, None], 1.0)
            conf = np.where(texture, np.minimum(vis0[x], vis0[y])[:, None], 1.0)
            diff = phi[y] - phi[x] - conf * (phi0[y] - phi0[x])
            scale2 = ((0.5 * (jnd_t[x] + jnd_t[y])) ** 2 + se[x] ** 2 + se[y] ** 2
                      + conf ** 2 * (se0[x] ** 2 + se0[y] ** 2))
            c = wk * diff ** 2 / scale2 / den
            pair_part["E_sheaf"][k0:k0 + wcsi._CHUNK] = c.sum(1)
            sheaf_coord += c.sum(0)
            for q in range(d.shape[1]):
                sheaf_tile_coord[:, q] += 0.5 * (np.bincount(x, c[:, q], N) + np.bincount(y, c[:, q], N))
        if len(corners):
            hol = sum(d[corners[:, (k + 1) % 4]] - d[corners[:, k]] for k in range(4)) / jnd_t[corners].mean(1)
            corner_part["E_sheaf"] = np.sum(hol ** 2, 1) / len(corners)
        sums["E_sheaf"] = float(pair_part["E_sheaf"].sum() + corner_part["E_sheaf"].sum())

    for k, v in pair_part.items():
        tile_part[k] += 0.5 * (np.bincount(I, v, N) + np.bincount(J, v, N))
    for k, v in corner_part.items():
        for q in range(4 if len(corners) else 0):
            tile_part[k] += 0.25 * np.bincount(corners[:, q], v, N)

    wts = dict(zip(wcsi.TERMS, p["weights"]))
    used = [k for k in wcsi.TERMS if det["terms"][k] is not None]
    F_tile = sum(wts[k] * tile_part[k] for k in used) if used else np.zeros(N)
    return dict(det=det, p=p, s=s, rects=rects, ids=ids, pos=pos, I=I, J=J, near=near, far=far, corners=corners,
                scored=scored, w=w, tile_part=tile_part, pair_part=pair_part, corner_part=corner_part,
                cyto_split=cyto_split, cyto_div=cyto_div, sheaf_coord=sheaf_coord, sheaf_tile_coord=sheaf_tile_coord, gj_raw=gj_raw,
                sums=sums, F_tile=F_tile, weights=wts, used=used, vis=vis, theta=theta, order=order,
                d=d, jnd_t=jnd_t, se=se, se0=se0, kappa=kappa, psi=psi, psi0=psi0,
                sr_rgb=sr_rgb, sr_white=sr_white, lr_rgb=lr_rgb, lr_white=lr_white,
                sr_lab=sr_lab, lr_lab=lr_lab, canv=canv, canv_rgb=canv_rgb, sigma=sigma, f_lo=f_lo)


def check(D):
    """Largest relative difference between wcsi_score's terms and the sums of the parts."""
    worst = 0.0
    for k in wcsi.TERMS:
        a, b = D["det"]["terms"][k], D["sums"][k]
        if (a is None) != (b is None):
            return float("inf")
        if a is not None:
            worst = max(worst, abs(a - b) / max(abs(a), 1e-12))
    return worst


# ---------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _rgb01(a, white):
    return np.clip(a.astype(np.float32) / white, 0, 1)


def _thumb(img, max_side=1600):
    step = max(1, math.ceil(max(img.shape[:2]) / max_side))
    return img[::step, ::step], step


def _tile_map(D, values):
    """Per LR pixel: mean of `values` over the tiles that cover it (NaN where none)."""
    H0, W0 = D["lr_rgb"].shape[:2]
    acc, cnt = np.zeros((H0, W0)), np.zeros((H0, W0))
    for v, (y0, x0, h, w_) in zip(values, D["rects"]):
        acc[y0:y0 + h, x0:x0 + w_] += v
        cnt[y0:y0 + h, x0:x0 + w_] += 1
    with np.errstate(invalid="ignore"):
        return acc / cnt


def _outline(ax, D, step, color="w", lw=0.4):
    from matplotlib.patches import Rectangle
    for y0, x0, h, w_ in D["rects"]:
        ax.add_patch(Rectangle((x0 / step - 0.5, y0 / step - 0.5), w_ / step, h / step,
                               fill=False, ec=color, lw=lw, alpha=0.6))


def fig_overview(D, out):
    plt = _plt()
    det = D["det"]
    lr, step = _thumb(_rgb01(D["lr_rgb"], D["lr_white"]))
    fig, ax = plt.subplots(1, 3, figsize=(19, 6), gridspec_kw=dict(width_ratios=[1.2, 1.2, 0.8]))
    ax[0].imshow(lr)
    _outline(ax[0], D, step)
    ax[0].set_title(f"LR with the {len(D['rects'])} tiles ({det['layout']['source']})")
    share = 100 * D["F_tile"] / max(det["F_total"], 1e-12)
    m = _tile_map(D, share)[::step, ::step]
    ax[1].imshow(lr, alpha=0.35)
    im = ax[1].imshow(m, cmap="inferno", alpha=0.8)
    fig.colorbar(im, ax=ax[1], fraction=0.035, label="tile's share of F (%)")
    worst = np.argsort(-D["F_tile"])[:5]
    for k in worst:
        y0, x0, h, w_ = D["rects"][k]
        ax[1].text((x0 + w_ / 2) / step, (y0 + h / 2) / step, str(D["ids"][k]), color="cyan",
                   ha="center", va="center", fontsize=8, weight="bold")
    ax[1].set_title("where F comes from (labels: the 5 most costly tile ids)")
    terms = [k for k in wcsi.TERMS]
    vals = [D["weights"][k] * det["terms"][k] if det["terms"][k] is not None else 0 for k in terms]
    bars = ax[2].bar(terms, vals, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"])
    for b_, k in zip(bars, terms):
        if det["terms"][k] is None:
            ax[2].text(b_.get_x() + b_.get_width() / 2, 0, "n/a", ha="center", va="bottom")
    ax[2].axhline(det["F_jnd"], color="k", ls="--", lw=0.8)
    ax[2].text(len(terms) - 0.5, det["F_jnd"], " F_JND (score 50 if F equals it)", ha="right", va="bottom", fontsize=8)
    ax[2].set_ylabel("w_k E_k  (JND units)")
    ax[2].set_title(f"WCSI {det['wcsi']:.2f} = 100 / (1 + {det['F_total']:.3f} / {det['F_jnd']:.1f})")
    for a in ax[:2]:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=110)
    plt.close(fig)


def fig_term_maps(D, out):
    plt = _plt()
    lr, step = _thumb(_rgb01(D["lr_rgb"], D["lr_white"]), 800)
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    info = dict(E_RG="colour of D_s(SR) vs LR, colour distribution vs LR's",
                E_GJ="seams (half of each seam to each tile)",
                E_PCP="orientation vs neighbours + corner defects",
                E_cyto="texture vs distant same-material tiles",
                E_sheaf="statistics vs LR prediction, all pairs")
    for a, k in zip(ax.flat, wcsi.TERMS):
        a.imshow(lr, alpha=0.3)
        if D["det"]["terms"][k] is None:
            a.set_title(f"{k}: n/a")
        else:
            im = a.imshow(_tile_map(D, D["tile_part"][k])[::step, ::step], cmap="viridis", alpha=0.85)
            fig.colorbar(im, ax=a, fraction=0.035)
            a.set_title(f"{k} = {D['det']['terms'][k]:.3f}: {info[k]}", fontsize=9)
        a.axis("off")
    a = ax.flat[5]
    a.imshow(lr, alpha=0.3)
    im = a.imshow(_tile_map(D, D["vis"])[::step, ::step], cmap="gray", vmin=0, vmax=1, alpha=0.85)
    fig.colorbar(im, ax=a, fraction=0.035)
    a.set_title("texture visibility v (weight of every texture comparison)", fontsize=9)
    a.axis("off")
    fig.suptitle("each tile's share of every term (pair costs split between the two tiles)")
    fig.tight_layout()
    fig.savefig(out / "term_maps.png", dpi=110)
    plt.close(fig)


def fig_pairs(D, out):
    plt = _plt()
    from matplotlib.collections import LineCollection
    N, I, J, w = len(D["rects"]), D["I"], D["J"], D["w"]
    fig, ax = plt.subplots(2, 3, figsize=(19, 11))
    a = ax[0, 0]
    if N <= 4000:
        A = np.eye(N)
        A[I, J] = A[J, I] = w
        im = a.imshow(A, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        fig.colorbar(im, ax=a, fraction=0.04)
        a.set_title("material affinity w_ij in the LR (tile order)")
    else:
        a.text(0.5, 0.5, f"{N} tiles: matrix not drawn", ha="center")
    a = ax[0, 1]
    bins = np.linspace(0, 1, 41)
    for m, lab in ((D["near"], "neighbours"), (D["far"], "distant"),
                   (~D["near"] & ~D["far"], "other (touching)")):
        if m.any():
            a.hist(w[m], bins, alpha=0.6, label=f"{lab} ({m.sum()})")
    a.set_yscale("log")
    a.set_xlabel("affinity w_ij")
    a.legend()
    a.set_title("affinity of tile pairs")

    a = ax[0, 2]
    lr, step = _thumb(_rgb01(D["lr_rgb"], D["lr_white"]))
    a.imshow(lr, alpha=0.5)
    cen = np.array([((x0 + w_ / 2) / step, (y0 + h / 2) / step) for y0, x0, h, w_ in D["rects"]])
    nk = np.flatnonzero(D["near"])
    cost = sum(D["weights"][k] * D["pair_part"][k][nk] for k in ("E_GJ", "E_PCP", "E_sheaf")
               if D["det"]["terms"][k] is not None) if len(nk) else np.zeros(0)
    if len(nk):
        lc = LineCollection(np.stack([cen[I[nk]], cen[J[nk]]], 1), cmap="plasma", linewidths=0.5 + 3 * w[nk])
        lc.set_array(cost)
        a.add_collection(lc)
        fig.colorbar(lc, ax=a, fraction=0.035, label="w_k-weighted E_GJ + E_PCP + E_sheaf parts")
    a.scatter(cen[:, 0], cen[:, 1], s=6, c="k")
    for c, cid in enumerate(D["corners"]):
        if D["scored"][c]:
            p_ = cen[cid].mean(0)
            a.plot(*p_, marker="x", color="red" if D["corner_part"]["E_PCP"][c] > 0 else "lime", ms=5)
    a.set_title("neighbour graph: edge colour = cost, width = affinity;\n"
                "x = scored corner (red: defect mismatch)", fontsize=9)
    a.axis("off")

    a = ax[1, 0]
    fk = np.flatnonzero(D["far"])
    if len(fk):
        a.scatter(w[fk], D["pair_part"]["E_cyto"][fk] / max(w[fk].max(), 1e-12), s=3, alpha=0.4,
                  c=D["cyto_split"][fk].argmax(1), cmap="brg", vmin=0, vmax=2)
        a.set_xlabel("affinity w_ij")
        a.set_ylabel("pair's part of E_cyto (scaled)")
        a.set_yscale("symlog", linthresh=1e-6)
        a.set_title("distant pairs: cost vs affinity\n(colour: dominant part blue=freq, red=alpha, green=Gabor)",
                    fontsize=9)
    a = ax[1, 1]
    if D["det"]["terms"]["E_cyto"] is not None:
        parts = D["cyto_split"].sum(0)
        a.bar(["freq", "alpha", "Gabor"], parts, color=["#4c72b0", "#c44e52", "#55a868"])
        a.set_title(f"E_cyto = {D['det']['terms']['E_cyto']:.3f} by statistic")
    a = ax[1, 2]
    if D["det"]["terms"]["E_sheaf"] is not None:
        a.bar(COORDS, D["sheaf_coord"], color=["#4c72b0"] * 3 + ["#dd8452"] * 6 + ["#55a868"] * 2 + ["#c44e52"] * 2)
        a.tick_params(axis="x", rotation=60)
        hol = D["corner_part"]["E_sheaf"].sum()
        a.set_title(f"E_sheaf = {D['det']['terms']['E_sheaf']:.3f} by coordinate (holonomy {hol:.1e})")
    fig.tight_layout()
    fig.savefig(out / "pairs.png", dpi=110)
    plt.close(fig)


def _seam_detail(img, lr, a, b, r0, r1, s, beta, jnd2, offsets=(3, 6), lr_offsets=(2, 3)):
    reach, reach0 = max(offsets) + 3, max(lr_offsets) + 3
    best = None
    for c in sorted({a, (a + b) // 2, b}):
        if c - reach < 0 or c + reach > img.shape[1]:
            continue
        ex, ctrl = wcsi._excess(img, c, r0, r1, beta, offsets)
        gate, c0 = 1.0, int(round(c / s))
        if reach0 <= c0 <= lr.shape[1] - reach0:
            ex0, ctrl0 = wcsi._excess(lr, c0, r0 // s, max(r1 // s, r0 // s + 1), beta, lr_offsets)
            gate = math.exp(-max(ex0, 0.0) / (jnd2 + ctrl0))
        v = gate * max(ex, 0.0) / (jnd2 + ctrl)
        if best is None or v > best["v"]:
            best = dict(c=c, v=v, ex=ex, ctrl=ctrl, gate=gate)
    if best is None:
        return None
    c = best["c"]
    best["jump"] = wcsi._jump(img, c, r0, r1, beta)
    best["control"] = np.median([wcsi._jump(img, c + o, r0, r1, beta) for o in offsets + tuple(-o for o in offsets)], 0)
    return best


def fig_seams(D, out, n=4):
    plt = _plt()
    ok = np.flatnonzero(~np.isnan(D["gj_raw"]))
    if not len(ok):
        return
    top = ok[np.argsort(-D["gj_raw"][ok])[:n]]
    s, p, rects = D["s"], D["p"], D["rects"]
    fig, ax = plt.subplots(len(top), 3, figsize=(20, 3.2 * len(top)), squeeze=False,
                           gridspec_kw=dict(width_ratios=[1.5, 1.5, 2]))
    sr01 = _rgb01(D["sr_rgb"], D["sr_white"])
    for row, k in enumerate(top):
        i, j = int(D["I"][k]), int(D["J"][k])
        ri, rj = rects[i], rects[j]
        y0, y1, x0, x1 = wcsi._shared(ri, rj)
        side = abs(ri[1] - rj[1]) / min(ri[3], rj[3]) >= abs(ri[0] - rj[0]) / min(ri[2], rj[2])
        title = f"tiles {D['ids'][i]} | {D['ids'][j]}: E_GJ part {D['gj_raw'][k]:.3f} JND"
        if D["canv"] is not None:
            Y0, Y1, X0, X1 = s * y0, s * y1, s * x0, s * x1
            (yi, xi), (yj, xj) = (s * ri[0], s * ri[1]), (s * rj[0], s * rj[1])
            A = D["canv_rgb"][i][Y0 - yi:Y1 - yi, X0 - xi:X1 - xi]
            B = D["canv_rgb"][j][Y0 - yj:Y1 - yj, X0 - xj:X1 - xj]
            dE = wcsi.ciede2000(D["canv"][i][Y0 - yi:Y1 - yi, X0 - xi:X1 - xi],
                                D["canv"][j][Y0 - yj:Y1 - yj, X0 - xj:X1 - xj])
            ax[row, 0].imshow(A)
            ax[row, 0].set_title(f"tile {D['ids'][i]} on the overlap")
            ax[row, 1].imshow(B)
            ax[row, 1].set_title(f"tile {D['ids'][j]} on the overlap")
            im = ax[row, 2].imshow(dE, cmap="magma")
            fig.colorbar(im, ax=ax[row, 2], fraction=0.03, label="dE00")
            ax[row, 2].set_title(title, fontsize=9)
            for a in ax[row]:
                a.axis("off")
            continue
        img, lr = D["sr_lab"], D["lr_lab"]
        rgb, a_, b_, r0, r1 = sr01, s * x0, s * x1, s * y0, s * y1
        if not side:
            img, lr, rgb = img.transpose(1, 0, 2), lr.transpose(1, 0, 2), rgb.transpose(1, 0, 2)
            a_, b_, r0, r1 = s * y0, s * y1, s * x0, s * x1
        det = _seam_detail(img, lr, a_, b_, r0, r1, s, p["beta_grad"], p["jnd_de"] ** 2)
        if det is None:
            continue
        c = det["c"]
        pad = 24
        # drawn with the seam running left to right, aligned with the profile on the right
        lo = max(c - pad, 0)
        crop = rgb[r0:r1, lo:c + pad].transpose(1, 0, 2)
        ax[row, 0].imshow(crop, extent=(r0 - 0.5, r1 - 0.5, c + pad - 0.5, lo - 0.5))
        ax[row, 0].axhline(c - 0.5, color="red", lw=1.2)
        ax[row, 0].set_title(f"SR across the seam (red line; the seam is {'vertical' if side else 'horizontal'} "
                             "in the image)", fontsize=8)
        c0 = int(round(c / s))
        lcrop = _rgb01(D["lr_rgb"], D["lr_white"])
        lcrop = lcrop if side else lcrop.transpose(1, 0, 2)
        l0, l1 = max(c0 - pad // s - 2, 0), c0 + pad // s + 2
        lcrop = lcrop[r0 // s:max(r1 // s, r0 // s + 1), l0:l1].transpose(1, 0, 2)
        ax[row, 1].imshow(lcrop, interpolation="nearest",
                          extent=(r0 - 0.5, r0 - 0.5 + s * lcrop.shape[1], s * l0 + s * lcrop.shape[0] - 0.5,
                                  s * l0 - 0.5))
        ax[row, 1].axhline(c - 0.5, color="red", lw=1.2)
        ax[row, 1].set_ylim(c + pad - 0.5, lo - 0.5)
        ax[row, 1].set_title("LR at the same place (SR pixel coordinates)", fontsize=8)
        rows = np.arange(r0, r0 + len(det["jump"]))
        ax[row, 2].plot(rows, np.maximum(det["jump"], 1e-3), lw=0.7, label="jump at the seam line")
        ax[row, 2].plot(rows, np.maximum(det["control"], 1e-3), lw=0.7,
                        label="median jump at control lines +-3, +-6 px")
        ax[row, 2].axhline(np.median(det["jump"]), color="C0", ls="--", lw=0.8)
        ax[row, 2].axhline(np.median(det["control"]), color="C1", ls="--", lw=0.8)
        ax[row, 2].set_yscale("log")
        ax[row, 2].set_xlabel("position along the seam (SR px); dashed: medians")
        ax[row, 2].set_ylabel("dLab^2 + beta dGrad^2")
        ax[row, 2].legend(fontsize=7)
        ax[row, 2].set_title(f"{title}\nmedian excess {det['ex']:.2f} / (1.5^2 + control {det['ctrl']:.2f}) "
                             f"x LR gate {det['gate']:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "seams.png", dpi=110)
    plt.close(fig)


def _radial(lum, f_lo):
    """Ring-averaged windowed power spectrum (as in wcsi.spectral) and the slope fit in the band."""
    h, w = lum.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F = sfft.fft2((lum - lum.mean()) * win)
    P = F.real ** 2 + F.imag ** 2
    kx, ky = wcsi._freqs(h, w)
    f = np.sqrt(kx ** 2 + ky ** 2)
    up = (f <= 0.5) & (ky >= 0) & (f > 0)
    ring = np.floor(f[up] * min(h, w)).astype(np.int64)
    cnt = np.bincount(ring)
    ps, fs = np.bincount(ring, P[up]), np.bincount(ring, f[up])
    good = (cnt > 0) & (ps > 0)
    fr, pr = fs[good] / cnt[good], ps[good] / cnt[good]
    band = fr >= f_lo
    fit = None
    if band.sum() >= 3:
        x, y, wt = np.log(fr[band]), np.log(pr[band]), cnt[good][band].astype(np.float64)
        fit = np.polyfit(x, y, 1, w=np.sqrt(wt))
    return fr, pr, band, fit, np.fft.fftshift(P)


def _tile_data(D, k):
    """SR canvas (Lab, RGB), LR tile (Lab, RGB), M, and the L* residuals of SR and LR."""
    s, sigma = D["s"], D["sigma"]
    y0, x0, h, w_ = D["rects"][k]
    if D["canv"] is not None:
        Hc, rgb = D["canv"][k], D["canv_rgb"][k]
    else:
        Hc = D["sr_lab"][s * y0:s * (y0 + h), s * x0:s * (x0 + w_)]
        rgb = _rgb01(D["sr_rgb"][s * y0:s * (y0 + h), s * x0:s * (x0 + w_)], D["sr_white"])
    Lc = D["lr_lab"][y0:y0 + h, x0:x0 + w_]
    lrgb = _rgb01(D["lr_rgb"][y0:y0 + h, x0:x0 + w_], D["lr_white"])
    M, R = wcsi.rg_split(Hc, s, sigma)
    _, R0 = wcsi.rg_split(Lc, s, sigma)
    return Hc, rgb, Lc, lrgb, M, np.ascontiguousarray(R[..., 0]), np.ascontiguousarray(R0[..., 0])


def select_far_pairs(D, n, min_affinity=0.5):
    """Distant pairs the LR calls one material (affinity >= min_affinity, or the top 20% of
    distant pairs by affinity if fewer than 2n qualify), ranked by their texture divergence
    (the E_cyto statistics before affinity weighting): the n most consistent and n least."""
    fk = np.flatnonzero(D["far"])
    if not len(fk):
        return [], [], None
    cand = fk[D["w"][fk] >= min_affinity]
    if len(cand) < 2 * n:
        cand = fk[np.argsort(-D["w"][fk])[:max(2 * n, math.ceil(0.2 * len(fk)))]]
    o = cand[np.argsort(D["cyto_div"][cand].sum(1), kind="stable")]
    best = o[:n].tolist()
    worst = [k for k in o[::-1] if k not in set(best)][:n]
    return best, worst, float(D["w"][cand].min())


def fig_far_pairs(D, out, n=3, min_affinity=0.5):
    """far_pairs_consistent.png / far_pairs_inconsistent.png: same-material distant pairs."""
    plt = _plt()
    from matplotlib.patches import Rectangle
    best, worst, thr = select_far_pairs(D, n, min_affinity)
    if not best:
        return []
    lr, step = _thumb(_rgb01(D["lr_rgb"], D["lr_white"]), 800)
    written = []
    for name, sel, label in (("consistent", best, "most consistent"), ("inconsistent", worst, "least consistent")):
        if not sel:
            continue
        fig, ax = plt.subplots(len(sel), 7, figsize=(31, 4.4 * len(sel)), squeeze=False,
                               gridspec_kw=dict(width_ratios=[1.3, 1, 1, 1, 1, 1.3, 1.3]))
        for row, q in enumerate(sel):
            i, j = int(D["I"][q]), int(D["J"][q])
            di, dj = _tile_data(D, i), _tile_data(D, j)
            fi = wcsi.tile_features(di[0], D["s"], D["sigma"], D["f_lo"])[1]
            fj = wcsi.tile_features(dj[0], D["s"], D["sigma"], D["f_lo"])[1]
            div = D["cyto_div"][q]
            a = ax[row, 0]
            a.imshow(lr)
            cen = []
            for k, c in ((i, "cyan"), (j, "magenta")):
                y0, x0, h, w_ = D["rects"][k]
                a.add_patch(Rectangle((x0 / step, y0 / step), w_ / step, h / step, fill=False, ec=c, lw=2))
                cen.append(((x0 + w_ / 2) / step, (y0 + h / 2) / step))
            a.plot(*zip(*cen), color="yellow", lw=1, ls="--")
            a.set_title(f"affinity w = {D['w'][q]:.2f}\ndivergence: freq {div[0]:.2f} + slope {div[1]:.2f} + "
                        f"Gabor {div[2]:.2f} = {div.sum():.2f} JND^2", fontsize=9)
            a.axis("off")
            lim = max(np.percentile(np.abs(di[5]), 99.5), np.percentile(np.abs(dj[5]), 99.5)) + 1e-6
            for col, (k, dd, ft, c) in enumerate(((i, di, fi, "cyan"), (j, dj, fj, "magenta"))):
                a = ax[row, 1 + col]
                a.imshow(dd[1])
                ins = a.inset_axes([0.0, 0.0, 0.32, 0.32])
                ins.imshow(dd[3], interpolation="nearest")
                ins.set_xticks([])
                ins.set_yticks([])
                for sp in ins.spines.values():
                    sp.set_edgecolor(c)
                a.set_title(f"SR tile {D['ids'][k]} (inset: LR)", fontsize=9, color="k")
                a.axis("off")
                a = ax[row, 3 + col]
                a.imshow(dd[5], cmap="RdBu_r", vmin=-lim, vmax=lim)
                a.set_title(f"residual L* of tile {D['ids'][k]}\nvisibility {D['vis'][k]:.2f}, "
                            f"RMS {dd[5].std():.2f} (shared colour scale)", fontsize=9)
                a.axis("off")
            a = ax[row, 5]
            for k, dd, ft, c in ((i, di, fi, "C0"), (j, dj, fj, "C3")):
                fr, pr, band, fit, _ = _radial(dd[5], D["f_lo"])
                a.loglog(fr, pr, ".", ms=3, color=c, alpha=0.5)
                if fit is not None:
                    a.loglog(fr[band], np.exp(np.polyval(fit, np.log(fr[band]))), color=c, lw=1.5,
                             label=f"tile {D['ids'][k]}: slope {ft['alpha']:.2f}, mean freq {ft['fbar']:.3f}")
            a.axvline(D["f_lo"], color="k", ls="--", lw=0.7)
            a.set_xlabel("radial frequency (cycles/px); band right of dashed line")
            a.set_title("radial power spectra of the residuals", fontsize=9)
            a.legend(fontsize=8)
            a = ax[row, 6]
            gi, gj = np.diagonal(fi["gram"]), np.diagonal(fj["gram"])
            xs = np.arange(len(gi))
            a.bar(xs - 0.2, gi, 0.4, color="C0", label=f"tile {D['ids'][i]}")
            a.bar(xs + 0.2, gj, 0.4, color="C3", label=f"tile {D['ids'][j]}")
            a.set_yscale("log")
            a.set_xticks(xs, [f"{k_ % 4 * 45}d\nf{k_ // 4}" for k_ in xs], fontsize=7)
            a.legend(fontsize=8)
            a.set_title(f"Gabor energies; pair's E_cyto part {D['pair_part']['E_cyto'][q]:.4f}, "
                        f"E_sheaf part {D['pair_part']['E_sheaf'][q]:.4f}", fontsize=9)
        fig.suptitle(f"distant pairs of one material in the LR (affinity >= {thr:.2f}): the {len(sel)} {label} "
                     "in the SR's invented texture", fontsize=13)
        fig.tight_layout()
        fig.savefig(out / f"far_pairs_{name}.png", dpi=90)
        plt.close(fig)
        written.append(f"far_pairs_{name}")
    return written


def fig_tile(D, k, out):
    plt = _plt()
    s, f_lo, sigma = D["s"], D["f_lo"], D["sigma"]
    y0, x0, h, w_ = D["rects"][k]
    Hc, rgb, Lc, lrgb, M, lum, lum0 = _tile_data(D, k)
    dE = wcsi.ciede2000(M, Lc)

    fig, ax = plt.subplots(3, 4, figsize=(20, 14.5))
    ax[0, 0].imshow(rgb)
    ax[0, 0].set_title(f"SR tile {D['ids'][k]}  (LR x={x0} y={y0} {w_}x{h})")
    ax[0, 1].imshow(lrgb, interpolation="nearest")
    ax[0, 1].set_title("LR tile")
    im = ax[0, 2].imshow(dE, cmap="magma", interpolation="nearest")
    fig.colorbar(im, ax=ax[0, 2], fraction=0.045, label="dE00")
    ax[0, 2].set_title(f"E_RG colour: dE00(D_s(SR), LR), RMS {math.sqrt(np.mean(dE ** 2)):.2f}\n"
                       f"(JND {D['p']['jnd_de']})", fontsize=9)
    lim = np.percentile(np.abs(lum), 99.5) + 1e-6
    ax[0, 3].imshow(lum, cmap="RdBu_r", vmin=-lim, vmax=lim)
    th, od = D["theta"][k], D["order"][k]
    cy, cx, L = lum.shape[0] / 2, lum.shape[1] / 2, 0.45 * min(lum.shape) * max(od, 0.05) ** 0.5
    ax[0, 3].plot([cx - L * math.cos(th), cx + L * math.cos(th)], [cy + L * math.sin(th), cy - L * math.sin(th)],
                  color="lime", lw=2)
    ax[0, 3].set_title(f"residual R = SR - U_s(D_s(SR)), L*\nstroke angle {math.degrees(th):.0f} deg, "
                       f"order {od:.2f}, visibility {D['vis'][k]:.2f}", fontsize=9)

    fr, pr, band, fit, Pc = _radial(lum, f_lo)
    fr0, pr0, band0, fit0, Pc0 = _radial(lum0, f_lo)
    for a, Pc_, title, (hh, ww) in ((ax[1, 0], Pc, "SR residual", lum.shape), (ax[1, 1], Pc0, "LR residual (one scale down)", lum0.shape)):
        a.imshow(np.log10(Pc_ + 1e-12), cmap="viridis")
        for f_ in (f_lo, 0.5):
            t = np.linspace(0, 2 * np.pi, 200)
            a.plot(ww / 2 + f_ * ww * np.cos(t), hh / 2 + f_ * hh * np.sin(t), color="w", lw=0.7, ls="--")
        a.set_title(f"log10 power spectrum of the windowed {title}\nband [1/(2s), 1/2] between the circles", fontsize=9)
        a.axis("off")
    dct = np.abs(sfft.dctn(lum, norm="ortho"))
    ax[1, 2].imshow(np.log10(dct + 1e-6), cmap="viridis")
    yy, xx = np.mgrid[:dct.shape[0], :dct.shape[1]]
    ax[1, 2].contour(np.hypot(xx / (2 * dct.shape[1]), yy / (2 * dct.shape[0])), [f_lo], colors="w",
                     linewidths=0.7, linestyles="--")
    ax[1, 2].set_title("log10 |DCT| of the SR residual (top left = low frequency);\n"
                       "right of / below the dashed line = the band", fontsize=9)
    ax[1, 2].axis("off")
    a = ax[1, 3]
    a.loglog(fr, pr, ".", ms=3, color="C0", alpha=0.5, label="SR residual")
    a.loglog(fr0, pr0, ".", ms=3, color="C1", alpha=0.5, label="LR residual")
    for fit_, fr_, band_, c in ((fit, fr, band, "C0"), (fit0, fr0, band0, "C1")):
        if fit_ is not None:
            xs = fr_[band_]
            a.loglog(xs, np.exp(np.polyval(fit_, np.log(xs))), color=c, lw=1.5)
    a.axvline(f_lo, color="k", ls="--", lw=0.7)
    a.set_xlabel("radial frequency (cycles/px)")
    a.set_ylabel("ring-mean power")
    ft = wcsi.tile_features(Hc, s, sigma, f_lo)[1]
    ft0 = wcsi.tile_features(Lc, s, sigma, f_lo)[1]
    a.set_title(f"radial spectrum, lines: slope fit in the band\nalpha SR {ft['alpha']:.2f}, LR {ft0['alpha']:.2f}; "
                f"mean radial freq SR {ft['fbar']:.3f}, LR {ft0['fbar']:.3f} cycles/px", fontsize=9)
    a.legend(fontsize=8)

    g, g0 = np.diagonal(ft["gram"]), np.diagonal(ft0["gram"])
    xs = np.arange(len(g))
    ax[2, 0].bar(xs - 0.2, g, 0.4, label="SR")
    ax[2, 0].bar(xs + 0.2, g0, 0.4, label="LR (one scale down)")
    ax[2, 0].set_yscale("log")
    ax[2, 0].set_xticks(xs, [f"{k_ % 4 * 45}d\nf{k_ // 4}" for k_ in xs], fontsize=7)
    ax[2, 0].legend(fontsize=8)
    ax[2, 0].set_title("Gabor energies G(R): orientation, octave (f0 = finest)", fontsize=9)

    z = D["d"][k] / np.sqrt(D["jnd_t"][k] ** 2 + D["se"][k] ** 2 + D["se0"][k] ** 2)
    ax[2, 1].bar(COORDS, z, color=["#4c72b0"] * 3 + ["#dd8452"] * 6 + ["#55a868"] * 2 + ["#c44e52"] * 2)
    ax[2, 1].axhline(1, color="k", ls=":", lw=0.7)
    ax[2, 1].axhline(-1, color="k", ls=":", lw=0.7)
    ax[2, 1].tick_params(axis="x", rotation=60)
    ax[2, 1].set_title("phi - phi0 (tile vs its LR, JND units; +-1 dotted)", fontsize=9)

    terms = list(wcsi.TERMS)
    vals = [D["weights"][t] * D["tile_part"][t][k] for t in terms]
    ax[2, 2].bar(terms, vals, color=["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"])
    share = 100 * D["F_tile"][k] / max(D["det"]["F_total"], 1e-12)
    ax[2, 2].set_title(f"this tile's share of F: {D['F_tile'][k]:.4f} ({share:.1f}% of F, "
                       f"{100 / len(D['rects']):.1f}% if uniform)", fontsize=9)
    ax[2, 2].tick_params(axis="x", rotation=30)

    a = ax[2, 3]
    a.axis("off")
    st = D["sheaf_tile_coord"][k]
    partners = []
    for key in ("E_sheaf", "E_cyto", "E_PCP", "E_GJ"):
        m = (D["I"] == k) | (D["J"] == k)
        v = D["pair_part"][key][m]
        if not len(v) or v.max() <= 0:
            continue
        for q in np.argsort(-v)[:3]:
            idx = np.flatnonzero(m)[q]
            other = D["J"][idx] if D["I"][idx] == k else D["I"][idx]
            partners.append(f"{key:8s} with tile {D['ids'][other]}: {v[q]:.4f} (w {D['w'][idx]:.2f})")
    lines = [f"advice: {D['det']['tiles'][k]['advice']}", "",
             "E_sheaf share by coordinate (largest 4):"]
    lines += [f"  {COORDS[q]:7s} {st[q]:.4f}" for q in np.argsort(-st)[:4]]
    lines += ["", "most costly partners:"] + ["  " + t for t in partners]
    a.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8)
    fig.tight_layout()
    name = str(D["ids"][k]).replace("/", "_")
    fig.savefig(out / f"tile_{name}.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------- tables

def write_tables(D, out, n_pairs=200):
    det = D["det"]
    with open(out / "tiles.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["id", "y0", "x0", "h", "w", "row", "col", *wcsi.TERMS, "F_i", "share_pct", "visibility", "advice"])
        for k, t in enumerate(det["tiles"]):
            wr.writerow([t["id"], t["y0"], t["x0"], t["h"], t["w"], t["row"], t["col"],
                         *(f"{D['tile_part'][x][k]:.6g}" for x in wcsi.TERMS), f"{D['F_tile'][k]:.6g}",
                         f"{100 * D['F_tile'][k] / max(det['F_total'], 1e-12):.3f}", f"{D['vis'][k]:.3f}",
                         t["advice"]])
    tot = sum(D["weights"][k] * D["pair_part"][k] for k in D["pair_part"] if det["terms"][k] is not None) \
        if D["used"] else np.zeros(len(D["I"]))
    order = np.argsort(-tot)[:n_pairs]
    with open(out / "pairs.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["tile_i", "tile_j", "relation", "affinity", "E_GJ", "E_PCP", "E_cyto", "E_sheaf", "weighted_total"])
        for q in order:
            rel = "neighbour" if D["near"][q] else ("distant" if D["far"][q] else "touching")
            wr.writerow([D["ids"][D["I"][q]], D["ids"][D["J"][q]], rel, f"{D['w'][q]:.4f}",
                         *(f"{D['pair_part'][x][q]:.6g}" for x in ("E_GJ", "E_PCP", "E_cyto", "E_sheaf")),
                         f"{tot[q]:.6g}"])
    summary = dict(wcsi=det["wcsi"], F_total=det["F_total"], F_jnd=det["F_jnd"], terms=det["terms"],
                   sums_of_parts=D["sums"], max_rel_mismatch=check(D), layout=det["layout"], scale=D["s"],
                   kappa=D["kappa"], notes=det["notes"],
                   E_sheaf_by_coordinate=dict(zip(COORDS, D["sheaf_coord"].tolist())),
                   E_cyto_by_statistic=dict(zip(("freq", "alpha", "gabor"), D["cyto_split"].sum(0).tolist())))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description="Figures and tables showing how each tile and pair contributes to WCSI.")
    ap.add_argument("sr")
    ap.add_argument("lr")
    ap.add_argument("--out", default="wcsi_viz", help="output directory (default: wcsi_viz)")
    ap.add_argument("--scale", type=int)
    ap.add_argument("--tile", type=int, help="tile size in LR pixels")
    ap.add_argument("--overlap", type=int)
    ap.add_argument("--tiles-jsonl", help="tile coordinates, one JSON object per line")
    ap.add_argument("--tile-space", choices=("lr", "sr"), default="lr")
    ap.add_argument("--tiles-dir", help="unblended tile outputs; sorted file names = tile order")
    ap.add_argument("--weights", help="w1..w5, comma separated")
    ap.add_argument("--kappa", default="auto")
    ap.add_argument("--tile-ids", nargs="*", help="tiles to draw in detail (default: the 3 most and 1 least costly)")
    ap.add_argument("--seams", type=int, default=4, help="number of worst seams to draw")
    ap.add_argument("--far-pairs", type=int, default=3,
                    help="same-material distant pairs to draw, most and least consistent (default 3 each)")
    ap.add_argument("--min-affinity", type=float, default=0.5,
                    help="LR affinity that makes two distant tiles 'one material' (default 0.5)")
    args = ap.parse_args(argv)
    if args.tiles_jsonl and (args.tile is not None or args.overlap is not None):
        ap.error("--tiles-jsonl replaces --tile/--overlap")

    params = {}
    if args.weights:
        params["weights"] = tuple(float(v) for v in args.weights.split(","))
    if args.kappa != "auto":
        params["kappa"] = float(args.kappa)
    tiles = None
    if args.tiles_dir:
        tiles = sorted((f for f in Path(args.tiles_dir).iterdir() if f.suffix.lower() in wcsi.IMAGE_EXT),
                       key=wcsi._natural_key)
    D = decompose(args.sr, args.lr, scale=args.scale, tile=args.tile, overlap=args.overlap, tiles=tiles,
                  tile_coords=args.tiles_jsonl, tile_space=args.tile_space, **params)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    det = D["det"]
    mis = check(D)
    print(f"WCSI {det['wcsi']:.2f}   F {det['F_total']:.4f}   {len(D['rects'])} tiles   "
          f"parts add up to the terms within {mis:.1e} (relative)")
    if mis > 1e-6:
        print("  warning: the decomposition does not match wcsi_score; wcsi.py may have changed")
    for k in wcsi.TERMS:
        v = det["terms"][k]
        print(f"  {k:8s} {'n/a' if v is None else f'{v:9.4f}'}")

    if args.tile_ids:
        want = [str(t) for t in args.tile_ids]
        sel = [k for k, i in enumerate(D["ids"]) if str(i) in want]
        if len(sel) != len(want):
            print(f"  warning: some tile ids not found (ids are {D['ids'][:10]}...)")
    else:
        o = np.argsort(-D["F_tile"])
        sel = list(dict.fromkeys(o[:3].tolist() + o[-1:].tolist()))

    write_tables(D, out)
    fig_overview(D, out)
    fig_term_maps(D, out)
    fig_pairs(D, out)
    fig_seams(D, out, args.seams)
    far = fig_far_pairs(D, out, args.far_pairs, args.min_affinity) if args.far_pairs > 0 else []
    for k in sel:
        fig_tile(D, k, out)
    print(f"  wrote {out}/: " + ", ".join(["overview", "term_maps", "pairs", "seams"] + far
                                          + [f"tile_{D['ids'][k]}" for k in sel])
          + " (.png), tiles.csv, pairs.csv, summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
