#!/usr/bin/env python3
"""Hybrid pixel + frozen-feature score WITH a ground-truth reference.

Pixel space keeps what encoders cannot see: coarse colour against the ground
truth, 1-px seams, and the Nyquist octave. All pixel terms come from WCSI-2 with
reference (genet/wcsi2.py, which superseded GMCI), plus a one-sided Nyquist probe.
Frozen features take the mid band and the non-local structure: gated
pointwise + Bures-Wasserstein texture fidelity, feature nematics and spectra,
DINOv2 token distance, and sheaf frustration over a DINOv2 material graph.
All terms are returned; terms.py turns them into every ablation configuration.
The legacy GMCI pixel terms (minutes per image) are computed only with with_gmci=True.

    python with_encoder/hybrid_ref.py SR.png GT.png LR.png --scale 8 --tile-sr 512 \
        --encoders with_encoder/configs/encoders.json --method VOSR --json out.json
    python with_encoder/hybrid_ref.py SR.png GT.png LR.png --scale 4 --encoders gabor   # CPU control

See DESIGN.md for the decisions behind every term.
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
from encoders.registry import Stack, build_stack, conflicts  # noqa: E402
import wcsi2  # noqa: E402
from wcsi2_backend import get_backend, group_rects  # noqa: E402

gmci, wcsi = fs.gmci, fs.wcsi

DEFAULTS = dict(
    k_pca=16,                 # PCA axes per layer for distances
    k_rho=4,                  # PCA axes per layer in the sheaf state (mean and spread)
    neff_frac=0.25,           # neighbouring feature tokens are correlated: effective samples per token
    basis_tiles=16,
    max_basis_tokens=200_000,
    gate="pix",               # pix | feat | none | all
    jnd_feat=0.2,             # 1 JND = 20% of the image's feature RMS spread in that layer (nominal)
    jnd_angle_deg=4.0,
    jnd_freq=0.05,
    jnd_peak=math.log(2.0),   # aliasing peak twice the GT's is one JND (nominal)
    jnd_nyq_energy=math.log(2.0),  # top-octave energy twice the GT's is one JND (nominal)
    n_q=8,
    n_proj_spectrum=12,
    tau_sem=0.3,              # centred-cosine threshold of the material graph (nominal)
    far_top=8,                # distant edges kept per tile
    prior_pairs=1.0,
    harmonic_frac=0.05,
    harmonic_cap=4,
    rho_max_n=6000,
    phenotype_master=1.0,
    phenotype_frust=1.0,
    rigidity_margin=16,       # context around a tile for the phase-congruency gate, HR px
    with_wcsi2=True,          # WCSI-2 reference pixel terms (pix_tone ... pix_seam)
    with_gmci=False,          # legacy GMCI pixel terms (gmci_*), slow
    device=None,              # WCSI-2 kernels: None (numpy), "cuda:k", "torch:cpu"
)


def _rgb01(a, white):
    return np.clip(np.asarray(a, np.float32) / white, 0, 1)


def _pix_nyq(sr_lab, gt_lab, rects, s, p):
    """One-sided: excess top-octave energy and excess aliasing peaks (ringing, checkerboards).
    Missing high frequencies (blur) belong to the mid-band feature terms."""
    vals = []
    for r in rects:
        _, k_s, e_s = fs.nyquist_stats(engine.crop(sr_lab, r, s)[..., 0])
        _, k_g, e_g = fs.nyquist_stats(engine.crop(gt_lab, r, s)[..., 0])
        vals.append((max(e_s - e_g, 0.0) / p["jnd_nyq_energy"]) ** 2 + (max(k_s - k_g, 0.0) / p["jnd_peak"]) ** 2)
    return float(np.mean(vals)), vals


def rigidity_tiles(lum, rects, s, margin, device=None):
    """Phase-congruency rigidity of each tile of a full-resolution luminance image, computed
    with `margin` px of context (WCSI-2's kernel, batched; GPU with device="cuda:k")."""
    be = get_backend(device)
    padded = be.pad(lum, margin)
    out = [None] * len(rects)
    inner = (slice(None), slice(margin, -margin if margin else None), slice(margin, -margin if margin else None))
    for _, idx in group_rects(rects).items():
        step = max(1, be.batch // 2)
        for b0 in range(0, len(idx), step):
            part = idx[b0:b0 + step]
            g = be.numpy(be.rigidity(be.crop(padded, rects, s, part, margin))[inner]).astype(np.float64)
            for k, t in enumerate(part):
                out[t] = g[k]
    return out


def hybrid_ref(sr, gt, lr, stack=None, scale=None, tile=None, overlap=None, tile_sr=None,
               method_families=(), return_details=False, **params):
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown parameters: {sorted(unknown)}")
    p = {**DEFAULTS, **params}
    stack = stack if isinstance(stack, Stack) else build_stack(stack)
    if lr is None:
        raise ValueError("hybrid_ref needs the LR input (it fixes the scale and the tile grid)")
    s, sr, sr_w, gt, gt_w, lr, lr_w, notes = wcsi2._match_ref(sr, gt, lr, scale)
    if tile_sr is not None:
        if tile_sr % s:
            raise ValueError(f"tile_sr {tile_sr} does not divide scale {s}")
        tile = tile_sr // s
    H0, W0 = lr.shape[:2]
    rects, _, _, layout, rnotes = wcsi2._rects(H0, W0, s, tile, overlap, None, "lr")
    notes += rnotes
    overlap = layout["overlap"]
    N = len(rects)
    E = {k: None for k, v in T.TERMS.items() if v["setting"] == "ref"}

    pix = {}
    if p["with_wcsi2"]:
        w2 = wcsi2.wcsi2_ref_score(sr, gt, lr, scale=s, tile=rects[0][2], overlap=overlap,
                                   device=p["device"], return_details=True)
        t2 = w2["terms"]
        E.update(pix_tone=t2["E_tone"], pix_colour=t2["E_colour"], pix_struct=t2["E_struct"],
                 pix_detail=t2["E_detail"], pix_consist=t2["E_consist"], pix_seam=t2["E_seam"])
        pix.update(wcsi2_ref=w2["wcsi2_ref"], rho_pixel=w2["rho_coherence"])
    if p["with_gmci"]:
        g = gmci.gmci_score(sr, gt, lr, scale=s, tile=rects[0][2], overlap=overlap, return_details=True)
        en = g["energies"]
        E.update(gmci_lr=en["E_lr"], gmci_seam=en["E_seam"], gmci_master=en["E_master"],
                 gmci_frust=en["E_frust"], gmci_emerge=en["E_emerge"])
        pix.update(gmci=g["gmci"], rho_gmci=g["rho_coherence"])
    sr_lab, gt_lab = wcsi.srgb_to_lab(sr, sr_w), wcsi.srgb_to_lab(gt, gt_w)
    E["pix_nyq"], nyq_tile = _pix_nyq(sr_lab, gt_lab, rects, s, p)

    roles = [r for r in ("texture", "latent", "semantic") if stack.has(r)]
    conflict = {r: stack.family(r) for r in roles if stack.family(r) and stack.family(r) in set(method_families)}
    if "latent" in conflict:
        roles.remove("latent")
        notes.append(f"latent encoder {conflict['latent']!r} is the method's own autoencoder: latent terms not scored")
    for r in set(conflict) - {"latent"}:
        notes.append(f"{r} encoder {conflict[r]!r} was used by the method: cross-check with another {r} encoder")
    sr01, gt01 = _rgb01(sr, sr_w), _rgb01(gt, gt_w)
    bases, gate = engine.fit_bases(stack, sr01, gt01, rects, s, roles, p)
    gammas = (rigidity_tiles(gt_lab[..., 0], rects, s, p["rigidity_margin"], p["device"])
              if p["gate"] == "pix" else [None] * N)
    per = [engine.tile_stats(stack, sr01, gt01, r, s, bases, gate, p, g) for r, g in zip(rects, gammas)]

    def tile_mean(role, fields):
        out = []
        for t in per:
            v = [sum(L[f] for f in fields if L.get(f) is not None)
                 for k, L in t["layers"].items() if k[0] == role]
            out.append(float(np.mean(v)) if v else None)
        return out

    wadd_t = tile_mean("texture", ("point", "bw"))
    lat_t = tile_mean("latent", ("point", "bw"))
    pcp_t = tile_mean("texture", ("q", "spec"))
    sem_t = tile_mean("semantic", ("point_all",))
    point_t = tile_mean("texture", ("point_all",))
    for name, vals in (("feat_wadd", wadd_t), ("lat_wadd", lat_t), ("feat_pcp", pcp_t),
                       ("feat_sem", sem_t), ("feat_point", point_t)):
        v = [x for x in vals if x is not None]
        E[name] = float(np.mean(v)) if v else None

    rho = None
    frust_t = np.zeros(N)
    graph = {}
    keys = engine.sheaf_keys(per)
    if N > 1 and keys:
        pos = wcsi._grid_positions(rects)
        I, J, near, far, _ = wcsi._tile_graph(rects, pos)
        dk = engine.semantic_key(per)
        cos = fs.centered_cosine(engine.descriptors(per, dk, "anchor"))
        w_far = fs.keep_top(I, J, fs.affinity(cos[I, J], p["tau_sem"]), far, N, p["far_top"])
        w = np.where(near, 1.0, np.where(far, w_far, 0.0))
        delta = engine.deltas(per, keys)
        edges = engine.build_edges(per, keys, I, J, w, p)
        E["feat_frust"], frust_t = fs.frustration(delta, edges, p["prior_pairs"])
        rho = fs.coherence(delta, edges, p["harmonic_frac"], p["harmonic_cap"], p["rho_max_n"])
        if rho is None and N * delta.shape[1] > p["rho_max_n"]:
            notes.append(f"rho_feature not computed: {N} tiles x {delta.shape[1]} coordinates > rho_max_n")
        graph = dict(descriptor_layer=list(dk), n_near=int(near.sum()), n_far_kept=int((w_far > 0).sum()),
                     sheaf_layers=[list(k) for k in keys], state_dim=int(delta.shape[1]))

    master_t = [sum(x for x in (a, b, c, d) if x is not None) for a, b, c, d in zip(wadd_t, lat_t, pcp_t, sem_t)]
    labels = [_phenotype(master_t[i], frust_t[i], p) for i in range(N)]
    scores = T.score_all(E, "ref")
    out = dict(score=scores["F_hybrid"]["score"], energies=E, configs=scores, rho_feature=rho,
               encoder_conflict=conflict, **pix)
    if not return_details:
        return out
    out.update(
        scale=s, n_tiles=N, gate=p["gate"], encoders=stack.describe(), graph=graph,
        layout=dict(tile_lr=rects[0][2], overlap_lr=overlap, tile_sr=s * rects[0][2]),
        phenotype_counts={k: labels.count(k) for k in wcsi2.PHENOTYPES},
        tiles=[dict(index=i, y0=r[0], x0=r[1], h=r[2], w=r[3], gamma_mean=per[i]["gamma_mean"],
                    feat_wadd=wadd_t[i], lat_wadd=lat_t[i], feat_pcp=pcp_t[i], feat_sem=sem_t[i],
                    pix_nyq=nyq_tile[i], frust=float(frust_t[i]), phenotype=labels[i])
               for i, r in enumerate(rects)],
        notes=notes,
    )
    return out


def _phenotype(master, frust, p):
    if master < p["phenotype_master"]:
        return wcsi2.PHENOTYPES[0] if frust < p["phenotype_frust"] else wcsi2.PHENOTYPES[3]
    return wcsi2.PHENOTYPES[1] if frust < p["phenotype_frust"] else wcsi2.PHENOTYPES[2]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Hybrid pixel + frozen-feature score with a ground truth.")
    ap.add_argument("sr")
    ap.add_argument("gt")
    ap.add_argument("lr")
    ap.add_argument("--scale", type=int)
    ap.add_argument("--tile", type=int, help="analysis tile in LR pixels")
    ap.add_argument("--tile-sr", type=int, help="analysis tile in full-resolution pixels")
    ap.add_argument("--overlap", type=int, help="overlap in LR pixels (default tile/16)")
    ap.add_argument("--encoders", default="gabor", help="encoder config JSON, or 'gabor' (weight-free control)")
    ap.add_argument("--method", help="method name, checked against the config's autoencoder conflicts")
    ap.add_argument("--gate", choices=("pix", "feat", "none", "all"), default="pix")
    ap.add_argument("--gmci", action="store_true", help="also compute the legacy GMCI pixel terms (slow)")
    ap.add_argument("--device", help="WCSI-2 kernels: numpy (default), cuda, cuda:k, torch:cpu")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    cfg = None if args.encoders == "gabor" else json.loads(Path(args.encoders).read_text())
    fam = conflicts(cfg, args.method) if (cfg and args.method) else set()
    d = hybrid_ref(args.sr, args.gt, args.lr, stack=cfg or "gabor", scale=args.scale, tile=args.tile,
                   tile_sr=args.tile_sr, overlap=args.overlap, method_families=fam, return_details=True,
                   gate=args.gate, with_gmci=args.gmci, device=args.device)
    rho = "n/a" if d["rho_feature"] is None else f"{d['rho_feature']:.3f}"
    print(f"hybrid (F) {d['score']:.2f}   rho_feature {rho}   {d['n_tiles']} tiles of {d['layout']['tile_sr']} px")
    for k, v in d["energies"].items():
        print(f"  {k:11s} {'n/a' if v is None else f'{v:9.4f}'}   {T.TERMS[k]['source']}")
    for name, c in d["configs"].items():
        if c["score"] is not None:
            print(f"  {name:22s} {c['score']:7.2f}" + (f"   missing {c['missing']}" if c["missing"] else ""))
    for n in d["notes"]:
        print(f"  note: {n}")
    if args.json:
        Path(args.json).write_text(json.dumps(d, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
