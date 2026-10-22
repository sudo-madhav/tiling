"""Every energy term, the band it owns, and the ablation configurations built from them.

An image is scored once; every configuration is a weighted sum over a subset of
the terms, so all ablation arms come from the same pass. Energies are in
just-noticeable-difference units (1 = one nominal JND), lower is better.

Pixel terms come from WCSI-2 (genet/wcsi2.py). The WCSI v1 and GMCI pixel terms are kept as
legacy arms (`wcsi_*`, `gmci_*`), computed only on request (with_v1 / with_gmci), because
both put the ground truth below generative outputs (see genet/WCSI2.md).
"""

import math

# space: pixel / feature / latent / semantic. band: which part of the signal the term owns.
TERMS = {
    # ---- with reference (ground truth available)
    "pix_tone": dict(setting="ref", space="pixel", band="ultra-low (D_s, low-passed CIELAB)",
                     source="WCSI-2 ref E_tone: coarse colour against the coarse-grained ground truth"),
    "pix_colour": dict(setting="ref", space="pixel", band="all (per-tile CIELAB distribution)",
                       source="WCSI-2 ref E_colour: sliced Wasserstein of the tile colour clouds"),
    "pix_struct": dict(setting="ref", space="pixel", band="phase-locked structure",
                       source="WCSI-2 ref E_struct: gradient error weighted by ground-truth phase congruency"),
    "pix_detail": dict(setting="ref", space="pixel", band="invented band [1/(2s), 1/2]",
                       source="WCSI-2 ref E_detail: per-tile band energies, orientation, chroma, peak vs the GT"),
    "pix_consist": dict(setting="ref", space="pixel", band="non-local, invented band",
                        source="WCSI-2 ref E_consist: tiles of one material departing differently"),
    "pix_seam": dict(setting="ref", space="pixel", band="1-px seam step and gradient",
                     source="WCSI-2 ref E_seam: seam steps beyond the ground truth's own"),
    "pix_nyq": dict(setting="ref", space="pixel", band="Nyquist octave [0.25, 0.5] cycles/px",
                    source="excess top-octave energy and aliasing peaks over the ground truth (one-sided)"),
    "gmci_lr": dict(setting="ref", space="pixel", band="ultra-low", source="legacy GMCI E_lr"),
    "gmci_seam": dict(setting="ref", space="pixel", band="1-px seam", source="legacy GMCI E_seam"),
    "gmci_master": dict(setting="ref", space="pixel", band="all", source="legacy GMCI E_master"),
    "gmci_frust": dict(setting="ref", space="pixel", band="non-local", source="legacy GMCI E_frust"),
    "gmci_emerge": dict(setting="ref", space="pixel", band="cross-scale phase", source="legacy GMCI E_emerge"),
    "feat_point": dict(setting="ref", space="feature", band="mid (2x-8x)",
                       source="ungated pointwise feature distance (an unweighted LPIPS analogue)"),
    "feat_wadd": dict(setting="ref", space="feature", band="mid (2x-8x)",
                      source="rigidity-gated pointwise + Bures-Wasserstein per texture layer"),
    "lat_wadd": dict(setting="ref", space="latent", band="mid (8x)",
                     source="rigidity-gated pointwise + Bures-Wasserstein on the VAE latent"),
    "feat_pcp": dict(setting="ref", space="feature", band="mid (2x-8x)",
                     source="feature nematic tensor + feature power-spectrum W1"),
    "feat_sem": dict(setting="ref", space="semantic", band="semantic (14 px tokens)",
                     source="token distance in the DINOv2 PCA basis"),
    "feat_frust": dict(setting="ref", space="feature", band="non-local",
                       source="feature sheaf frustration, DINOv2 graph, Gaussian-OT transport"),
    # ---- without reference
    "nr_low": dict(setting="noref", space="pixel", band="ultra-low",
                   source="WCSI-2 E_low: coarse colour against the LR, minus the natural floor"),
    "nr_seam": dict(setting="noref", space="pixel", band="1-px seam step and gradient",
                    source="WCSI-2 E_seam: seam log-ratio beyond the LR's own, minus the natural floor"),
    "nr_detail": dict(setting="noref", space="pixel", band="invented band [1/(2s), 1/2]",
                      source="WCSI-2 E_detail: per-tile invented detail against the calibrated natural target"),
    "nr_consist": dict(setting="noref", space="pixel", band="non-local, invented band",
                       source="WCSI-2 E_consist: same-material tiles departing differently, beyond natural"),
    "nr_nyq": dict(setting="noref", space="pixel", band="Nyquist aliasing peaks",
                   source="aliasing-peak ratio above the calibrated natural floor"),
    "nr_feat_detail": dict(setting="noref", space="feature", band="mid (2x-8x)",
                           source="per-layer feature energy of each tile against the natural upgrade "
                                  "predicted from the bicubic anchor (calibrated)"),
    "nr_feat_consist": dict(setting="noref", space="feature", band="non-local",
                            source="feature sheaf frustration of the SR-minus-bicubic upgrade, "
                                   "beyond the natural level (calibrated)"),
    "wcsi_rg": dict(setting="noref", space="pixel", band="ultra-low", source="legacy WCSI v1 E_RG"),
    "wcsi_seam": dict(setting="noref", space="pixel", band="1-px seam", source="legacy WCSI v1 E_GJ"),
    "wcsi_pcp": dict(setting="noref", space="pixel", band="neighbour stroke angle", source="legacy WCSI v1 E_PCP"),
    "wcsi_cyto": dict(setting="noref", space="pixel", band="non-local texture", source="legacy WCSI v1 E_cyto"),
    "wcsi_sheaf": dict(setting="noref", space="pixel", band="non-local", source="legacy WCSI v1 E_sheaf"),
    "wcsi_cyto_dino": dict(setting="noref", space="pixel", band="non-local texture",
                           source="legacy WCSI v1 E_cyto divergences re-weighted by the DINOv2 graph"),
}

PIX_REF = ["pix_tone", "pix_colour", "pix_struct", "pix_detail", "pix_consist", "pix_seam"]
FEAT_REF = ["feat_wadd", "lat_wadd", "feat_pcp", "feat_sem", "feat_frust"]

CONFIGS_REF = {
    "B_pointwise_feature": ["feat_point"],
    "C_pixel_wcsi2": PIX_REF,
    "C_pixel_gmci": ["gmci_master", "gmci_frust", "gmci_seam", "gmci_emerge", "gmci_lr"],
    "D_feature_latent": ["feat_wadd", "lat_wadd", "feat_pcp"],
    "E_feature_sheaf": FEAT_REF,
    "F_hybrid": ["pix_tone", "pix_seam", "pix_nyq"] + FEAT_REF,
    "F_union": PIX_REF + ["pix_nyq"] + FEAT_REF,
    "F_minus_pixel_low": ["pix_seam", "pix_nyq"] + FEAT_REF,
    "F_minus_pixel_seam": ["pix_tone", "pix_nyq"] + FEAT_REF,
    "F_minus_pixel_nyq": ["pix_tone", "pix_seam"] + FEAT_REF,
    "F_minus_latent": ["pix_tone", "pix_seam", "pix_nyq", "feat_wadd", "feat_pcp", "feat_sem", "feat_frust"],
    "F_minus_sheaf": ["pix_tone", "pix_seam", "pix_nyq", "feat_wadd", "lat_wadd", "feat_pcp", "feat_sem"],
}

CONFIGS_NOREF = {
    "N0_wcsi": ["wcsi_rg", "wcsi_seam", "wcsi_pcp", "wcsi_cyto", "wcsi_sheaf"],
    "N1_wcsi_dino_graph": ["wcsi_rg", "wcsi_seam", "wcsi_pcp", "wcsi_cyto_dino", "wcsi_sheaf"],
    "N0_wcsi2": ["nr_low", "nr_seam", "nr_detail", "nr_consist"],
    "N2_feature": ["nr_feat_detail", "nr_feat_consist"],
    "N3_hybrid": ["nr_low", "nr_seam", "nr_detail", "nr_feat_detail", "nr_feat_consist"],
    "N4_union": ["nr_low", "nr_seam", "nr_detail", "nr_consist", "nr_nyq", "nr_feat_detail", "nr_feat_consist"],
}


def combine(energies, terms, alphas=None, kind="ref"):
    """Score of one configuration. Missing terms (None) are left out and listed; a
    configuration with no term computed has score None.

    kind="ref": 100 exp(-F / sum alpha), as GMCI. kind="noref": 100 / (1 + F / sum alpha), as WCSI.
    """
    alphas = alphas or {}
    used, missing = [], []
    for t in terms:
        v = energies.get(t)
        (missing if v is None else used).append(t)
    if not used:
        return dict(score=None, F=None, F_norm=None, used=used, missing=missing)
    F = sum(alphas.get(t, 1.0) * energies[t] for t in used)
    norm = sum(alphas.get(t, 1.0) for t in used)
    if norm <= 0:
        return dict(score=100.0, F=0.0, F_norm=0.0, used=used, missing=missing)
    x = F / norm
    score = 100.0 * math.exp(-min(x, 60.0)) if kind == "ref" else 100.0 / (1.0 + x)
    return dict(score=score, F=F, F_norm=x, used=used, missing=missing)


def score_all(energies, kind="ref", alphas=None):
    configs = CONFIGS_REF if kind == "ref" else CONFIGS_NOREF
    return {name: combine(energies, terms, alphas, kind) for name, terms in configs.items()}
