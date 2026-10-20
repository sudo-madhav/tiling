"""Every energy term, the band it owns, and the ablation configurations built from them.

An image is scored once; every configuration is a weighted sum over a subset of
the terms, so all ablation arms come from the same pass. Energies are in
just-noticeable-difference units (1 = one nominal JND), lower is better.
"""

import math

# space: pixel / feature / latent / semantic. band: which part of the signal the term owns.
TERMS = {
    # ---- with reference (ground truth available)
    "pix_lr": dict(setting="ref", space="pixel", band="ultra-low (D_s block means, CIELAB)",
                   source="GMCI E_lr: break of the LR input beyond the ground truth's own"),
    "pix_seam": dict(setting="ref", space="pixel", band="1-px seam step and gradient",
                     source="GMCI E_seam: seams beyond the ground truth's own edges"),
    "pix_nyq": dict(setting="ref", space="pixel", band="Nyquist octave [0.25, 0.5] cycles/px",
                    source="excess top-octave energy and aliasing peaks over the ground truth (one-sided)"),
    "pix_master": dict(setting="ref", space="pixel", band="all (colour transport, Frank, spectrum)",
                       source="GMCI E_master"),
    "pix_frust": dict(setting="ref", space="pixel", band="non-local, 14-D pixel state",
                      source="GMCI E_frust"),
    "pix_emerge": dict(setting="ref", space="pixel", band="cross-scale phase coupling",
                       source="GMCI E_emerge"),
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
    "nr_rg": dict(setting="noref", space="pixel", band="ultra-low", source="WCSI E_RG"),
    "nr_seam": dict(setting="noref", space="pixel", band="1-px seam step and gradient", source="WCSI E_GJ"),
    "nr_pcp": dict(setting="noref", space="pixel", band="neighbour stroke angle", source="WCSI E_PCP"),
    "nr_cyto": dict(setting="noref", space="pixel", band="non-local texture", source="WCSI E_cyto"),
    "nr_sheaf": dict(setting="noref", space="pixel", band="non-local, 14-D pixel state", source="WCSI E_sheaf"),
    "nr_cyto_dino": dict(setting="noref", space="pixel", band="non-local texture",
                         source="WCSI E_cyto divergences re-weighted by the DINOv2 graph"),
    "nr_nyq": dict(setting="noref", space="pixel", band="Nyquist aliasing peaks",
                   source="aliasing-peak ratio above a natural-image floor"),
    "nr_feat_sheaf": dict(setting="noref", space="feature", band="non-local",
                          source="feature sheaf frustration of the SR-minus-bicubic upgrade"),
}

CONFIGS_REF = {
    "B_pointwise_feature": ["feat_point"],
    "C_pixel_morphogen": ["pix_master", "pix_frust", "pix_seam", "pix_emerge", "pix_lr"],
    "D_feature_latent": ["feat_wadd", "lat_wadd", "feat_pcp"],
    "E_feature_sheaf": ["feat_wadd", "lat_wadd", "feat_pcp", "feat_sem", "feat_frust"],
    "F_hybrid": ["pix_lr", "pix_seam", "pix_nyq", "feat_wadd", "lat_wadd", "feat_pcp", "feat_sem", "feat_frust"],
    "F_union": ["pix_lr", "pix_seam", "pix_nyq", "pix_master", "pix_frust", "pix_emerge",
                "feat_wadd", "lat_wadd", "feat_pcp", "feat_sem", "feat_frust"],
    "F_minus_pixel_low": ["pix_seam", "pix_nyq", "feat_wadd", "lat_wadd", "feat_pcp", "feat_sem", "feat_frust"],
    "F_minus_pixel_seam": ["pix_lr", "pix_nyq", "feat_wadd", "lat_wadd", "feat_pcp", "feat_sem", "feat_frust"],
    "F_minus_pixel_nyq": ["pix_lr", "pix_seam", "feat_wadd", "lat_wadd", "feat_pcp", "feat_sem", "feat_frust"],
    "F_minus_latent": ["pix_lr", "pix_seam", "pix_nyq", "feat_wadd", "feat_pcp", "feat_sem", "feat_frust"],
    "F_minus_sheaf": ["pix_lr", "pix_seam", "pix_nyq", "feat_wadd", "lat_wadd", "feat_pcp", "feat_sem"],
}

CONFIGS_NOREF = {
    "N0_wcsi": ["nr_rg", "nr_seam", "nr_pcp", "nr_cyto", "nr_sheaf"],
    "N1_wcsi_dino_graph": ["nr_rg", "nr_seam", "nr_pcp", "nr_cyto_dino", "nr_sheaf"],
    "N2_feature_sheaf": ["nr_feat_sheaf"],
    "N3_hybrid": ["nr_rg", "nr_seam", "nr_nyq", "nr_feat_sheaf"],
    "N4_union": ["nr_rg", "nr_seam", "nr_pcp", "nr_cyto_dino", "nr_sheaf", "nr_nyq", "nr_feat_sheaf"],
}


def combine(energies, terms, alphas=None, kind="ref"):
    """Score of one configuration. Missing terms (None) are left out and listed.

    kind="ref": 100 exp(-F / sum alpha), as GMCI. kind="noref": 100 / (1 + F / sum alpha), as WCSI.
    """
    alphas = alphas or {}
    used, missing = [], []
    for t in terms:
        v = energies.get(t)
        (missing if v is None else used).append(t)
    F = sum(alphas.get(t, 1.0) * energies[t] for t in used)
    norm = sum(alphas.get(t, 1.0) for t in used)
    if norm <= 0:
        return dict(score=100.0, F=0.0, used=used, missing=missing)
    x = F / norm
    score = 100.0 * math.exp(-min(x, 60.0)) if kind == "ref" else 100.0 / (1.0 + x)
    return dict(score=score, F=F, F_norm=x, used=used, missing=missing)


def score_all(energies, kind="ref", alphas=None):
    configs = CONFIGS_REF if kind == "ref" else CONFIGS_NOREF
    return {name: combine(energies, terms, alphas, kind) for name, terms in configs.items()}
