"""Per-tile feature statistics of a test image against an anchor image, shared by both scores.

anchor = ground truth (with reference) or bicubic-upsampled LR (without reference).
Two passes over the encoder stack:
  1. fit one PCA basis per layer on tokens from up to `basis_tiles` evenly spaced tiles
     of anchor and test (so the basis spans both) and collect gate statistics;
  2. encode every tile of both images once and keep only small statistics.
Full feature maps are never held for more than one tile.
"""

import math

import numpy as np

import featstats as fs

SHEAF_ROLES = ("texture", "latent")


def crop(img, rect, s):
    y0, x0, h, w = rect
    return img[s * y0:s * (y0 + h), s * x0:s * (x0 + w)]


def _even(n, m):
    if n <= m:
        return list(range(n))
    return sorted({int(round(v)) for v in np.linspace(0, n - 1, m)})


def _grad_mag(fk):
    g = np.zeros(fk.shape[:2])
    for c in range(fk.shape[-1]):
        gy, gx = np.gradient(fk[..., c])
        g += gx ** 2 + gy ** 2
    return np.sqrt(g)


def _log_energy(fmap):
    X = fs.tokens(fmap)
    return float(np.log(np.mean(np.sum((X - X.mean(0)) ** 2, 1)) + 1e-12))


def _gate_layer(keys):
    lat = [k for k in keys if k[0] == "latent"]
    return lat[0] if lat else min((k for k in keys if k[0] == "texture"), key=lambda k: k[2], default=None)


def fit_bases(stack, test, anchor, rects, s, roles, p):
    idx = _even(len(rects), p["basis_tiles"])
    per_tile = max(256, p["max_basis_tokens"] // (2 * len(idx)))
    blocks, strides = {}, {}
    for t in idx:
        for img in (anchor, test):
            enc = stack.encode(crop(img, rects[t], s), roles)
            for role, maps in enc.items():
                for name, f in maps.items():
                    key = (role, name)
                    strides[key] = f.stride
                    blocks.setdefault(key, []).append(fs.subsample(fs.tokens(f.data), per_tile, seed=t))
    bases = {}
    for key, b in blocks.items():
        B = fs.fit_basis(b, p["k_pca"])
        B["stride"] = strides[key]
        bases[(key[0], key[1], strides[key])] = B
    gate = None
    gk = _gate_layer(list(bases))
    if p["gate"] == "feat" and gk is not None:
        mags = []
        for t in idx:
            f = stack.encode(crop(anchor, rects[t], s), (gk[0],))[gk[0]][gk[1]]
            mags.append(_grad_mag(fs.project(f.data, bases[gk])).ravel())
        m = np.concatenate(mags)
        gate = dict(key=gk, mu=float(m.mean()), sd=float(m.std() + 1e-12))
    return bases, gate


def _gamma_full(kind, shape, gamma_pix, gate, anchor_maps, bases):
    if kind == "pix":
        return gamma_pix
    if kind == "none":
        return np.zeros(shape)
    if kind == "all":
        return np.ones(shape)
    if kind == "feat":
        if gate is None:
            return np.zeros(shape)
        role, name, stride = gate["key"]
        fk = fs.project(anchor_maps[role][name].data, bases[gate["key"]])
        z = (_grad_mag(fk) - gate["mu"]) / gate["sd"]
        g = 1.0 / (1.0 + np.exp(-z))
        g = np.repeat(np.repeat(g, stride, 0), stride, 1)
        out = np.zeros(shape)
        h, w = min(shape[0], g.shape[0]), min(shape[1], g.shape[1])
        out[:h, :w] = g[:h, :w]
        return out
    raise ValueError(f"unknown gate {kind!r}")


def tile_stats(stack, test, anchor, rect, s, bases, gate, p, gamma_pix=None):
    """Statistics of one tile. gamma_pix: full-resolution rigidity of the anchor tile (gate="pix")."""
    roles = sorted({k[0] for k in bases})
    T = crop(test, rect, s)
    A = crop(anchor, rect, s)
    et, ea = stack.encode(T, roles), stack.encode(A, roles)
    shape = A.shape[:2]
    if gamma_pix is None:
        gamma_pix = np.zeros(shape)
    gamma = _gamma_full(p["gate"], shape, gamma_pix, gate, ea, bases)
    rec = dict(layers={}, gamma_mean=float(np.mean(gamma)))
    kr = p["k_rho"]
    jq = 2 * math.sin(math.radians(p["jnd_angle_deg"]))
    for key, B in bases.items():
        role, name, stride = key
        ft, fa = fs.project(et[role][name].data, B), fs.project(ea[role][name].data, B)
        unit = p["jnd_feat"] ** 2 * B["trace"]
        L = dict(role=role, stride=stride)
        d2 = np.sum((ft - fa) ** 2, -1)
        L["point_all"] = float(d2.mean()) / unit
        g = fs.pool_to(gamma, d2.shape, stride)
        L["point"] = float(np.mean(g * d2)) / unit
        (mt, St), (ma, Sa) = fs.gauss(ft), fs.gauss(fa)
        L["bw"] = fs.bures_sq(mt, St, ma, Sa) / unit
        if role == "semantic":
            L["desc_t"], L["desc_a"] = ft.reshape(-1, B["k"]).mean(0), fa.reshape(-1, B["k"]).mean(0)
        else:
            # log feature energy in the encoder's own channels: unlike the PCA coordinates it
            # means the same thing in every image, so a natural model can be fitted across images
            L["loge_t"], L["loge_a"] = _log_energy(et[role][name].data), _log_energy(ea[role][name].data)
            Qt, _ = fs.feature_nematic(ft, p["n_q"])
            Qa, _ = fs.feature_nematic(fa, p["n_q"])
            L["q"] = float(np.sum((Qt - Qa) ** 2)) / jq ** 2
            sp = fs.feature_spectrum_w1(ft, fa, stride / (2.0 * s), p["n_proj_spectrum"])
            L["spec"] = None if sp is None else (sp[0] / max(p["jnd_freq"] * sp[1], 1e-3)) ** 2
            k = min(kr, B["k"])
            a = p["jnd_feat"] * np.sqrt(np.maximum(B["var"][:k], 1e-12))
            aa = np.concatenate([a, a])
            xt, xa = ft[..., :k].reshape(-1, k), fa[..., :k].reshape(-1, k)
            neff = max(2.0, p["neff_frac"] * len(xa))
            L["psi_t"] = np.concatenate([xt.mean(0), xt.std(0)]) / aa
            L["psi_a"] = np.concatenate([xa.mean(0), xa.std(0)]) / aa
            # sampling variance of psi_t - psi_a, in JND^2: mean ~ sd/sqrt(n), sd ~ sd/sqrt(2n)
            sd2 = (xt.std(0) ** 2 + xa.std(0) ** 2) / neff
            L["var_delta"] = np.concatenate([sd2, sd2 / 2]) / aa ** 2
            L["cov_a"] = fs.shrunk_cov(xa)
            L["std_a"] = xa.std(0)
            L["axis"] = a
            L["neff"] = neff
        rec["layers"][key] = L
    return rec


def sheaf_keys(per, exclude_roles=()):
    keys = [k for k in per[0]["layers"] if k[0] in SHEAF_ROLES and k[0] not in exclude_roles]
    return sorted(keys, key=lambda k: (k[0], k[2], k[1]))


def deltas(per, keys):
    return np.stack([np.concatenate([t["layers"][k]["psi_t"] - t["layers"][k]["psi_a"] for k in keys])
                     for t in per])


def transport(Li, Lj, p):
    """Map of a departure in tile i to the matching departure in tile j, in JND-normalised axes:
    the Gaussian optimal transport between the anchor tiles on the means, the ratio of anchor
    spreads on the standard deviations."""
    a = Li["axis"]
    k = len(a)
    eps = float(np.mean(a ** 2))
    T = fs.monge_linear(Li["cov_a"], Lj["cov_a"], eps)
    Tn = (T * a[None, :]) / a[:, None]
    # Positive-part James-Stein shrinkage towards the identity: between two samples of one
    # Gaussian, ||T - I||_F^2 is about k^2 / (2 n_eff) from sampling alone, and an unresolved
    # transport multiplied by a large departure would read as frustration.
    noise = k * k / (2.0 * min(Li["neff"], Lj["neff"]))
    dev = float(np.sum((Tn - np.eye(k)) ** 2))
    kappa = max(0.0, 1.0 - noise / dev) if dev > 0 else 0.0
    Tn = np.eye(k) + kappa * (Tn - np.eye(k))
    ratio = np.clip((Lj["std_a"] + 1e-12) / (Li["std_a"] + 1e-12), 0.25, 4.0)
    ratio = 1.0 + kappa * (ratio - 1.0)
    A = np.zeros((2 * k, 2 * k))
    A[:k, :k] = Tn
    A[k:, k:] = np.diag(ratio)
    return A


def build_edges(per, keys, I, J, w, p):
    """Edges (i, j, A, S, w). The residual S (A d_i - d_j) is in JND units, widened by the
    sampling error of both tiles' statistics (as WCSI widens its spectral terms)."""
    edges = []
    for e in np.flatnonzero(w > 0):
        i, j = int(I[e]), int(J[e])
        blocks = [transport(per[i]["layers"][k], per[j]["layers"][k], p) for k in keys]
        var = np.concatenate([per[i]["layers"][k]["var_delta"] + per[j]["layers"][k]["var_delta"] for k in keys])
        d = sum(b.shape[0] for b in blocks)
        A = np.zeros((d, d))
        o = 0
        for b in blocks:
            A[o:o + b.shape[0], o:o + b.shape[0]] = b
            o += b.shape[0]
        edges.append((i, j, A, 1.0 / np.sqrt(1.0 + var), float(w[e])))
    return edges


def semantic_key(per):
    sem = [k for k in per[0]["layers"] if k[0] == "semantic"]
    if sem:
        return sorted(sem, key=lambda k: k[2])[-1]
    tex = [k for k in per[0]["layers"] if k[0] in SHEAF_ROLES]
    return max(tex, key=lambda k: k[2]) if tex else None


def descriptors(per, key, which):
    """Pooled per-tile descriptors from the semantic layer, or the coarsest texture layer."""
    out = []
    for t in per:
        L = t["layers"][key]
        if "desc_a" in L:
            out.append(L["desc_a" if which == "anchor" else "desc_t"])
        else:
            out.append(L["psi_a" if which == "anchor" else "psi_t"])
    return np.stack(out)


def layer_mean(per, roles, field, keys=None):
    vals = []
    for t in per:
        v = [L[field] for k, L in t["layers"].items()
             if k[0] in roles and (keys is None or k in keys) and L.get(field) is not None]
        if v:
            vals.append(float(np.mean(v)))
    return (float(np.mean(vals)) if vals else None), vals
