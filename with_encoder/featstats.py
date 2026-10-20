"""Statistics of frozen feature maps, shared by the reference and no-reference scores.

All distances are computed in a per-image, per-layer PCA basis fitted on the
anchor image (ground truth with a reference, bicubic-upsampled LR without), so
the tiles of one image share coordinates and the cost does not depend on the
channel count of the backbone.
"""

import math
import sys
from pathlib import Path

import numpy as np
from scipy import fft as sfft
from scipy import ndimage

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "genet"))
sys.path.insert(0, str(_ROOT / "with_ref_genet"))
import gmci  # noqa: E402
import wcsi  # noqa: E402


# ---------------------------------------------------------------- basis and Gaussians

def tokens(fmap):
    a = np.asarray(fmap, np.float64)
    return a.reshape(-1, a.shape[-1])


def subsample(X, n, seed=0):
    if len(X) <= n:
        return X
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    return X[np.sort(idx)]


def fit_basis(blocks, k):
    """PCA of the concatenated token blocks: mean (C,), basis (C, k), eigenvalues (k,), trace."""
    X = np.concatenate([tokens(b) for b in blocks], 0)
    mean = X.mean(0)
    Xc = X - mean
    cov = Xc.T @ Xc / max(len(X) - 1, 1)
    w, V = np.linalg.eigh(0.5 * (cov + cov.T))
    k = int(min(k, len(w)))
    order = np.argsort(w)[::-1][:k]
    var = np.clip(w[order], 0, None)
    return dict(mean=mean, basis=V[:, order], var=var, trace=float(var.sum()) + 1e-12, k=k)


def project(fmap, B):
    a = np.asarray(fmap, np.float64)
    return (a - B["mean"]) @ B["basis"]


def shrunk_cov(X):
    """Ledoit-Wolf shrinkage towards a scaled identity. X: (n, d) tokens."""
    n, d = X.shape
    if n < 2:
        return np.eye(d) * 1e-12
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / n
    mu = float(np.trace(S)) / d
    d2 = float(np.sum((S - mu * np.eye(d)) ** 2))
    if d2 <= 1e-30:
        return S
    b2 = (float(np.sum(np.sum(Xc ** 2, 1) ** 2)) - n * float(np.sum(S ** 2))) / n ** 2
    shrink = min(max(b2, 0.0), d2) / d2
    return shrink * mu * np.eye(d) + (1 - shrink) * S


def gauss(fk):
    X = tokens(fk)
    return X.mean(0), shrunk_cov(X)


def bures_sq(m1, S1, m2, S2):
    return float(wcsi.bures_w2(m1, S1, m2, S2))


def monge_linear(Si, Sj, eps):
    """Linear part of the optimal transport map from N(., Si) to N(., Sj)."""
    d = Si.shape[0]
    Si = Si + eps * np.eye(d)
    Sj = Sj + eps * np.eye(d)
    w, V = np.linalg.eigh(0.5 * (Si + Si.T))
    w = np.clip(w, eps, None)
    rh = (V * np.sqrt(w)) @ V.T
    irh = (V / np.sqrt(w)) @ V.T
    M = wcsi._sqrtm_psd(rh @ Sj @ rh)
    return irh @ M @ irh


# ---------------------------------------------------------------- geometry of a feature map

def feature_nematic(fk, n_ch):
    """Q = order (cos 2theta, sin 2theta) of the structure tensor summed over the first n_ch channels."""
    f = np.asarray(fk, np.float64)[..., :n_ch]
    if min(f.shape[:2]) < 3:
        return np.zeros(2), 0.0
    jxx = jxy = jyy = 0.0
    for c in range(f.shape[-1]):
        gx = ndimage.correlate(f[..., c], wcsi._SCHARR_X.astype(np.float64), mode="reflect")
        gy = ndimage.correlate(f[..., c], wcsi._SCHARR_Y.astype(np.float64), mode="reflect")
        jxx += float(np.mean(gx * gx))
        jxy += float(np.mean(gx * gy))
        jyy += float(np.mean(gy * gy))
    theta = (0.5 * math.atan2(2 * jxy, jxx - jyy) + math.pi / 2) % math.pi
    order = (math.hypot(jxx - jyy, 2 * jxy) / (jxx + jyy + 1e-12)) ** 2
    return np.array([order * math.cos(2 * theta), order * math.sin(2 * theta)]), theta


def _spectrum_cloud(fk, f_lo):
    f = np.asarray(fk, np.float64)
    h, w = f.shape[:2]
    win = np.outer(np.hanning(h), np.hanning(w))
    P = np.zeros((h, w))
    for c in range(f.shape[-1]):
        F = sfft.fft2((f[..., c] - f[..., c].mean()) * win)
        P += F.real ** 2 + F.imag ** 2
    kx, ky = wcsi._freqs(h, w)
    fr = np.sqrt(kx ** 2 + ky ** 2)
    mask = (fr >= f_lo) & (fr <= 0.5) & (ky >= 0)
    pts = np.stack([np.broadcast_to(kx, P.shape)[mask], np.broadcast_to(ky, P.shape)[mask]], 1)
    wt = P[mask]
    if wt.sum() <= 0 or len(pts) < 4:
        return None
    wt = wt / wt.sum()
    return pts, wt, float(np.sqrt(np.sum(pts ** 2, 1)) @ wt)


def feature_spectrum_w1(fk, fk_star, f_lo, n_proj, seed=0):
    """Sliced W1 between band-limited power clouds of two feature maps, and the reference mean |k|.

    f_lo is in cycles per feature pixel. Returns None when the band is empty on this grid."""
    if f_lo >= 0.45 or min(np.shape(fk)[:2]) < 8:
        return None
    a, b = _spectrum_cloud(fk, f_lo), _spectrum_cloud(fk_star, f_lo)
    if a is None or b is None:
        return None
    rng = np.random.default_rng(seed)
    acc = 0.0
    for _ in range(n_proj):
        t = rng.random() * math.pi
        u = np.array([math.cos(t), math.sin(t)])
        acc += gmci.w1_1d(a[0] @ u, a[1], b[0] @ u, b[1])
    return acc / n_proj, b[2]


def pool_to(mask, shape, stride):
    """Average a full-resolution map onto a feature grid of `shape` with `stride`."""
    h, w = shape
    m = np.asarray(mask, np.float64)
    ph, pw = h * stride - m.shape[0], w * stride - m.shape[1]
    if ph > 0 or pw > 0:
        m = np.pad(m, ((0, max(ph, 0)), (0, max(pw, 0))), mode="edge")
    return m[:h * stride, :w * stride].reshape(h, stride, w, stride).mean((1, 3))


# ---------------------------------------------------------------- pixel Nyquist probe

def nyquist_stats(lum):
    """Top-octave statistics of one luminance tile: (slope, aliasing peak, log energy).

    slope: least-squares 1/f^alpha fit over rings in [0.25, 0.5] cycles/px (diagnostic only:
    ill-conditioned when the octave is nearly empty, as after bicubic upsampling).
    peak: log of the mean power within 0.02 cycles/px of the aliasing points (1/2, 0), (0, 1/2),
    (1/2, 1/2) over the median power of the annulus [0.4, 0.5]. A checkerboard or 2-px ringing
    raises it; natural images keep it near zero.
    log energy: log of the mean power in the octave.
    """
    lum = np.asarray(lum, np.float64)
    h, w = lum.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    F = sfft.fft2((lum - lum.mean()) * win)
    P = F.real ** 2 + F.imag ** 2 + 1e-20
    kx, ky = wcsi._freqs(h, w)
    kx, ky = np.broadcast_to(kx, P.shape), np.broadcast_to(ky, P.shape)
    f = np.sqrt(kx ** 2 + ky ** 2)
    top = (f >= 0.25) & (f <= 0.5)
    ring = np.floor(f[top] * min(h, w)).astype(np.int64)
    cnt = np.bincount(ring)
    ps, fs = np.bincount(ring, P[top]), np.bincount(ring, f[top])
    ok = cnt > 0
    alpha = 0.0
    if ok.sum() >= 3:
        alpha = -float(np.polyfit(np.log(fs[ok] / cnt[ok]), np.log(ps[ok] / cnt[ok]), 1)[0])
    ax, ay = np.abs(kx), np.abs(ky)
    near = np.zeros(P.shape, bool)
    for cx, cy in ((0.5, 0.0), (0.0, 0.5), (0.5, 0.5)):
        near |= np.hypot(ax - cx, ay - cy) <= 0.02
    ann = (f >= 0.4) & (f <= 0.5) & ~near
    peak = 0.0
    if near.any() and ann.any():
        peak = float(np.log(P[near].mean() / np.median(P[ann])))
    return alpha, peak, float(np.log(P[top].mean()))


# ---------------------------------------------------------------- graph and sheaf

def centered_cosine(desc):
    """Cosine similarity after removing the image-mean descriptor (DINO features share a
    large common component, so raw cosines between unrelated tiles are high)."""
    D = np.asarray(desc, np.float64)
    D = D - D.mean(0)
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-12
    return D @ D.T


def affinity(cos, tau):
    return np.clip((cos - tau) / max(1.0 - tau, 1e-6), 0.0, 1.0)


def keep_top(I, J, w, mask, n_tiles, m):
    """Zero all but each tile's m strongest edges among `mask` (an edge survives if either end keeps it)."""
    keep = np.zeros(len(I), bool)
    idx = np.flatnonzero(mask & (w > 0))
    for t in range(n_tiles):
        mine = idx[(I[idx] == t) | (J[idx] == t)]
        keep[mine[np.argsort(-w[mine])[:m]]] = True
    return np.where(mask & ~keep, 0.0, w)


def frustration(delta, edges, prior):
    """sum_e w ||S (A d_i - d_j)||^2 / (sum_e w + prior), per-edge values, per-tile means."""
    N = len(delta)
    per_edge, ws = [], []
    tile_sum, deg = np.zeros(N), np.zeros(N)
    for i, j, A, S, w in edges:
        r = S * (A @ delta[i] - delta[j])
        v = float(r @ r)
        per_edge.append(v)
        ws.append(w)
        tile_sum[[i, j]] += w * v
        deg[[i, j]] += w
    if not edges or sum(ws) <= 0:
        return None, np.zeros(N)
    E = float(np.dot(ws, per_edge) / (sum(ws) + prior))
    return E, tile_sum / np.maximum(deg, 1e-12)


def coherence(delta, edges, harmonic_frac, harmonic_cap, max_n):
    """Fraction of ||delta||^2 (in S units) lying in the kernel and lowest modes of the sheaf
    Laplacian built from `edges`. None when there is no departure or the system is too large."""
    N, d = delta.shape
    n = N * d
    if N < 2 or not edges or n > max_n:
        return None
    S0 = np.mean([e[3] for e in edges], 0)
    x = (delta * S0).ravel()
    if float(x @ x) < 1e-12:
        return None
    L = np.zeros((n, n))
    for i, j, A, S, w in edges:
        # in S0 units: residual S (A d_i - d_j) with d = x / S0
        SA = (S[:, None] * A) / S0[None, :]
        Sj = S / S0
        ii, jj = slice(i * d, (i + 1) * d), slice(j * d, (j + 1) * d)
        L[ii, ii] += w * (SA.T @ SA)
        L[jj, jj] += w * np.diag(Sj ** 2)
        L[ii, jj] -= w * (SA.T * Sj)
        L[jj, ii] -= w * (Sj[:, None] * SA)
    evals, evecs = np.linalg.eigh(0.5 * (L + L.T))
    lam_max = float(max(evals[-1], 1e-12))
    kernel = np.flatnonzero(evals <= 1e-6 * lam_max + 1e-8)
    smooth = [m for m in np.flatnonzero(evals <= harmonic_frac * lam_max) if m not in set(kernel.tolist())]
    keep = np.concatenate([kernel, np.asarray(smooth[:harmonic_cap], np.int64)]).astype(np.int64)
    if not len(keep):
        keep = np.array([0])
    proj = evecs[:, keep].T @ x
    return float(proj @ proj / (x @ x))
