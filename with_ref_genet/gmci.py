#!/usr/bin/env python3
"""GMCI: Ground-truth Morphogenetic & Cohomological Index.

Scores a super-resolved image against the low-resolution input AND the
ground-truth high-resolution image. WCSI never sees the ground truth. GMCI
does, and it uses that to separate three different questions:

  - master fidelity: is this tile the same kind of painting as the ground truth
    (colour transport, stroke elasticity, spectral rhythm), with pixel position
    enforced only where the ground truth itself is phase-locked
  - sheaf frustration: where the tiles depart from the ground truth, do they
    depart together
  - the low-resolution anchor: does the super-resolved image break the input
    by more than the ground truth already does

    python with_ref_genet/gmci.py SR.png GT.png LR.png --tile 64 --scale 8

    from gmci import gmci_score
    out = gmci_score("sr.png", "gt.png", "lr.png", scale=8, tile=64, return_details=True)

The score is in (0, 100]. It is 100 when every energy is zero, and 100/e
(about 36.8) when the energies that exist average one just-noticeable
difference. rho_coherence is the fraction of the departure from the ground
truth that lies in the smooth modes of the ground-truth sheaf; it is undefined
when the super-resolved image matches the ground truth, because there is no
departure to classify.

See PROTOCOL.md for every place the implementation makes the specification
computable. genet/wcsi.py is used and not modified.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import fft as sfft
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "genet"))
import wcsi  # noqa: E402

DEFAULTS = dict(
    sigma_cyto=5.0,
    tau_band=0.69,
    prior_pairs=1.0,
    beta_grad=1.0,
    K0=1.0, K1=1.0, K3=1.0,
    n_proj_colour=16,
    n_proj_spectrum=12,
    harmonic_frac=0.05,       # eigenvectors with lambda <= this fraction of lambda_max
    harmonic_cap=4,
    phenotype_master=1.0,     # JND; below this, master fidelity is "low"
    phenotype_frust=1.0,
    vis_floor=0.05,           # both residuals below this: texture coordinates carry no deviation
    alphas=(1.0, 1.0, 1.0, 1.0, 1.0),  # master, frust, seam, emerge, lr
    jnd_de=1.5,
    jnd_angle_deg=4.0,
    jnd_freq=0.05,
    jnd_alpha=0.14,
    jnd_cov=0.2,
    jnd_corr=0.2,             # phase-harmonic correlation
    jnd_splay_px=16.0,        # a 4 degree director turn over this many pixels is one JND
    min_wave=3.0,
    n_scale=4,
    n_orient=4,
)

TERMS = ("E_master", "E_frust", "E_seam", "E_emerge", "E_lr")
PHENOTYPES = ("clonal masterpiece", "allometric polymorphism",
              "somatic mutation", "isolated clonal island")
_IU = np.triu_indices(3)


# ---------------------------------------------------------------- alignment

def align_triplet(sr, gt, lr, scale):
    """Crop SR, GT and LR to a common integer scale. SR and GT are the full resolution."""
    sr, sr_white = wcsi.load_rgb(sr)
    gt, gt_white = wcsi.load_rgb(gt)
    lr, lr_white = wcsi.load_rgb(lr)
    notes = []
    H, W = min(sr.shape[0], gt.shape[0]), min(sr.shape[1], gt.shape[1])
    if (H, W) != tuple(sr.shape[:2]) or (H, W) != tuple(gt.shape[:2]):
        notes.append(f"cropped SR and ground truth to the common {W}x{H}")
    sr, gt = sr[:H, :W], gt[:H, :W]
    H0, W0 = lr.shape[:2]
    s = int(scale) if scale else int(round((H / H0 + W / W0) / 2))
    if s < 2 or abs(H - s * H0) >= s or abs(W - s * W0) >= s:
        raise ValueError(f"full-resolution {W}x{H} is not an integer multiple (>= 2) of LR {W0}x{H0}"
                         + (f" at scale {scale}" if scale else ""))
    h0, w0 = min(H0, H // s), min(W0, W // s)
    if (h0, w0) != (H0, W0) or (s * h0, s * w0) != (H, W):
        notes.append(f"cropped to LR {w0}x{h0} / full-resolution {s * w0}x{s * h0}")
    return (s, sr[:s * h0, :s * w0], gt[:s * h0, :s * w0], lr[:h0, :w0],
            sr_white, gt_white, lr_white, notes)


def _rects_for(h0, w0, tile, overlap):
    th, tw = min(tile, h0), min(tile, w0)
    if not 0 <= overlap < min(th, tw):
        raise ValueError(f"overlap {overlap} must be in [0, {min(th, tw)})")
    return [(y, x, th, tw) for y in wcsi.tile_starts(h0, th, overlap) for x in wcsi.tile_starts(w0, tw, overlap)]


# ---------------------------------------------------------------- wavelets: rigidity and phase harmonics

def _quadrature_filters(h, w, n_scale, n_orient, min_wave):
    """Even log-Gabor and the sign that turns it into its Hilbert partner, per scale and orientation."""
    ky = sfft.fftfreq(h)[:, None]
    kx = sfft.fftfreq(w)[None, :]
    f = np.hypot(kx, ky)
    ang = np.arctan2(ky, kx)
    sigma_theta = math.pi / n_orient
    sig_l = math.log(0.55)
    evens, signs = [], []
    for level in range(n_scale):
        f0 = 1.0 / (min_wave * 2 ** level)
        radial = np.exp(-0.5 * (np.log(np.maximum(f, 1e-8) / f0) / sig_l) ** 2)
        radial = np.where(f > 1e-8, radial, 0.0)
        for k in range(n_orient):
            theta = math.pi * k / n_orient
            d1 = (ang - theta + np.pi) % (2 * np.pi) - np.pi
            d2 = (ang - theta - np.pi + np.pi) % (2 * np.pi) - np.pi
            angular = np.exp(-0.5 * (d1 / sigma_theta) ** 2) + np.exp(-0.5 * (d2 / sigma_theta) ** 2)
            evens.append((radial * angular).astype(np.float32))
            direction = kx * math.cos(theta) + ky * math.sin(theta)
            signs.append(np.sign(direction).astype(np.float32))
    shape = (n_scale, n_orient, h, w)
    return np.stack(evens).reshape(shape), np.stack(signs).reshape(shape)


_FILTERS = {}


def wavelet_bank(lum, n_scale=4, n_orient=4, min_wave=3.0):
    """Quadrature log-Gabor responses, shape (scale, orientation, h, w).

    Phase is atan2(odd, even). A step edge sits at the same phase on every octave,
    which a one-sided filter does not do.
    """
    h, w = lum.shape
    key = (h, w, n_scale, n_orient, round(min_wave, 3))
    filt = _FILTERS.get(key)
    if filt is None:
        filt = _quadrature_filters(h, w, n_scale, n_orient, min_wave)
        _FILTERS[key] = filt
    even_f, sign_f = filt
    spec = sfft.fft2(np.asarray(lum, np.float32))
    out = np.empty((n_scale, n_orient, h, w), np.complex64)
    for i in range(n_scale):
        for j in range(n_orient):
            # even_f is real and even, so its inverse transform is real.
            # even_f * sign is real and odd, so its inverse transform is imaginary.
            even = sfft.ifft2(spec * even_f[i, j]).real
            odd = sfft.ifft2(spec * even_f[i, j] * sign_f[i, j]).imag
            out[i, j] = even + 1j * odd
    return out


def rigidity_map(lum, n_scale=4, n_orient=4, min_wave=3.0):
    """Where the ground truth is phase-locked, in [0, 1].

    Pairwise phase congruence: the amplitude-weighted agreement of quadrature phase
    across octaves. The specification writes a doubled coarse phase; once phase is
    measured from a quadrature pair, a step has phase 0 at every scale, so the
    doubled and the plain difference agree on the features rigidity is for. A noise
    floor at the median energy sets flat regions, noise and one-scale gratings to 0.
    """
    bank = wavelet_bank(lum, n_scale, n_orient, min_wave)
    acc = np.zeros(lum.shape, np.complex128)
    weight = np.zeros(lum.shape, np.float64)
    for k in range(bank.shape[0] - 1):
        for j in range(bank.shape[1]):
            fine, coarse = bank[k, j], bank[k + 1, j]
            amp = np.sqrt(np.abs(fine) * np.abs(coarse)).astype(np.float64)
            acc += amp * np.exp(1j * (np.angle(fine) - np.angle(coarse)))
            weight += amp
    energy = np.abs(acc)
    pc = np.maximum(energy - np.median(energy), 0.0) / (weight + 1e-8)
    return np.clip(pc, 0.0, 1.0).astype(np.float32)


def phase_harmonics(bank):
    """Cross-scale phase-harmonic correlation, real and imaginary part per scale pair and orientation."""
    feats = []
    n_scale, n_orient = bank.shape[:2]
    for k in range(n_scale - 1):
        fine, coarse = bank[k], bank[k + 1]
        for j in range(n_orient):
            coupled = fine[j] * np.exp(-2j * np.angle(coarse[j]))
            denom = math.sqrt(float(np.mean(np.abs(fine[j]) ** 2))) + 1e-8
            c = np.mean(coupled) / denom
            feats.append(float(np.real(c)))
            feats.append(float(np.imag(c)))
    return np.asarray(feats, np.float64)


# ---------------------------------------------------------------- transport distances

def _quantile(values, weights, qs):
    values = np.asarray(values, np.float64).ravel()
    weights = np.clip(np.asarray(weights, np.float64).ravel(), 0, None)
    if weights.sum() < 1e-12:
        return np.zeros_like(qs)
    order = np.argsort(values)
    values, weights = values[order], weights[order] / weights.sum()
    cdf = np.cumsum(weights) - 0.5 * weights
    return np.interp(qs, cdf, values)


def sliced_w2_sq(a, b, n_proj, seed):
    """Mean over random Lab directions of the squared 1D Wasserstein-2 distance."""
    a = np.asarray(a, np.float64).reshape(-1, 3)
    b = np.asarray(b, np.float64).reshape(-1, 3)
    step_a, step_b = max(1, len(a) // 2048), max(1, len(b) // 2048)
    a, b = a[::step_a], b[::step_b]
    rng = np.random.default_rng(seed)
    qs = np.linspace(0.02, 0.98, 49)
    acc = 0.0
    for _ in range(n_proj):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        qa, qb = _quantile(a @ u, np.ones(len(a)), qs), _quantile(b @ u, np.ones(len(b)), qs)
        acc += float(np.mean((qa - qb) ** 2))
    return acc / n_proj


def w1_1d(a, wa, b, wb):
    """1D Wasserstein-1 between two weighted samples."""
    qs = np.linspace(0.02, 0.98, 49)
    return float(np.mean(np.abs(_quantile(a, wa, qs) - _quantile(b, wb, qs))))


def spectrum_cloud(lum, f_lo):
    """Windowed power in the band as a weighted point cloud of (kx, ky)."""
    h, w = lum.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F = sfft.fft2((np.asarray(lum, np.float32) - lum.mean()) * win)
    P = (F.real ** 2 + F.imag ** 2).astype(np.float64)
    kx, ky = wcsi._freqs(h, w)
    f = np.sqrt(kx ** 2 + ky ** 2)
    mask = (f >= f_lo) & (f <= 0.5) & (ky >= 0)
    pts = np.stack([np.broadcast_to(kx, P.shape)[mask], np.broadcast_to(ky, P.shape)[mask]], 1)
    wt = P[mask]
    total = wt.sum()
    if total <= 0 or len(pts) < 4:
        return np.zeros((1, 2)), np.ones(1)
    return pts, wt / total


def sliced_spectral_w1(lum, lum_star, f_lo, n_proj, seed):
    a, wa = spectrum_cloud(lum, f_lo)
    b, wb = spectrum_cloud(lum_star, f_lo)
    rng = np.random.default_rng(seed)
    acc = 0.0
    for _ in range(n_proj):
        theta = rng.random() * math.pi
        u = np.array([math.cos(theta), math.sin(theta)])
        acc += w1_1d(a @ u, wa, b @ u, wb)
    return acc / n_proj


# ---------------------------------------------------------------- director field

def _director(lum, sigma):
    """Local nematic director (headless), order, and Q, smoothed by `sigma` pixels."""
    gx = ndimage.correlate(lum, wcsi._SCHARR_X, mode="reflect")
    gy = ndimage.correlate(lum, wcsi._SCHARR_Y, mode="reflect")
    jxx = ndimage.gaussian_filter(gx * gx, sigma)
    jxy = ndimage.gaussian_filter(gx * gy, sigma)
    jyy = ndimage.gaussian_filter(gy * gy, sigma)
    theta = (0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2) % np.pi
    order = (np.hypot(jxx - jyy, 2 * jxy) / (jxx + jyy + 1e-12)) ** 2
    Qxx = order * np.cos(2 * theta)
    Qxy = order * np.sin(2 * theta)
    return theta.astype(np.float64), order.astype(np.float64), Qxx.astype(np.float64), Qxy.astype(np.float64)


def _vectors(theta):
    return np.cos(theta), np.sin(theta)


def frank_energy(lum, lum_star, gamma, p):
    """Pointwise Frank energy where the ground truth is rigid, distributional where it is not.

    Q is headless, so it needs no gauge. Splay is not: the vector is flipped where it
    points opposite the ground-truth director before the derivatives are taken.
    """
    theta, _, Qxx, Qxy = _director(lum, sigma=1.5)
    theta_s, order_s, Qxx_s, Qxy_s = _director(lum_star, sigma=1.5)
    sx, sy = _vectors(theta)
    nx, ny = _vectors(theta_s)
    flip = (sx * nx + sy * ny) < 0
    sx = np.where(flip, -sx, sx)
    sy = np.where(flip, -sy, sy)
    splay = np.gradient(sx, axis=1) + np.gradient(sy, axis=0)
    omega = np.gradient(sy, axis=1) - np.gradient(sx, axis=0)
    bend = np.stack([-sy * omega, sx * omega], -1)
    splay_s = np.gradient(nx, axis=1) + np.gradient(ny, axis=0)
    omega_s = np.gradient(ny, axis=1) - np.gradient(nx, axis=0)
    bend_s = np.stack([-ny * omega_s, nx * omega_s], -1)

    jnd_q = 2 * math.sin(math.radians(p["jnd_angle_deg"]))
    jnd_d = math.radians(p["jnd_angle_deg"]) / p["jnd_splay_px"]
    point = (p["K0"] * ((Qxx - Qxx_s) ** 2 + (Qxy - Qxy_s) ** 2) / jnd_q ** 2
             + p["K1"] * (splay - splay_s) ** 2 / jnd_d ** 2
             + p["K3"] * np.sum((bend - bend_s) ** 2, -1) / jnd_d ** 2)
    rigid = gamma * order_s
    if rigid.sum() > 1e-6:
        e_point = float((rigid * point).sum() / rigid.sum())
    else:
        e_point = 0.0
    loose = (1.0 - gamma) * order_s
    stride = max(1, lum.shape[0] // 48)
    sl = (slice(None, None, stride), slice(None, None, stride))
    wt = loose[sl].ravel()
    parts = []
    for field, field_s, jnd in ((Qxx, Qxx_s, jnd_q), (Qxy, Qxy_s, jnd_q),
                                (splay, splay_s, jnd_d), (np.linalg.norm(bend, axis=-1),
                                                          np.linalg.norm(bend_s, axis=-1), jnd_d)):
        dist = w1_1d(field[sl], wt, field_s[sl], wt)
        parts.append((dist / jnd) ** 2)
    e_dist = float(np.mean(parts)) if wt.sum() > 1e-6 else 0.0
    return e_point + e_dist


# ---------------------------------------------------------------- state, transport, sheaf

def _psi(ft):
    return np.concatenate([ft["mu"], ft["cov"][_IU], np.asarray(ft["Q"], np.float64),
                           np.asarray(ft["kbar"], np.float64), [ft["alpha"]]])


def _jnd_vector(ft, p):
    var = np.diag(ft["cov"])
    j = np.empty(14)
    j[0:3] = p["jnd_de"]
    j[3:9] = p["jnd_cov"] * (np.sqrt(np.maximum(var[_IU[0]] * var[_IU[1]], 0)) + p["jnd_de"] ** 2)
    j[9:11] = 2 * math.sin(math.radians(p["jnd_angle_deg"]))
    fbar = max(float(ft["fbar"]), 1e-3)
    j[11:13] = p["jnd_freq"] * fbar
    j[13] = p["jnd_alpha"]
    return j


def _nematic_delta(th_j, th_i):
    return (th_j - th_i + math.pi / 2) % math.pi - math.pi / 2


def _cov_map(Si, Sj, eps):
    """6x6 map on vech(δΣ) induced by Σ -> B Σ B^T, B = sqrt(Sj+εI) sqrt(Si+εI)^{-1}."""
    Bi = wcsi._sqrtm_psd(Si + eps * np.eye(3))
    Bj = wcsi._sqrtm_psd(Sj + eps * np.eye(3))
    B = Bj @ np.linalg.inv(Bi)
    A = np.zeros((6, 6))
    for k, (a, b) in enumerate(zip(*_IU)):
        M = np.zeros((3, 3))
        M[a, b] = M[b, a] = 1.0
        A[:, k] = (B @ M @ B.T)[_IU]
    return A


def _A_matrix(cov_i, cov_j, th_i, th_j, eps):
    """A*_ij. Identity on colour means, wavevector and slope. Covariance is pushed by the
    master covariance transport; Q is rotated by the master stroke-angle change."""
    A = np.eye(14)
    A[3:9, 3:9] = _cov_map(cov_i, cov_j, eps)
    d = _nematic_delta(th_j, th_i)
    c, s = math.cos(2 * d), math.sin(2 * d)
    A[9, 9], A[9, 10] = c, -s
    A[10, 9], A[10, 10] = s, c
    return A


def _affinity(feats, p):
    """Material affinity between ground-truth tiles. Same kernel as WCSI."""
    N = len(feats)
    I, J = np.triu_indices(N, 1)
    mu = np.stack([f["mu"] for f in feats])
    cov = np.stack([f["cov"] for f in feats])
    floor = p["jnd_de"] ** 2 / 4
    band = np.log(np.stack([np.diagonal(f["gram"]) + floor for f in feats]))
    w = np.ones(len(I))
    for k0 in range(0, len(I), wcsi._CHUNK):
        a, b = I[k0:k0 + wcsi._CHUNK], J[k0:k0 + wcsi._CHUNK]
        w[k0:k0 + wcsi._CHUNK] = np.exp(
            -wcsi.bures_w2(mu[a], cov[a], mu[b], cov[b]) / (2 * p["sigma_cyto"] ** 2)
            - np.mean((band[a] - band[b]) ** 2, 1) / (2 * p["tau_band"] ** 2))
    return I, J, w


def _coherence(delta, edge_list, p):
    """Fraction of ||δΨ||^2 in the low eigenvalues of the ground-truth sheaf Laplacian."""
    N, d = delta.shape
    if N < 2 or float(np.sum(delta ** 2)) < 1e-10:
        return None
    n = N * d
    L = np.zeros((n, n))
    for i, j, A, S, w in edge_list:
        # residual S (A δ_i - δ_j); S is diagonal
        SA = S[:, None] * A
        ii, jj = slice(i * d, (i + 1) * d), slice(j * d, (j + 1) * d)
        L[ii, ii] += w * (SA.T @ SA)
        L[jj, jj] += w * np.diag(S ** 2)
        L[ii, jj] += -w * (SA.T * S)
        L[jj, ii] += -w * (S[:, None] * SA)
    evals, evecs = np.linalg.eigh(0.5 * (L + L.T))
    lam_max = float(max(evals[-1], 1e-12))
    # The whole kernel counts. Gated-off coordinates are unconstrained and must not be
    # dropped by the cap, or a shared departure is measured against an arbitrary basis.
    kernel = np.flatnonzero(evals <= 1e-6 * lam_max + 1e-8)
    smooth = np.flatnonzero(evals <= p["harmonic_frac"] * lam_max)
    extra = [i for i in smooth if i not in set(kernel.tolist())]
    keep = np.concatenate([kernel, extra[:p["harmonic_cap"]]]) if len(kernel) or extra else np.array([0])
    flat = delta.ravel()
    proj = sum(float(evecs[:, m] @ flat) ** 2 for m in keep)
    return proj / float(flat @ flat)


# ---------------------------------------------------------------- seams

def _separated_excess(img, c, r0, r1, beta, offsets=(3, 6)):
    """Median colour excess and median gradient excess at column c, over rows r0:r1."""
    colour = wcsi._jump(img, c, r0, r1, 0)
    both = wcsi._jump(img, c, r0, r1, beta) if beta else colour
    grad = both - colour
    cols, grads = [], []
    for o in offsets + tuple(-o for o in offsets):
        if 3 <= c + o < img.shape[1] - 3:
            ctrl = wcsi._jump(img, c + o, r0, r1, 0)
            cols.append(ctrl)
            grads.append((wcsi._jump(img, c + o, r0, r1, beta) - ctrl) if beta else np.zeros_like(ctrl))
    if not cols:
        return None
    cmed, gmed = np.median(cols, 0), np.median(grads, 0)
    return float(np.median(colour - cmed)), float(np.median(cmed)), float(np.median(grad - gmed)), float(np.median(gmed))


def seam_against_master(sr, gt, gammas, rects, pairs, s, beta, jnd2):
    """Seam penalty of a stitched image, zero when SR equals the ground truth.

    Colour excess over the ground truth's own edge is weighted by the rigidity
    along the line, so a contour must not step and a texture may. Gradient excess
    counts either way. Separate unblended canvases are not required.
    """
    reach = 9
    penalties = []

    def one(im_s, im_g, gamma, a, b, r0, r1):
        best = None
        for c in sorted({int(a), int((a + b) // 2), int(b)}):
            if c - reach < 0 or c + reach >= im_s.shape[1] or r1 - r0 < 4:
                continue
            got_s = _separated_excess(im_s, c, r0, r1, beta)
            got_g = _separated_excess(im_g, c, r0, r1, beta)
            if got_s is None or got_g is None:
                continue
            cs, ctrls, gs, ctrlsg = got_s
            cg, ctrlg, gg, ctrlgg = got_g
            g = float(np.mean(gamma[r0:r1, max(c - 1, 0):c + 1])) if gamma is not None else 1.0
            colour = g * max(cs - cg, 0.0) / (jnd2 + max(ctrlg, 0.0))
            gradient = max(gs - gg, 0.0) / (jnd2 + max(ctrlgg, 0.0))
            v = colour + gradient
            best = v if best is None else max(best, v)
        return best

    srT, gtT = sr.transpose(1, 0, 2), gt.transpose(1, 0, 2)
    for i, j in pairs:
        ri, rj = rects[i], rects[j]
        y0, y1, x0, x1 = wcsi._shared(ri, rj)
        if y1 <= y0 or x1 <= x0:
            continue
        side = abs(ri[1] - rj[1]) / min(ri[3], rj[3]) >= abs(ri[0] - rj[0]) / min(ri[2], rj[2])
        if side:
            v = one(sr, gt, _embed(gammas[i], sr.shape[:2], s * ri[0], s * ri[1]), s * x0, s * x1, s * y0, s * y1)
        else:
            v = one(srT, gtT, _embed(gammas[i], sr.shape[:2], s * ri[0], s * ri[1]).T, s * y0, s * y1, s * x0, s * x1)
        if v is not None:
            penalties.append(v)
    return penalties


def _embed(tile_gamma, shape, y, x):
    """Place a tile rigidity map back into image coordinates. Outside the tile, rigidity is 0."""
    full = np.zeros(shape, np.float32)
    h, w = tile_gamma.shape
    full[y:y + h, x:x + w] = tile_gamma
    return full


# ---------------------------------------------------------------- the score

def _phenotype(master, frust, p):
    low_m, low_f = master < p["phenotype_master"], frust < p["phenotype_frust"]
    if low_m and low_f:
        return PHENOTYPES[0]
    if low_f:
        return PHENOTYPES[1]
    if low_m:
        return PHENOTYPES[3]
    return PHENOTYPES[2]


def gmci_score(sr, gt, lr, scale=None, tile=None, overlap=None, tile_sr=None,
               return_details=False, **params):
    """GMCI of super-resolved `sr` against ground truth `gt` and low-resolution `lr`.

    tile, overlap are in LR pixels, as in WCSI. tile_sr is the analysis tile in
    full-resolution pixels and, when given, overrides tile (it must divide the scale).
    """
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown parameters: {sorted(unknown)}")
    p = {**DEFAULTS, **params}
    if len(p["alphas"]) != len(TERMS):
        raise ValueError(f"alphas needs {len(TERMS)} values, one per {TERMS}")
    s, sr, gt, lr, sr_w, gt_w, lr_w, notes = align_triplet(sr, gt, lr, scale)
    if tile_sr is not None:
        if tile_sr % s:
            raise ValueError(f"tile_sr {tile_sr} does not divide scale {s}")
        tile = tile_sr // s
    H0, W0 = lr.shape[:2]
    if tile is None:
        tile = max(8, min(1024 // s, min(H0, W0) // 3))
        notes.append(f"nominal grid (tile {tile} LR px, {s * tile} full-resolution px)")
    if overlap is None:
        overlap = tile // 16
    rects = _rects_for(H0, W0, tile, overlap)
    sr_lab = wcsi.srgb_to_lab(sr, sr_w)
    gt_lab = wcsi.srgb_to_lab(gt, gt_w)
    lr_lab = wcsi.srgb_to_lab(lr, lr_w)
    sigma, f_lo = 0.4 * s, 1.0 / (2 * s)
    jnd2 = p["jnd_de"] ** 2
    eps = jnd2

    per = []
    for y0, x0, h, w_ in rects:
        Hs = sr_lab[s * y0:s * (y0 + h), s * x0:s * (x0 + w_)]
        Hg = gt_lab[s * y0:s * (y0 + h), s * x0:s * (x0 + w_)]
        Lc = lr_lab[y0:y0 + h, x0:x0 + w_]
        gamma = rigidity_map(np.ascontiguousarray(Hg[..., 0]), p["n_scale"], p["n_orient"], p["min_wave"])
        g_macro = gamma.reshape(h, s, w_, s).mean((1, 3))
        Ms, fs = wcsi.tile_features(Hs, s, sigma, f_lo)
        Mg, fg = wcsi.tile_features(Hg, s, sigma, f_lo)
        de_pos = float(np.mean(wcsi.ciede2000(Ms, Mg) ** 2 * g_macro)) / jnd2
        sw = sliced_w2_sq(Hs.reshape(-1, 3), Hg.reshape(-1, 3), p["n_proj_colour"], seed=0) / jnd2
        Rs = np.ascontiguousarray(wcsi.rg_split(Hs, s, sigma)[1][..., 0])
        Rg = np.ascontiguousarray(wcsi.rg_split(Hg, s, sigma)[1][..., 0])
        e_pcp = frank_energy(Rs, Rg, gamma, p)
        fbar = 0.5 * (fs["fbar"] + fg["fbar"])
        jnd_w1 = max(p["jnd_freq"] * max(fbar, 1e-3), 1e-3)
        e_clock = (sliced_spectral_w1(Rs, Rg, f_lo, p["n_proj_spectrum"], seed=0) / jnd_w1) ** 2
        e_clock += (fs["alpha"] - fg["alpha"]) ** 2 / p["jnd_alpha"] ** 2
        harm_s = phase_harmonics(wavelet_bank(np.ascontiguousarray(Hs[..., 0]), p["n_scale"], p["n_orient"], p["min_wave"]))
        harm_g = phase_harmonics(wavelet_bank(np.ascontiguousarray(Hg[..., 0]), p["n_scale"], p["n_orient"], p["min_wave"]))
        e_emerge = float(np.mean((harm_s - harm_g) ** 2)) / p["jnd_corr"] ** 2
        de_lr_s = float(np.mean(wcsi.ciede2000(Ms, Lc) ** 2))
        de_lr_g = float(np.mean(wcsi.ciede2000(Mg, Lc) ** 2))
        e_lr = max(0.0, de_lr_s - de_lr_g) / jnd2
        per.append(dict(gamma=gamma, gamma_mean=float(gamma.mean()), fs=fs, fg=fg,
                        psi=_psi(fs), psi_star=_psi(fg), jnd=_jnd_vector(fg, p),
                        vis=min(1.0, fs["band_var"] / jnd2), vis_star=min(1.0, fg["band_var"] / jnd2),
                        E_wadd=de_pos + sw, E_wadd_pos=de_pos, E_wadd_sw=sw, E_pcp=e_pcp,
                        E_clock=e_clock, E_emerge=e_emerge, E_lr=e_lr))

    N = len(per)
    delta = np.stack([t["psi"] - t["psi_star"] for t in per])
    for i, t in enumerate(per):
        if t["vis"] < p["vis_floor"] and t["vis_star"] < p["vis_floor"]:
            delta[i, 9:] = 0.0
    pos = wcsi._grid_positions(rects)
    I, J, near, far, _corners = wcsi._tile_graph(rects, pos)
    _Ia, _Ja, aff = _affinity([t["fg"] for t in per], p) if N > 1 else (I, J, np.zeros(len(I)))
    # _affinity rebuilds triu order, which matches _tile_graph's I, J
    edge_w = np.where(near, 1.0, np.where(far, aff, 0.0))

    frust_edge = np.zeros(len(I))
    edge_list = []
    for e, (i, j) in enumerate(zip(I, J)):
        if edge_w[e] <= 0:
            continue
        i, j = int(i), int(j)
        A = _A_matrix(per[i]["fg"]["cov"], per[j]["fg"]["cov"], per[i]["fg"]["theta"], per[j]["fg"]["theta"], eps)
        raw = A @ delta[i] - delta[j]
        scale = np.maximum(0.5 * (per[i]["jnd"] + per[j]["jnd"]), 1e-8)
        vis = min(per[i]["vis"], per[j]["vis"], per[i]["vis_star"], per[j]["vis_star"])
        S = 1.0 / scale
        S[9:] *= vis
        resid = S * raw
        frust_edge[e] = float(resid @ resid)
        edge_list.append((i, j, A, S, float(edge_w[e])))

    used_w = edge_w.copy()
    e_frust = None
    frust_tile = np.zeros(N)
    if used_w.sum() > 0:
        e_frust = float(np.sum(used_w * frust_edge) / (used_w.sum() + p["prior_pairs"]))
        degree = np.zeros(N)
        for e, (i, j) in enumerate(zip(I.tolist(), J.tolist())):
            if used_w[e] <= 0:
                continue
            frust_tile[i] += used_w[e] * frust_edge[e]
            frust_tile[j] += used_w[e] * frust_edge[e]
            degree[i] += used_w[e]
            degree[j] += used_w[e]
        frust_tile = frust_tile / np.maximum(degree, 1e-12)
    rho = _coherence(delta, edge_list, p) if edge_list else None

    pairs = list(zip(I[near].tolist(), J[near].tolist())) if near.any() else []
    seam_vals = seam_against_master(sr_lab, gt_lab, [t["gamma"] for t in per], rects, pairs, s,
                                     p["beta_grad"], jnd2) if pairs else []
    e_seam = float(np.mean(seam_vals)) if seam_vals else None

    master_tile = np.array([t["E_wadd"] + t["E_pcp"] + t["E_clock"] for t in per])
    energies = dict(
        E_master=float(master_tile.mean()),
        E_frust=e_frust,
        E_seam=e_seam,
        E_emerge=float(np.mean([t["E_emerge"] for t in per])),
        E_lr=float(np.mean([t["E_lr"] for t in per])),
    )
    for name, value in energies.items():
        if value is None:
            notes.append(f"{name}: undefined for this layout ({N} tiles), left out of GMCI")
    alphas = dict(zip(TERMS, p["alphas"]))
    used = [(alphas[k], energies[k]) for k in TERMS if energies[k] is not None]
    F = sum(a * v for a, v in used)
    E_jnd = sum(a for a, _ in used)
    score = 100.0 * math.exp(-min(F / E_jnd, 60.0)) if E_jnd > 0 else 100.0

    labels = [_phenotype(master_tile[i], frust_tile[i], p) for i in range(N)]
    if not return_details:
        return score
    counts = {name: labels.count(name) for name in PHENOTYPES}
    return dict(
        gmci=score, rho_coherence=rho, F=F, E_jnd=E_jnd, energies=energies,
        parts=dict(
            E_wadd=float(np.mean([t["E_wadd"] for t in per])),
            E_wadd_positional=float(np.mean([t["E_wadd_pos"] for t in per])),
            E_wadd_sliced=float(np.mean([t["E_wadd_sw"] for t in per])),
            E_pcp=float(np.mean([t["E_pcp"] for t in per])),
            E_clock=float(np.mean([t["E_clock"] for t in per])),
            gamma_mean=float(np.mean([t["gamma_mean"] for t in per])),
            phenotype_counts=counts,
        ),
        scale=s, n_tiles=N,
        layout=dict(tile_lr=rects[0][2] if rects else None, overlap_lr=overlap,
                    tile_sr=s * rects[0][2] if rects else None),
        tiles=[dict(index=i, y0=rects[i][0], x0=rects[i][1], h=rects[i][2], w=rects[i][3],
                    E_master=float(master_tile[i]), E_frust=float(frust_tile[i]),
                    E_wadd=per[i]["E_wadd"], E_pcp=per[i]["E_pcp"], E_clock=per[i]["E_clock"],
                    E_emerge=per[i]["E_emerge"], E_lr=per[i]["E_lr"],
                    gamma_mean=per[i]["gamma_mean"], phenotype=labels[i])
               for i in range(N)],
        notes=notes,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="GMCI of a super-resolved image against ground truth and the LR input.")
    ap.add_argument("sr")
    ap.add_argument("gt")
    ap.add_argument("lr")
    ap.add_argument("--scale", type=int)
    ap.add_argument("--tile", type=int, help="analysis tile in LR pixels")
    ap.add_argument("--tile-sr", type=int, help="analysis tile in full-resolution pixels (overrides --tile)")
    ap.add_argument("--overlap", type=int, help="overlap in LR pixels (default tile/16)")
    ap.add_argument("--json", help="write the full breakdown to this file")
    args = ap.parse_args(argv)
    d = gmci_score(args.sr, args.gt, args.lr, scale=args.scale, tile=args.tile, tile_sr=args.tile_sr,
                   overlap=args.overlap, return_details=True)
    rho = "undefined" if d["rho_coherence"] is None else f"{d['rho_coherence']:.3f}"
    g = d["layout"]
    print(f"GMCI {d['gmci']:.2f}   rho_coherence {rho}   scale {d['scale']}x   "
          f"{d['n_tiles']} tiles of {g['tile_sr']} px")
    labels = dict(E_master="colour transport, Frank elasticity, spectral rhythm vs ground truth",
                  E_frust="departures from the ground truth that disagree between tiles",
                  E_seam="seams beyond the ground truth's own edges, rigidity-weighted",
                  E_emerge="cross-scale phase coupling vs the ground truth",
                  E_lr="extra break of the low-resolution input, beyond the ground truth")
    for k in TERMS:
        v = d["energies"][k]
        print(f"  {k:10s} {'n/a' if v is None else f'{v:9.4f}'}   {labels[k]}")
    counts = d["parts"]["phenotype_counts"]
    print("  phenotypes: " + ", ".join(f"{name} {counts[name]}" for name in PHENOTYPES))
    for n in d["notes"]:
        print(f"  note: {n}")
    if args.json:
        Path(args.json).write_text(json.dumps(d, indent=1))
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
