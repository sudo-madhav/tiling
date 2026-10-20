#!/usr/bin/env python3
"""WCSI: Waddington Canalization & Sheaf Index for tiled super-resolution.

Scores a super-resolved (SR) image against its low-resolution (LR) reference,
tile by tile. WCSI is in (0, 100]: 100 when every error term is zero, 50 when
the error terms average one just-noticeable difference (JND).

What it measures: agreement of the SR's coarse scale with the LR, seams, and
whether tiles that are one material in the LR invented the same texture. It
does not measure detail: an output with no invented detail (bicubic) scores
near the top, and natural photographs, whose fine texture differs between
tiles of similar colour, score low. Compare outputs of the same kind, e.g.
tiled against untiled inference of one model.

    python genet/wcsi.py SR.png LR.png --tile 256 --overlap 16
    python genet/wcsi.py SR.png LR.png --json scores.json --advice
    python genet/wcsi.py SR.png LR.png --tiles-dir tiles/   # unblended tile outputs

    from wcsi import wcsi_score
    score = wcsi_score("sr.png", "lr.png", tile=256, overlap=16)       # float
    details = wcsi_score(sr, lr, tile=256, overlap=16, return_details=True)

Inputs: any file OpenCV or Pillow reads (png, jpg, jpeg, bmp, tif, webp),
8 or 16 bit, or arrays / tensors (HxWx3 or 3xHxW, uint8, uint16 or float in
[0, 1]). The scale is the integer size ratio SR/LR (2, 4, 8, ...). `tile` and
`overlap` are in LR pixels and should match how the SR model was tiled; the
defaults are a nominal grid. Students are tiles, the original painting is the
LR image, a student's canvas is the SR tile.

Pipeline (CIELAB, D65; s = scale):
  Stage 1  M = D_s(SR): Gaussian (sigma = 0.4 s SR px), then s x s block mean.
           R = SR - U_s(M), U_s = bicubic.
  Stage 2  psi_i in R^14 per tile: Lab mean (3) and covariance (6) of the SR
           tile; nematic tensor of R's luminance (2); mean wavevector and
           spectral slope of R's luminance in the band [1/(2s), 1/2] (3).
  Stage 3  E_RG, E_GJ, E_PCP, E_cyto, E_sheaf, each in JND units.
  Stage 4  F = sum_k w_k E_k, WCSI = 100 / (1 + F / F_JND), F_JND = sum_k w_k.
           Per tile: r_i = sum_j w_ij (Phi_ji(psi_j) - psi_i) + eta (psi0_i - psi_i)
           and a one-line instruction.

Choices the spec leaves open:
  - Band [1/(2s), 1/2] cycles/px: frequencies above the LR Nyquist limit.
    The spec's lower edge 1/8 at 8x would leave an empty band at 2x.
  - psi0_i is psi of the LR tile one scale down: the same split and
    statistics applied to the LR tile, covariance times kappa. E_RG uses its
    colour part, the corner winding numbers its orientation, E_sheaf all of it.
  - kappa ("master expansion factor"): default "auto", the pooled trace ratio
    of SR to LR tile covariances, so only per-tile deviations from the common
    detail gain count. A uniformly flat output (bicubic) is then not penalised
    for missing detail; pass a number (e.g. measured on HR/LR pairs) for that.
  - Material affinity w_ij: the spec's cytoneme affinity exp(-W2^2 / (2 sigma^2))
    between Gaussians fitted to the two LR tiles, times exp(-m / (2 tau^2)),
    m the mean squared difference of the LR tiles' log Gabor energies, so a
    textured and a flat tile of one colour are not one material. The spec
    defines it for E_cyto; E_PCP and E_sheaf use it too, so sky above water
    is not asked to share a texture.
  - E_PCP, E_cyto and E_sheaf are affinity-weighted means over pairs, so the
    score does not grow with the number of tiles, with `prior_pairs`
    pseudo-pairs of zero divergence in the denominator: the spec's normalised
    mean does not change when every affinity shrinks, so without them an
    image whose distant tiles share no material would be scored on its least
    different pairs.
  - E_GJ. On one stitched image both tiles have the same pixel in their
    overlap, so the spec's overlap difference is zero. Without per-tile
    outputs, E_GJ scores each overlap band by the excess colour + gradient
    jump at its candidate seam lines (start, middle, end) over control lines
    3 and 6 px away, as a median over rows (a seam steps along the whole
    line, an image edge crossing it covers a few rows), divided by 1.5^2 plus
    the control jump (masking by the local texture) and discounted where the
    LR steps at the same line (an image edge along the line). With `tiles=`
    it is the spec's overlap comparison.
  - E_PCP and E_cyto compare the SR tiles' own statistics, as written. R_ij
    in E_PCP is the identity: one global frame on a flat image.
  - E_sheaf. Phi_ij(psi) = psi + c_ij (psi0_j - psi0_i), with c_ij = 1 on the
    colour coordinates as in the spec, and on the texture coordinates
    c_ij = min(v0_i, v0_j), the visibility of the LR tiles' own residual:
    where the LR's top band is visible the expected texture difference is
    the LR's; where it is flat or noise there is no reference and
    same-material tiles must invent the same texture. The loop holonomy of
    the spec's map telescopes to zero for any image; it is computed and
    reported as a check.
  - Wavevector. The upper-half-plane mean flips sign when the dominant
    wavevector crosses the k_x axis, so comparisons use its length (the
    power-weighted mean radial frequency) and leave orientation to Q. psi
    reports the spec's k.
  - Corner winding numbers: the LR orientation is that of the LR tile's own
    residual one scale down. Corners count only where all four SR and all
    four LR tiles have nematic order times visibility >= defect_min_order;
    orientation is undefined on isotropic or invisible residuals.
  - G(R): even Gabor energies (4 orientations x octave scales inside the
    band), computed from the windowed power spectrum by Parseval. alpha is
    the least-squares fit over rings weighted by their number of samples.
  - Texture comparisons. Orientation, frequency and slope of an invisible
    residual mean nothing (the order parameter ignores amplitude), so their
    pair terms are weighted by min(v_i, v_j), v = min(1, var_band(R^L) / 1.5^2)
    with var_band the residual's variance inside the band: full weight once
    its RMS there reaches one dE JND. Frequency and
    slope differences are measured against the JND widened by the sampling
    standard errors of both tiles (and of their LR references where used),
    so two crops of one texture do not count as inconsistent.
  - JND units: dE00 1.5 per Lab coordinate; for Q the rotation of a fully
    oriented texture by 4 degrees; for frequency 5% of the pair's mean.
    Nominal values where the spec gives none: 0.14 for alpha (power-spectrum
    slope; human slope discrimination is best for slopes of 2.8 to 3.6,
    Knill, Field & Kersten, JOSA A 1990); 20% for covariance entries and
    Gabor energies (about 10% RMS contrast), with a floor of a 1.5 dE texture
    so that invisible residuals do not count; half a defect per corner.
  - Advice. r_i is the spec's vector (raw units). The printed instruction uses
    `step`, the change that reaches the optimum: psi0_i - psi_i on the colour
    coordinates (the minimiser of F matches the LR) and the affinity-weighted
    mean of the restriction-map differences on the texture coordinates
    (same-material, visible tiles).
  - The LR reference is the degraded input. E_RG is therefore nonzero for a
    perfect SR output; the other terms compare tiles with each other, where a
    shared bias cancels.

Requires numpy and scipy; OpenCV or Pillow to read files.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import fft as sfft
from scipy import ndimage

DEFAULTS = dict(
    eta=1.0,                # covariance weight in E_RG, anchor weight in r_i
    kappa="auto",           # LR -> SR covariance expansion
    beta_grad=1.0,          # gradient weight in E_GJ
    mu_defect=1.0,          # defect weight in E_PCP
    sigma_cyto=5.0,         # Lab bandwidth of the material affinity
    tau_band=0.69,          # LR band-energy bandwidth of the affinity (natural log, ~ factor 2)
    prior_pairs=1.0,        # pseudo-pairs of zero divergence in the affinity-weighted means
    defect_min_order=0.01,
    weights=(1.0, 1.0, 1.0, 1.0, 1.0),
    jnd_de=1.5,             # dE00
    jnd_angle_deg=4.0,      # stroke angle
    jnd_freq=0.05,          # wavevector, relative to the mean radial frequency
    jnd_alpha=0.14,         # spectral slope
    jnd_cov=0.2,            # covariance entries, relative
    jnd_gram=0.2,           # Gabor energies, relative
)

_CHUNK = 4096

TERMS = ("E_RG", "E_GJ", "E_PCP", "E_cyto", "E_sheaf")
DIMS = ("mu_L", "mu_a", "mu_b", "cov_LL", "cov_La", "cov_Lb", "cov_aa", "cov_ab", "cov_bb",
        "Q_xx", "Q_xy", "k_x", "k_y", "alpha")
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"}


# ---------------------------------------------------------------- input / colour

def _read_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import cv2
        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if a is not None:
            return a[..., 2::-1] if a.ndim == 3 and a.shape[2] >= 3 else a
    except ImportError:
        pass
    from PIL import Image
    with Image.open(path) as im:
        if im.mode.startswith("I;16"):
            return np.array(im).astype(np.uint16)
        return np.asarray(im.convert("RGB"))


def load_rgb(src):
    """Path, array or tensor -> (HxWx3 array in its native dtype, value of white)."""
    if isinstance(src, (str, Path)):
        src = _read_file(src)
    if hasattr(src, "detach"):
        src = src.detach().cpu().numpy()
    a = np.asarray(src)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[2] not in (1, 3, 4):
        a = np.moveaxis(a, 0, -1)
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim != 3 or a.shape[2] not in (1, 3, 4):
        raise ValueError(f"expected an RGB or grey image, got shape {a.shape}")
    if a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    a = a[..., :3]
    if a.dtype == np.uint8:
        return a, 255.0
    if a.dtype == np.uint16:
        return a, 65535.0
    if np.issubdtype(a.dtype, np.integer):
        return a, (65535.0 if a.max() > 255 else 255.0)
    return a, (255.0 if float(np.nanmax(a)) > 1.5 else 1.0)


_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]])
_D65 = np.array([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb, white=1.0, rows=512):
    """sRGB -> CIELAB (D65, 2 degree observer), float32, in row chunks."""
    m = (_RGB2XYZ / _D65[:, None]).T.astype(np.float32)
    out = np.empty(rgb.shape[:2] + (3,), np.float32)
    for r0 in range(0, rgb.shape[0], rows):
        c = np.clip(rgb[r0:r0 + rows].astype(np.float32) / np.float32(white), 0.0, 1.0)
        lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        t = lin @ m
        f = np.where(t > 216 / 24389, np.cbrt(t), (24389 / 27 * t + 16) / 116)
        out[r0:r0 + rows, :, 0] = 116 * f[..., 1] - 16
        out[r0:r0 + rows, :, 1] = 500 * (f[..., 0] - f[..., 1])
        out[r0:r0 + rows, :, 2] = 200 * (f[..., 1] - f[..., 2])
    return out


def ciede2000(lab1, lab2):
    """CIEDE2000 colour difference (kL = kC = kH = 1), elementwise over [..., 3]."""
    L1, a1, b1 = np.moveaxis(np.asarray(lab1, np.float64), -1, 0)
    L2, a2, b2 = np.moveaxis(np.asarray(lab2, np.float64), -1, 0)
    c7 = ((np.hypot(a1, b1) + np.hypot(a2, b2)) / 2) ** 7
    g = 0.5 * (1 - np.sqrt(c7 / (c7 + 25.0 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    achrom = (c1p * c2p) == 0
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(achrom, 0.0, dh)
    dL, dC = L2 - L1, c2p - c1p
    dH = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh) / 2)
    Lm, Cm, hs = (L1 + L2) / 2, (c1p + c2p) / 2, h1p + h2p
    hm = np.where(np.abs(h1p - h2p) <= 180, hs / 2, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2))
    hm = np.where(achrom, hs, hm)
    T = (1 - 0.17 * np.cos(np.radians(hm - 30)) + 0.24 * np.cos(np.radians(2 * hm))
         + 0.32 * np.cos(np.radians(3 * hm + 6)) - 0.20 * np.cos(np.radians(4 * hm - 63)))
    cm7 = Cm ** 7
    rt = -np.sin(np.radians(60 * np.exp(-(((hm - 275) / 25) ** 2)))) * 2 * np.sqrt(cm7 / (cm7 + 25.0 ** 7))
    tl = dL / (1 + 0.015 * (Lm - 50) ** 2 / np.sqrt(20 + (Lm - 50) ** 2))
    tc = dC / (1 + 0.045 * Cm)
    th = dH / (1 + 0.015 * Cm * T)
    return np.sqrt(tl ** 2 + tc ** 2 + th ** 2 + rt * tc * th)


# ---------------------------------------------------------------- stage 1: RG split

def _keys(x, a=-0.5):
    x = abs(x)
    if x <= 1:
        return (a + 2) * x ** 3 - (a + 3) * x ** 2 + 1
    if x < 2:
        return a * x ** 3 - 5 * a * x ** 2 + 8 * a * x - 4 * a
    return 0.0


def _upsample_axis(img, s, axis):
    img = np.moveaxis(img, axis, 0)
    n = img.shape[0]
    pad = np.concatenate([np.repeat(img[:1], 2, 0), img, np.repeat(img[-1:], 2, 0)], 0)
    out = np.empty((n, s) + img.shape[1:], np.float32)
    for r in range(s):
        u = (r + 0.5) / s - 0.5
        base = math.floor(u)
        acc = 0.0
        for t in (-1, 0, 1, 2):
            k = base + t + 2
            acc = acc + np.float32(_keys(u - base - t)) * pad[k:k + n]
        out[:, r] = acc
    return np.moveaxis(out.reshape((n * s,) + img.shape[1:]), 0, axis)


def upsample_bicubic(img, s):
    """U_s: bicubic (Keys, a = -0.5) upsampling by an integer factor, pixel centres aligned."""
    return _upsample_axis(_upsample_axis(np.asarray(img, np.float32), s, 0), s, 1)


def coarse_grain(img, s, sigma):
    """D_s: Gaussian anti-aliasing, then the mean of every s x s block."""
    h, w = img.shape[0] // s, img.shape[1] // s
    sig = (sigma, sigma) + (0,) * (img.ndim - 2)
    blur = ndimage.gaussian_filter(img[:h * s, :w * s], sigma=sig, mode="reflect", truncate=4.0)
    return blur.reshape((h, s, w, s) + img.shape[2:]).mean(axis=(1, 3))


def rg_split(img, s, sigma):
    """Stage 1: macro image M = D_s(img) and residual R = img - U_s(M)."""
    H, W = img.shape[:2]
    ph, pw = (-H) % s, (-W) % s
    x = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect") if ph or pw else img
    M = coarse_grain(x, s, sigma)
    return M, (x - upsample_bicubic(M, s))[:H, :W]


# ---------------------------------------------------------------- stage 2: psi

_SCHARR_X = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], np.float32) / 32
_SCHARR_Y = -_SCHARR_X.T  # y points up the image, so angles are counter-clockwise


def nematic(lum, eps=1e-12):
    """Dominant stroke angle theta in [0, pi), order s in [0, 1], Q = s (cos 2theta, sin 2theta)."""
    gx = ndimage.correlate(lum, _SCHARR_X, mode="reflect")
    gy = ndimage.correlate(lum, _SCHARR_Y, mode="reflect")
    jxx, jxy, jyy = float(np.mean(gx * gx)), float(np.mean(gx * gy)), float(np.mean(gy * gy))
    theta = (0.5 * math.atan2(2 * jxy, jxx - jyy) + math.pi / 2) % math.pi
    order = (math.hypot(jxx - jyy, 2 * jxy) / (jxx + jyy + eps)) ** 2
    return theta, order, (order * math.cos(2 * theta), order * math.sin(2 * theta))


def _freqs(h, w):
    return sfft.fftfreq(w)[None, :], -sfft.fftfreq(h)[:, None]


_BANKS = {}


def gabor_bank(h, w, f_lo, n_orient=4, rel_bw=0.3):
    """Even Gabor transfer functions (n_filters, h*w): octave scales inside [f_lo, 1/2]."""
    key = (h, w, round(f_lo, 9))
    if key not in _BANKS:
        kx, ky = _freqs(h, w)
        centres, fc = [], 0.5 / math.sqrt(2)
        while fc >= f_lo * math.sqrt(2) * (1 - 1e-6):
            centres.append(fc)
            fc /= 2
        filters = []
        for fc in centres or [(f_lo + 0.5) / 2]:
            var = 2 * (rel_bw * fc) ** 2
            for k in range(n_orient):
                cx, cy = fc * math.cos(math.pi * k / n_orient), fc * math.sin(math.pi * k / n_orient)
                g = (np.exp(-((kx - cx) ** 2 + (ky - cy) ** 2) / var)
                     + np.exp(-((kx + cx) ** 2 + (ky + cy) ** 2) / var))
                filters.append(g.astype(np.float32).ravel())
        _BANKS[key] = np.stack(filters)
    return _BANKS[key]


_ENBW = 1.5  # equivalent noise bandwidth of the Hann window, bins per axis


def spectral(lum, f_lo):
    """Texture statistics of `lum` in the band [f_lo, 1/2].

    Returns the mean wavevector (upper half-plane), mean radial frequency, spectral slope
    alpha (least squares over rings, weighted by their number of Fourier samples), Gabor Gram,
    the sampling standard errors of the mean radial frequency and of alpha (periodogram
    ordinates as independent exponentials, widened by the window's noise bandwidth), and the
    variance of `lum` inside the band.
    """
    h, w = lum.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F = sfft.fft2((lum - lum.mean()) * win, workers=-1)
    P = F.real ** 2 + F.imag ** 2
    kx, ky = _freqs(h, w)
    f = np.sqrt(kx ** 2 + ky ** 2)
    upper = (f >= f_lo) & (f <= 0.5) & (ky >= 0)
    pu, fu = P[upper].astype(np.float64), f[upper]
    tot = pu.sum()
    kbar, fbar, se_f = (0.0, 0.0), 0.5 * (f_lo + 0.5), 0.0
    if tot > 0:
        kbar = (float((np.broadcast_to(kx, P.shape)[upper] * pu).sum() / tot),
                float((np.broadcast_to(ky, P.shape)[upper] * pu).sum() / tot))
        fbar = float((fu * pu).sum() / tot)
        se_f = _ENBW * math.sqrt(float(((fu - fbar) ** 2 * pu ** 2).sum())) / tot
    ring = np.floor(fu * min(h, w)).astype(np.int64)
    cnt = np.bincount(ring)
    psum, p2sum, fsum = (np.bincount(ring, weights=v) for v in (pu, pu ** 2, fu))
    ok = (cnt > 0) & (psum > 0)
    alpha, se_a = 0.0, 0.0
    if ok.sum() >= 3:
        x, y, wt = np.log(fsum[ok] / cnt[ok]), np.log(psum[ok] / cnt[ok]), cnt[ok].astype(np.float64)
        alpha = -float(np.polyfit(x, y, 1, w=np.sqrt(wt))[0])
        dx = x - (wt * x).sum() / wt.sum()
        keff = psum[ok] ** 2 / p2sum[ok]
        se_a = _ENBW * math.sqrt(float((wt ** 2 * dx ** 2 / keff).sum())) / float((wt * dx ** 2).sum())
    norm = h * w * float((win ** 2).sum())
    band_var = float(P[(f >= f_lo) & (f <= 0.5)].astype(np.float64).sum()) / norm
    bank = gabor_bank(h, w, f_lo)
    gram = ((bank * P.ravel()[None, :]) @ bank.T).astype(np.float64) / norm
    return kbar, fbar, alpha, gram, se_f, se_a, band_var


def tile_features(canvas, s, sigma, f_lo):
    """Stage 2 on one Lab canvas: (M, dict of mu, cov, theta, order, Q, kbar, fbar, alpha, gram,
    se_f, se_alpha, band_var)."""
    M, R = rg_split(canvas, s, sigma)
    px = canvas.reshape(-1, 3).astype(np.float64)
    lum = np.ascontiguousarray(R[..., 0])
    theta, order, Q = nematic(lum)
    ft = dict(mu=px.mean(0), cov=np.cov(px, rowvar=False), theta=theta, order=order, Q=Q)
    (ft["kbar"], ft["fbar"], ft["alpha"], ft["gram"], ft["se_f"], ft["se_alpha"],
     ft["band_var"]) = spectral(lum, f_lo)
    return M, ft


def _sqrtm_psd(A):
    w, V = np.linalg.eigh(0.5 * (A + np.swapaxes(A, -1, -2)))
    return (V * np.sqrt(np.clip(w, 0, None))[..., None, :]) @ np.swapaxes(V, -1, -2)


def bures_w2(m1, S1, m2, S2):
    """Squared 2-Wasserstein distance between Gaussians N(m1, S1), N(m2, S2); batched."""
    r2 = _sqrtm_psd(S2)
    cross = _sqrtm_psd(r2 @ S1 @ r2)
    tr = np.trace(S1 + S2 - 2 * cross, axis1=-2, axis2=-1)
    return np.sum((m1 - m2) ** 2, axis=-1) + np.clip(tr, 0, None)


# ---------------------------------------------------------------- stage 3 helpers

def tile_starts(n, tile, overlap):
    """Tile origins along one axis; the last tile is flush with the border."""
    if tile >= n:
        return [0]
    starts = list(range(0, n - tile + 1, tile - overlap))
    if starts[-1] != n - tile:
        starts.append(n - tile)
    return starts


def _grads(x):
    gx = np.stack([ndimage.correlate(x[..., k], _SCHARR_X, mode="nearest") for k in range(3)], -1)
    gy = np.stack([ndimage.correlate(x[..., k], _SCHARR_Y, mode="nearest") for k in range(3)], -1)
    return gx, gy


def _jump(img, c, r0, r1, beta):
    """Per-row colour + gradient step between columns c - 1 and c over rows r0:r1."""
    strip = np.ascontiguousarray(img[r0:r1, c - 3:c + 3], np.float32)
    val = np.sum((strip[:, 3] - strip[:, 2]) ** 2, -1)
    if beta:
        gx, gy = _grads(strip)
        val = val + beta * (np.sum((gx[:, 3] - gx[:, 2]) ** 2, -1) + np.sum((gy[:, 3] - gy[:, 2]) ** 2, -1))
    return val


def _excess(im, c, r0, r1, beta, offsets):
    """Median over rows of the jump at line c minus the median jump at control lines c +- offsets,
    and the median control jump."""
    ctrl = np.median([_jump(im, c + o, r0, r1, beta) for o in offsets + tuple(-o for o in offsets)], 0)
    return float(np.median(_jump(im, c, r0, r1, beta) - ctrl)), float(np.median(ctrl))


def _gj_stitched(img, lr, ys, xs, th, tw, s, beta, jnd2, offsets=(3, 6), lr_offsets=(2, 3)):
    """Seam score of every overlap band of a stitched image, in JND units.

    At each candidate line (start, middle and end of the band) the excess jump over control lines
    nearby, as a median over rows: a seam is a step along the whole line, an image edge crossing it
    covers a few rows. The excess is divided by jnd2 plus the control jump (masking by the local
    texture) and multiplied by exp(-the same ratio in the LR at that line), because a step the LR
    also has is image content. The band's score is its largest candidate.
    """
    reach, reach0 = max(offsets) + 3, max(lr_offsets) + 3

    def seam(im, im0, a, b, r0, r1):
        best = None
        for c in sorted({a, (a + b) // 2, b}):
            if c - reach < 0 or c + reach > im.shape[1]:
                continue
            ex, ctrl = _excess(im, c, r0, r1, beta, offsets)
            gate, c0 = 1.0, int(round(c / s))
            if reach0 <= c0 <= im0.shape[1] - reach0:
                ex0, ctrl0 = _excess(im0, c0, r0 // s, max(r1 // s, r0 // s + 1), beta, lr_offsets)
                gate = math.exp(-max(ex0, 0.0) / (jnd2 + ctrl0))
            v = gate * max(ex, 0.0) / (jnd2 + ctrl)
            best = v if best is None else max(best, v)
        return best

    vals, imT, lrT = [], img.transpose(1, 0, 2), lr.transpose(1, 0, 2)
    for y0 in ys:
        for c in range(len(xs) - 1):
            vals.append(seam(img, lr, s * xs[c + 1], s * (xs[c] + tw), s * y0, s * (y0 + th)))
    for r in range(len(ys) - 1):
        for x0 in xs:
            vals.append(seam(imT, lrT, s * ys[r + 1], s * (ys[r] + th), s * x0, s * (x0 + tw)))
    return [v for v in vals if v is not None]


def _gj_canvases(canv, ys, xs, th, tw, s, beta):
    """The spec's E_GJ: two neighbouring canvases compared on their shared pixels."""
    vals, nc = [], len(xs)
    for r in range(len(ys)):
        for c in range(nc):
            for rr, cc in ((r, c + 1), (r + 1, c)):
                if rr >= len(ys) or cc >= nc:
                    continue
                yi, xi, yj, xj = s * ys[r], s * xs[c], s * ys[rr], s * xs[cc]
                y_lo, y_hi = max(yi, yj), min(yi, yj) + s * th
                x_lo, x_hi = max(xi, xj), min(xi, xj) + s * tw
                if y_hi <= y_lo or x_hi <= x_lo:
                    continue
                A = canv[r * nc + c][y_lo - yi:y_hi - yi, x_lo - xi:x_hi - xi]
                B = canv[rr * nc + cc][y_lo - yj:y_hi - yj, x_lo - xj:x_hi - xj]
                v = float(np.sum((A - B) ** 2, -1).mean())
                if beta:
                    (ax, ay), (bx, by) = _grads(A), _grads(B)
                    v += beta * float((np.sum((ax - bx) ** 2, -1) + np.sum((ay - by) ** 2, -1)).mean())
                vals.append(v)
    return vals


def _wmean(w, x, prior):
    """Affinity-weighted mean with `prior` pseudo-pairs of zero divergence."""
    return float(np.sum(w * x)) / (float(np.sum(w)) + prior)


def _winding(thetas):
    """Winding number of a headless direction field around a cycle, in {0, +-1/2, +-1}."""
    tot = 0.0
    for a in range(len(thetas)):
        dlt = thetas[(a + 1) % len(thetas)] - thetas[a]
        tot += (dlt + math.pi / 2) % math.pi - math.pi / 2
    return round(2 * tot / (2 * math.pi)) / 2


def _advice(step, psi, scale, fbar, vis):
    """One instruction from the largest entry of the corrective step in JND units."""
    z = step / scale
    z[9:] *= vis
    k = int(np.argmax(np.abs(z)))
    if abs(z[k]) < 1:
        return "within one JND on every coordinate"
    v = step[k]
    if k == 0:
        return f"{'raise' if v > 0 else 'lower'} lightness by {abs(v):.1f} L* units"
    if k == 1:
        return f"shift colour toward {'red (+a*)' if v > 0 else 'green (-a*)'} by {abs(v):.1f} units"
    if k == 2:
        return f"shift colour toward {'yellow (+b*)' if v > 0 else 'blue (-b*)'} by {abs(v):.1f} units"
    if k in (3, 6, 8):
        name = {3: "lightness", 6: "red-green", 8: "yellow-blue"}[k]
        pct = 100 * v / max(psi[k], 1e-9)
        return f"{'increase' if v > 0 else 'decrease'} {name} contrast (variance) by {abs(pct):.0f}%"
    if k in (4, 5, 7):
        return f"change the {DIMS[k][4:]} colour covariance by {v:+.1f}"
    if k in (9, 10):
        q0, q1 = psi[9:11], psi[9:11] + step[9:11]
        rot = math.degrees(0.5 * ((math.atan2(q1[1], q1[0]) - math.atan2(q0[1], q0[0]) + math.pi)
                                  % (2 * math.pi) - math.pi))
        if abs(rot) >= 1:
            return f"rotate the invented texture {abs(rot):.0f} degrees {'counter-clockwise' if rot > 0 else 'clockwise'}"
        return f"make the invented texture {'more' if math.hypot(*q1) > math.hypot(*q0) else 'less'} oriented"
    if k == 11:
        return (f"make the invented texture {'finer' if v > 0 else 'coarser'} by "
                f"{100 * abs(v) / max(fbar, 1e-12):.0f}% in frequency")
    return ("texture too smooth: add finer grain" if v < 0 else "texture too grainy: soften it") + \
        f" (spectral slope {v:+.2f})"


def _match_scale(sr, lr, scale):
    H, W = sr.shape[:2]
    H0, W0 = lr.shape[:2]
    s = int(scale) if scale else int(round((H / H0 + W / W0) / 2))
    if s < 2 or abs(H - s * H0) >= s or abs(W - s * W0) >= s:
        raise ValueError(f"SR {W}x{H} is not an integer multiple (>= 2) of LR {W0}x{H0}"
                         + (f" at scale {scale}" if scale else ""))
    h0, w0 = min(H0, H // s), min(W0, W // s)
    notes = []
    if (h0, w0) != (H0, W0) or (s * h0, s * w0) != (H, W):
        notes.append(f"cropped to LR {w0}x{h0} / SR {s * w0}x{s * h0} to match the scale")
    return s, sr[:s * h0, :s * w0], lr[:h0, :w0], notes


# ---------------------------------------------------------------- the score

def wcsi_score(sr, lr, scale=None, tile=None, overlap=None, tiles=None, return_details=False, **params):
    """WCSI of a super-resolved image `sr` against its low-resolution reference `lr`.

    sr, lr   paths, arrays or tensors. scale: integer SR/LR factor (inferred if None).
    tile, overlap   the SR model's tile grid in LR pixels (nominal grid if None).
    tiles    optional unblended SR tile outputs (paths or arrays), row-major over the
             grid, each (scale*tile)^2; enables the spec's overlap comparison in E_GJ.
    params   any key of DEFAULTS (eta, kappa, weights, sigma_cyto, jnd_*, ...).
    Returns the score in (0, 100], or a dict with every term, per-tile psi, psi0,
    r_i and advice when return_details is True.
    """
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown parameters: {sorted(unknown)}")
    p = {**DEFAULTS, **params}
    if len(p["weights"]) != len(TERMS):
        raise ValueError(f"weights needs {len(TERMS)} values, one per term {TERMS}")
    sr_rgb, sr_white = load_rgb(sr)
    lr_rgb, lr_white = load_rgb(lr)
    s, sr_rgb, lr_rgb, notes = _match_scale(sr_rgb, lr_rgb, scale)
    H0, W0 = lr_rgb.shape[:2]
    if tile is None:
        tile = max(8, min(1024 // s, min(H0, W0) // 3))
        notes.append(f"nominal grid (tile {tile} LR px); pass tile/overlap to match the SR model")
    if overlap is None:
        overlap = tile // 16
    th, tw = min(tile, H0), min(tile, W0)
    if not 0 <= overlap < min(th, tw):
        raise ValueError(f"overlap {overlap} must be in [0, {min(th, tw)})")
    ys, xs = tile_starts(H0, th, overlap), tile_starts(W0, tw, overlap)
    nr, nc = len(ys), len(xs)
    N = nr * nc
    sr_lab = srgb_to_lab(sr_rgb, sr_white)
    lr_lab = srgb_to_lab(lr_rgb, lr_white)
    del sr_rgb, lr_rgb

    canv = None
    if tiles is not None:
        tiles = list(tiles)
        if len(tiles) != N:
            raise ValueError(f"got {len(tiles)} tile outputs for a {nr}x{nc} grid")
        canv = []
        for t in tiles:
            a, wh = load_rgb(t)
            if a.shape[:2] != (s * th, s * tw):
                raise ValueError(f"tile output {a.shape[1]}x{a.shape[0]}, expected {s * tw}x{s * th}")
            canv.append(srgb_to_lab(a, wh))

    sigma, f_lo = 0.4 * s, 1.0 / (2 * s)
    feats, feats0, de2 = [], [], []
    for r, y0 in enumerate(ys):
        for c, x0 in enumerate(xs):
            Hc = canv[r * nc + c] if canv is not None else sr_lab[s * y0:s * (y0 + th), s * x0:s * (x0 + tw)]
            Lc = lr_lab[y0:y0 + th, x0:x0 + tw]
            M, ft = tile_features(Hc, s, sigma, f_lo)
            de2.append(float(np.mean(ciede2000(M, Lc) ** 2)))
            feats.append(ft)
            feats0.append(tile_features(Lc, s, sigma, f_lo)[1])

    def col(key, src=feats):
        return np.array([f[key] for f in src], np.float64)

    keys = ("mu", "cov", "Q", "kbar", "fbar", "alpha", "gram", "theta", "order", "band_var", "se_f", "se_alpha")
    mu, cov, Q, kbar, fbar, alpha, gram, theta, order, band_var, se_f, se_a = (col(k) for k in keys)
    mu0, cov0, Q0, kbar0, fbar0, alpha0, gram0, theta0, order0, band_var0, se_f0, se_a0 = (col(k, feats0) for k in keys)

    if p["kappa"] == "auto":
        kappa = float(np.trace(cov, axis1=1, axis2=2).sum() / max(np.trace(cov0, axis1=1, axis2=2).sum(), 1e-12))
    else:
        kappa = float(p["kappa"])

    # psi and psi0 as in the spec (psi0: the LR tile one scale down, covariance times kappa). The 13
    # compared coordinates (phi) replace the wavevector by the mean radial frequency. Per tile and
    # coordinate: JND and sampling standard error; visibility of the residual texture.
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
                            np.full((N, 2), jnd_q),
                            p["jnd_freq"] * fbar[:, None],
                            np.full((N, 1), jnd_a)], 1)
    se, se0 = np.zeros_like(jnd_t), np.zeros_like(jnd_t)
    se[:, 11], se[:, 12], se0[:, 11], se0[:, 12] = se_f, se_a, se_f0, se_a0
    vis, vis0 = np.minimum(1.0, band_var / jnd_de ** 2), np.minimum(1.0, band_var0 / jnd_de ** 2)
    texture = np.arange(d.shape[1]) >= 9
    G = gram + floor * np.eye(gram.shape[1])
    gnorm = np.sqrt(np.sum(G ** 2, (1, 2)))
    lr_band = np.log(np.diagonal(gram0, axis1=1, axis2=2) + floor)

    # E_RG: coarse-grained SR against LR, plus the canalized colour distribution
    w2 = bures_w2(mu, cov, mu0, kappa * cov0)
    e_rg = float(np.mean(np.array(de2) + p["eta"] * w2) / (jnd_de ** 2 * (1 + p["eta"])))

    # all tile pairs, their grid relation and material affinity in the LR
    rc = [(r, c) for r in range(nr) for c in range(nc)]
    I, J = np.triu_indices(N, 1)
    dr, dc = np.abs(I // nc - J // nc), np.abs(I % nc - J % nc)
    near, far = dr + dc == 1, np.maximum(dr, dc) > 1
    corners = np.array([[r * nc + c, r * nc + c + 1, (r + 1) * nc + c + 1, (r + 1) * nc + c]
                        for r in range(nr - 1) for c in range(nc - 1)], np.int64).reshape(-1, 4)
    w = np.empty(len(I))
    for k0 in range(0, len(I), _CHUNK):
        a, b = I[k0:k0 + _CHUNK], J[k0:k0 + _CHUNK]
        w[k0:k0 + _CHUNK] = np.exp(-bures_w2(mu0[a], cov0[a], mu0[b], cov0[b]) / (2 * p["sigma_cyto"] ** 2)
                                   - np.mean((lr_band[a] - lr_band[b]) ** 2, 1) / (2 * p["tau_band"] ** 2))

    # E_GJ: seam continuity
    if canv is not None:
        gj = [v / jnd_de ** 2 for v in _gj_canvases(canv, ys, xs, th, tw, s, p["beta_grad"])]
        notes.append("E_GJ: overlap comparison of the tile outputs")
    else:
        gj = _gj_stitched(sr_lab, lr_lab, ys, xs, th, tw, s, p["beta_grad"], jnd_de ** 2)
        notes.append("E_GJ: stitched image, excess jump at seam lines (pass tiles= for the overlap comparison)")
    e_gj = float(np.mean(gj)) if gj else None

    # E_PCP: nematic elasticity between same-material neighbours + corner defects
    e_pcp = pcp_edges = pcp_defects = None
    n_corners = 0
    if near.any():
        a, b = I[near], J[near]
        pcp_edges = _wmean(w[near], np.minimum(vis[a], vis[b]) * np.sum((Q[a] - Q[b]) ** 2, 1) / jnd_q ** 2,
                           p["prior_pairs"])
        dq = [abs(_winding(theta[ids]) - _winding(theta0[ids])) for ids in corners
              if min((order * vis)[ids].min(), (order0 * vis0)[ids].min()) >= p["defect_min_order"]]
        n_corners = len(dq)
        pcp_defects = float(np.mean(dq)) / 0.5 if dq else 0.0
        e_pcp = pcp_edges + p["mu_defect"] * pcp_defects

    # E_cyto: distant tiles that are the same material in LR must invent the same texture
    e_cyto = None
    cyto_parts = {}
    if far.any():
        a, b, wf = I[far], J[far], w[far]
        div = np.empty((len(a), 3))
        for k0 in range(0, len(a), _CHUNK):
            x, y = a[k0:k0 + _CHUNK], b[k0:k0 + _CHUNK]
            jf2 = (p["jnd_freq"] * 0.5 * (fbar[x] + fbar[y])) ** 2 + se_f[x] ** 2 + se_f[y] ** 2
            ja2 = jnd_a ** 2 + se_a[x] ** 2 + se_a[y] ** 2
            jg = p["jnd_gram"] * 0.5 * (gnorm[x] + gnorm[y])
            v = np.minimum(vis[x], vis[y])
            div[k0:k0 + _CHUNK] = np.stack([v * (fbar[x] - fbar[y]) ** 2 / jf2, v * (alpha[x] - alpha[y]) ** 2 / ja2,
                                            np.sum((G[x] - G[y]) ** 2, (1, 2)) / jg ** 2], 1)
        cyto_parts = {k: _wmean(wf, div[:, n], p["prior_pairs"]) for n, k in enumerate(("freq", "alpha", "gabor"))}
        cyto_parts["affinity_sum"] = float(wf.sum())
        e_cyto = cyto_parts["freq"] + cyto_parts["alpha"] + cyto_parts["gabor"]
        if wf.sum() < p["prior_pairs"]:
            notes.append(f"E_cyto: distant tiles share little material in the LR (total affinity {wf.sum():.2f}), "
                         "so the term is shrunk toward 0")

    # E_sheaf over all pairs, the loop holonomy, and the per-tile sums of the corrective vector
    e_sheaf = sheaf_edges = None
    hol_max = 0.0
    cons, wsum = np.zeros_like(d), np.zeros_like(d)
    sheaf_parts = {}
    if len(I):
        num = np.zeros(d.shape[1])
        for k0 in range(0, len(I), _CHUNK):
            x, y = I[k0:k0 + _CHUNK], J[k0:k0 + _CHUNK]
            wk = w[k0:k0 + _CHUNK, None] * np.where(texture, np.minimum(vis[x], vis[y])[:, None], 1.0)
            conf = np.where(texture, np.minimum(vis0[x], vis0[y])[:, None], 1.0)
            diff = phi[y] - phi[x] - conf * (phi0[y] - phi0[x])
            scale2 = ((0.5 * (jnd_t[x] + jnd_t[y])) ** 2 + se[x] ** 2 + se[y] ** 2
                      + conf ** 2 * (se0[x] ** 2 + se0[y] ** 2))
            num += (wk * diff ** 2 / scale2).sum(0)
            for k in range(d.shape[1]):
                cons[:, k] += np.bincount(x, wk[:, k] * diff[:, k], N) - np.bincount(y, wk[:, k] * diff[:, k], N)
                wsum[:, k] += np.bincount(x, wk[:, k], N) + np.bincount(y, wk[:, k], N)
        num /= float(w.sum()) + p["prior_pairs"]
        sheaf_parts = dict(mean=float(num[:3].sum()), cov=float(num[3:9].sum()), Q=float(num[9:11].sum()),
                           freq=float(num[11]), alpha=float(num[12]))
        sheaf_edges = float(num.sum())
        hol_e = 0.0
        if len(corners):
            hol = sum(d[corners[:, (k + 1) % 4]] - d[corners[:, k]] for k in range(4)) / jnd_t[corners].mean(1)
            hol_e = float(np.mean(np.sum(hol ** 2, 1)))
            hol_max = float(np.abs(hol).max())
        e_sheaf = sheaf_edges + hol_e

    # r_i = -1/2 grad F as in the spec (raw units) and step = the change that reaches the optimum,
    # mapped to psi: the wavevector scales with the relative change of radial frequency
    r_c = cons - p["eta"] * ~texture * d
    step_c = np.where(texture, cons / np.maximum(wsum, 1e-12), -d)

    def to_psi(v):
        return np.concatenate([v[:, :11], kbar * (v[:, 11:12] / np.maximum(fbar[:, None], 1e-12)), v[:, 12:]], 1)

    rvec, step = to_psi(r_c), to_psi(step_c)

    values = dict(zip(TERMS, (e_rg, e_gj, e_pcp, e_cyto, e_sheaf)))
    used = [(wt, values[k]) for wt, k in zip(p["weights"], TERMS) if values[k] is not None]
    F = sum(wt * v for wt, v in used)
    F_jnd = sum(wt for wt, _ in used)
    score = 100.0 / (1.0 + F / F_jnd) if F_jnd > 0 else 100.0
    for k in TERMS:
        if values[k] is None:
            notes.append(f"{k}: undefined for a {nr}x{nc} grid, left out of F")
    if not return_details:
        return score

    return dict(
        wcsi=score, F_total=F, F_jnd=F_jnd, terms=values,
        parts=dict(E_RG_dE00_sq=float(np.mean(de2)), E_RG_W2_sq=float(np.mean(w2)),
                   E_PCP_edges=pcp_edges, E_PCP_defects=pcp_defects, corners_scored=n_corners,
                   E_cyto=cyto_parts, E_sheaf_edges=sheaf_edges, E_sheaf=sheaf_parts, holonomy_max_abs=hol_max,
                   affinity_mean=float(w.mean()) if len(w) else None),
        scale=s, kappa=kappa,
        grid=dict(rows=nr, cols=nc, tile=[th, tw], overlap=overlap, ys=ys, xs=xs),
        jnd=dict(dE00=jnd_de, Q=jnd_q, freq_rel=p["jnd_freq"], alpha=jnd_a,
                 cov_rel=p["jnd_cov"], gram_rel=p["jnd_gram"]),
        dims=list(DIMS),
        tiles=[dict(row=r, col=c, y0=ys[r], x0=xs[c], psi=psi[i].tolist(),
                    psi0=psi0[i].tolist(), r=rvec[i].tolist(), step=step[i].tolist(),
                    texture_visibility=float(vis[i]),
                    advice=_advice(step_c[i], psi[i], np.sqrt(jnd_t[i] ** 2 + se[i] ** 2 + se0[i] ** 2),
                                   fbar[i], vis[i]))
               for i, (r, c) in enumerate(rc)],
        notes=notes,
    )


# ---------------------------------------------------------------- CLI

def _natural_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="WCSI of a super-resolved image against its LR reference.")
    ap.add_argument("sr", help="super-resolved image")
    ap.add_argument("lr", help="low-resolution reference")
    ap.add_argument("--scale", type=int, help="SR/LR factor (default: inferred)")
    ap.add_argument("--tile", type=int, help="tile size in LR pixels, as used by the SR model")
    ap.add_argument("--overlap", type=int, help="tile overlap in LR pixels (default: tile // 16)")
    ap.add_argument("--tiles-dir", help="unblended tile outputs; sorted file names = row-major grid order")
    ap.add_argument("--weights", help="w1..w5 for E_RG,E_GJ,E_PCP,E_cyto,E_sheaf, comma separated")
    ap.add_argument("--kappa", default="auto", help="covariance expansion LR -> SR, or 'auto'")
    ap.add_argument("--json", help="write every term and the per-tile vectors to this file")
    ap.add_argument("--advice", action="store_true", help="print one instruction per tile")
    args = ap.parse_args(argv)

    params = {}
    if args.weights:
        params["weights"] = tuple(float(v) for v in args.weights.split(","))
        if len(params["weights"]) != 5:
            ap.error("--weights needs 5 values")
    if args.kappa != "auto":
        params["kappa"] = float(args.kappa)
    tiles = None
    if args.tiles_dir:
        tiles = sorted((f for f in Path(args.tiles_dir).iterdir() if f.suffix.lower() in IMAGE_EXT), key=_natural_key)

    d = wcsi_score(args.sr, args.lr, scale=args.scale, tile=args.tile, overlap=args.overlap,
                   tiles=tiles, return_details=True, **params)
    g = d["grid"]
    print(f"WCSI {d['wcsi']:.2f}   scale {d['scale']}x, {g['rows']}x{g['cols']} tiles of "
          f"{g['tile'][1]}x{g['tile'][0]} LR px, overlap {g['overlap']}")
    labels = dict(E_RG="coarse-grained colour fidelity to LR", E_GJ="seam continuity",
                  E_PCP="texture orientation between neighbours, corner defects",
                  E_cyto="texture agreement of distant same-material tiles",
                  E_sheaf="tile statistics against the LR's prediction, all same-material pairs")
    for k in TERMS:
        v = d["terms"][k]
        print(f"  {k:8s} {'n/a' if v is None else f'{v:9.4f}'}   {labels[k]}")
    print(f"  F_total {d['F_total']:.4f}   F_JND {d['F_jnd']:.1f}   kappa {d['kappa']:.3f}   "
          f"holonomy max {d['parts']['holonomy_max_abs']:.1e}")
    for n in d["notes"]:
        print(f"  note: {n}")
    if args.advice:
        for t in d["tiles"]:
            print(f"  tile r{t['row']}c{t['col']}: {t['advice']}")
    if args.json:
        Path(args.json).write_text(json.dumps(d, indent=1))
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
