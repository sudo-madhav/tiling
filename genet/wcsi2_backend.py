"""Array kernels of WCSI-2, on numpy (default) or torch (CPU or CUDA).

Only the per-pixel and per-tile heavy work lives here: CIELAB conversion, the coarse /
residual split, the windowed tile spectra and the phase-congruency rigidity map. Everything
downstream works on small per-tile tables in numpy. Both backends compute the same
numbers (float32 images, complex64 spectra); `test_wcsi2.py` checks that they agree.

    be = get_backend(None)        # numpy + scipy
    be = get_backend("cuda:0")    # torch on that device
    be = get_backend("torch:cpu") # torch on the CPU (parity tests)
"""

import math

import numpy as np
from scipy import fft as sfft
from scipy import ndimage

import wcsi

ENBW = 1.5  # equivalent noise bandwidth of the Hann window, bins per axis


# ---------------------------------------------------------------- band layout (shared)

class BandLayout:
    """Partition of the band [1/(2s), 1/2] cycles/px of an h x w periodogram.

    Octaves are geometric between 1/(2s) and 1/2 (log2 s of them, at least one), octave 0
    just above the LR Nyquist limit. Orientations are four 45 degree sectors of the
    (headless) wavevector angle. Rows of `lum` are the octave x orientation bands followed by
    the three structure-tensor moments kx^2, kx ky, ky^2 over the whole band; rows of
    `chroma` are the octaves.
    """

    def __init__(self, h, w, s, n_orient=4):
        self.h, self.w, self.s, self.n_orient = h, w, s, n_orient
        self.f_lo = 1.0 / (2 * s)
        self.n_oct = max(1, int(round(math.log2(s))))
        kx = np.broadcast_to(sfft.fftfreq(w)[None, :], (h, w))
        ky = np.broadcast_to(-sfft.fftfreq(h)[:, None], (h, w))
        f = np.hypot(kx, ky)
        edges = self.f_lo * (0.5 / self.f_lo) ** (np.arange(self.n_oct + 1) / self.n_oct)
        inband = (f >= self.f_lo) & (f <= 0.5)
        octv = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, self.n_oct - 1)
        ang = np.arctan2(ky, kx) % math.pi
        sector = np.floor((ang + math.pi / (2 * n_orient)) / (math.pi / n_orient)).astype(int) % n_orient
        rows = [(inband & (octv == o) & (sector == t)).ravel() for o in range(self.n_oct) for t in range(n_orient)]
        self.n_lum = len(rows)
        band = np.stack(rows).astype(np.float32)
        moments = np.stack([(kx * kx) * inband, (kx * ky) * inband, (ky * ky) * inband]).reshape(3, -1)
        self.lum = np.concatenate([band, moments.astype(np.float32)], 0)
        self.chroma = np.stack([(inband & (octv == o)).ravel() for o in range(self.n_oct)]).astype(np.float32)
        self.top = (inband & (octv == self.n_oct - 1)).ravel()
        self.lum_count = band.sum(1)
        self.chroma_count = self.chroma.sum(1)
        win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
        self.win = win
        self.win_sum = float(win.sum())
        self.norm = float(h * w * (win.astype(np.float64) ** 2).sum())


_LAYOUTS = {}


def band_layout(h, w, s):
    key = (h, w, s)
    if key not in _LAYOUTS:
        _LAYOUTS[key] = BandLayout(h, w, s)
    return _LAYOUTS[key]


def _keys_weight(x, a=-0.5):
    x = abs(x)
    if x <= 1:
        return (a + 2) * x ** 3 - (a + 3) * x ** 2 + 1
    if x < 2:
        return a * x ** 3 - 5 * a * x ** 2 + 8 * a * x - 4 * a
    return 0.0


def _log_gabor(h, w, n_scale, n_orient, min_wave):
    """Even log-Gabor transfer functions times (1 + sign): one inverse FFT gives even + i odd."""
    ky = sfft.fftfreq(h)[:, None]
    kx = sfft.fftfreq(w)[None, :]
    f = np.hypot(kx, ky)
    ang = np.arctan2(ky, kx)
    sig_theta, sig_l = math.pi / n_orient, math.log(0.55)
    out = np.empty((n_orient, n_scale, h, w), np.float32)
    for level in range(n_scale):
        f0 = 1.0 / (min_wave * 2 ** level)
        radial = np.where(f > 1e-8, np.exp(-0.5 * (np.log(np.maximum(f, 1e-8) / f0) / sig_l) ** 2), 0.0)
        for k in range(n_orient):
            theta = math.pi * k / n_orient
            d1 = (ang - theta + np.pi) % (2 * np.pi) - np.pi
            d2 = (ang - theta) % (2 * np.pi) - np.pi
            angular = np.exp(-0.5 * (d1 / sig_theta) ** 2) + np.exp(-0.5 * (d2 / sig_theta) ** 2)
            sign = np.sign(kx * math.cos(theta) + ky * math.sin(theta))
            out[k, level] = radial * angular * (1.0 + sign)
    return out


_GABORS = {}


def log_gabor(h, w, n_scale=4, n_orient=4, min_wave=3.0):
    key = (h, w, n_scale, n_orient, round(min_wave, 3))
    if key not in _GABORS:
        _GABORS[key] = _log_gabor(h, w, n_scale, n_orient, min_wave)
    return _GABORS[key]


def group_rects(rects):
    """Indices of the rectangles grouped by (h, w), so each group stacks into one batch."""
    groups = {}
    for k, r in enumerate(rects):
        groups.setdefault((r[2], r[3]), []).append(k)
    return groups


# ---------------------------------------------------------------- numpy backend

class NumpyBackend:
    name = "numpy"
    batch = 16

    def lab(self, rgb, white):
        return wcsi.srgb_to_lab(rgb, white)

    def asarray(self, a):
        return np.ascontiguousarray(a, np.float32)

    def numpy(self, a):
        return np.asarray(a)

    def split(self, lab, s, sigma):
        """Macro image M = D_s(lab) (numpy) and residual R = lab - U_s(M). A size that is not a
        multiple of s is padded symmetrically for the split and cropped back."""
        H, W = lab.shape[:2]
        ph, pw = (-H) % s, (-W) % s
        x = np.pad(lab, ((0, ph), (0, pw), (0, 0)), mode="symmetric") if ph or pw else lab
        M = wcsi.coarse_grain(x, s, sigma)
        return M, (x - wcsi.upsample_bicubic(M, s))[:H, :W]

    def lowpass(self, img, sigma):
        return ndimage.gaussian_filter(img, (sigma, sigma, 0), mode="reflect", truncate=4.0)

    def pad(self, img, margin):
        """Symmetric padding of the first two axes (scipy's "reflect"), once per image."""
        img = np.ascontiguousarray(img, np.float32)
        if not margin:
            return img
        return np.pad(img, ((margin, margin), (margin, margin)) + ((0, 0),) * (img.ndim - 2), mode="symmetric")

    def crop(self, img, rects, s, idx, margin=0):
        """Stack of the tiles idx (same size): s x the LR rects, widened by `margin` px on each
        side. With a margin, img must be `pad(img, margin)`."""
        return np.stack([img[s * rects[k][0]:s * (rects[k][0] + rects[k][2]) + 2 * margin,
                             s * rects[k][1]:s * (rects[k][1] + rects[k][3]) + 2 * margin] for k in idx])

    def spectra(self, R, lay):
        """Band energies of a tile stack R (n, h, w, 3) in CIELAB residual units.

        Returns float64 arrays: lum (n, n_lum + 3) = band energies and structure-tensor
        moments, lum_sq (n, n_lum) = sum of squared periodogram ordinates per band,
        chroma (n, n_oct), chroma_sq, peak (n,) = max / mean of the top octave.
        """
        win = lay.win
        x = R - (np.tensordot(R, win, axes=([1, 2], [0, 1])) / lay.win_sum)[:, None, None, :]
        F = sfft.fft2(x * win[None, :, :, None], axes=(1, 2), workers=-1)
        P = (F.real ** 2 + F.imag ** 2) / np.float32(lay.norm)
        n = len(R)
        Pl = P[..., 0].reshape(n, -1)
        Pc = (P[..., 1] + P[..., 2]).reshape(n, -1)
        out = dict(lum=(Pl @ lay.lum.T).astype(np.float64),
                   lum_sq=((Pl * Pl) @ lay.lum[:lay.n_lum].T).astype(np.float64),
                   chroma=(Pc @ lay.chroma.T).astype(np.float64),
                   chroma_sq=((Pc * Pc) @ lay.chroma.T).astype(np.float64))
        top = Pl[:, lay.top]
        out["peak"] = (top.max(1) / np.maximum(top.mean(1), 1e-20)).astype(np.float64) if top.shape[1] else np.ones(n)
        return out

    def moments(self, tiles):
        """Per-tile Lab mean (n, 3) and covariance (n, 3, 3), float64."""
        x = tiles.reshape(len(tiles), -1, 3).astype(np.float64)
        mu = x.mean(1)
        xc = x - mu[:, None]
        return mu, np.einsum("npi,npj->nij", xc, xc) / max(x.shape[1] - 1, 1)

    def rigidity(self, lum, n_scale=4, n_orient=4, min_wave=3.0):
        """Phase congruency across octave pairs of a tile stack (n, h, w), in [0, 1]."""
        n, h, w = lum.shape
        filt = log_gabor(h, w, n_scale, n_orient, min_wave)
        spec = sfft.fft2(lum.astype(np.float32), axes=(1, 2), workers=-1)
        acc = np.zeros((n, h, w), np.complex64)
        weight = np.zeros((n, h, w), np.float32)
        for k in range(n_orient):
            prev = None
            for level in range(n_scale):
                z = sfft.ifft2(spec * filt[k, level], axes=(1, 2), workers=-1).astype(np.complex64)
                if prev is not None:
                    amp = np.sqrt(np.abs(prev) * np.abs(z))
                    acc += prev * np.conj(z) / np.maximum(amp, 1e-12)
                    weight += amp
                prev = z
        energy = np.abs(acc)
        med = np.median(energy.reshape(n, -1), 1)[:, None, None]
        return np.clip(np.maximum(energy - med, 0) / (weight + 1e-8), 0, 1).astype(np.float32)

    def gradients(self, lum):
        """Scharr gradients of a stack (n, h, w)."""
        gx = ndimage.correlate(lum, wcsi._SCHARR_X[None], mode="reflect")
        gy = ndimage.correlate(lum, wcsi._SCHARR_Y[None], mode="reflect")
        return gx, gy


# ---------------------------------------------------------------- torch backend

class TorchBackend:
    name = "torch"

    def __init__(self, device):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.batch = 32 if self.device.type == "cuda" else 8
        self._cache = {}

    def _t(self, a, dtype=None):
        t = self.torch
        if isinstance(a, t.Tensor):
            return a.to(self.device) if dtype is None else a.to(self.device, dtype)
        return t.as_tensor(np.ascontiguousarray(a), device=self.device, dtype=dtype)

    def _const(self, key, make):
        if key not in self._cache:
            self._cache[key] = self._t(make())
        return self._cache[key]

    def asarray(self, a):
        return self._t(a, self.torch.float32)

    def numpy(self, a):
        return a.detach().cpu().numpy() if isinstance(a, self.torch.Tensor) else np.asarray(a)

    def lab(self, rgb, white, rows=1024):
        t = self.torch
        m = self._const("rgb2xyz", lambda: (wcsi._RGB2XYZ / wcsi._D65[:, None]).T.astype(np.float32))
        src = self._t(rgb if rgb.dtype != np.uint16 else rgb.astype(np.int32))
        out = t.empty(rgb.shape[:2] + (3,), dtype=t.float32, device=self.device)
        for r0 in range(0, rgb.shape[0], rows):
            c = t.clamp(src[r0:r0 + rows].to(t.float32) / float(white), 0.0, 1.0)
            lin = t.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
            x = lin @ m
            f = t.where(x > 216 / 24389, t.sign(x) * t.abs(x) ** (1.0 / 3.0), (24389 / 27 * x + 16) / 116)
            out[r0:r0 + rows, :, 0] = 116 * f[..., 1] - 16
            out[r0:r0 + rows, :, 1] = 500 * (f[..., 0] - f[..., 1])
            out[r0:r0 + rows, :, 2] = 200 * (f[..., 1] - f[..., 2])
        return out

    def _symmetric_index(self, n, r, r_end=None):
        # scipy's "reflect": d c b a | a b c d | d c b a
        idx = np.arange(-r, n + (r if r_end is None else r_end))
        period = 2 * n
        idx = np.mod(idx, period)
        idx = np.where(idx >= n, period - 1 - idx, idx)
        return self._t(idx.astype(np.int64))

    def _blur_axis(self, x, sigma, axis):
        t = self.torch
        r = int(4.0 * sigma + 0.5)
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        k = (k / k.sum()).astype(np.float32)
        n = x.shape[axis]
        xp = t.index_select(x, axis, self._symmetric_index(n, r))
        out = t.zeros_like(x)
        for j, kj in enumerate(k):
            out += float(kj) * t.narrow(xp, axis, j, n)
        return out

    def lowpass(self, img, sigma):
        return self._blur_axis(self._blur_axis(self._t(img), sigma, 0), sigma, 1)

    def _upsample_axis(self, x, s, axis):
        t = self.torch
        x = t.movedim(x, axis, 0)
        n = x.shape[0]
        pad = t.cat([x[:1], x[:1], x, x[-1:], x[-1:]], 0)
        out = t.empty((n, s) + tuple(x.shape[1:]), dtype=x.dtype, device=self.device)
        for r in range(s):
            u = (r + 0.5) / s - 0.5
            base = math.floor(u)
            acc = t.zeros_like(x)
            for tap in (-1, 0, 1, 2):
                k = base + tap + 2
                acc += float(np.float32(_keys_weight(u - base - tap))) * pad[k:k + n]
            out[:, r] = acc
        return t.movedim(out.reshape((n * s,) + tuple(x.shape[1:])), 0, axis)

    def split(self, lab, s, sigma):
        t = self.torch
        x = self._t(lab)
        H, W = x.shape[:2]
        ph, pw = (-H) % s, (-W) % s
        if ph or pw:
            x = t.index_select(t.index_select(x, 0, self._symmetric_index(H, 0, ph)), 1,
                               self._symmetric_index(W, 0, pw))
        h, w = x.shape[0] // s, x.shape[1] // s
        blur = self._blur_axis(self._blur_axis(x, sigma, 0), sigma, 1)
        M = blur.reshape(h, s, w, s, -1).mean((1, 3))
        U = self._upsample_axis(self._upsample_axis(M, s, 0), s, 1)
        return self.numpy(M), (x - U)[:H, :W]

    def pad(self, img, margin):
        t = self.torch
        img = self._t(img, t.float32)
        if not margin:
            return img
        H, W = img.shape[:2]
        return t.index_select(t.index_select(img, 0, self._symmetric_index(H, margin)), 1,
                              self._symmetric_index(W, margin))

    def crop(self, img, rects, s, idx, margin=0):
        t = self.torch
        img = self._t(img)
        return t.stack([img[s * rects[k][0]:s * (rects[k][0] + rects[k][2]) + 2 * margin,
                            s * rects[k][1]:s * (rects[k][1] + rects[k][3]) + 2 * margin] for k in idx])

    def spectra(self, R, lay):
        t = self.torch
        key = (lay.h, lay.w, lay.s)
        win = self._const(("win",) + key, lambda: lay.win)
        Wl = self._const(("lum",) + key, lambda: lay.lum)
        Wc = self._const(("chroma",) + key, lambda: lay.chroma)
        top = self._const(("top",) + key, lambda: np.flatnonzero(lay.top).astype(np.int64))
        R = self._t(R)
        mean = (R * win[None, :, :, None]).sum((1, 2)) / lay.win_sum
        x = (R - mean[:, None, None, :]) * win[None, :, :, None]
        F = t.fft.fft2(x, dim=(1, 2))
        P = (F.real ** 2 + F.imag ** 2) / float(lay.norm)
        n = R.shape[0]
        Pl = P[..., 0].reshape(n, -1)
        Pc = (P[..., 1] + P[..., 2]).reshape(n, -1)
        out = dict(lum=Pl @ Wl.T, lum_sq=(Pl * Pl) @ Wl[:lay.n_lum].T,
                   chroma=Pc @ Wc.T, chroma_sq=(Pc * Pc) @ Wc.T)
        if len(top):
            tp = Pl[:, top]
            out["peak"] = tp.max(1).values / t.clamp(tp.mean(1), min=1e-20)
        else:
            out["peak"] = t.ones(n, device=self.device)
        return {k: self.numpy(v).astype(np.float64) for k, v in out.items()}

    def moments(self, tiles):
        x = self._t(tiles).reshape(tiles.shape[0], -1, 3).double()
        mu = x.mean(1)
        xc = x - mu[:, None]
        cov = xc.transpose(1, 2) @ xc / max(x.shape[1] - 1, 1)
        return self.numpy(mu), self.numpy(cov)

    def rigidity(self, lum, n_scale=4, n_orient=4, min_wave=3.0):
        t = self.torch
        lum = self._t(lum, t.float32)
        n, h, w = lum.shape
        filt = self._const(("gabor", h, w, n_scale, n_orient, min_wave),
                           lambda: log_gabor(h, w, n_scale, n_orient, min_wave))
        spec = t.fft.fft2(lum, dim=(1, 2))
        acc = t.zeros((n, h, w), dtype=t.complex64, device=self.device)
        weight = t.zeros((n, h, w), dtype=t.float32, device=self.device)
        for k in range(n_orient):
            prev = None
            for level in range(n_scale):
                z = t.fft.ifft2(spec * filt[k, level], dim=(1, 2))
                if prev is not None:
                    amp = t.sqrt(prev.abs() * z.abs())
                    acc += prev * t.conj(z) / t.clamp(amp, min=1e-12)
                    weight += amp
                prev = z
        energy = acc.abs()
        med = energy.reshape(n, -1).median(1).values[:, None, None]
        return t.clamp(t.clamp(energy - med, min=0) / (weight + 1e-8), 0, 1)

    def gradients(self, lum):
        t = self.torch
        lum = self._t(lum, t.float32)
        n, h, w = lum.shape
        x = t.index_select(t.index_select(lum, 1, self._symmetric_index(h, 1)), 2, self._symmetric_index(w, 1))
        kx = self._const("scharr_x", lambda: wcsi._SCHARR_X.astype(np.float32))
        ky = self._const("scharr_y", lambda: wcsi._SCHARR_Y.astype(np.float32))
        gx = t.zeros_like(lum)
        gy = t.zeros_like(lum)
        for a in range(3):
            for b in range(3):
                sl = x[:, a:a + h, b:b + w]
                gx += kx[a, b] * sl
                gy += ky[a, b] * sl
        return gx, gy


def get_backend(device=None):
    """None or "numpy": numpy + scipy. "torch:cpu", "cuda", "cuda:k": torch on that device."""
    if device in (None, "numpy", "cpu"):
        return NumpyBackend()
    if isinstance(device, (NumpyBackend, TorchBackend)):
        return device
    dev = device[len("torch:"):] if str(device).startswith("torch:") else device
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError(f"device {device!r} needs torch, which is not installed")
    return TorchBackend(dev)
