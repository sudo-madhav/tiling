"""CPU tests for wcsi2.py and wcsi2_backend.py:  pytest -q genet/test_wcsi2.py"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wcsi  # noqa: E402
import wcsi2  # noqa: E402
from wcsi2_backend import get_backend  # noqa: E402

S, TILE_SR = 4, 64


def natural(h, w, seed, slope=1.0):
    """Colour image with a 1/f^slope amplitude spectrum: scale invariant like a photograph."""
    rng = np.random.default_rng(seed)
    f = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :])
    f[0, 0] = 1.0
    amp = f ** -slope
    amp[0, 0] = 0.0

    def field():
        x = np.real(np.fft.ifft2(amp * np.exp(2j * np.pi * rng.random((h, w)))))
        return x / x.std()

    lum = field()
    img = 0.5 + 0.12 * lum[..., None] + 0.03 * np.stack([field() for _ in range(3)], -1)
    return np.clip(img, 0.02, 0.98).astype(np.float32)


def area_down(img, s):
    h, w = img.shape[0] // s, img.shape[1] // s
    return img[:h * s, :w * s].reshape(h, s, w, s, -1).mean((1, 3)).astype(np.float32)


def residual_split(img, s):
    M = wcsi.coarse_grain(img, s, 0.4 * s)
    up = wcsi.upsample_bicubic(M, s)
    return up, img - up


def per_tile(img, s, fn, seed=0):
    """Apply fn(tile, rng) on the analysis grid of TILE_SR (no overlap) and return the result."""
    out = img.copy()
    rng = np.random.default_rng(seed)
    for y in range(0, img.shape[0], TILE_SR):
        for x in range(0, img.shape[1], TILE_SR):
            out[y:y + TILE_SR, x:x + TILE_SR] = fn(out[y:y + TILE_SR, x:x + TILE_SR], rng)
    return np.clip(out, 0, 1).astype(np.float32)


@pytest.fixture(scope="module")
def gt():
    return natural(256, 320, 0)


@pytest.fixture(scope="module")
def lr(gt):
    return area_down(gt, S)


@pytest.fixture(scope="module")
def calib(tmp_path_factory):
    pairs = [(g, area_down(g, S)) for g in (natural(256, 320, k) for k in range(10, 16))]
    out = tmp_path_factory.mktemp("calib") / f"wcsi2_x{S}_t{TILE_SR}.json"
    cal = wcsi2.calibrate(pairs, S, tile_sr=TILE_SR, overlap=0, out=out, note="synthetic 1/f")
    return out, cal


def ref(sr, gt, lr=None, **kw):
    return wcsi2.wcsi2_ref_score(sr, gt, lr, scale=S, tile_sr=TILE_SR, overlap=0, return_details=True, **kw)


def noref(sr, lr, calib, **kw):
    return wcsi2.wcsi2_score(sr, lr, scale=S, tile_sr=TILE_SR, overlap=0, calib=calib, return_details=True, **kw)


def test_coordinates_follow_the_scale():
    assert len(wcsi2.coord_names(8)) == 3 * 4 + 2 + 3 + 1
    assert len(wcsi2.coord_names(4)) == 2 * 4 + 2 + 2 + 1
    assert wcsi2.coord_names(4)[wcsi2._q_slice(4)] == ["Q_xx", "Q_xy"]


def test_saturation_keeps_orientation_linear():
    p = dict(wcsi2.DEFAULTS)
    eps = np.full((1, len(wcsi2.coord_names(4))), 40.0)
    out = wcsi2.saturate(eps, 4, p)
    assert np.all(out[0, wcsi2._q_slice(4)] == 40.0)
    assert np.all(np.abs(np.delete(out[0], [8, 9])) <= p["sat_loge"])


def test_ref_identity_is_100(gt, lr):
    d = ref(gt, gt, lr)
    assert d["wcsi2_ref"] == pytest.approx(100.0)
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in d["terms"].values() if v is not None)
    assert d["parts"]["phenotype_counts"]["faithful"] == d["layout"]["n_tiles"]


def test_ref_ranks_degradations_below_the_ground_truth(gt, lr):
    up, R = residual_split(gt, S)
    bic = np.clip(wcsi.upsample_bicubic(lr, S), 0, 1).astype(np.float32)
    blur = np.clip(up + 0.3 * R, 0, 1).astype(np.float32)
    scores = {k: ref(v, gt, lr) for k, v in dict(bicubic=bic, blur=blur).items()}
    assert scores["blur"]["wcsi2_ref"] < 100 and scores["bicubic"]["wcsi2_ref"] < scores["blur"]["wcsi2_ref"]
    assert scores["bicubic"]["terms"]["E_detail"] > 1
    advice = " ".join(t["advice"] for t in scores["bicubic"]["tiles"])
    assert "too little invented detail" in advice or "chroma detail missing" in advice
    assert np.all(np.array(list(scores["bicubic"]["parts"]["detail_mean_z"].values())[:8]) < 0)


def test_ref_separates_shared_from_tile_wise_departure(gt, lr):
    up, R = residual_split(gt, S)
    shared = np.clip(up + 1.8 * R, 0, 1).astype(np.float32)
    tilewise = per_tile(R, S, lambda t, rng: t * rng.choice([0.2, 1.8]), seed=3)
    tilewise = np.clip(up + tilewise, 0, 1).astype(np.float32)
    a, b = ref(shared, gt, lr), ref(tilewise, gt, lr)
    assert a["terms"]["E_consist"] < b["terms"]["E_consist"]
    assert a["rho_coherence"] > b["rho_coherence"]
    assert a["parts"]["phenotype_counts"]["inconsistent departure"] <= b["parts"]["phenotype_counts"]["inconsistent departure"]


def test_ref_structure_term_sees_misplaced_edges(gt, lr):
    shifted = np.roll(gt, 2, axis=1)
    d = ref(shifted, gt, lr)
    assert d["terms"]["E_struct"] > 1


def test_colour_jitter_raises_seam_and_tone(gt, lr):
    jit = per_tile(gt, S, lambda t, rng: t + rng.normal(0, 0.03, 3).astype(np.float32), seed=1)
    d = ref(jit, gt, lr)
    assert d["terms"]["E_seam"] > 0.5 and d["terms"]["E_tone"] > 0.5


def test_calibration_round_trip(calib, tmp_path):
    path, cal = calib
    loaded = wcsi2.load_calibration(str(path), S)
    np.testing.assert_allclose(loaded["B"], np.asarray(cal["B"]))
    assert loaded["tile_lr"] == TILE_SR // S and cal["n_images"] == 6
    assert all(v > 0 for v in cal["sigma_nat2"]) and all(v > 0 for v in cal["sigma_pair2"])
    for t in (64, 256):
        (tmp_path / f"wcsi2_x{S}_t{t}.json").write_text(path.read_text())
    assert wcsi2.find_calibration(S, 20, tmp_path).name == f"wcsi2_x{S}_t64.json"
    assert wcsi2.find_calibration(S, 50, tmp_path).name == f"wcsi2_x{S}_t256.json"
    assert wcsi2.find_calibration(8, 20, tmp_path) is None
    with pytest.raises(ValueError):
        wcsi2.load_calibration(str(path), 8)  # other scale, other coordinates
    fallback = wcsi2.load_calibration(None, S)
    assert fallback["source"] is None and np.allclose(fallback["B"], np.eye(len(wcsi2.coord_names(S))))


def test_noref_ground_truth_beats_missing_and_inconsistent_detail(calib):
    path, _ = calib
    g = natural(256, 320, 1)
    low = area_down(g, S)
    up, R = residual_split(g, S)
    bic = np.clip(wcsi.upsample_bicubic(low, S), 0, 1).astype(np.float32)
    tilewise = np.clip(up + per_tile(R, S, lambda t, rng: t * rng.choice([0.0, 2.5]), seed=2), 0, 1).astype(np.float32)
    d_gt, d_bic, d_tw = (noref(x, low, str(path)) for x in (g, bic, tilewise))
    assert d_gt["wcsi2"] > d_bic["wcsi2"] and d_gt["wcsi2"] > d_tw["wcsi2"]
    assert d_bic["terms"]["E_detail"] > d_gt["terms"]["E_detail"]
    assert d_tw["terms"]["E_consist"] > d_gt["terms"]["E_consist"]
    assert d_gt["calibration"]["source"] == str(path)


def test_noref_uncalibrated_scale_is_flagged(gt, lr):
    d = noref(gt, lr, None)
    assert 0 < d["wcsi2"] <= 100
    assert any("uncalibrated" in n for n in d["notes"])


def test_tile_coords_match_the_grid(gt, lr):
    t = TILE_SR // S
    coords = [dict(x=x, y=y, w=t, h=t) for y in wcsi.tile_starts(lr.shape[0], t, 0)
              for x in wcsi.tile_starts(lr.shape[1], t, 0)]
    a = ref(gt * 0.9 + 0.05, gt, lr)
    b = wcsi2.wcsi2_ref_score(gt * 0.9 + 0.05, gt, lr, scale=S, tile_coords=coords, return_details=True)
    assert a["wcsi2_ref"] == pytest.approx(b["wcsi2_ref"], rel=1e-9)
    with pytest.raises(ValueError):
        wcsi2.wcsi2_ref_score(gt, gt, lr, scale=S, tile=t, tile_coords=coords)


def test_unknown_parameter_raises(gt, lr):
    with pytest.raises(TypeError):
        ref(gt, gt, lr, no_such_param=1)


def test_non_multiple_size_splits(gt):
    be = get_backend(None)
    lab = be.lab(gt[:250, :317], 1.0)
    M, R = be.split(lab, S, 0.4 * S)
    assert M.shape[:2] == (63, 80) and R.shape == lab.shape


def test_torch_backend_matches_numpy(gt, lr, calib):
    pytest.importorskip("torch")
    path, _ = calib
    sr = np.clip(wcsi.upsample_bicubic(lr, S), 0, 1).astype(np.float32)
    lab = get_backend(None).lab(gt[:250, :317], 1.0)
    M0, R0 = get_backend(None).split(lab, S, 0.4 * S)
    tb = get_backend("torch:cpu")
    M1, R1 = tb.split(tb.asarray(lab), S, 0.4 * S)
    np.testing.assert_allclose(M1, M0, atol=1e-3)
    np.testing.assert_allclose(tb.numpy(R1), R0, atol=1e-3)
    for fn in (lambda dev: ref(sr, gt, lr, device=dev)["wcsi2_ref"],
               lambda dev: noref(sr, lr, str(path), device=dev)["wcsi2"]):
        assert fn("torch:cpu") == pytest.approx(fn(None), rel=2e-3)


def test_cli(tmp_path, gt, lr, calib):
    PIL = pytest.importorskip("PIL.Image")
    paths = {}
    for k, a in dict(gt=gt, lr=lr, sr=np.clip(wcsi.upsample_bicubic(lr, S), 0, 1)).items():
        paths[k] = tmp_path / f"{k}.png"
        PIL.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).save(paths[k])
    out = tmp_path / "r.json"
    assert wcsi2.main(["ref", str(paths["sr"]), str(paths["gt"]), str(paths["lr"]), "--scale", str(S),
                       "--tile-sr", str(TILE_SR), "--overlap", "0", "--json", str(out), "--advice"]) == 0
    assert 0 < json.loads(out.read_text())["wcsi2_ref"] < 100
    assert wcsi2.main(["noref", str(paths["sr"]), str(paths["lr"]), "--scale", str(S), "--tile-sr", str(TILE_SR),
                       "--calib", str(calib[0]), "--json", str(out)]) == 0
    assert json.loads(out.read_text())["calibration"]["source"] == str(calib[0])
    cal_out = tmp_path / "c.json"
    (tmp_path / "g").mkdir(), (tmp_path / "l").mkdir()
    paths["gt"].rename(tmp_path / "g" / "a.png"), paths["lr"].rename(tmp_path / "l" / "a.png")
    assert wcsi2.main(["calibrate", "--gt", str(tmp_path / "g"), "--lr", str(tmp_path / "l"), "--scale", str(S),
                       "--tile-sr", str(TILE_SR), "--out", str(cal_out)]) == 0
    assert json.loads(cal_out.read_text())["coords"] == wcsi2.coord_names(S)
