"""CPU tests for wcsi.py:  pytest -q genet/test_wcsi.py"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wcsi  # noqa: E402


def smooth_field(h, w, sigma, seed):
    return ndimage.gaussian_filter(np.random.default_rng(seed).standard_normal((h, w)), sigma, mode="wrap")


def make_pair(s, h0=48, w0=64, seed=0):
    """LR image and a consistent SR: bicubic of the LR plus one stationary grain."""
    base = np.stack([smooth_field(h0, w0, 6, seed + k) for k in range(3)], -1)
    lr = np.clip(0.5 + 0.6 * base / base.std(), 0.05, 0.95).astype(np.float32)
    grain = smooth_field(s * h0, s * w0, 0.7, seed + 9)
    sr = wcsi.upsample_bicubic(lr, s) + 0.03 * (grain / grain.std())[..., None]
    return lr, np.clip(sr, 0, 1).astype(np.float32)


def checker(lr_shape, s, tile, overlap):
    ys, xs = wcsi.tile_starts(lr_shape[0], tile, overlap), wcsi.tile_starts(lr_shape[1], tile, overlap)
    m = np.zeros((s * lr_shape[0], s * lr_shape[1], 1), np.float32)
    for r, y0 in enumerate(ys):
        for c, x0 in enumerate(xs):
            if (r + c) % 2:
                m[s * y0:s * (y0 + tile), s * x0:s * (x0 + tile)] = 1
    return m


def test_lab_reference_values():
    rgb = np.array([[[1, 1, 1], [0, 0, 0], [1, 0, 0], [0.5, 0.5, 0.5]]], np.float32)
    lab = wcsi.srgb_to_lab(rgb)[0]
    np.testing.assert_allclose(lab[0], [100, 0, 0], atol=0.01)
    np.testing.assert_allclose(lab[1], [0, 0, 0], atol=0.01)
    np.testing.assert_allclose(lab[2], [53.24, 80.09, 67.20], atol=0.05)
    np.testing.assert_allclose(lab[3], [53.39, 0, 0], atol=0.05)


def test_ciede2000_sharma_pairs():
    # Sharma, Wu & Dalal (2005), pairs 1, 7, 13, 17, 25
    a = [(50, 2.6772, -79.7751), (50, 0, 0), (50, 2.49, -0.001), (50, 2.5, 0), (60.2574, -34.0099, 36.2677)]
    b = [(50, 0, -82.7485), (50, -1, 2), (50, -2.49, 0.0009), (73, 25, -18), (60.4626, -34.1751, 39.4387)]
    np.testing.assert_allclose(wcsi.ciede2000(np.array(a), np.array(b)),
                               [2.0425, 2.3669, 7.1792, 27.1492, 1.2644], atol=1e-4)


@pytest.mark.parametrize("s", [2, 4, 8])
def test_bicubic_constant_and_ramp(s):
    const = np.full((6, 7, 3), 0.25, np.float32)
    np.testing.assert_allclose(wcsi.upsample_bicubic(const, s), 0.25, atol=1e-6)
    ramp = np.tile(np.arange(10, dtype=np.float32)[None, :, None], (4, 1, 1))
    up = wcsi.upsample_bicubic(ramp, s)[0, :, 0]
    x = (np.arange(10 * s) + 0.5) / s - 0.5
    inner = (x >= 1) & (x <= 8)
    np.testing.assert_allclose(up[inner], x[inner], atol=1e-5)


def test_rg_split_of_constant_is_zero():
    M, R = wcsi.rg_split(np.full((32, 40, 3), 60.0, np.float32), 4, 1.6)
    assert M.shape == (8, 10, 3)
    np.testing.assert_allclose(R, 0, atol=1e-4)


@pytest.mark.parametrize("s", [2, 4, 8])
def test_scale_inferred(s):
    lr, sr = make_pair(s, 32, 40)
    d = wcsi.wcsi_score(sr, lr, tile=16, overlap=2, return_details=True)
    assert d["scale"] == s
    assert 0 < d["wcsi"] <= 100
    d = wcsi.wcsi_score(sr[:-1, :-1], lr, tile=16, overlap=2, return_details=True)
    assert d["scale"] == s and any("cropped" in n for n in d["notes"])


def test_non_integer_scale_rejected():
    lr, sr = make_pair(2, 32, 40)
    with pytest.raises(ValueError):
        wcsi.wcsi_score(sr[: 3 * 32 // 2], lr[:, :20])


@pytest.mark.parametrize("s", [2, 4])
def test_tile_colour_shift_lowers_score(s):
    lr, sr = make_pair(s)
    shifted = np.clip(sr + 0.04 * checker(lr.shape, s, 16, 2) * np.array([1, 0, 0], np.float32), 0, 1)
    good = wcsi.wcsi_score(sr, lr, tile=16, overlap=2, return_details=True)
    bad = wcsi.wcsi_score(shifted, lr, tile=16, overlap=2, return_details=True)
    assert good["wcsi"] > bad["wcsi"] + 5
    for k in ("E_RG", "E_GJ", "E_sheaf"):
        assert bad["terms"][k] > good["terms"][k]
    assert good["parts"]["holonomy_max_abs"] < 1e-9


def test_details_shapes_and_advice():
    lr, sr = make_pair(4, 32, 48)
    d = wcsi.wcsi_score(sr, lr, tile=16, overlap=2, return_details=True)
    g = d["grid"]
    assert len(d["tiles"]) == g["rows"] * g["cols"]
    t = d["tiles"][0]
    assert len(t["psi"]) == len(t["psi0"]) == len(t["r"]) == len(t["step"]) == 14
    assert isinstance(t["advice"], str) and t["advice"]
    assert set(d["terms"]) == set(wcsi.TERMS)
    json.dumps(d)


def test_tile_outputs_overlap_comparison():
    s, tile, ov = 2, 16, 4
    lr, sr = make_pair(s, 32, 44)
    ys, xs = wcsi.tile_starts(32, tile, ov), wcsi.tile_starts(44, tile, ov)
    canv = [sr[s * y:s * (y + tile), s * x:s * (x + tile)].copy() for y in ys for x in xs]
    d = wcsi.wcsi_score(sr, lr, tile=tile, overlap=ov, tiles=canv, return_details=True)
    assert d["terms"]["E_GJ"] == pytest.approx(0, abs=1e-9)
    canv[1] = np.clip(canv[1] + 0.05, 0, 1)
    d = wcsi.wcsi_score(sr, lr, tile=tile, overlap=ov, tiles=canv, return_details=True)
    assert d["terms"]["E_GJ"] > 1
    with pytest.raises(ValueError):
        wcsi.wcsi_score(sr, lr, tile=tile, overlap=ov, tiles=canv[:-1])


def test_single_tile_grid():
    lr, sr = make_pair(2, 24, 24)
    d = wcsi.wcsi_score(sr, lr, tile=64, overlap=0, return_details=True)
    assert (d["grid"]["rows"], d["grid"]["cols"]) == (1, 1)
    assert d["terms"]["E_RG"] is not None
    assert all(d["terms"][k] is None for k in ("E_GJ", "E_PCP", "E_cyto", "E_sheaf"))
    assert 0 < d["wcsi"] <= 100


def test_file_inputs_and_layouts(tmp_path):
    PIL = pytest.importorskip("PIL.Image")
    lr, sr = make_pair(2, 32, 40)
    u8 = lambda a: (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    PIL.fromarray(u8(lr)).save(tmp_path / "lr.png")
    PIL.fromarray(u8(sr)).save(tmp_path / "sr.jpg", quality=95)
    PIL.fromarray(u8(sr)).save(tmp_path / "sr.png")
    v_png = wcsi.wcsi_score(tmp_path / "sr.png", str(tmp_path / "lr.png"), tile=16, overlap=2)
    v_jpg = wcsi.wcsi_score(str(tmp_path / "sr.jpg"), tmp_path / "lr.png", tile=16, overlap=2)
    assert 0 < v_png <= 100 and 0 < v_jpg <= 100
    v_arr = wcsi.wcsi_score(u8(sr), u8(lr), tile=16, overlap=2)
    v_chw = wcsi.wcsi_score(np.moveaxis(u8(sr), -1, 0), np.moveaxis(u8(lr), -1, 0), tile=16, overlap=2)
    assert v_arr == pytest.approx(v_png, abs=1e-9) and v_chw == pytest.approx(v_arr, abs=1e-9)
    a16, white = wcsi.load_rgb((sr * 65535).astype(np.uint16))
    assert white == 65535.0 and a16.dtype == np.uint16


def test_cli(tmp_path, capsys):
    PIL = pytest.importorskip("PIL.Image")
    lr, sr = make_pair(4, 32, 40)
    u8 = lambda a: (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    PIL.fromarray(u8(lr)).save(tmp_path / "lr.png")
    PIL.fromarray(u8(sr)).save(tmp_path / "sr.png")
    out = tmp_path / "d.json"
    assert wcsi.main([str(tmp_path / "sr.png"), str(tmp_path / "lr.png"), "--tile", "16", "--overlap", "2",
                      "--json", str(out), "--advice"]) == 0
    assert "WCSI" in capsys.readouterr().out
    assert json.loads(out.read_text())["scale"] == 4


def test_unknown_parameter_rejected():
    lr, sr = make_pair(2, 24, 24)
    with pytest.raises(TypeError):
        wcsi.wcsi_score(sr, lr, sigma=3)
