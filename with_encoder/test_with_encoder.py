"""CPU tests for with_encoder. Learned encoders are never loaded; the weight-free Gabor
encoder stands in for them, which tests the plumbing and the invariances that do not
depend on what an encoder learned."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablate  # noqa: E402
import corruptions as C  # noqa: E402
import featstats as fs  # noqa: E402
import terms as T  # noqa: E402
from encoders import EncoderUnavailable, FeatureMap, build_encoder  # noqa: E402
from encoders.gabor import GaborEncoder  # noqa: E402
from encoders.registry import Stack, build_stack, conflicts  # noqa: E402
import hybrid_noref as hybrid_noref_mod  # noqa: E402
from hybrid_noref import hybrid_noref  # noqa: E402
from hybrid_ref import hybrid_ref  # noqa: E402

wcsi = fs.wcsi
S = 4


def scene(n=256, seed=0):
    """Two materials (oriented stripes, isotropic grain) and a sharp bright bar."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    stripes = 0.5 + 0.18 * np.sin(2 * np.pi * (0.11 * x + 0.04 * y)) + 0.03 * rng.standard_normal((n, n))
    grain = 0.45 + 0.08 * rng.standard_normal((n, n))
    from scipy import ndimage
    grain = ndimage.gaussian_filter(grain, 0.8)
    img = np.empty((n, n, 3), np.float32)
    img[:, : n // 2] = np.stack([stripes, 0.9 * stripes, 0.7 * stripes], -1)[:, : n // 2]
    img[:, n // 2:] = np.stack([0.7 * grain, grain, 0.8 * grain], -1)[:, n // 2:]
    img[n // 2 - 3:n // 2 + 3, 20:n - 20] = (0.95, 0.95, 0.9)
    return np.clip(img, 0, 1)


def down(img, s=S):
    h, w = img.shape[0] // s, img.shape[1] // s
    return img[:h * s, :w * s].reshape(h, s, w, s, 3).mean((1, 3))


@pytest.fixture(scope="module")
def data():
    gt = scene()
    lr = down(gt)
    up = np.clip(wcsi.upsample_bicubic(lr, S), 0, 1)
    return gt, lr, up, gt - up


# ---------------------------------------------------------------- encoders

def test_gabor_shapes_and_padding():
    e = GaborEncoder()
    out = e.encode(np.random.default_rng(0).random((70, 53, 3), dtype=np.float32))
    for name, f in out.items():
        assert f.data.shape[:2] == (math.ceil(70 / f.stride), math.ceil(53 / f.stride)), name
        assert np.isfinite(f.data).all()
    assert {f.role for f in out.values()} == {"texture", "semantic"}


def test_learned_encoders_never_download():
    for spec in ({"type": "dinov2", "path": "/nonexistent/dinov2"},
                 {"type": "vgg19", "path": "/nonexistent/vgg19.pth"},
                 {"type": "vae", "path": "/nonexistent/vae"}):
        with pytest.raises(EncoderUnavailable):
            build_encoder(spec, device="cpu")


def test_stack_relabels_role():
    g = GaborEncoder()
    st = Stack(dict(texture=g, latent=g))
    out = st.encode(np.zeros((32, 32, 3), np.float32), ("texture", "latent"))
    assert set(out["texture"]) == {"g1", "g2", "g3"}
    assert all(f.role == "latent" for f in out["latent"].values()) and len(out["latent"]) == 4


def test_conflict_prefix():
    cfg = {"conflicts": {"_note": "x", "vosr": ["qwen-image-vae", "dinov2"], "seedvr": ["s"]}}
    assert conflicts(cfg, "vosr_tile512") == {"qwen-image-vae", "dinov2"}
    assert conflicts(cfg, "vosrx") == set()
    assert conflicts(cfg, "casr") == set()


# ---------------------------------------------------------------- statistics

def test_monge_and_bures():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((5, 5))
    B = rng.standard_normal((5, 5))
    Si, Sj = A @ A.T + np.eye(5), B @ B.T + np.eye(5)
    Tm = fs.monge_linear(Si, Sj, 0.0)
    assert np.allclose(Tm @ Si @ Tm.T, Sj, atol=1e-6)
    assert np.allclose(Tm, Tm.T, atol=1e-8)
    m = rng.standard_normal(5)
    assert fs.bures_sq(m, Si, m, Si) == pytest.approx(0.0, abs=1e-8)
    assert fs.bures_sq(m, Si, m + 1, Si) == pytest.approx(5.0, rel=1e-6)


def test_shrunk_cov_is_psd_with_few_samples():
    X = np.random.default_rng(2).standard_normal((6, 16))
    S_ = fs.shrunk_cov(X)
    assert np.linalg.eigvalsh(S_).min() > 0


def test_nyquist_peak_sees_checkerboard():
    rng = np.random.default_rng(3)
    from scipy import ndimage
    nat = ndimage.gaussian_filter(rng.standard_normal((128, 128)), 1.0) * 10 + 50
    y, x = np.mgrid[0:128, 0:128]
    cb = nat + 2.0 * ((x + y) % 2 * 2 - 1)
    assert fs.nyquist_stats(cb)[1] > fs.nyquist_stats(nat)[1] + math.log(4)
    assert fs.nyquist_stats(cb)[2] > fs.nyquist_stats(nat)[2]


def test_centered_cosine_separates_materials():
    D = np.array([[1.0, 0.1, 5.0], [1.1, 0.0, 5.0], [0.0, 1.0, 5.0], [0.1, 1.1, 5.0]])
    raw = D / np.linalg.norm(D, axis=1, keepdims=True)
    assert (raw @ raw.T)[0, 2] > 0.9  # the shared component hides the difference
    c = fs.centered_cosine(D)
    assert c[0, 1] > 0.9 and c[0, 2] < 0


# ---------------------------------------------------------------- with reference

def test_clone_scores_100(data):
    gt, lr, _, _ = data
    d = hybrid_ref(gt, gt, lr, scale=S, tile=16)
    assert d["score"] == pytest.approx(100.0)
    assert all(v is None or abs(v) < 1e-9 for v in d["energies"].values())
    assert all(c["score"] is None or c["score"] == pytest.approx(100.0) for c in d["configs"].values())
    assert d["wcsi2_ref"] == pytest.approx(100.0)
    assert d["energies"]["gmci_master"] is None and d["configs"]["C_pixel_gmci"]["score"] is None


def test_gate_spares_resampled_texture(data):
    """F1 moves above-Nyquist texture where the GT is not phase-locked. The gated distributional
    term must react less than with the gate forced on everywhere, less than the ungated pointwise
    distance, and far less than to over-smoothing."""
    gt, lr, _, _ = data
    rects = C.layout(lr.shape[0], lr.shape[1], 16, 1)
    f1 = C.corrupt("F1_texture_resample", gt, lr, S, rects)
    f5 = C.corrupt("F5_oversmooth", gt, lr, S, rects)
    e = hybrid_ref(f1, gt, lr, scale=S, tile=16, with_gmci=False)["energies"]
    e_all = hybrid_ref(f1, gt, lr, scale=S, tile=16, with_gmci=False, gate="all")["energies"]
    e5 = hybrid_ref(f5, gt, lr, scale=S, tile=16, with_gmci=False)["energies"]
    assert np.abs(down(f1) - lr).mean() < 0.02
    assert e["feat_wadd"] < 0.6 * e_all["feat_wadd"]
    assert e["feat_wadd"] < e["feat_point"]
    assert e["feat_wadd"] < 0.3 * e5["feat_wadd"]


def within_material_clash(up, hf):
    """Top half sharp, bottom half blurry: splits both materials (the scene's materials are
    left/right, so a left/right split would be a fidelity error, not an inconsistency)."""
    clash = up + hf
    clash[128:] = up[128:] + 0.3 * hf[128:]
    return np.clip(clash, 0, 1)


def test_coherent_vs_clash(data):
    gt, lr, up, hf = data
    cons = np.clip(up + 1.4 * hf, 0, 1)
    clash = within_material_clash(up, hf)
    a = hybrid_ref(cons, gt, lr, scale=S, tile=16, with_gmci=False)
    b = hybrid_ref(clash, gt, lr, scale=S, tile=16, with_gmci=False)
    assert b["energies"]["feat_frust"] > 3 * a["energies"]["feat_frust"]
    assert a["rho_feature"] > b["rho_feature"]


def test_pixel_terms_catch_what_they_own(data):
    gt, lr, _, _ = data
    rects = C.layout(lr.shape[0], lr.shape[1], 16, 1)
    drift = hybrid_ref(C.corrupt("F2_colour_drift", gt, lr, S, rects), gt, lr, scale=S, tile=16)["energies"]
    seam = hybrid_ref(C.corrupt("F3_seam_step", gt, lr, S, rects), gt, lr, scale=S, tile=16)["energies"]
    # a +-3.5 b* ramp is a smaller CIEDE2000 difference, about 1 JND on average here
    assert drift["pix_tone"] > 0.5
    assert seam["pix_seam"] > 0.1


def test_latent_conflict_drops_latent_terms(data):
    gt, lr, up, hf = data

    class Lat(GaborEncoder):
        family = "fake-vae"

    st = Stack(dict(texture=GaborEncoder(), semantic=GaborEncoder(), latent=Lat()))
    img = np.clip(up + 1.2 * hf, 0, 1)
    ok = hybrid_ref(img, gt, lr, stack=st, scale=S, tile=16, with_gmci=False)
    bad = hybrid_ref(img, gt, lr, stack=st, scale=S, tile=16, with_gmci=False,
                     method_families={"fake-vae"}, return_details=True)
    assert ok["energies"]["lat_wadd"] is not None and ok["energies"]["lat_wadd"] > 0
    assert bad["energies"]["lat_wadd"] is None
    assert bad["encoder_conflict"] == {"latent": "fake-vae"}
    assert any("own autoencoder" in n for n in bad["notes"])


def test_semantic_conflict_is_flagged_not_dropped(data):
    gt, lr, up, hf = data
    st = build_stack({"device": "cpu", "stack": {"texture": {"type": "gabor"},
                                                 "semantic": {"type": "gabor", "family": "dinov2"}}})
    d = hybrid_ref(np.clip(up + 1.2 * hf, 0, 1), gt, lr, stack=st, scale=S, tile=16, with_gmci=False,
                   method_families=conflicts({"conflicts": {"vosr": ["dinov2"]}}, "vosr_tile512"),
                   return_details=True)
    assert d["encoder_conflict"] == {"semantic": "dinov2"}
    assert d["energies"]["feat_sem"] is not None
    assert any("cross-check" in n for n in d["notes"])


@pytest.mark.parametrize("gate", ["feat", "none", "all"])
def test_other_gates_run(data, gate):
    gt, lr, up, hf = data
    e = hybrid_ref(np.clip(up + hf * 1.2, 0, 1), gt, lr, scale=S, tile=16, gate=gate, with_gmci=False)["energies"]
    assert e["feat_wadd"] is not None and np.isfinite(e["feat_wadd"])


# ---------------------------------------------------------------- without reference

@pytest.fixture(scope="module")
def feat_calib(tmp_path_factory):
    pairs = [(g, down(g)) for g in (scene(seed=k) for k in (11, 12, 13))]
    out = tmp_path_factory.mktemp("fc") / "feat_x4_t16.json"
    return hybrid_noref_mod.calibrate(pairs, "gabor", S, tile=16, out=out, note="synthetic scenes"), out


def test_noref_uncalibrated_consistency(data):
    gt, lr, up, hf = data
    n_gt = hybrid_noref(gt, lr, scale=S, tile=16, calib=None, return_details=True)
    n_clash = hybrid_noref(within_material_clash(up, hf), lr, scale=S, tile=16, calib=None)
    n_bic = hybrid_noref(up, lr, scale=S, tile=16, calib=None)
    assert n_clash["energies"]["nr_feat_consist"] > n_gt["energies"]["nr_feat_consist"] + 1
    # without the anchor kernel, the bar tiles (more SR gain than plain stripes) read as frustration
    loose = hybrid_noref(gt, lr, scale=S, tile=16, calib=None, tau_anchor=1e9, return_details=True)
    raw = lambda d: np.mean([t["frust"] for t in d["tiles"]])  # noqa: E731
    assert raw(loose) > 1.5 * raw(n_gt)
    # uncalibrated, doing nothing is perfectly consistent and the detail term is not computed
    assert n_bic["energies"]["nr_feat_consist"] == pytest.approx(0.0, abs=1e-9)
    assert n_bic["energies"]["nr_feat_detail"] is None
    assert any("uncalibrated" in n for n in n_gt["notes"])
    assert n_gt["energies"]["wcsi_cyto"] is None


def test_noref_calibrated_puts_ground_truth_first(data, feat_calib):
    gt, lr, up, hf = data
    cal, path = feat_calib
    assert cal["n_images"] == 3 and all(v["sigma2"] > 0 and v["nu_pair"] > 0 for v in cal["layers"].values())
    kw = dict(scale=S, tile=16, calib=None, feat_calib=str(path))
    n_gt, n_bic = hybrid_noref(gt, lr, **kw), hybrid_noref(up, lr, **kw)
    n_clash = hybrid_noref(within_material_clash(up, hf), lr, **kw)
    # missing detail now costs (it scored 0 before calibration), tile-wise clash still costs most
    assert n_bic["energies"]["nr_feat_detail"] > n_gt["energies"]["nr_feat_detail"] + 0.3
    assert n_clash["energies"]["nr_feat_consist"] > n_gt["energies"]["nr_feat_consist"] + 1
    assert n_gt["configs"]["N2_feature"]["score"] > n_bic["configs"]["N2_feature"]["score"]
    assert n_gt["configs"]["N2_feature"]["score"] > n_clash["configs"]["N2_feature"]["score"]
    assert n_gt["score"] > n_bic["score"]
    with pytest.raises(ValueError):
        hybrid_noref(gt, lr, scale=S, tile=16, calib=None, feat_calib={**cal, "scale": 8})


def test_noref_legacy_v1_terms(data):
    gt, lr, _, _ = data
    d = hybrid_noref(gt, lr, scale=S, tile=16, calib=None, with_v1=True)
    assert d["wcsi"] is not None and d["energies"]["wcsi_cyto_dino"] is not None
    assert d["configs"]["N0_wcsi"]["score"] is not None


# ---------------------------------------------------------------- terms and corruptions

def test_combine_skips_missing():
    E = dict(a=2.0, b=None)
    c = T.combine(E, ["a", "b"], kind="ref")
    assert c["missing"] == ["b"] and c["score"] == pytest.approx(100 * math.exp(-2.0))
    assert T.combine(E, ["a"], kind="noref")["score"] == pytest.approx(100 / 3)
    for name, ts in {**T.CONFIGS_REF, **T.CONFIGS_NOREF}.items():
        assert all(t in T.TERMS for t in ts), name
    assert T.combine(dict(a=None), ["a"])["score"] is None


def test_lab_roundtrip():
    rgb = np.random.default_rng(4).random((16, 16, 3), dtype=np.float32)
    back = C.lab_to_srgb(wcsi.srgb_to_lab(rgb, 1.0))
    assert np.abs(back - rgb).max() < 2e-3


def test_corruptions_geometry(data):
    gt, lr, _, _ = data
    rects = C.layout(lr.shape[0], lr.shape[1], 16, 1)
    for k in C.KINDS:
        out = C.corrupt(k, gt, lr, S, rects)
        assert out.shape == gt.shape and np.isfinite(out).all(), k
    drift = C.corrupt("F2_colour_drift", gt, lr, S, rects)
    db = wcsi.srgb_to_lab(drift, 1.0)[..., 2] - wcsi.srgb_to_lab(gt, 1.0)[..., 2]
    assert db[:, :8].mean() > 2.5 and db[:, -8:].mean() < -2.5
    seam = C.corrupt("F3_seam_step", gt, lr, S, rects)
    assert np.allclose(seam[:, :S * 15], gt[:, :S * 15])  # first tile column untouched


def test_ablate_end_to_end(tmp_path, data):
    from PIL import Image
    gt, lr, _, _ = data
    gt, lr = gt[:128, :128], lr[:32, :32]
    d = tmp_path / "img0"
    d.mkdir()
    save = lambda p, a: Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).save(p)  # noqa: E731
    save(d / "gt.png", gt)
    save(d / "lr.png", lr)
    rects = C.layout(32, 32, 16, 1)
    C._tiles_jsonl(d / "tiles.jsonl", rects)
    man = tmp_path / "manifest.jsonl"
    with man.open("w") as fh:
        for k in ("clean", "F1_texture_resample", "F2_colour_drift", "F3_seam_step", "F6_coherent_sharpen"):
            save(d / f"{k}.png", C.corrupt(k, gt, lr, S, rects))
            fh.write(json.dumps(dict(stem="img0", kind=k, sr=str(d / f"{k}.png"), gt=str(d / "gt.png"),
                                     lr=str(d / "lr.png"), tiles=str(d / "tiles.jsonl"), scale=S,
                                     tile_lr=16, overlap_lr=1)) + "\n")
    res = tmp_path / "res.jsonl"
    assert ablate.main(["run", "--manifest", str(man), "--results", str(res), "--gates", "pix,none"]) == 0
    rows = [json.loads(x) for x in res.read_text().splitlines()]
    assert len(rows) == 10 and not any("error" in r for r in rows), [r.get("error") for r in rows]
    assert ablate.main(["summarize", str(res), "--out", str(tmp_path / "sum")]) == 0
    md = (tmp_path / "sum" / "summary.md").read_text()
    assert "ref:F_hybrid" in md and "noref:N3_hybrid" in md and "base:A_mse" in md


def test_build_stack_gabor_default():
    st = build_stack("gabor")
    assert st.has("texture") and st.has("semantic") and not st.has("latent")
    assert isinstance(next(iter(st.encode(np.zeros((32, 32, 3), np.float32), ("texture",))["texture"].values())),
                      FeatureMap)
