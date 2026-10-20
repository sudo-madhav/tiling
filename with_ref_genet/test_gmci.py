"""CPU tests for GMCI. No DIV8K.  pytest -q with_ref_genet/test_gmci.py"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "with_ref_genet"))
import gmci  # noqa: E402


def scene(h=96, w=128, seed=0):
    yy = np.linspace(0, 1, h)[:, None]
    xx = np.linspace(0, 1, w)[None, :]
    img = np.stack([
        0.35 + 0.25 * np.broadcast_to(yy, (h, w)),
        0.40 + 0.15 * np.broadcast_to(xx, (h, w)),
        0.50 - 0.10 * np.broadcast_to(yy, (h, w)),
    ], -1).astype(np.float32)
    rng = np.random.default_rng(seed)
    img += 0.012 * rng.standard_normal(img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def down(img, s=2):
    h, w = img.shape[0] // s, img.shape[1] // s
    return img[:h * s, :w * s].reshape(h, s, w, s, 3).mean((1, 3)).astype(np.float32)


def paint(img, s, tile, delta):
    """Add an RGB offset per LR tile. delta(row, col) is a scalar or None."""
    out = img.copy()
    ys = range(0, img.shape[0] // s - tile + 1, tile)
    xs = range(0, img.shape[1] // s - tile + 1, tile)
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            d = delta(r, c)
            if not d:
                continue
            out[s * y:s * (y + tile), s * x:s * (x + tile)] = np.clip(
                out[s * y:s * (y + tile), s * x:s * (x + tile)] + d, 0, 1)
    return out


def score(sr, gt):
    return gmci.gmci_score(sr, gt, down(gt), scale=2, tile=16, overlap=0, return_details=True)


def test_rigidity_locks_an_edge_and_not_noise():
    lum = np.full((64, 96), 0.35, np.float32)
    lum[:, 48:] = 0.85
    edge = gmci.rigidity_map(lum)[:, 45:52].mean()
    noise = gmci.rigidity_map(np.random.default_rng(0).standard_normal((64, 96)).astype(np.float32))
    assert edge > 0.35
    assert noise.mean() < edge / 2


def test_clone_scores_high_and_has_no_seam():
    gt = scene()
    d = gmci.gmci_score(gt, gt, down(gt), scale=2, tile=16, overlap=4, return_details=True)
    assert d["gmci"] > 90
    assert d["energies"]["E_master"] < 0.15
    assert d["energies"]["E_frust"] < 0.15
    assert d["energies"]["E_seam"] is not None and d["energies"]["E_seam"] < 0.05
    assert d["energies"]["E_lr"] < 1e-6
    assert d["parts"]["phenotype_counts"]["clonal masterpiece"] == d["n_tiles"]


def test_coherent_shift_is_not_the_same_failure_as_a_checkerboard():
    gt = scene()
    coherent = paint(gt, 2, 16, lambda r, c: 0.08)
    checker = paint(gt, 2, 16, lambda r, c: 0.08 if (r + c) % 2 == 0 else -0.08)
    a, b = score(coherent, gt), score(checker, gt)
    assert a["energies"]["E_frust"] < b["energies"]["E_frust"] / 3
    assert a["rho_coherence"] > 0.7
    assert b["rho_coherence"] < 0.4
    assert a["energies"]["E_master"] > 1  # the departure is real; it is just shared


def test_phenotypes_separate_an_island_from_its_neighbors():
    gt = scene()
    sr = paint(gt, 2, 16, lambda r, c: 0.0 if (r, c) == (1, 1) else 0.08)
    d = score(sr, gt)
    tiles = {(t["y0"] // 16, t["x0"] // 16): t for t in d["tiles"]}
    assert tiles[(1, 1)]["phenotype"] == "isolated clonal island"
    assert tiles[(0, 0)]["phenotype"] == "allometric polymorphism"
    assert tiles[(1, 1)]["E_master"] < tiles[(0, 0)]["E_master"] / 3
    assert tiles[(1, 1)]["E_frust"] > tiles[(0, 0)]["E_frust"] * 3


def test_a_moved_edge_costs_more_than_a_moved_texture():
    # At 8x the coarse-graining that the positional term uses removes a texture shift
    # and keeps a shifted contour.
    edge = np.full((96, 128, 3), 0.40, np.float32)
    edge[:, 48:] = 0.75
    moved_edge = np.full_like(edge, 0.40)
    moved_edge[:, 56:] = 0.75
    rng = np.random.default_rng(2)
    noise = np.clip(0.5 + 0.25 * rng.standard_normal((96, 128, 3)).astype(np.float32), 0, 1)
    moved_noise = np.roll(noise, 8, axis=1)

    def run(sr, gt):
        return gmci.gmci_score(sr, gt, down(gt, 8), scale=8, tile=8, overlap=0, return_details=True)

    edge_cost = run(moved_edge, edge)["parts"]["E_wadd_positional"]
    noise_cost = run(moved_noise, noise)["parts"]["E_wadd_positional"]
    assert edge_cost > noise_cost * 2
