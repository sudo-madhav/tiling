"""CPU tests for the ablation harness. No DIV8K.  pytest -q genet_ablation/test_ablation.py"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "genet"))
sys.path.insert(0, str(ROOT / "genet_ablation"))
import common  # noqa: E402
import score_sweep  # noqa: E402
import summarize  # noqa: E402
from test_wcsi import make_pair  # noqa: E402


def _png(path, array):
    PIL = pytest.importorskip("PIL.Image")
    u8 = (np.clip(array, 0, 1) * 255 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    PIL.fromarray(u8).save(path)


def _dataset(tmp_path, scale=2, n=2):
    gt, lr, sr = tmp_path / "gt", tmp_path / "lr", tmp_path / "sr"
    for i in range(n):
        low, high = make_pair(scale, 32, 48, seed=i)
        _png(gt / f"im{i}.png", high)
        _png(lr / f"im{i}.png", low)
        shifted = np.clip(high + 0.02, 0, 1)
        _png(sr / f"im{i}.jpg", shifted)
    cfg = {
        "dataset": "toy",
        "gt": str(gt),
        "out": str(tmp_path / "out"),
        "scales": {"2": {"lr": str(lr), "lr_recipe": "test",
                         "methods": {"bicubic": "", "vosr": str(sr), "casr": ""}}},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path, cfg


def test_sr_tile_converts_to_lr_pixels():
    assert common.analysis_grid(8, 1024, 64, 0) == (128, 8, 0)
    assert common.analysis_grid(8, 512, 32, 0) == (64, 4, 0)
    assert common.analysis_grid(8, 256, 16, 0) == (32, 2, 0)
    assert common.analysis_grid(4, 1024, 64, 0) == (256, 16, 0)
    assert common.default_overlap_sr(1024) == 64
    with pytest.raises(ValueError):
        common.analysis_grid(8, 100, 0, 0)
    with pytest.raises(ValueError):
        common.analysis_grid(8, 32, 0, 0)  # 4 LR px, below the minimum of 8


def test_offset_lattice_does_not_flush_to_the_border():
    rects = common.lattice_coords(100, 90, tile=32, overlap=0, offset=16)
    xs = sorted({r["x"] for r in rects})
    assert xs[0] == 16 and 90 - 32 not in xs


def test_job_count_and_missing_scale(tmp_path):
    path, _ = _dataset(tmp_path)
    cfg = common.load_config(path)
    tiles = [32]
    overlaps = common.parse_overlaps("tile/16", tiles)
    offsets = common.parse_offsets("0,half", tiles)
    jobs = score_sweep.build_jobs(cfg, [2], ["bicubic", "vosr", "casr"], {"sr", "gt"},
                                  tiles, overlaps, offsets, limit=0, shard=None)
    # 2 stems x 2 offsets x (gt + bicubic + vosr); casr path empty is skipped
    assert len(jobs) == 2 * 2 * 3
    assert {j["method"] for j in jobs} == {"gt", "bicubic", "vosr"}
    cfg["scales"]["4"] = {"lr": str(tmp_path / "no_such_lr"), "methods": {"vosr": ""}}
    assert score_sweep.build_jobs(cfg, [4], ["vosr"], {"sr"}, tiles, overlaps, offsets, 0, None) == []
    with pytest.raises(ValueError):
        score_sweep.build_jobs(cfg, [9], ["vosr"], {"sr"}, tiles, overlaps, offsets, 0, None)


def test_score_resume_and_summary(tmp_path):
    path, _ = _dataset(tmp_path)
    assert score_sweep.main(["--config", str(path), "--scales", "2", "--methods", "bicubic,vosr",
                             "--targets", "sr,gt", "--tiles", "32", "--overlaps", "tile/16",
                             "--offsets", "0", "--workers", "2"]) == 0
    first = list((tmp_path / "out").rglob("*.json"))
    assert len(first) == 2 * 3  # stems x (gt, bicubic, vosr)
    mtimes = {p: p.stat().st_mtime_ns for p in first}
    assert score_sweep.main(["--config", str(path), "--scales", "2", "--methods", "bicubic,vosr",
                             "--targets", "sr,gt", "--tiles", "32", "--workers", "1"]) == 0
    assert {p: p.stat().st_mtime_ns for p in first} == mtimes
    assert summarize.main(["--out", str(tmp_path / "out"), "--dataset", "toy"]) == 0
    text = (tmp_path / "out" / "toy" / "summary.md").read_text()
    assert "Against the ground truth" in text
    paired = list(csv_rows(tmp_path / "out" / "toy" / "vs_gt.csv"))
    assert {r["method"] for r in paired} == {"bicubic", "vosr"}
    assert all(int(r["n"]) == 2 for r in paired)


def test_missing_stem_is_recorded(tmp_path):
    path, _ = _dataset(tmp_path, n=1)
    cfg = json.loads(path.read_text())
    for p in Path(cfg["scales"]["2"]["methods"]["vosr"]).glob("im0.*"):
        p.unlink()
    score_sweep.main(["--config", str(path), "--scales", "2", "--methods", "vosr",
                      "--targets", "sr", "--tiles", "32", "--workers", "1"])
    row = json.loads(next((tmp_path / "out").rglob("*.json")).read_text())
    assert row["status"] == "missing"


def test_half_tile_offset_scores(tmp_path):
    path, _ = _dataset(tmp_path, n=1)
    assert score_sweep.main(["--config", str(path), "--scales", "2", "--methods", "vosr",
                             "--targets", "gt", "--tiles", "32", "--offsets", "half",
                             "--workers", "1"]) == 0
    row = json.loads(next((tmp_path / "out").rglob("*.json")).read_text())
    assert row["status"] == "ok" and row["offset_sr"] == 16
    assert 0 < row["wcsi"] <= 100


def csv_rows(path):
    import csv
    with path.open() as fh:
        yield from csv.DictReader(fh)
