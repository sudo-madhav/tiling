"""CPU tests for wcsi_viz.py:  pytest -q genet/test_wcsi_viz.py"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wcsi  # noqa: E402
import wcsi_viz  # noqa: E402
from test_wcsi import checker, make_pair  # noqa: E402


def _assert_adds_up(D):
    assert wcsi_viz.check(D) < 1e-9
    assert D["F_tile"].sum() == pytest.approx(D["det"]["F_total"], rel=1e-9)


@pytest.mark.parametrize("s", [2, 4])
def test_parts_add_up_on_a_grid(s):
    lr, sr = make_pair(s)
    sr = np.clip(sr + 0.04 * checker(lr.shape, s, 16, 2) * np.array([1, 0, 0], np.float32), 0, 1)
    _assert_adds_up(wcsi_viz.decompose(sr, lr, tile=16, overlap=2))


def test_parts_add_up_on_other_layouts():
    lr, sr = make_pair(2, 48, 64)
    rects = [dict(x=x, y=0, w=16, h=16) for x in (0, 16, 32, 48)] + \
            [dict(x=x, y=16, w=16, h=16) for x in (8, 24, 40)] + \
            [dict(x=x, y=32, w=16, h=16) for x in (0, 16, 32, 48)]
    _assert_adds_up(wcsi_viz.decompose(sr, lr, tile_coords=rects))
    ys, xs = wcsi.tile_starts(48, 16, 4), wcsi.tile_starts(64, 16, 4)
    canv = [sr[2 * y:2 * (y + 16), 2 * x:2 * (x + 16)].copy() for y in ys for x in xs]
    canv[3] = np.clip(canv[3] + 0.05, 0, 1)
    _assert_adds_up(wcsi_viz.decompose(sr, lr, tile=16, overlap=4, tiles=canv))


def test_figures_written(tmp_path):
    pytest.importorskip("matplotlib")
    PIL = pytest.importorskip("PIL.Image")
    lr, sr = make_pair(4, 32, 48)
    u8 = lambda a: (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    PIL.fromarray(u8(lr)).save(tmp_path / "lr.png")
    PIL.fromarray(u8(sr)).save(tmp_path / "sr.png")
    out = tmp_path / "viz"
    assert wcsi_viz.main([str(tmp_path / "sr.png"), str(tmp_path / "lr.png"), "--tile", "16", "--overlap", "2",
                          "--out", str(out), "--tile-ids", "0", "4"]) == 0
    for name in ("overview", "term_maps", "pairs", "seams", "tile_0", "tile_4"):
        assert (out / f"{name}.png").stat().st_size > 0
    n = len(wcsi.tile_starts(32, 16, 2)) * len(wcsi.tile_starts(48, 16, 2))
    assert (out / "tiles.csv").read_text().count("\n") == 1 + n
