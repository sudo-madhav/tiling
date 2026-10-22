#!/usr/bin/env python3
"""Aggregate per-image WCSI JSON into tables, including the gt comparison.

    python genet_ablation/summarize.py --out /path/to/wcsi_ablation_out --dataset div8k
    python genet_ablation/summarize.py --out /path/to/wcsi_ablation_out --dataset div8k --metric wcsi2

WCSI v1 rewards internal consistency, not fidelity: natural ground truth scores low because
tiles of one material do not share a texture. WCSI-2 (no reference) subtracts what natural
images score, so the ground truth should come out on top; frac_above_gt is then the share of
images on which a method beats the photograph. wcsi2_ref has no gt rows (the ground truth is
the reference). See PROTOCOL.md and genet/WCSI2.md.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def load_rows(out_dir, dataset, metric="wcsi"):
    """Rows of one metric, with the score under "score" (v1 rows only have "wcsi")."""
    root = Path(out_dir) / dataset
    rows = []
    if not root.is_dir():
        raise FileNotFoundError(root)
    for path in sorted(root.rglob("*.json")):
        row = json.loads(path.read_text())
        if "method" not in row or row.get("metric", "wcsi") != metric:
            continue
        if "score" not in row:
            if "wcsi" not in row:
                continue
            row["score"] = row["wcsi"]
        rows.append(row)
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    mid = len(xs) // 2
    if len(xs) % 2:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def _std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _key(row):
    return (row["scale"], row["tile_sr"], row["overlap_sr"], row["offset_sr"], row["image"])


def _fmt(value, digits):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def condition_table(rows, terms=common.TERMS):
    groups = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        groups[(row["scale"], row["method"], row["tile_sr"], row["overlap_sr"], row["offset_sr"])].append(row)
    table = []
    for key in sorted(groups):
        group = groups[key]
        rec = dict(scale=key[0], method=key[1], tile_sr=key[2], overlap_sr=key[3], offset_sr=key[4],
                   n=len(group), score_mean=_mean([r["score"] for r in group]),
                   score_median=_median([r["score"] for r in group]),
                   score_std=_std([r["score"] for r in group]))
        for term in terms:
            rec[f"{term}_mean"] = _mean([r.get(term) for r in group])
        table.append(rec)
    return table


def paired_table(rows, terms=common.TERMS):
    """One row per SR method and analysis grid: score(method) against score(gt) on the same images."""
    gt = {_key(r): r for r in rows if r["method"] == "gt" and r.get("status") == "ok"}
    groups = defaultdict(list)
    for row in rows:
        if row["method"] == "gt" or row.get("status") != "ok":
            continue
        ref = gt.get(_key(row))
        if ref is None or ref["score"] in (None, 0):
            continue
        groups[(row["scale"], row["method"], row["tile_sr"], row["overlap_sr"], row["offset_sr"])].append((row, ref))
    table = []
    for key in sorted(groups):
        pairs = groups[key]
        delta = [a["score"] - b["score"] for a, b in pairs]
        ratio = [a["score"] / b["score"] for a, b in pairs]
        rec = dict(scale=key[0], method=key[1], tile_sr=key[2], overlap_sr=key[3], offset_sr=key[4],
                   n=len(pairs), score_mean=_mean([a["score"] for a, _ in pairs]),
                   score_gt_mean=_mean([b["score"] for _, b in pairs]),
                   delta_mean=_mean(delta), delta_median=_median(delta),
                   ratio_mean=_mean(ratio),
                   frac_above_gt=sum(d > 0 for d in delta) / len(delta))
        for term in terms:
            rec[f"delta_{term}"] = _mean([a[term] - b[term] for a, b in pairs
                                          if a.get(term) is not None and b.get(term) is not None])
        table.append(rec)
    return table


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _md_table(rows, columns):
    if not rows:
        return "(no rows)\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"{value:.3f}" if abs(value) < 100 else f"{value:.2f}")
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body]) + "\n"


_INTRO = dict(
    wcsi="WCSI compares each image with the low-resolution input. The ground-truth column is WCSI of the "
         "original image against that same input. A method scoring above the ground truth is more "
         "self-consistent than the photograph; it is not closer to the photograph.",
    wcsi2="WCSI-2 compares each image with the low-resolution input and subtracts what natural images "
          "score (calibrated on ground-truth / LR pairs), so the photograph is the expected optimum. "
          "frac_above_gt is the share of images on which a method beats the photograph.",
    wcsi2_ref="WCSI-2 with reference compares each image with the ground truth, tile by tile; 100 is "
              "the ground truth itself. E_detail and E_consist split the departure into a shared part "
              "and a tile-to-tile part.",
)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Summarize WCSI ablation JSON into CSV and Markdown.")
    ap.add_argument("--out", required=True, help="the config's out directory")
    ap.add_argument("--dataset", default="div8k")
    ap.add_argument("--metric", default="wcsi", choices=sorted(common.METRICS))
    args = ap.parse_args(argv)
    terms = common.metric_terms(args.metric)
    rows = load_rows(args.out, args.dataset, args.metric)
    ok = [r for r in rows if r.get("status") == "ok"]
    bad = [r for r in rows if r.get("status") != "ok"]
    by_method = condition_table(rows, terms)
    paired = paired_table(rows, terms)
    dest = Path(args.out) / args.dataset
    if args.metric != "wcsi":
        dest = dest / args.metric
        dest.mkdir(parents=True, exist_ok=True)
    _write_csv(dest / "by_condition.csv", by_method)
    _write_csv(dest / "vs_gt.csv", paired)
    name = args.metric.upper()
    lines = [
        f"# {name} ablation summary ({args.dataset})",
        "",
        f"{len(ok)} scored images, {len(bad)} missing or failed.",
        "",
        _INTRO[args.metric],
        "",
        "## Per condition",
        "",
        _md_table(by_method, ["scale", "method", "tile_sr", "overlap_sr", "offset_sr", "n",
                              "score_mean", "score_median", "score_std",
                              *[f"{t}_mean" for t in terms]]),
        "",
    ]
    if args.metric != "wcsi2_ref":
        lines += [
            "## Against the ground truth, same images and same analysis grid",
            "",
            f"delta = {name}(method) - {name}(gt). delta_E_* = term(method) - term(gt); positive means the "
            "method is worse than the photograph on that term.",
            "",
            _md_table(paired, ["scale", "method", "tile_sr", "overlap_sr", "offset_sr", "n",
                               "score_mean", "score_gt_mean", "delta_mean", "ratio_mean", "frac_above_gt",
                               *[f"delta_{t}" for t in terms]]),
            "",
        ]
    if bad:
        lines.append("## Failures (first 30)\n")
        lines.append(_md_table(bad[:30], ["scale", "method", "image", "tile_sr", "status", "error"]))
    text = "\n".join(lines)
    (dest / "summary.md").write_text(text)
    print(text)
    print(f"wrote {dest / 'by_condition.csv'}, {dest / 'vs_gt.csv'}, {dest / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
