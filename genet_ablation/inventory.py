#!/usr/bin/env python3
"""List which DIV8K stems pair across gt, lr and each SR method. Scores nothing.

    python genet_ablation/inventory.py --config genet_ablation/config.json
    python genet_ablation/inventory.py --config genet_ablation/config.json --check-sizes 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def _dir_status(path):
    path = Path(path) if path else None
    if path is None or str(path) == "":
        return "unset", {}
    if not path.is_dir():
        return "missing", {}
    try:
        return "ok", common.index_images(path)
    except ValueError as exc:
        return f"bad ({exc})", {}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check that gt, lr and SR directories pair by filename stem.")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.json"))
    ap.add_argument("--check-sizes", type=int, default=0,
                    help="open this many headers per method and report SR/LR size ratios")
    args = ap.parse_args(argv)
    cfg = common.load_config(args.config)
    gt_state, gt = _dir_status(cfg["gt"])
    print(f"dataset {cfg.get('dataset', '?')}  gt {cfg['gt']}  [{gt_state}]  {len(gt)} images")
    if gt_state != "ok":
        return 1
    for scale, spec in cfg["scales"].items():
        lr_state, lr = _dir_status(spec["lr"])
        paired = sorted(set(gt) & set(lr))
        print(f"\nscale {scale}x  lr {spec['lr']}  [{lr_state}]  {len(lr)} images  "
              f"paired with gt {len(paired)}  ({spec.get('lr_recipe', '')})")
        if lr_state != "ok":
            continue
        only_gt = sorted(set(gt) - set(lr))[:5]
        only_lr = sorted(set(lr) - set(gt))[:5]
        if only_gt:
            print(f"  gt stems with no lr (first few): {only_gt}")
        if only_lr:
            print(f"  lr stems with no gt (first few): {only_lr}")
        for name, raw in spec["methods"].items():
            info = common.method_spec(raw)
            if name == "bicubic" and info["path"] == "":
                print(f"  {name:24s} synthesized from lr at score time  ({len(paired)} stems)")
                continue
            state, files = _dir_status(info["path"])
            both = sorted(set(paired) & set(files))
            tile = f"  model_tile_sr {info['model_tile_sr']}" if info["model_tile_sr"] else ""
            print(f"  {name:24s} [{state}]  {len(files)} files  paired {len(both)}{tile}")
            if state == "ok" and args.check_sizes and both:
                for stem in both[:args.check_sizes]:
                    gw, gh = common.header_size(gt[stem])
                    lw, lh = common.header_size(lr[stem])
                    sw, sh = common.header_size(files[stem])
                    print(f"    {stem}: gt {gw}x{gh}  lr {lw}x{lh}  sr {sw}x{sh}  "
                          f"sr/lr {sw / lw:.3f} x {sh / lh:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
