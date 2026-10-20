#!/usr/bin/env bash
# 20 stems, both scales, primary methods, three analysis tiles. Does not run SR models.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CFG="${GENET_ABLATION_CONFIG:-genet_ablation/config.json}"
python3 genet_ablation/score_sweep.py --config "$CFG" \
    --scales 8,4 \
    --methods bicubic,vosr,casr,seedvr \
    --targets sr,gt \
    --tiles 1024,512,256 \
    --overlaps tile/16 \
    --offsets 0 \
    --limit 20 \
    --workers 4
