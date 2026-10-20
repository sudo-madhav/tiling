#!/usr/bin/env bash
# Seam probe on 20 stems: slide the analysis grid by half a tile, and vary the overlap.
# 32 SR px is VOSR's default blend; 128 SR px is the usual SeedVR VAE overlap.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CFG="${GENET_ABLATION_CONFIG:-genet_ablation/config.json}"
python3 genet_ablation/score_sweep.py --config "$CFG" \
    --scales 8,4 \
    --methods bicubic,vosr,casr,seedvr \
    --targets sr,gt \
    --tiles 1024,512 \
    --overlaps 0,32,128 \
    --offsets 0,half \
    --limit 20 \
    --workers 4
