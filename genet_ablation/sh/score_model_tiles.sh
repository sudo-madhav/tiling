#!/usr/bin/env bash
# Model-tile folders (vosr_tile*, seedvr_vae*). Empty paths are skipped.
# Each folder is scored at 256, 512 and 1024, so model tile and analysis tile are not confounded.
# Pilot size. Point the config at the re-inferred directories before running.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CFG="${GENET_ABLATION_CONFIG:-genet_ablation/config.json}"
python3 genet_ablation/score_sweep.py --config "$CFG" \
    --scales 8,4 \
    --methods vosr_tile256,vosr_tile512,vosr_tile1024,seedvr_vae256,seedvr_vae512,seedvr_vae1024 \
    --targets sr \
    --tiles 1024,512,256 \
    --overlaps tile/16 \
    --offsets 0 \
    --limit 20 \
    --workers 4
