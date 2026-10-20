#!/usr/bin/env bash
# Step 1: controlled corruptions of DIV8K ground truth at 8x and 4x (ABLATION.md, part 1).
# 100 images per scale is enough to rank the arms; raise LIMIT for the paper table.
# CPU only, one shard per core group. Does not download DIV8K.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CFG="${GENET_ABLATION_CONFIG:-genet_ablation/config.json}"
OUT="${WITH_ENCODER_OUT:?set WITH_ENCODER_OUT}/synth"
LIMIT="${LIMIT:-100}"
SHARDS="${SHARDS:-8}"
mkdir -p "$OUT"
for SCALE in 8 4; do
  for ((k = 0; k < SHARDS; k++)); do
    python3 with_encoder/corruptions.py --config "$CFG" --scale "$SCALE" --tile-sr 512 \
        --out "$OUT" --limit "$LIMIT" --shard "$k/$SHARDS" > "$OUT.x$SCALE.shard$k.log" 2>&1 &
  done
  wait
done
