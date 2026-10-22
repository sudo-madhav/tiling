#!/usr/bin/env bash
# Step 3: hybrid scores of the real SR outputs (bicubic, VOSR, CASR, SeedVR) and of the ground truth,
# at 8x and 4x, analysis tile 512 SR px. One process per GPU, sharded by image. Resumable.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CFG="${GENET_ABLATION_CONFIG:-genet_ablation/config.json}"
ENC="${WITH_ENCODER_CONFIG:-with_encoder/configs/encoders.json}"
GPUS="${GPUS:-8}"
TAG="${TAG:-learned}"
LOGDIR="${LOGDIR:-/tmp/with_encoder_logs}"
mkdir -p "$LOGDIR"
OUT="${WITH_ENCODER_OUT:-}"
for SCALE in 8 4; do
  FC=(); [ -n "$OUT" ] && [ -f "$OUT/calib/feat_${TAG}_x${SCALE}_t512.json" ] && FC=(--feat-calib "$OUT/calib/feat_${TAG}_x${SCALE}_t512.json")
  for ((g = 0; g < GPUS; g++)); do
    CUDA_VISIBLE_DEVICES=$g python3 with_encoder/score_methods.py --config "$CFG" --encoders "$ENC" \
        --tag "$TAG" --scale "$SCALE" --tile-sr 512 --methods bicubic,vosr,casr,seedvr --targets sr,gt \
        --device cuda:0 "${FC[@]}" \
        --shard "$g/$GPUS" > "$LOGDIR/x$SCALE.gpu$g.log" 2>&1 &
  done
  wait
  for Q in score energy; do
    python3 with_encoder/score_methods.py --summarize --quantity "$Q" --config "$CFG" --tag "$TAG" \
        --scale "$SCALE" --tile-sr 512
  done
done
