#!/usr/bin/env bash
# Step 2: every ablation arm on the corruption manifests, one process per GPU.
# Runs the learned stack with all four gates, then the weight-free Gabor control (CPU-capable),
# then summarises. Results are resumable: rerun to fill in errors or new rows.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
ENC="${WITH_ENCODER_CONFIG:-with_encoder/configs/encoders.json}"
OUT="${WITH_ENCODER_OUT:?set WITH_ENCODER_OUT}"
GPUS="${GPUS:-8}"
DATASET="${DATASET:-div8k}"
mkdir -p "$OUT/ablation"
for SCALE in 8 4; do
  MAN=("$OUT/synth/$DATASET/x$SCALE"/manifest*.jsonl)
  # feature natural model from sh/calibrate_noref.sh; without it the no-reference feature terms are uncalibrated
  FC=(); [ -f "$OUT/calib/feat_learned_x${SCALE}_t512.json" ] && FC=(--feat-calib "$OUT/calib/feat_learned_x${SCALE}_t512.json")
  FG=(); [ -f "$OUT/calib/feat_gabor_x${SCALE}_t512.json" ] && FG=(--feat-calib "$OUT/calib/feat_gabor_x${SCALE}_t512.json")
  for ((g = 0; g < GPUS; g++)); do
    CUDA_VISIBLE_DEVICES=$g python3 with_encoder/ablate.py run --manifest "${MAN[@]}" --encoders "$ENC" \
        --tag "learned_x$SCALE" --gates pix,feat,none,all --shard "$g/$GPUS" --device cuda:0 "${FC[@]}" \
        --results "$OUT/ablation/learned_x$SCALE.shard$g.jsonl" > "$OUT/ablation/learned_x$SCALE.shard$g.log" 2>&1 &
  done
  wait
  for ((g = 0; g < GPUS; g++)); do
    python3 with_encoder/ablate.py run --manifest "${MAN[@]}" --encoders gabor \
        --tag "gabor_x$SCALE" --gates pix --shard "$g/$GPUS" "${FG[@]}" \
        --results "$OUT/ablation/gabor_x$SCALE.shard$g.jsonl" > "$OUT/ablation/gabor_x$SCALE.shard$g.log" 2>&1 &
  done
  wait
  python3 with_encoder/ablate.py summarize "$OUT"/ablation/*_x$SCALE.shard*.jsonl --out "$OUT/ablation/summary_x$SCALE"
done
