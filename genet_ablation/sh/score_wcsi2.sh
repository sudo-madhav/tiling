#!/usr/bin/env bash
# WCSI-2 without and with reference on every paired stem, one worker per GPU.
# GPUS: comma list of devices (cuda:0*2 = two workers on GPU 0); empty = numpy on WORKERS processes.
# CALIB: genet/calib lookup ("auto"), a JSON fitted with `genet/wcsi2.py calibrate`, or "none".
# Does not run SR models and does not download DIV8K. Read PROTOCOL.md for the runtime.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CFG="${GENET_ABLATION_CONFIG:-genet_ablation/config.json}"
GPUS="${GPUS:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}"
WORKERS="${WORKERS:-8}"
CALIB="${CALIB:-auto}"
TILES="${TILES:-1024,512,256}"
par=(--workers "$WORKERS")
[ -n "$GPUS" ] && par=(--devices "$GPUS")
for metric in wcsi2 wcsi2_ref; do
    python3 genet_ablation/score_sweep.py --config "$CFG" --metric "$metric" --calib "$CALIB" \
        --scales 8,4 \
        --methods bicubic,vosr,casr,seedvr \
        --targets sr,gt \
        --tiles "$TILES" \
        --overlaps tile/16 \
        --offsets 0 \
        "${par[@]}"
done
