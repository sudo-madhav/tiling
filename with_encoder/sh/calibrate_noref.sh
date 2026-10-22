#!/usr/bin/env bash
# Step 1b: natural models for the no-reference score, fitted on ground-truth / LR pairs that are
# NOT scored later (a held-out split): the WCSI-2 pixel model and the feature model, per scale,
# for the learned stack and the Gabor control, at the 512 SR px analysis tile.
# CALIB_GT: ground-truth directory; CALIB_LR_8 / CALIB_LR_4: the matching LR directories,
# degraded exactly like the evaluated inputs. CALIB_LIMIT images (default 50).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
ENC="${WITH_ENCODER_CONFIG:-with_encoder/configs/encoders.json}"
OUT="${WITH_ENCODER_OUT:?set WITH_ENCODER_OUT}"
GT="${CALIB_GT:?set CALIB_GT}"
LIMIT="${CALIB_LIMIT:-50}"
mkdir -p "$OUT/calib"
for SCALE in 8 4; do
  LRV="CALIB_LR_$SCALE"
  LR="${!LRV:?set $LRV}"
  python3 genet/wcsi2.py calibrate --gt "$GT" --lr "$LR" --scale "$SCALE" --tile-sr 512 --limit "$LIMIT" \
      --device cuda:0 --note "held-out calibration split, ${LIMIT} images" --out "$OUT/calib/wcsi2_x${SCALE}_t512.json"
  CUDA_VISIBLE_DEVICES=0 python3 with_encoder/hybrid_noref.py calibrate --gt "$GT" --lr "$LR" --scale "$SCALE" \
      --tile-sr 512 --limit "$LIMIT" --encoders "$ENC" --out "$OUT/calib/feat_learned_x${SCALE}_t512.json"
  python3 with_encoder/hybrid_noref.py calibrate --gt "$GT" --lr "$LR" --scale "$SCALE" \
      --tile-sr 512 --limit "$LIMIT" --encoders gabor --out "$OUT/calib/feat_gabor_x${SCALE}_t512.json"
done
echo "Copy $OUT/calib/wcsi2_x*_t512.json to genet/calib/ (or pass --calib) to use the refitted pixel model."
