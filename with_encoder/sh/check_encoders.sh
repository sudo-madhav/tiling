#!/usr/bin/env bash
# Step 0 on the GPU machine: every encoder in the config loads from local files and returns
# feature maps of the expected shape. Downloads nothing; an absent file is reported as UNAVAILABLE.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
: "${WITH_ENCODER_WEIGHTS:?set WITH_ENCODER_WEIGHTS to the directory that already holds the weights}"
ENC="${WITH_ENCODER_CONFIG:-with_encoder/configs/encoders.json}"
python3 with_encoder/encoders/registry.py --check "$ENC"
