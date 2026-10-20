#!/usr/bin/env bash
# Pairing report only. Does not score.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CFG="${GENET_ABLATION_CONFIG:-genet_ablation/config.json}"
python3 genet_ablation/inventory.py --config "$CFG" --check-sizes 3
