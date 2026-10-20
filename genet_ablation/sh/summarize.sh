#!/usr/bin/env bash
# Tables from whatever JSON the sweeps have written. Reads the out path from the config.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CFG="${GENET_ABLATION_CONFIG:-genet_ablation/config.json}"
OUT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["out"])' "$CFG")"
DATASET="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("dataset","div8k"))' "$CFG")"
python3 genet_ablation/summarize.py --out "$OUT" --dataset "$DATASET"
