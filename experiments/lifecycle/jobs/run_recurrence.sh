#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-$ROOT/results/b3_composite_v2/lifecycle_recurrence_qwen_low_t06}"
"$PYTHON_BIN" "$ROOT/experiments/lifecycle/b3_lifecycle.py" --manifest "$ROOT/src/b3_suite_manifest.json" --out-dir "$OUT_DIR" --protocol recurrence --partition-mode noniid --seeds 0 44 56 --models "$ROOT/src/benchmark_models.json" --resume
