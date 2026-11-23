#!/usr/bin/env bash
# One axis of the author reproduction: train (alpha=0.5) then validation eval.
# Usage: bash scripts/run_axis.sh <protan|deutan> <gpu_id> <out_root> [extra_train_args]
set -euo pipefail
cd "$(dirname "$0")/.."
AXIS="$1"; GPU="$2"; OUT="$3"
EXTRA="${4:-}"   # unquoted on purpose: word-split extra train.py flags, e.g. "--sim-train linearized"
PY="${PY:-/home/mahai/miniconda3/envs/image_enhance/bin/python}"
mkdir -p "$OUT"
trap 'echo "FAILED $AXIS $(date -Is)" > "$OUT/STATUS.txt"' ERR
echo "TRAIN $AXIS $(date -Is)" > "$OUT/STATUS.txt"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py --axis "$AXIS" --alpha 0.5 \
    --sim-train author $EXTRA \
    --out "$OUT/author_${AXIS}_alpha0.5" > "$OUT/train_${AXIS}.log" 2>&1
echo "EVAL $AXIS $(date -Is)" >> "$OUT/STATUS.txt"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" evaluate.py --axis "$AXIS" \
    --checkpoint "$OUT/author_${AXIS}_alpha0.5/final.pt" --split validation \
    --out "$OUT/eval_${AXIS}" > "$OUT/eval_${AXIS}.log" 2>&1
echo "COMPLETE $AXIS $(date -Is)" > "$OUT/STATUS.txt"
