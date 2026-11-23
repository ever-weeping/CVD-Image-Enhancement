#!/usr/bin/env bash
# One alpha branch of the author reproduction: finetune only from a shared
# alpha=0 pretrain (--init-from), then validation eval.
# Usage: bash scripts/run_branch.sh <protan|deutan> <alpha> <gpu_id> <out_root> [init_from]
set -euo pipefail
cd "$(dirname "$0")/.."
AXIS="$1"; ALPHA="$2"; GPU="$3"; OUT="$4"
INIT="${5:-$HOME/mahai_ext/cvd_runs/reproduction_v3_20261122/$AXIS/author_${AXIS}_alpha0.5/pretrained_alpha0.pt}"
PY="${PY:-/home/mahai/miniconda3/envs/image_enhance/bin/python}"
RUN="$OUT/author_${AXIS}_alpha${ALPHA}"
mkdir -p "$OUT"
trap 'echo "FAILED $AXIS alpha$ALPHA $(date -Is)" > "$OUT/STATUS.txt"' ERR
echo "TRAIN $AXIS alpha$ALPHA $(date -Is)" > "$OUT/STATUS.txt"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py --axis "$AXIS" --alpha "$ALPHA" \
    --sim-train author --init-from "$INIT" --out "$RUN" \
    > "$OUT/train_${AXIS}_alpha${ALPHA}.log" 2>&1
echo "EVAL $AXIS alpha$ALPHA $(date -Is)" >> "$OUT/STATUS.txt"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" evaluate.py --axis "$AXIS" \
    --checkpoint "$RUN/final.pt" --split validation \
    --out "$OUT/eval_${AXIS}_alpha${ALPHA}" > "$OUT/eval_${AXIS}_alpha${ALPHA}.log" 2>&1
echo "COMPLETE $AXIS alpha$ALPHA $(date -Is)" > "$OUT/STATUS.txt"
