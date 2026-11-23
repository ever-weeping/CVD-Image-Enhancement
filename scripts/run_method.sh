#!/usr/bin/env bash
# One alpha model of the OWN method (linearized simulation loss; promoted 2026-11-23).
# Full two-phase training (alpha=0 pretrain + finetune) unless an init checkpoint
# (a linearized pretrained_alpha0.pt) is given, then finetune-only branch.
# Usage: bash scripts/run_method.sh <protan|deutan> <alpha> <gpu_id> <out_root> [init_from]
set -euo pipefail
cd "$(dirname "$0")/.."
AXIS="$1"; ALPHA="$2"; GPU="$3"; OUT="$4"; INIT="${5:-}"
PY="${PY:-/home/mahai/miniconda3/envs/image_enhance/bin/python}"
RUN="$OUT/method_${AXIS}_alpha${ALPHA}"
mkdir -p "$OUT"
trap 'echo "FAILED $AXIS alpha$ALPHA $(date -Is)" > "$OUT/STATUS.txt"' ERR
echo "TRAIN $AXIS alpha$ALPHA $(date -Is)" > "$OUT/STATUS.txt"
if [ -n "$INIT" ]; then
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py --axis "$AXIS" --alpha "$ALPHA" \
        --sim-train linearized --init-from "$INIT" --out "$RUN" \
        > "$OUT/train_${AXIS}_alpha${ALPHA}.log" 2>&1
else
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py --axis "$AXIS" --alpha "$ALPHA" \
        --sim-train linearized --out "$RUN" \
        > "$OUT/train_${AXIS}_alpha${ALPHA}.log" 2>&1
fi
echo "EVAL $AXIS alpha$ALPHA $(date -Is)" >> "$OUT/STATUS.txt"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" evaluate.py --axis "$AXIS" \
    --checkpoint "$RUN/final.pt" --split validation \
    --out "$OUT/eval_${AXIS}_alpha${ALPHA}" > "$OUT/eval_${AXIS}_alpha${ALPHA}.log" 2>&1
echo "COMPLETE $AXIS alpha$ALPHA $(date -Is)" > "$OUT/STATUS.txt"
