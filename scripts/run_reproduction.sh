#!/usr/bin/env bash
# Pure author-method reproduction queue: deutan then protan, train + validation eval.
# Outputs (weights, logs, reports) go to the external disk, never /home.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/mahai/miniconda3/envs/image_enhance/bin/python}"
GPU="${CVD_GPU:-0}"
ALPHA="${ALPHA:-0.5}"
OUT_ROOT="${OUT_ROOT:-$HOME/mahai_ext/cvd_runs/reproduction_$(date +%Y%m%d)}"
RESUME_FLAG="${RESUME:+--resume}"
mkdir -p "$OUT_ROOT"
echo "RUNNING $OUT_ROOT alpha=$ALPHA gpu=$GPU" > "$OUT_ROOT/STATUS.txt"
for AXIS in deutan protan; do
  RUN="$OUT_ROOT/author_${AXIS}_alpha${ALPHA}"
  echo "TRAIN $AXIS $(date -Is)" >> "$OUT_ROOT/STATUS.txt"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py --axis "$AXIS" --alpha "$ALPHA" \
      --out "$RUN" $RESUME_FLAG > "$OUT_ROOT/train_${AXIS}.log" 2>&1
  echo "EVAL $AXIS $(date -Is)" >> "$OUT_ROOT/STATUS.txt"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" evaluate.py --axis "$AXIS" \
      --checkpoint "$RUN/final.pt" --split validation \
      --out "$OUT_ROOT/eval_${AXIS}" > "$OUT_ROOT/eval_${AXIS}.log" 2>&1
done
echo "COMPLETE $OUT_ROOT $(date -Is)" > "$OUT_ROOT/STATUS.txt"
echo "Done: $OUT_ROOT"
