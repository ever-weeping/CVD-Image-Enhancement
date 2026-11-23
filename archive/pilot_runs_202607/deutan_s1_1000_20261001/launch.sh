#!/usr/bin/env bash
set -uo pipefail
cd /home/mahai/PPWorkspace/CVD-Image-Enhancement/code
/home/mahai/miniconda3/envs/image_enhance/bin/python -u cvd_recolor_experiments.py \
 --data /home/mahai/mahai_ext/xiaolunwen_data/val2017 \
 --stress-data /home/mahai/mahai_ext/xiaolunwen_data/stress_dataset \
 --out runs/deutan_s1_1000_20261001 \
 --size 128 --train-images 1000 --eval-images 128 --stress-images-per-axis 24 \
 --batch-size 8 --steps 1000 --width 12 --seeds 7 --split-seed 2026 \
 --variants baseline ours_full --device cuda:0 --num-workers 2 \
 --axes deutan --train-severities 1.0 --eval-severities 1.0 \
 --interpolation-severities 1.0 --stress-severities 1.0 \
 --log-every 20 --sample-images 2 --save-checkpoints \
 2>&1 | tee runs/deutan_s1_1000_20261001/run.log
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > runs/deutan_s1_1000_20261001/exit_code.txt
exit "$status"
