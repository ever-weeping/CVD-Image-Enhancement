"""Data and run locations for the author-method reproduction.

Default manifest is cvd_fixed_v3.csv (protocol decision 2026-11-22, user):
all synthetic rows are train-only, so validation/test contain natural images
only -- matching the paper, where the artificial confusing-colour images
(Sect. 4.2) merely aid training and quantitative evaluation uses real photos
(Sect. 6.1). v3 was derived from v2 by moving its 27,636 synthetic
validation/test rows to train; natural splits are unchanged
(31,453/1,753/1,747), total rows unchanged (311,090).
v2 remains frozen: the completed reproduction_v2_20261122 run references it
via its own fixed_manifest.csv snapshot. v1 remains frozen for historical runs.

Roots verified against cvd_fixed_v2.csv on 2026-11-22: natural rows carry
"train|validation|test/<sha256>.jpg" prefixes (places365_cvd_localccpr, the
ICCP12 Eq.15 local-neighbour CCPR filter, 34,953 images; see the dataset
folder's FILTER_METHOD_zh.md), synthetic rows carry
"Color_cvd_{P,D}_experiment_100000/<n>.png" prefixes (relative to DATA_ROOT,
inherited unchanged from v1).
``data.read_manifest`` still resolves and validates every row, so any future
drift fails loudly. Override per run via CLI flags.
"""
from pathlib import Path

# All reproduction data lives on the external disk (home partition is full).
DATA_ROOT = Path.home() / "mahai_ext" / "xiaolunwen_data"

# Fixed train/validation/test manifest with per-image SHA256 (no re-splitting).
# v3: synthetic images train-only; validation/test = natural images only.
MANIFEST = DATA_ROOT / "cvd_fixed_v3.csv"

# 34,953 natural images, local-neighbour CCPR filter, explicit split subdirs.
NATURAL_ROOT = DATA_ROOT / "places365_cvd_localccpr"

# Both axes' synthetic images live under DATA_ROOT with per-axis prefixes.
SYNTHETIC_ROOT = DATA_ROOT

# Checkpoints, logs and evaluation outputs go to the external disk, never /home.
RUN_ROOT = Path.home() / "mahai_ext" / "cvd_runs"
