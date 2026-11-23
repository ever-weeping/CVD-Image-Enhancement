# CVD-aware conditional recoloring experiments

This directory is an independent Git repository. Raw images (`val2017/`), run
artifacts (`runs/`) and checkpoints are intentionally ignored. A run is
reproduced from the tracked code plus its own `config.json` and
`split_manifest.csv`; checkpoints are disabled unless explicitly requested.

## Terminology and scope

The experiment interface uses `axis + simulator_severity`:

- `protan`, `0 < s < 1`: simulated protanomaly condition;
- `deutan`, `0 < s < 1`: simulated deuteranomaly condition;
- `protan`, `s = 1`: simulated protanopia endpoint;
- `deutan`, `s = 1`: simulated deuteranopia endpoint.

`s` is a simulator-control parameter. It is not a clinical diagnosis,
anomaloscope measurement or clinically validated severity grade. Machado
(2009) is used for training and primary evaluation; Brettel (1997) is used only
as a second-model sensitivity check at the dichromat endpoint. The two models
belong to a related color-vision modelling lineage, so this is not an
independent real-observer validation.

## Tracked modules

- `cvd_simulation.py`: differentiable Machado, Brettel, Viénot and smoke-test
  simulators, condition labels and condition vectors;
- `cvd_metrics.py`: fixed confusion masks, train losses and per-image metrics;
- `cvd_models.py`: parameter-matched conditional baseline and three-branch
  model/ablations;
- `cvd_recolor_experiments.py`: deterministic splits, multi-seed training,
  COCO evaluation, optional existing synthetic stress test, paired bootstrap
  confidence intervals and Wilcoxon/Holm tests;
- `tests/`: simulation, metric, model and CPU–MPS numerical regression tests.

## Evaluation outcomes

The script avoids the ambiguous legacy names TCC/CD/CVD_D. It reports:

- `cvd_contrast_gain` and `cvd_contrast_ratio` within an input-fixed confusion
  mask (higher is better);
- `delta_e00_mean` and `delta_e00_nonconf` as CIEDE2000 fidelity costs (lower);
- `ssim_luma` is SSIM on CIELAB L*/100 for structure only (higher; not a
  color-naturalness claim);
- `gamut_preclip_rate` before output clipping (lower);
- runtime and trainable parameter count.

These are computational proxy outcomes. They do not establish benefit for real
people with CVD without observer/task validation.

## Tests

This Mac uses the Anaconda Python that contains PyTorch and colour-science:

```bash
/opt/anaconda3/bin/python3 -m unittest discover -s tests -v
```

Use the Python executable for the target environment on other machines.

## Reproducible local pilot

```bash
/opt/anaconda3/bin/python3 cvd_recolor_experiments.py \
  --data val2017 \
  --stress-data ../dataset_gen/CVDdataset \
  --out runs/pilot_v2_20260720_fixed \
  --size 64 \
  --train-images 128 \
  --eval-images 24 \
  --stress-images-per-axis 12 \
  --batch-size 4 \
  --steps 40 \
  --width 12 \
  --seeds 7 19 \
  --train-simulator machado \
  --robustness-simulator brettel \
  --device mps \
  --sample-images 1 \
  --log-every 10 \
  --bootstrap-samples 2000
```

The default variant list contains the parameter-matched baseline, the full
model, branch removals and the no-decoupling-loss ablation. The script also
evaluates identity and a clearly labelled engineering error-compensation
baseline. The latter is not claimed as an exact reproduction of a published
algorithm.

## Run outputs

- `config.json`: arguments, resolved device, versions, starting Git commit and
  terminology note;
- `split_manifest.csv`: disjoint train/eval file list;
- `per_image_metrics.csv`: source table for all later aggregation;
- `summary_by_seed.csv`, `summary_conditions.csv`, `summary_overall.csv`;
- `paired_statistics.csv`: seed-averaged per-image paired comparisons,
  bootstrap 95% confidence intervals and Holm-adjusted p values;
- `train_history.csv` and compact visual panels under `samples/`.

The synthetic CVDdataset already present in the workspace is used only as a
stress set. It is not a substitute for independent natural images or a human
observer study, and this script never generates or downloads new datasets.

## Frozen pilot evidence

The paper uses `runs/pilot_v2_20260720_fixed` as its primary frozen run (24
evaluation images, two seeds). At the one preset operating point, the full
model has descriptive mean `cvd_contrast_gain=0.124042` and
`delta_e00_mean=5.834555`; the parameter-matched baseline has `0.111595` and
`6.012034`, respectively. The contrast-gain comparisons do not survive Holm
correction, while all 14 per-condition Delta-E differences have the same
direction. These results are pilot evidence, not a general superiority claim.

Two supplemental runs expose important boundaries:

- `runs/condition_probe_20260720`: swapping the FiLM axis or fixing `s=0.5`
  changes the main metrics only around `1e-6` to `1e-5`, so the 40-step model
  has not learned an identifiable explicit condition mechanism.
- `runs/no_mask_ablation_20260720`: replacing the complete mask input/routing/
  fusion/output-gating chain with ones increases mean
  contrast gain to `0.239056` but also increases mean Delta-E to `11.865320`
  and lowers SSIM-L* to `0.861705`. This is a composite intervention that
  changes residual magnitude; it identifies a system-level trade-off, not the
  causal effect of one routing location.
