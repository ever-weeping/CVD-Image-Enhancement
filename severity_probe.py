"""Step-0 probe: training-free severity scaling ("blend ray") of the endpoint model.

Hypothesis under test (Machado 2009 simulation is itself linear in severity s:
sim_s = (1-s)*identity + s*endpoint, see cvd_simulation.py).  If the *correction*
an observer needs scales approximately linearly in s, then a single endpoint
model trained at s=1.0 can serve severity-s observers via the scalar blend

    out_w = x + w * (model(x) - x),        w in [0, 1],

with w(s) calibrated offline -- no new training and no dataset growth.

PRE-REGISTERED PROTOCOL (written 2026-11-24, before any result was seen):
  * Data: natural validation split only (1,753 images, cvd_fixed_v3.csv),
    same order/seed as evaluate.py; test split untouched and sealed.
  * Model: linearized alpha=0.5 checkpoint (own-method default), both axes.
  * Grids: w in 0.00..1.00 step 0.05; s in {0.25, 0.50, 0.75, 1.00}.
  * Metrics: evaluate.py conventions reproduced exactly -- TCC gain (paired
    per-image pair-sampling seed = 7*1000003 + index), SSIM naturalness,
    cd_fidelity = 1 - CD/252.7 (severity-independent).  The TCC is computed by
    batched_tcc below, self-checked against metrics.tcc to <1e-6 (sanity (3)).
  * Selection rule (primary readout):
      w*(s) := argmax_w mean TCC gain(s, w) over grid w subject to
      mean cd_fidelity(w) >= 0.9400 (the claimed fidelity band); ties within
      1e-12 broken toward the SMALLER w (closer to the original).  If no grid
      point satisfies the constraint, fall back to the unconstrained argmax and
      mark the cell "band-violating".
  * Sanity checks (fatal if broken):
      (1) w=0 gives gain numerically 0 (<1e-4; bitwise exactness is impossible
          because cuBLAS batch shapes differ between input and blended rows);
      (2) w=1, s=1.0 reproduces the run's own eval per_image_metrics.csv
          (reference CSV passed via --reference-csv), max |Δ| reported;
      (3) batched_tcc matches metrics.tcc on random tensors to <1e-6.
  * Decision rule (no self-comfort):
      - Calibration HELPS at severity s iff paired Δgain = gain(w*) - gain(1)
        has a 95% CI strictly above 0 AND mean effect >= 0.3 TCC (the smallest
        gap treated as decision-relevant in this project, cf. the 0.09-0.4
        differences inside the alpha table).
      - Scaling hypothesis CONSISTENT iff w*(s) is non-decreasing in s AND
        |w*(s) - s| <= 0.10 for every s in {0.25, 0.50, 0.75}.
      - Both must hold to justify cancelling the training-based conditioning
        step; a mixed result stays mixed and the mixed-s training step is on.
      - LIMITATION stated up front: without a dedicated fixed-s fine-tuned
        control model this probe only bounds the calibration benefit from
        below; it CANNOT show the blend reaches the ceiling of a trained
        conditional model.

Outputs into --out (must be fresh): config.json, grid_summary.csv (per s,w),
per_image_grid.csv (per image x w, gains as per-s columns), REPORT_zh.md,
readout.json, COMPLETE.json.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import paths
from data import seed_everything, read_manifest, group_paths, ImagePathDataset, make_loader
from metrics import tcc, CD_SRGB_MAX
from models.author import make_model
from cvd_simulation import simulate_cvd
from losses.ssim import SSIMLoss
from kornia.color import rgb_to_lab

_SSIM = SSIMLoss()


def _lab(rgb):
    return rgb_to_lab(rgb.clamp(0, 1))


def batched_tcc(labs, seed):
    """metrics.tcc vectorized over the batch dim. labs: [n,3,H,W] -> [n].

    Pair coordinates are drawn in the same order/values as metrics.tcc for one
    seed, so batched_tcc(lab[i], seed) == tcc(lab[i], seed) for every row.
    """
    n, _, h, w = labs.shape
    m = 5  # metrics.tcc default window=11 -> radius 5
    device = labs.device
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, h, (20000,), generator=g).to(device)
    x = torch.randint(0, w, (20000,), generator=g).to(device)
    y2 = torch.randint(0, h, (20000,), generator=g).to(device)
    x2 = torch.randint(0, w, (20000,), generator=g).to(device)
    interior = labs[:, :, m:h - m, m:w - m]
    local = []
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = labs[:, :, m + dy:h - m + dy, m + dx:w - m + dx]
            local.append((interior - shifted).abs().sum(1).mean(dim=(1, 2)))
    local_term = torch.stack(local).mean(0)
    flat = labs.reshape(n, 3, -1)
    idx1 = y * w + x
    idx2 = y2 * w + x2
    global_term = (flat[:, :, idx1] - flat[:, :, idx2]).abs().sum(1).mean(1)
    return global_term + local_term


def selfcheck_batched_tcc(device):
    torch.manual_seed(20261124)
    labs = torch.rand(4, 3, 256, 256, device=device)
    batched = batched_tcc(labs, 12345)
    ref = torch.stack([tcc(labs[i], seed=12345) for i in range(4)])
    diff = float((batched - ref).abs().max())
    if diff > 1e-6:
        raise RuntimeError(f'batched_tcc deviates from metrics.tcc by {diff:g}')
    return diff


@torch.no_grad()
def sweep_axis(inps, outs, axis, severities, ws, seed=7):
    """Returns (per-image rows, grid aggregates, per-(s,w) paired gain arrays).

    inps/outs: CPU [N,3,256,256] float tensors in manifest order; image index i
    seeds TCC pair sampling exactly like evaluate.py (seed*1000003 + i).
    """
    device = torch.device('cuda')
    n = inps.shape[0]
    ws_t = torch.tensor(ws, device=device).view(-1, 1, 1, 1)
    rows = []
    gains_by_sw = {(s, w): np.empty(n) for s in severities for w in ws}
    ssim_by_w = np.empty((len(ws), n))
    fid_by_w = np.empty((len(ws), n))
    t_start = time.perf_counter()
    for i in range(n):
        # Streaming one image at a time: co-tenant GPU jobs made resident
        # tensors OOM on 2026-11-24; per-image transfer of 2x0.8MB is negligible.
        x = inps[i][None].to(device)
        d = ((outs[i] - inps[i])[None]).to(device)
        out_w = x + ws_t * d  # [W,3,H,W]
        tseed = seed * 1000003 + i
        lab_x = _lab(x)
        lab_outw = _lab(out_w)
        chroma = ((lab_outw[:, 1:] - lab_x[:, 1:]) ** 2).sum(1).sqrt().mean(dim=(1, 2))
        ssim_map = _SSIM(x.expand(len(ws), -1, -1, -1), out_w, as_loss=False)
        for k, w in enumerate(ws):
            fid_by_w[k, i] = 1.0 - float(chroma[k]) / CD_SRGB_MAX
            ssim_by_w[k, i] = float(ssim_map[k].mean())
        for s in severities:
            tin = batched_tcc(_lab(simulate_cvd(x, axis, s, 'machado')), tseed)
            tout = batched_tcc(_lab(simulate_cvd(out_w, axis, s, 'machado')), tseed)
            for k, w in enumerate(ws):
                gains_by_sw[(s, w)][i] = float(tout[k] - tin[0])
        for k, w in enumerate(ws):
            r = dict(index=i, w=w, ssim=ssim_by_w[k, i], cd_fidelity=fid_by_w[k, i])
            for s in severities:
                r[f'gain_s{s:.2f}'] = gains_by_sw[(s, w)][i]
            rows.append(r)
        if (i + 1) % 200 == 0 or i + 1 == n:
            el = time.perf_counter() - t_start
            print(f'{axis}: sweep {i + 1}/{n} images, {el:.0f}s elapsed, '
                  f'eta {el / (i + 1) * (n - i - 1):.0f}s', flush=True)
    grid = []
    for s in severities:
        for k, w in enumerate(ws):
            gains = gains_by_sw[(s, w)]
            grid.append(dict(severity=s, w=w, n=n,
                             mean_gain=float(gains.mean()),
                             se_gain=float(gains.std(ddof=1) / math.sqrt(n)),
                             mean_ssim=float(ssim_by_w[k].mean()),
                             mean_fidelity=float(fid_by_w[k].mean())))
    return rows, grid, gains_by_sw, ssim_by_w, fid_by_w, ws


def paired_ci(gains_by_sw, s, w_best, w_ref, n):
    d = gains_by_sw[(s, w_best)] - gains_by_sw[(s, w_ref)]
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    return float(m), float(m - 1.96 * se), float(m + 1.96 * se)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--axis', choices=['protan', 'deutan'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference-csv', type=Path, default=None,
                   help='per_image_metrics.csv of the same checkpoint at s=1.0 for parity check')
    p.add_argument('--severities', default='0.25,0.50,0.75,1.00')
    p.add_argument('--ws', default='0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,'
                                  '0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00')
    p.add_argument('--manifest', type=Path, default=paths.MANIFEST)
    p.add_argument('--natural-root', type=Path, default=paths.NATURAL_ROOT)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    if (a.out / 'config.json').exists():
        raise ValueError('Use a fresh output directory')
    severities = [float(v) for v in a.severities.split(',')]
    ws = [float(v) for v in a.ws.split(',')]
    fidelity_floor = 0.9400

    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.8)
    seed_everything(a.seed)
    device = torch.device('cuda')
    a.out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if ck['config']['axis'] != a.axis:
        raise ValueError('checkpoint axis mismatch')
    if ck['config']['manifest_sha256'] != sha:
        raise ValueError('checkpoint was trained on a different manifest')
    model = make_model().to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    dev_err = selfcheck_batched_tcc(device)
    (a.out / 'config.json').write_text(json.dumps(
        dict(axis=a.axis, checkpoint=str(a.checkpoint), severities=severities, ws=ws,
             fidelity_floor=fidelity_floor, seed=a.seed,
             tcc_batched_vs_ref_deviation=dev_err, manifest_sha256=sha,
             protocol='pre-registered in module docstring'), indent=2) + '\n')

    # ---- model outputs on the natural validation set, evaluate.py order ----
    rows_m = read_manifest(a.manifest, a.natural_root, paths.SYNTHETIC_ROOT)
    group = group_paths(rows_m, 'natural', 'validation', None, a.limit, 2026)
    ds = ImagePathDataset(a.natural_root, 256, group)
    loader = make_loader(ds, range(len(ds)), a.batch_size, device, a.workers,
                         shuffle=False, seed=a.seed)
    inps, outs = [], []
    with torch.no_grad():
        for inp, _ in loader:
            inp = inp.to(device)
            out = (model(inp * 2 - 1) + 1) / 2
            inps.append(inp.cpu())
            outs.append(out.cpu())
    inps = torch.cat(inps)
    outs = torch.cat(outs)
    print(f'{a.axis}: forward done on {inps.shape[0]} images', flush=True)

    rows, grid, gains_by_sw, ssim_by_w, fid_by_w, ws = sweep_axis(
        inps, outs, a.axis, severities, ws, seed=a.seed)
    n = inps.shape[0]

    # ---- sanity (1): w=0 gain numerically zero ----
    # Not bitwise zero: input TCC runs on a batch-1 tensor while blended rows
    # run in the batch-|ws| tensor; cuBLAS picks different accumulation orders
    # per shape, giving ~1e-6 wobbles on the ~70-scale TCC.  Logic errors would
    # be orders of magnitude larger, so 1e-4 is the tolerance.
    for s in severities:
        g0 = gains_by_sw[(s, 0.0)]
        if float(np.abs(g0).max()) > 1e-4:
            raise RuntimeError(f'w=0 sanity broken at s={s}: max|gain|={np.abs(g0).max():g}')

    # ---- sanity (2): w=1, s=1.0 parity vs the run's own eval CSV ----
    parity = None
    if a.reference_csv:
        if not a.reference_csv.exists():
            raise FileNotFoundError(a.reference_csv)
        with a.reference_csv.open() as h:
            ref_rows = [r for r in csv.DictReader(h) if r['dataset'] == 'natural_validation']
        if len(ref_rows) != n:
            raise RuntimeError(f'row count mismatch: ref {len(ref_rows)} vs probe {n}')
        worst = 0.0
        for k, w in enumerate(ws):
            if abs(w - 1.0) < 1e-9:
                for i in range(n):
                    worst = max(worst,
                                abs(gains_by_sw[(1.0, w)][i] - float(ref_rows[i]['tcc_gain'])),
                                abs(fid_by_w[k, i] - float(ref_rows[i]['cd_fidelity'])),
                                abs(ssim_by_w[k, i] - float(ref_rows[i]['ssim_naturalness'])))
        parity = worst
        if worst > 5e-4:
            raise RuntimeError(f'PARITY BROKEN at s=1.0,w=1: max |delta|={worst:g}')

    # ---- CSVs ----
    with (a.out / 'per_image_grid.csv').open('w', newline='') as h:
        wr = csv.writer(h)
        wr.writerow(['index', 'w', 'ssim', 'cd_fidelity'] + [f'gain_s{s:.2f}' for s in severities])
        for r in rows:
            wr.writerow([r['index'], f"{r['w']:.2f}", f"{r['ssim']:.6f}",
                         f"{r['cd_fidelity']:.6f}"] + [f"{r[f'gain_s{s:.2f}']:.6f}" for s in severities])
    with (a.out / 'grid_summary.csv').open('w', newline='') as h:
        wr = csv.DictWriter(h, fieldnames=list(grid[0].keys()))
        wr.writeheader()
        wr.writerows(grid)

    # ---- pre-registered readout ----
    gmap = {(g['severity'], g['w']): g for g in grid}
    readout = []
    for s in severities:
        feas = [w for w in ws if gmap[(s, w)]['mean_fidelity'] >= fidelity_floor]
        band_ok = len(feas) > 0
        pool = feas if feas else ws
        best = max(gmap[(s, w)]['mean_gain'] for w in pool)
        w_star = min(w for w in pool if abs(gmap[(s, w)]['mean_gain'] - best) <= 1e-12)
        m, lo, hi = paired_ci(gains_by_sw, s, w_star, 1.0, n)
        readout.append(dict(severity=s, w_star=w_star, band_ok=band_ok,
                            gain_star=gmap[(s, w_star)]['mean_gain'],
                            gain_w1=gmap[(s, 1.0)]['mean_gain'],
                            delta=m, ci_lo=lo, ci_hi=hi,
                            fid_star=gmap[(s, w_star)]['mean_fidelity'],
                            ssim_star=gmap[(s, w_star)]['mean_ssim'],
                            helps=bool(lo > 0 and m >= 0.3),
                            consistent=bool(abs(w_star - s) <= 0.10)))
    monotone = all(readout[i]['w_star'] <= readout[i + 1]['w_star'] + 1e-9
                   for i in range(len(readout) - 1))

    hdr = [w for w in ws if round(w * 100) % 10 == 0]
    lines = ['# 第 0 步探针：端点模型的免训练严重度缩放（blend ray）', '',
             f'- 轴 **{a.axis}**，检查点 `{a.checkpoint.name}`，自然验证图 {n} 张，'
             f'清单 sha `{sha[:12]}…`，评估 seed {a.seed}。',
             f'- w 网格 {len(ws)} 档 × s ∈ {severities}；保真约束带 ≥{fidelity_floor}（预注册）。',
             f'- 自检：batched-TCC vs metrics.tcc 偏差 {dev_err:.2e}；w=0 全档增益数值为 0（<1e-4）通过；'
             + (f's=1.0∧w=1.0 与既有 eval 逐图最大偏差 {parity:.2e}（阈值 5e-4，通过）。'
                if parity is not None else '未提供 reference CSV，跳过对表。'), '',
             '## 预注册判读', '',
             '| s | w* | 增益(w*) | 增益(w=1) | Δ(w*−1) [95%CI] | 保真(w*) | 标定有增益? | w*≈s? |',
             '|---|---|---|---|---|---|---|---|']
    for r in readout:
        lines.append(f"|{r['severity']:.2f}|{r['w_star']:.2f}|{r['gain_star']:.3f}|"
                     f"{r['gain_w1']:.3f}|{r['delta']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]|"
                     f"{r['fid_star']:.4f}|{'是' if r['helps'] else '否'}|"
                     f"{'是' if r['consistent'] else '否'}|")
    lines += ['', f"- w*(s) 单调不减：{'满足' if monotone else '不满足'}。",
              '- 判读规则（运行前写死）：helps = Δ 的 95%CI 下界>0 且均值≥0.3；'
              'consistent = |w*−s|≤0.10 且单调。两者同时成立才可认定免训练缩放足够。',
              '- 局限（预注册）：无"固定 s 专训"对照，本探针只能给免训练校准的下界证据。', '',
              '## 全网格：每格 平均增益 | 保真（·= 掉出约束带）', '',
              '| s\\w |' + '|'.join(f'{w:.1f}' for w in hdr) + '|']
    for s in severities:
        cells = []
        for w in hdr:
            g = gmap[(s, w)]
            flag = '' if g['mean_fidelity'] >= fidelity_floor else '·'
            cells.append(f"{g['mean_gain']:.2f}{flag}/{g['mean_fidelity']:.3f}")
        lines.append(f"|**{s:.2f}**|" + '|'.join(cells) + '|')
    lines += ['', '逐图数据 per_image_grid.csv；全网格（含 0.05 步长）grid_summary.csv。']
    (a.out / 'REPORT_zh.md').write_text('\n'.join(lines) + '\n')
    (a.out / 'readout.json').write_text(json.dumps(
        dict(axis=a.axis, readout=readout, monotone=monotone, parity_vs_eval=parity,
             tcc_dev=dev_err), indent=2) + '\n')
    (a.out / 'COMPLETE.json').write_text(json.dumps({'rows': len(rows), 'images': n}) + '\n')
    print(f'{a.axis}: DONE', flush=True)


if __name__ == '__main__':
    main()
