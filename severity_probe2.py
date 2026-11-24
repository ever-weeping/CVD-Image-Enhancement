"""Step-1 probe (a.k.a. probe2): target-alignment axis for the severity blend ray.

Motivation (2026-11-24, user hypothesis): the endpoint model's correction,
optionally scaled by w, might be "right-sized" for lower severities.  The
existing TCC/SSIM/cd_fidelity trio cannot adjudicate this because TCC rewards
any separation increase and never penalizes over-enhancement (survey
Zhu21 p.3002: "In a relative metric, *not over-enhanced* contrast, but JND or
contrast identical to that in the reference image is better").

This probe adds a DEFICIT-CLOSURE / OVERSHOOT pair metric with the survey's
appear-match target: for every sampled pixel pair, the s-severity observer's
goal is to see the separation a normal observer sees in the ORIGINAL image.

    d_in  = L1 separation of the pair in Lab(sim_s(x))          (user's view now)
    d_out = L1 separation of the pair in Lab(sim_s(out_w))      (user's view after)
    d_ref = L1 separation of the pair in Lab(x)                 (normal view; target)

Per-image metrics (20,000 global pairs, coords identical to metrics.tcc
sampling: same generator order, seed = 7*1000003 + image_index -- so all
(s,w) cells share one pair set per image and are strictly paired):

    MATCH(s,w)     = -mean_pairs |d_out - d_ref|                 (primary score)
    CLOSURE(s,w)   = mean over deficit pairs (d_ref - d_in > tau)
                     of clip((d_out - d_in)/(d_ref - d_in), 0, 1) (benefit, capped)
    OVERSHOOT(s,w) = mean over visible pairs (d_ref - d_in <= tau)
                     of max(d_out - d_ref, 0)                    (harm on colors
                     the user already sees fine)

Metric provenance (declared): d_ref target = appear-match goal statement
standard in the field, but as a single named metric this construction is OUR
design; anchors: scheme-1 simulation-domain comparison (Zhu21 p.3002),
contrast-identical-to-reference principle (same line), tau adapted from
CCPR's DeltaE76 tau=6 now used on Lab L1 sums (disclosed adaptation,
sensitivity at tau in {3,6,10} pre-registered as robustness only).

PRE-REGISTERED READOUT (written BEFORE any number was computed):
  Grids: same w (21 values) x s {0.25,0.50,0.75,1.00}; same 1,753 natural
  validation images, same model checkpoints (linearized alpha=0.5), seed 7;
  test split sealed.
  Primary question ("severity matching"): does the alignment-optimal
  operating point move with severity?  Declare SUPPORTED iff ALL of:
    (1) w_match(s) non-decreasing in s across {0.25,0.5,0.75,1.0};
    (2) w_match(0.25) <= 0.70 (interior: mild observers want a smaller push
        than the endpoint correction);
    (3) paired 95% CI of MATCH(0.25, w_match(0.25)) - MATCH(0.25, 1.0) > 0.
  Secondary: OVERSHOOT(0.25, w=1) significantly > 0 (CI over images) --
  the endpoint model measurably over-pushes colors the mild observer already
  sees; and OVERSHOOT decreasing in w (mechanism check).
  Degeneracy audit (lesson from probe 1, done EXPLICITLY this time): report
  whether MATCH(s,w) is monotone in w at each s.  If MATCH is monotone
  increasing at s=0.25 (never overshoots d_ref within the grid), then an
  interior optimum is mathematically impossible and readout (2) failing is
  EVIDENCE that endpoint correction UNDER-shoots even mild observers --
  the opposite conclusion from the over-enhancement story.  Both directions
  are legitimate results; do not spin.
  Limits: single-seed; new metric unvalidated against human judgments (no
  subjective data available; flagged as design-metric, to be labeled as such
  in the paper; Machado s is a convention, not a clinical measurement --
  claim stays compensation-side only).
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
from metrics import CD_SRGB_MAX
from models.author import make_model
from cvd_simulation import simulate_cvd
from severity_probe import _lab
from kornia.color import rgb_to_lab

TAUS = [3.0, 6.0, 10.0]          # tau=6 primary; others sensitivity
TAU_PRIMARY = 6.0


def pair_coords(h, w, seed):
    """Exactly the coordinate sequence of metrics.tcc / batched_tcc."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, h, (20000,), generator=g)
    x = torch.randint(0, w, (20000,), generator=g)
    y2 = torch.randint(0, h, (20000,), generator=g)
    x2 = torch.randint(0, w, (20000,), generator=g)
    return y, x, y2, x2


def pair_l1(lab, y, x, y2, x2):
    """[3,H,W] Lab -> per-pair L1 distances [20000] (sum over channels)."""
    a = lab[:, y, x]
    b = lab[:, y2, x2]
    return (a - b).abs().sum(0)


@torch.no_grad()
def sweep_match(inps, outs, axis, severities, ws, seed=7):
    device = torch.device('cuda')
    n, _, H, W = inps.shape
    ws_t = torch.tensor(ws, device=device).view(-1, 1, 1, 1)
    # accumulators
    match = {(s, w): np.empty(n) for s in severities for w in ws}
    close = {(t, s, w): np.zeros(n, dtype=np.float32) for t in TAUS for s in severities for w in ws}
    close_cnt = {(t, s): np.zeros(n) for t in TAUS for s in severities}
    over = {(t, s, w): np.zeros(n, dtype=np.float32) for t in TAUS for s in severities for w in ws}
    over_cnt = {(t, s): np.zeros(n) for t in TAUS for s in severities}
    fid = np.empty((len(ws), n))
    t0 = time.perf_counter()
    for i in range(n):
        x = inps[i][None].to(device)
        d = ((outs[i] - inps[i])[None]).to(device)
        out_w = x + ws_t * d
        tseed = seed * 1000003 + i
        y, xs, y2, x2 = [c.to(device) for c in pair_coords(H, W, tseed)]
        lab_x = _lab(x)[0]
        d_ref = pair_l1(lab_x, y, xs, y2, x2)
        lab_in, lab_outw = _lab(x), _lab(out_w)
        chroma = ((lab_outw[:, 1:] - lab_in[:, 1:]) ** 2).sum(1).sqrt()
        for k, w in enumerate(ws):
            fid[k, i] = 1.0 - float(chroma[k].mean()) / CD_SRGB_MAX
        idx1 = y * W + xs
        idx2 = y2 * W + x2
        for s in severities:
            d_in = pair_l1(_lab(simulate_cvd(x, axis, s, 'machado'))[0], y, xs, y2, x2).float()
            lab_out_sim = _lab(simulate_cvd(out_w, axis, s, 'machado'))  # [21,3,H,W]
            flat = lab_out_sim.reshape(len(ws), 3, -1)
            d_out = (flat[:, :, idx1] - flat[:, :, idx2]).abs().sum(1)  # [21,20000]
            gap = (d_ref - d_in)  # >0 == deficit pair (user sees less than normal)
            denom = gap.clamp_min(1e-6)
            for t in TAUS:
                close_cnt[(t, s)][i] = float((gap > t).float().mean())
                over_cnt[(t, s)][i] = float((gap <= t).float().mean())
            # closure / overshoot per tau, vectorized over w rows
            for t in TAUS:
                dm = (gap > t)[None]
                clos = ((d_out - d_in[None]) / denom[None]).clamp(0.0, 1.0)
                push = (d_out - d_ref[None]).clamp_min(0.0)
                vm = ~dm
                per_w_close = (clos * dm).sum(1) / dm.sum(1).clamp_min(1)
                per_w_over = (push * vm).sum(1) / vm.sum(1).clamp_min(1)
                for k, w in enumerate(ws):
                    close[(t, s, w)][i] = float(per_w_close[k])
                    over[(t, s, w)][i] = float(per_w_over[k])
            match_vec = -(d_out - d_ref[None]).abs().mean(1)
            for k, w in enumerate(ws):
                match[(s, w)][i] = float(match_vec[k])
        if (i + 1) % 200 == 0 or i + 1 == n:
            el = time.perf_counter() - t0
            print(f'{axis}: {i + 1}/{n} images, {el:.0f}s, eta {el / (i + 1) * (n - i - 1):.0f}s', flush=True)
    return match, close, close_cnt, over, over_cnt, fid


def mean_ci(v):
    v = np.asarray(v)
    m = v.mean()
    se = v.std(ddof=1) / math.sqrt(len(v))
    return float(m), float(m - 1.96 * se), float(m + 1.96 * se)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--axis', choices=['protan', 'deutan'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--severities', default='0.25,0.50,0.75,1.00')
    p.add_argument('--ws', default='0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,'
                                  '0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00')
    p.add_argument('--manifest', type=Path, default=paths.MANIFEST)
    p.add_argument('--natural-root', type=Path, default=paths.NATURAL_ROOT)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    if (a.out / 'config.json').exists():
        raise ValueError('Use a fresh output directory')
    severities = [float(v) for v in a.severities.split(',')]
    ws = [float(v) for v in a.ws.split(',')]
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.6)
    seed_everything(a.seed)
    device = torch.device('cuda')
    a.out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if ck['config']['axis'] != a.axis or ck['config']['manifest_sha256'] != sha:
        raise ValueError('checkpoint/manifest mismatch')
    model = make_model().to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    (a.out / 'config.json').write_text(json.dumps(
        dict(axis=a.axis, checkpoint=str(a.checkpoint), severities=severities, ws=ws,
             taus=TAUS, tau_primary=TAU_PRIMARY, seed=a.seed, manifest_sha256=sha,
             protocol='pre-registered in module docstring'), indent=2) + '\n')

    rows_m = read_manifest(a.manifest, a.natural_root, paths.SYNTHETIC_ROOT)
    group = group_paths(rows_m, 'natural', 'validation', None, a.limit, 2026)
    ds = ImagePathDataset(a.natural_root, 256, group)
    loader = make_loader(ds, range(len(ds)), a.batch_size, device, a.workers, shuffle=False, seed=a.seed)
    inps, outs = [], []
    with torch.no_grad():
        for inp, _ in loader:
            inp = inp.to(device)
            out = (model(inp * 2 - 1) + 1) / 2
            inps.append(inp.cpu())
            outs.append(out.cpu())
    inps = torch.cat(inps)
    outs = torch.cat(outs)
    print(f'{a.axis}: forward done', flush=True)

    match, close, close_cnt, over, over_cnt, fid = sweep_match(
        inps, outs, a.axis, severities, ws, seed=a.seed)
    n = inps.shape[0]

    # sanity: w=0 => out_w=x => d_out==d_in; MATCH(0) = -mean|d_in-d_ref|; OVERSHOOT(0) ~ 0 on visible pairs
    for s in severities:
        m0, _, _ = mean_ci(match[(s, 0.0)])
        print(f'sanity s={s}: MATCH(w=0) mean {m0:.4f}', flush=True)

    # ---- readout ----
    readout = []
    for s in severities:
        w_match = max(ws, key=lambda w: match[(s, w)].mean())
        ties = [w for w in ws if abs(match[(s, w)].mean() - match[(s, w_match)].mean()) < 1e-12]
        w_match = min(ties)
        d = match[(s, w_match)] - match[(s, 1.0)]
        dm, dlo, dhi = mean_ci(d) if w_match != 1.0 else (0.0, 0.0, 0.0)
        mono_inc = all(match[(s, ws[i])].mean() <= match[(s, ws[i + 1])].mean() + 1e-9 for i in range(len(ws) - 1))
        ov1, ovlo, ovhi = mean_ci(over[(TAU_PRIMARY, s, 1.0)])
        readout.append(dict(severity=s, w_match=w_match,
                            match_at_match=float(match[(s, w_match)].mean()),
                            match_at_w1=float(match[(s, 1.0)].mean()),
                            delta_vs_w1=dm, ci=[dlo, dhi],
                            match_monotone_increasing=mono_inc,
                            overshoot_w1=ov1, overshoot_ci=[ovlo, ovhi],
                            closure_w1=float(close[(TAU_PRIMARY, s, 1.0)].mean()),
                            fid_w1=float(fid[ws.index(1.0)].mean())))
    crit1 = all(readout[i]['w_match'] <= readout[i + 1]['w_match'] + 1e-9 for i in range(len(readout) - 1))
    crit2 = readout[0]['w_match'] <= 0.70
    crit3 = readout[0]['ci'][0] > 0
    supported = crit1 and crit2 and crit3

    hdr = [w for w in ws if round(w * 100) % 10 == 0]
    with (a.out / 'grid_match.csv').open('w', newline='') as h:
        wr = csv.writer(h)
        wr.writerow(['severity', 'w', 'match_mean', 'match_ci_lo', 'match_ci_hi',
                     'closure', 'overshoot', 'fid'])
        for s in severities:
            for w in ws:
                m, lo, hi = mean_ci(match[(s, w)])
                o, _, _ = mean_ci(over[(TAU_PRIMARY, s, w)])
                wr.writerow([s, f'{w:.2f}', f'{m:.6f}', f'{lo:.6f}', f'{hi:.6f}',
                             f"{close[(TAU_PRIMARY, s, w)].mean():.6f}",
                             f'{o:.6f}', f'{fid[ws.index(w)].mean():.6f}'])

    with (a.out / 'tau_sensitivity.csv').open('w', newline='') as h:
        wr = csv.writer(h)
        wr.writerow(['tau', 'severity', 'w', 'closure_mean', 'overshoot_mean', 'deficit_frac'])
        for t in TAUS:
            for s in severities:
                for w in ws:
                    wr.writerow([t, s, f'{w:.2f}',
                                 f"{close[(t, s, w)].mean():.6f}",
                                 f"{over[(t, s, w)].mean():.6f}",
                                 f"{close_cnt[(t, s)].mean():.6f}"])

    lines = ['# 探针2：blend ray 的目标对齐轴（deficit-closure / overshoot）', '',
             f'- 轴 **{a.axis}**，α0.5 端点模型，自然验证图 {n} 张，seed {a.seed}，τ={TAU_PRIMARY}（L1，CCPR 惯例适配，已声明）。',
             '- MATCH = −平均|仿真_s(输出) 像素对分离度 − 正常视图分离度|；CLOSURE=缺陷对闭合率(封顶1)；OVERSHOOT=可见对超调。', '',
             '## 预注册判读', '',
             '| s | w_match | MATCH(w_match) | MATCH(w=1) | Δ 及95%CI | MATCH随w单调升? | OVERSHOOT(w=1) 及CI | CLOSURE(w=1) |',
             '|---|---|---|---|---|---|---|---|']
    for r in readout:
        lines.append(f"|{r['severity']:.2f}|{r['w_match']:.2f}|{r['match_at_match']:.4f}|"
                     f"{r['match_at_w1']:.4f}|{r['delta_vs_w1']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]|"
                     f"{'是' if r['match_monotone_increasing'] else '否'}|"
                     f"{r['overshoot_w1']:.3f} [{r['overshoot_ci'][0]:.3f}, {r['overshoot_ci'][1]:.3f}]|"
                     f"{r['closure_w1']:.3f}|")
    lines += ['', f"三条判据（预注册）：单调不减={crit1}；w_match(0.25)≤0.70={crit2}；Δ CI 下界>0={crit3}",
              f"**严重度匹配假说：{'支持' if supported else '不支持'}**",
              '- 退化性说明：若 MATCH 对 w 单调升（尤其 s=0.25），则内部最优在数学上不可能，此时结论是"端点修正连轻度观察者都仍欠力度"（反向发现），不粉饰。',
              '- 局限：新指标未经人类判断校验（声明为设计型指标）；s 是建模约定非临床量；单 seed。', '',
              '- 缺陷对占比（d_ref−d_in>τ，τ=6）：'
              + '，'.join(f"s={s:.2f}: {close_cnt[(TAU_PRIMARY, s)].mean() * 100:.1f}%" for s in severities), '',
              '## MATCH 网格（越大越好；行=s，列=w 0.1 步长）', '',
              '| s\\w |' + '|'.join(f'{w:.1f}' for w in hdr) + '|']
    for s in severities:
        lines.append(f"|**{s:.2f}**|" + '|'.join(f"{match[(s, w)].mean():.3f}" for w in hdr) + '|')
    lines += ['', '## OVERSHOOT 网格（越小越好）', '',
              '| s\\w |' + '|'.join(f'{w:.1f}' for w in hdr) + '|']
    for s in severities:
        lines.append(f"|**{s:.2f}**|" + '|'.join(f"{over[(TAU_PRIMARY, s, w)].mean():.3f}" for w in hdr) + '|')
    (a.out / 'REPORT_zh.md').write_text('\n'.join(lines) + '\n')
    (a.out / 'readout.json').write_text(json.dumps(
        dict(axis=a.axis, readout=readout, criteria=dict(c1=crit1, c2=crit2, c3=crit3),
             supported=supported), indent=2) + '\n')
    (a.out / 'COMPLETE.json').write_text(json.dumps({'images': n}) + '\n')
    print(f'{a.axis}: DONE supported={supported}', flush=True)


if __name__ == '__main__':
    main()
