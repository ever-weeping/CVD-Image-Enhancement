"""Evaluation metrics as defined by the paper (Chen et al., NCAA 2024).

TCC  (Eq. 10): total color contrast = mean L1 color distance over 20,000
     random global pixel pairs + mean L1 over 11x11 local windows. Computed
     in CIE Lab of the *CVD-simulated* image, i.e. the contrast a CVD observer
     perceives. n1=20,000 and window 11 follow Sect. 6.1.
SSIM (Eq. 6/7): naturalness = SSIM(recolored, original) on RGB in [0,1],
     11x11 gaussian sigma=1.5, c1=0.0001, c2=0.0009 (the paper's constants).
CD   (Eq. 11): chromatic difference with lambda=0, i.e. mean sqrt(da^2+db^2)
     in CIE Lab between recolored and original.

The paper does not fully specify the Lab backend, window boundary handling or
pair sampling; those conventions are fixed here (kornia rgb_to_lab, interior
pixels for local windows, per-image seeded CPU sampling) and recorded in
docs/AUTHOR_REPRODUCTION_zh.md. Training keeps the author code's own loss
conventions (losses/author.py); these metrics are for evaluation only.
"""
import torch
from kornia.color import rgb_to_lab

from cvd_simulation import simulate_cvd
from losses.ssim import SSIMLoss

# Display constant: max possible per-pixel CD (Eq.11, lambda=0) between two
# in-gamut sRGB colours in our exact Lab backend (kornia rgb_to_lab), i.e. the
# chroma-plane diameter of the sRGB gamut surface (25k-point 65-per-face grid,
# measured 2026-11-23; a in [-86.2, 98.2], b in [-107.9, 94.5]).
# cd_fidelity = 1 - CD / CD_SRGB_MAX is strictly in [0, 1]. Monotone linear
# map: comparisons are convention-independent; raw CD remains the reported
# Eq.11 quantity everywhere.
CD_SRGB_MAX = 252.7

_SSIM = SSIMLoss()


def _lab(rgb):
    return rgb_to_lab(rgb.clamp(0, 1))


def tcc(lab, n_global=20000, window=11, seed=0):
    """Total color contrast of one Lab image (3,H,W); higher = more contrast."""
    _, h, w = lab.shape
    m = window // 2
    interior = lab[:, m:h - m, m:w - m]
    local = []
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            if dy == dx == 0:
                continue
            shifted = lab[:, m + dy:h - m + dy, m + dx:w - m + dx]
            local.append((interior - shifted).abs().sum(0).mean())
    local_term = torch.stack(local).mean()
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, h, (n_global,), generator=g)
    x = torch.randint(0, w, (n_global,), generator=g)
    y2 = torch.randint(0, h, (n_global,), generator=g)
    x2 = torch.randint(0, w, (n_global,), generator=g)
    global_term = (lab[:, y, x] - lab[:, y2, x2]).abs().sum(0).mean()
    return global_term + local_term


@torch.no_grad()
def batch_metrics(inp, out, axis, severity, simulator, index_offset=0, seed=7):
    """Per-image paper metrics for a batch. Returns dict of name -> list[float].

    ``index_offset`` is the batch's first image index in the evaluation order;
    it seeds TCC pair sampling so results are reproducible per image position.
    """
    sim_in = simulate_cvd(inp, axis, severity, simulator)
    sim_out = simulate_cvd(out, axis, severity, simulator)
    lab_sim_in, lab_sim_out = _lab(sim_in), _lab(sim_out)
    lab_in, lab_out = _lab(inp), _lab(out)
    ssim_map = _SSIM(inp, out, as_loss=False)
    chroma = ((lab_out[:, 1:] - lab_in[:, 1:]) ** 2).sum(1).sqrt()
    rows = {k: [] for k in ('tcc_input_sim', 'tcc_output_sim', 'ssim_naturalness', 'cd_mean')}
    for b in range(inp.shape[0]):
        s = seed * 1000003 + index_offset + b
        rows['tcc_input_sim'].append(float(tcc(lab_sim_in[b], seed=s)))
        rows['tcc_output_sim'].append(float(tcc(lab_sim_out[b], seed=s)))
        rows['ssim_naturalness'].append(float(ssim_map[b].mean()))
        rows['cd_mean'].append(float(chroma[b].mean()))
    return rows
