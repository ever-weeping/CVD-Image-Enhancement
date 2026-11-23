"""Hassan19 — "Flexible color contrast enhancement method for red-green
deficiency" (Hassan, MDSSP 30:1975-1989, 2019; DOI 10.1007/s11045-019-00638-7).

Closed-form, per-pixel, CPU colourimetric recoloring (no training, no
iteration). Self-implementation (no official code exists; see literature
note `hassan-flexible-contrast`, tier-1, formulas verified against the PDF).

Pipeline (paper Sect. 2-3, Eq. 1-14; all steps per-pixel and vectorised):
  1. Perceived image C_s = Brettel97 / Vienot99 dichromat simulation of the
     input at severity 1.0 (endpoint; axis = protan/deutan).
  2. RGB -> XYZ with Eq. 1 (Travis 1991 matrix, PDF p.1977), for original and
     perceived image; each set normalized to Y = 100 by its OWN Y (paper
     p.1977 literal: "both sets of tristimulus values are normalized
     (Y = 100)"). Consequence to note: our brettel backend is not exactly
     Y-preserving, so Y_s != Y_o and the E_x/E_z magnitudes depend on this
     per-image normalization choice (ledger-adjacent; literal reading kept).
  3. Error E_x=|X_o-X_s|, E_z=|Z_o-Z_s| (Eq. 2-3); elevation angles
     theta_o, theta_s (Eq. 4-5) and rotation parameter phi=|theta_o-theta_s|
     (Eq. 6); rotate the error vector in the XZ plane (Eq. 7-8).
  4. Distribute back on the normalized ORIGINAL tristimulus values:
     X_n = X_o + Ehat_x        (Eq. 12)
     Y_n = Y_o                 (Eq. 13, luminance preserved)
     Z_n = Z_o + beta*(Ehat_x + Ehat_z)   (Eq. 14; note Ehat_x enters BOTH X
     and Z; beta = 1 recovers Hassan17 Eq. 11, beta > 1 buys contrast with a
     blue shift and loses the hue premise).
  5. Un-normalize (multiply X_n, Z_n by Y_o/100; the Y channel of the output
     XYZ is a direct copy of Y_o, so luminance invariance is exact before
     clipping), apply the inverse of Eq. 1, convert back to sRGB and clip.

Out-of-gamut handling (paper silent -> ledger 4.2 "clip to [0,1]"): we always
clip. selftest() prints the measured clip fraction (channel values outside
[0,1] before clipping) on the test images: ~17-29% on the saturated
artificial colour bars, rising monotonically with beta (the blue
exaggeration is what pushes Z out of gamut, matching chen's "blue
saturation" paraphrase). Natural-photo clip rates were NOT measured locally
(no validation images on this machine) — record them during the server smoke
run if needed. Stats are not returned by `recolor` (harness contract is a
plain tensor).

Numeric conventions fixed by our implementation (each is a 欠定 new entry,
also mirrored in PARAM_DOC):
  * All math runs in float64 internally; the output is cast back to the
    input's dtype/device (contract).
  * theta = atan2(Z_n, X_n) — identical to the paper's tan^-1(Z/X) for the
    X>0 half-plane that in-gamut colours occupy; atan2 keeps black pixels
    (X=Z=0) finite (theta=0).
  * Y=100 normalization divides each set by its own max(Y, 1e-6) (guard only);
    the un-normalization multiplies by the TRUE Y_o, so black pixels map to
    black exactly and no epsilon leaks into mid-tones.
  * lightdomain='gamma' applies Eq. 1 to gamma-encoded values verbatim
    (literal transcription of the paper, which never states the light
    domain); the Brettel/Vienot backend itself always linearizes internally
    and returns sRGB-encoded values — that perceived image is what Eq. 1
    then consumes, so even in 'gamma' mode the simulator is unchanged.

Protocol (docs/BASELINES_zh.md Sec. 2/4.2): severity fixed 1.0; beta default
4.0, sweep with `--param beta=X` over {1,2,3,4}; default lightdomain
'linearized' (chromaticity matrices on gamma values are physically
inconsistent; matches our own linearization convention); default matrix
'paper' (Eq. 1 verbatim, Travis 1991); default simulator 'brettel'.
"""
import numpy as np
import torch

from cvd_simulation import (linear_to_srgb, simulate_brettel_cvd,
                            simulate_vienot_cvd, srgb_to_linear)

# Eq. 1 (Travis 1991, PDF p.1977, transcription verified). White maps to
# Y ~ 100, i.e. the matrix is already in the "illuminant = 100" convention.
_PAPER_XYZ = [
    [40.9568, 35.5041, 17.9167],
    [21.3389, 70.6743, 7.98680],
    [1.86297, 11.4620, 91.2367],
]
# Sensitivity variant (台账 4.2 row 2): standard sRGB->XYZ/D65 matrix
# (IEC 61966-2-1 / Lindbloom values; Y(white)=1 — harmless, everything is
# normalized to Y=100 anyway).
_SRGB_D65_XYZ = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
]

_MATRICES = {'paper': _PAPER_XYZ, 'srgb_d65': _SRGB_D65_XYZ}
_SIMULATORS = {'brettel': simulate_brettel_cvd, 'vienot': simulate_vienot_cvd}

DEFAULTS = {
    'beta': 4.0,
    'lightdomain': 'linearized',
    'matrix': 'paper',
    'simulator': 'brettel',
}

PARAM_DOC = {
    'beta': '对比参数 β (Eq.14), 全局标量, 放大 Ẑ=Z_o+β(Ê_x+Ê_z) 的误差再分配; '
            'β=1 ≡ Hassan17 (Eq.11); 论文 p.1980 推荐 1≤β≤4, 主观实验用 β=4; '
            '台账 4.2: chen 表 β 未报→我方自报, 扫参用 --param beta=X 逐档单独跑。',
    'lightdomain': 'linearized(默认, 台账 4.2: 色度学矩阵作用于 γ 值物理不自洽, '
                   '且与我方线性化消融口径一致): Eq.1 前先 srgb_to_linear, 逆变换后 '
                   '再 linear_to_srgb | gamma: Eq.1 直接作用于 γ 编码值(字面复现; '
                   '原文全文零光域语句)。双版并报作敏感性; 差异=同系数矩阵作用的'
                   '信号域不同, 重着色结果明显不同(自检打印对照数字)。',
    'matrix': 'paper(默认, 台账 4.2 照抄论文 Eq.1 = Travis1991 系数, 白点 Y≈100) '
              '| srgb_d65(标准 sRGB/D65 矩阵敏感性变体; 原文系数与标准值有出入, '
              '如 Y 行 70.6743 vs 71.52, 荧光粉基色口径不明)。',
    'simulator': '误差项 E_x/E_z 所用红绿色觉感知图的来源: brettel(默认) | vienot, '
                 '均取 severity=1.0 二色端点(本方法无连续 severity 概念, 台账 4.2); '
                 '原文仅写 Brettel1997/Vienot1999 未指明用哪个→默认 brettel, '
                 'vienot 为敏感性变体(protan 尤需)。',
}

# Pure closed-form transform; the harness may wrap us in torch.no_grad().
NO_GRAD = True

_Y_EPS = 1e-6  # guard for the Y=100 normalization division ONLY (float64 scale)


def _matrix(name, device):
    rows = _MATRICES[name]
    m = torch.tensor(rows, dtype=torch.float64, device=device)
    return m, torch.linalg.inv(m)


def _hassan19_xyz(xyz_o, xyz_s, beta):
    """Eq. 1-14 in normalized XYZ space; returns the un-normalized recolored
    XYZ. The Y channel is a bit-exact copy of Y_o (Eq. 13), so luminance
    invariance holds strictly pre-clip. float64 in/out, (B,3,H,W)."""
    y_o = xyz_o[:, 1]
    y_s = xyz_s[:, 1]
    # Eq. 2-5 use Y=100-normalized values; each set divided by its OWN Y
    # (paper: "both sets ... are normalized (Y = 100)").
    yo_s = y_o.clamp_min(_Y_EPS)
    ys_s = y_s.clamp_min(_Y_EPS)
    x_o = 100.0 * xyz_o[:, 0] / yo_s
    z_o = 100.0 * xyz_o[:, 2] / yo_s
    x_s = 100.0 * xyz_s[:, 0] / ys_s
    z_s = 100.0 * xyz_s[:, 2] / ys_s

    e_x = (x_o - x_s).abs()                                     # Eq. 2
    e_z = (z_o - z_s).abs()                                     # Eq. 3
    theta_o = torch.atan2(z_o, x_o)                             # Eq. 4
    theta_s = torch.atan2(z_s, x_s)                             # Eq. 5
    phi = (theta_o - theta_s).abs()                             # Eq. 6
    cos_p, sin_p = torch.cos(phi), torch.sin(phi)
    ehat_x = e_x * cos_p - e_z * sin_p                          # Eq. 7
    ehat_z = e_x * sin_p + e_z * cos_p                          # Eq. 8

    # distribute back (Eq. 12-14); Ehat_x enters X AND Z, beta scales only
    # the Z-side sum. In normalized space Y_n == 100 == Y_o.
    x_n = x_o + ehat_x                                           # Eq. 12
    z_n = z_o + beta * (ehat_x + ehat_z)                         # Eq. 14

    # un-normalize by the TRUE Y_o; Y channel pinned to Y_o exactly (Eq. 13).
    f = y_o / 100.0
    return torch.stack((x_n * f, y_o, z_n * f), dim=1)


def _recolor_full(inp, axis, beta, lightdomain, matrix, simulator):
    """Returns (out, clip_fraction). out has inp's dtype/device."""
    if axis not in ('protan', 'deutan'):
        raise ValueError(f'hassan19: axis must be protan/deutan, got {axis!r}')
    if inp.ndim != 4 or inp.shape[1] != 3:
        raise ValueError(f'hassan19: expected (B,3,H,W), got {tuple(inp.shape)}')
    if lightdomain not in ('linearized', 'gamma'):
        raise ValueError(f'hassan19: bad lightdomain {lightdomain!r}')
    if matrix not in _MATRICES:
        raise ValueError(f'hassan19: bad matrix {matrix!r}')
    if simulator not in _SIMULATORS:
        raise ValueError(f'hassan19: bad simulator {simulator!r}')
    beta = float(beta)

    # 1. red-green dichromat perceived image (severity 1.0 by protocol).
    sim = _SIMULATORS[simulator](inp, axis, 1.0)

    m_fwd, m_inv = _matrix(matrix, inp.device)

    # 2. to float64 working domain, then RGB -> XYZ (Eq. 1).
    if lightdomain == 'linearized':
        rgb_o = srgb_to_linear(inp).to(torch.float64)
        rgb_s = srgb_to_linear(sim).to(torch.float64)
    else:
        rgb_o = inp.to(torch.float64)
        rgb_s = sim.to(torch.float64)
    xyz_o = torch.einsum('bchw,dc->bdhw', rgb_o, m_fwd)
    xyz_s = torch.einsum('bchw,dc->bdhw', rgb_s, m_fwd)

    xyz_n = _hassan19_xyz(xyz_o, xyz_s, beta)                   # Eq. 2-14
    rgb_out = torch.einsum('bchw,dc->bdhw', xyz_n, m_inv)       # inverse Eq. 1

    clip_fraction = float(((rgb_out < 0.0) | (rgb_out > 1.0)).float().mean())
    if lightdomain == 'linearized':
        out = linear_to_srgb(rgb_out, clamp=True).to(inp.dtype)  # clamps [0,1]
    else:
        out = rgb_out.clamp(0.0, 1.0).to(inp.dtype)
    return out, clip_fraction


def recolor(inp, axis, **params):
    """(B,3,H,W) sRGB [0,1] in -> recolored (B,3,H,W) out, same dtype/device."""
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'hassan19: unknown params {sorted(unknown)}')
    merged = {**DEFAULTS, **params}
    out, _ = _recolor_full(inp, axis, merged['beta'], merged['lightdomain'],
                           merged['matrix'], merged['simulator'])
    return out


# --------------------------------------------------------------------------
# selftest (tier-A invariants + one tier-B trend check, docs/BASELINES_zh.md §3)
# --------------------------------------------------------------------------

def _srgb_to_linear_np(x):
    return np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)


def _linear_to_srgb_np(v):
    v = np.maximum(v, 0.0)
    return np.clip(np.where(v > 0.0031308, 1.055 * v ** (1.0 / 2.4) - 0.055,
                            12.92 * v), 0.0, 1.0)


def _reference_hassan17_numpy(inp_np, sim_np):
    """Independent numpy float64 implementation of the HASSAN17 path
    (Eq. 1-11, i.e. Eq. 14 with beta=1), written from the paper text only.
    inp_np/sim_np are gamma-encoded sRGB [0,1], shape (B,H,W,3)."""
    m = np.array(_PAPER_XYZ, dtype=np.float64)
    rgb_o = _srgb_to_linear_np(inp_np.astype(np.float64))
    rgb_s = _srgb_to_linear_np(sim_np.astype(np.float64))
    xyz_o = rgb_o @ m.T
    xyz_s = rgb_s @ m.T
    y_o = xyz_o[..., 1]
    y_s = xyz_s[..., 1]
    xo, zo = (100.0 * xyz_o[..., 0] / np.maximum(y_o, _Y_EPS),
              100.0 * xyz_o[..., 2] / np.maximum(y_o, _Y_EPS))
    xs, zs = (100.0 * xyz_s[..., 0] / np.maximum(y_s, _Y_EPS),
              100.0 * xyz_s[..., 2] / np.maximum(y_s, _Y_EPS))
    ex, ez = np.abs(xo - xs), np.abs(zo - zs)                    # Eq. 2-3
    phi = np.abs(np.arctan2(zo, xo) - np.arctan2(zs, xs))        # Eq. 4-6
    ehat_x = ex * np.cos(phi) - ez * np.sin(phi)                # Eq. 7
    ehat_z = ex * np.sin(phi) + ez * np.cos(phi)                # Eq. 8
    xn = xo + ehat_x                                             # Eq. 9
    zn = zo + ehat_x + ehat_z                                    # Eq. 11 (beta=1)
    f = y_o / 100.0
    xyz_n = np.stack((xn * f, y_o, zn * f), axis=-1)             # Eq. 10
    rgb_lin = xyz_n @ np.linalg.inv(m).T
    return _linear_to_srgb_np(rgb_lin)                           # clip


def _colorbar_image():
    """(1,3,64,64) artificial colour bars (8 vertical bars, mixed hues)."""
    colors = torch.tensor([[0.90, 0.10, 0.10],
                           [0.10, 0.70, 0.20],
                           [0.10, 0.15, 0.85],
                           [0.85, 0.80, 0.20],
                           [0.20, 0.80, 0.85],
                           [0.80, 0.20, 0.75],
                           [1.00, 1.00, 1.00],
                           [0.00, 0.00, 0.00]])
    bars = colors.t().repeat_interleave(8, dim=1)  # (3, 64)
    return bars.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 64)  # (1,3,64,64)


def selftest() -> None:
    torch.manual_seed(0)

    # ---- (a) beta=1 must equal an independent Hassan17 numpy reference ----
    inp = torch.rand(2, 3, 8, 8)
    worst = 0.0
    for axis in ('protan', 'deutan'):
        sim = _SIMULATORS['brettel'](inp, axis, 1.0).detach().numpy()
        ref = _reference_hassan17_numpy(np.transpose(inp.numpy(), (0, 2, 3, 1)),
                                       np.transpose(sim, (0, 2, 3, 1)))
        out = recolor(inp, axis, beta=1.0)
        diff = float(np.abs(np.transpose(out.numpy().astype(np.float64),
                                         (0, 2, 3, 1)) - ref).max())
        worst = max(worst, diff)
        assert diff < 1e-5, f'(a) Hassan17 equivalence failed: {diff:.3e}'
    print(f'(a) beta=1 == Hassan17 reference path: max|diff| = {worst:.2e} (< 1e-5)')

    # ---- (b) luminance invariance: output XYZ Y channel == Y_o exactly ----
    inp = torch.rand(1, 3, 8, 8)
    m_fwd, _ = _matrix('paper', inp.device)
    sim = _SIMULATORS['brettel'](inp, 'deutan', 1.0)
    xyz_o = torch.einsum('bchw,dc->bdhw',
                         srgb_to_linear(inp).to(torch.float64), m_fwd)
    xyz_s = torch.einsum('bchw,dc->bdhw',
                         srgb_to_linear(sim).to(torch.float64), m_fwd)
    xyz_n = _hassan19_xyz(xyz_o, xyz_s, beta=4.0)
    assert torch.equal(xyz_n[:, 1], xyz_o[:, 1]), '(b) Y channel not bit-exact pre-clip'
    print('(b) Y_n == Y_o bit-exact pre-clip (Eq. 13 luminance invariance)')

    # ---- (c) range/finiteness/dtype/device across all knob combinations ----
    inp = torch.rand(2, 3, 32, 32)
    for axis in ('protan', 'deutan'):
        for ld in ('linearized', 'gamma'):
            for mtx in ('paper', 'srgb_d65'):
                for sm in ('brettel', 'vienot'):
                    for beta in (1.0, 4.0):
                        out, frac = _recolor_full(inp, axis, beta, ld, mtx, sm)
                        assert out.dtype == inp.dtype and out.device == inp.device
                        assert torch.isfinite(out).all(), f'non-finite {axis}/{ld}'
                        assert out.min() >= 0.0 and out.max() <= 1.0, 'range'
    print('(c) output in [0,1] & finite & dtype/device-preserved for all 32 '
          'knob combos (2 axis x 2 lightdomain x 2 matrix x 2 simulator x beta{1,4})')

    # clip statistics (paper silent -> clip always, 台账 4.2; informational)
    bars = _colorbar_image()
    for axis in ('protan', 'deutan'):
        for beta in (1.0, 2.0, 4.0):
            _, frac = _recolor_full(bars, axis, beta, 'linearized', 'paper', 'brettel')
            print(f'    clip-fraction (bars, {axis}, beta={beta}): {frac:.4f}')

    # ---- (d) tier-B trend: E_nat (Eq.15, mean ||C - Chat||^2 in sRGB) rises
    #      monotonically with beta on a colour-bar image, for both axes ----
    for axis in ('protan', 'deutan'):
        vals = []
        for beta in (1.0, 2.0, 4.0):
            out = recolor(bars, axis, beta=beta)
            vals.append(float(((bars - out) ** 2).mean()))
        print(f'(d) E_nat vs beta [{axis}]: ' +
              ', '.join(f'{b:g}->{v:.6f}' for b, v in zip((1, 2, 4), vals)))
        assert vals[0] < vals[1] < vals[2], \
            f'(d) E_nat not monotone in beta for {axis}: {vals}'

    # lightdomain sensitivity is informational (documented in PARAM_DOC)
    for axis in ('protan', 'deutan'):
        o_lin = recolor(bars, axis)
        o_gam = recolor(bars, axis, lightdomain='gamma')
        mse = float(((o_lin - o_gam) ** 2).mean())
        print(f'    lightdomain sensitivity ({axis}, beta=4): '
              f'MSE(linearized, gamma) = {mse:.6f}')

    print('hassan19 selftest: PASS')


if __name__ == '__main__':
    selftest()
