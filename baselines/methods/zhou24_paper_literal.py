"""PAPER-LITERAL sensitivity variant of Zhou24 (moved 2026-11-24).

Not the reported baseline: underdetermined choices here were guessed from the
paper/citation chain (|P|=2N neighbour pairs, k=8, interpolated LUT steps) and
DISAGREE with the user's hand-reviewed 2025 reproduction (/Volumes/sea_disk/
颜色与视觉实验室/zhou_method/zhouMethod.py). Primary module: zhou24.py (port of
the user pipeline, protocol-adapted). Where the two conflict, the user version
wins (docs/BASELINES_zh.md 4.3).

STATUS 2026-11-24: NOT RUNNABLE — recolor() crashes on a 32x32 input
(np.linalg.norm on an inhomogeneous sequence; superseded draft, never
finished its own gates). Kept only as documentation of the literal-paper
reading; do not wire into any run. Repair = optional follow-up, off the
critical path.
"""

import time

import numpy as np
import torch
from kornia.color import lab_to_rgb, rgb_to_lab

from cvd_simulation import simulate_machado_cvd

DEFAULTS = {
    'beta_protan': 0.3,
    'beta_deutan': 0.2,
    'k': 8,
    'neighbor_radius': 1,
    'eta': 1.0,
    'lightdomain': 'linearized',
    'lut_mode': 'interp',
    'alpha': 'auto',
    'gamut_grid_step': 4,
    'lut_L_step': 1.0,
    'lut_b_step': 1.0,
}

PARAM_DOC = {
    'beta_protan': 'Eq.4 C_R=beta*R_i; Table 1 protan@100% = 0.3 (paper-given).',
    'beta_deutan': 'Table 1 deutan@100% = 0.2 (paper-given; 0.3 applies only at '
                   '40/60/80%). Sensitivity: 0.3.',
    'k': 'k-means classes (Fig.4 direction denoising) — paper gives NO value '
         '(ledger 4.3); default 8 ours, sensitivity {4,8,16}.',
    'neighbor_radius': 'Pair-set neighbour window — Zhou24/Huang22/Wang21 give no value '
                       '(ledger 4.3); 1 = 4-neighbourhood (Manhattan <= r), consistent '
                       'with Wang21 "1 neighbour + 1 global random" (|P|=2N). '
                       'Sensitivity: 2.',
    'eta': 'Eq.8 target-contrast multiplier; paper states eta=1 (Sec.3.8); >1 '
           'amplifies contrast (their Ishihara demo used x10).',
    'lightdomain': 'Light domain of Machado sim/gamut/LUT — paper silent (ledger 4.3); '
                   'default linearized = our evaluation convention; variant "gamma" = '
                   'matrix on encoded sRGB. Report both.',
    'lut_mode': "Gamut-surface query for the back-projection target: 'interp' = "
                "bilinear over the (L~,b~) surface table, 'nn' = nearest node. "
                "Paper: 'smaller steps: more accurate, slower', no values "
                "(ledger 4.3) — our choice; selftest prints both residuals.",
    'alpha': "'auto' = Eq.5 alpha in [-1,1], step 0.1, 21-point exhaustive (grid "
             "given in paper). Numeric (e.g. 0.5) fixes alpha — selftest/ablation "
             'convenience, not from paper.',
    'gamut_grid_step': 'RGB sampling step of the gamut grid; paper gives 4 '
                       '(= 262,144 points; Huang22 used 5). Paper-given; change '
                       'only for debugging.',
    'lut_L_step': 'surface-table node spacing along L~ in Lab units — paper not '
                  'specific (underspec: LUT resolution); ours: 1.0.',
    'lut_b_step': 'surface-table node spacing along b~ in Lab units — paper not '
                  'specific (underspec: LUT resolution); ours: 1.0.',
}

_PAIR_SEED = 0        # deterministic pixel-pair sampling (paper: "random")
_KMEANS_SEED = 0      # deterministic k-means init
_KMEANS_ITERS = 25
_ALPHA_GRID = np.linspace(-1.0, 1.0, 21)

_GAMUT_CACHE = {}


# ----------------------------------------------------------------------------- gamut

def _lab_of(rgb_t: torch.Tensor) -> np.ndarray:
    """(...,3,H,W) sRGB float tensor -> (M,3) Lab (metrics.py backend:
    kornia rgb_to_lab on encoded sRGB)."""
    return rgb_to_lab(rgb_t.double()).reshape(3, -1).t().contiguous().numpy()


def _nearest_valid_fill(table: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid nodes of an (nL,nb,C) table with the mean of valid
    8-neighbours, iterated (node grids are tiny; equivalent to nearest-valid
    for our smooth surface tables)."""
    out = table.copy()
    v = valid.copy()
    nL, nb = v.shape
    C = out.shape[2]
    flat_out = out.reshape(-1, C)
    flat_v = v.reshape(-1)
    for _ in range(max(nL, nb)):
        fill_rows = np.flatnonzero(~flat_v)
        if fill_rows.size == 0:
            break
        padded_v = np.zeros((nL + 2, nb + 2), dtype=bool)
        padded_v[1:-1, 1:-1] = v
        padded_o = np.zeros((nL + 2, nb + 2, C), dtype=out.dtype)
        padded_o[1:-1, 1:-1] = out
        fmask = ~v
        acc = np.zeros((fill_rows.size, C), dtype=out.dtype)
        cnt = np.zeros(fill_rows.size, dtype=np.float64)
        any_src = np.zeros(fill_rows.size, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == dx == 0:
                    continue
                mv = padded_v[1 + dy:1 + dy + nL, 1 + dx:1 + dx + nb][fmask]
                mo = padded_o[1 + dy:1 + dy + nL, 1 + dx:1 + dx + nb][fmask]
                acc[mv] += mo[mv]
                cnt[mv] += 1.0
                any_src |= mv
        if not any_src.any():
            break
        rows = fill_rows[any_src]
        flat_out[rows] = acc[any_src] / cnt[any_src][:, None]
        flat_v[rows] = True
    return out


def _build_gamut(axis: str, lightdomain: str, step: int,
                 lut_L_step: float, lut_b_step: float) -> dict:
    """Content-independent gamut surface table (paper Sec.3.2). Uses the EXACT
    simulate_machado_cvd + kornia rgb_to_lab pipeline on the whole RGB grid."""
    if lightdomain not in ('linearized', 'gamma'):
        raise ValueError(f"lightdomain must be 'linearized' or 'gamma', got {lightdomain!r}")
    levels = np.arange(0, 256, step, dtype=np.float64) / 255.0
    rr, gg, bb = np.meshgrid(levels, levels, levels, indexing='ij')
    grid = np.stack([rr.ravel(), gg.ravel(), bb.ravel()], axis=1)      # (G,3)
    g_t = torch.from_numpy(grid).float().permute(1, 0).reshape(1, 3, 1, -1)
    linearize = lightdomain == 'linearized'
    with torch.no_grad():
        sim_srgb = simulate_machado_cvd(g_t, axis, 1.0, linearize=linearize)
        lab_s = _lab_of(sim_srgb)          # perceived Lab of each grid colour (G,3)
        lab_d = _lab_of(g_t.double())      # displayed Lab
    band_dist = float(np.linalg.norm(lab_d - lab_s, axis=1).mean())  # grid self-sim gap

    L_all, a_all, b_all = lab_s[:, 0], lab_s[:, 1], lab_s[:, 2]
    b_lo, b_hi = float(np.floor(b_all.min())), float(np.ceil(b_all.max()))

    # --- gamut surface table a = f(L~, b~) from cell means (our fit)
    nL = int(np.floor(100.0 / lut_L_step)) + 1
    nb = int(np.floor((b_hi - b_lo) / lut_b_step)) + 1
    iL = np.clip(np.rint(L_all / lut_L_step).astype(np.int64), 0, nL - 1)
    ib = np.clip(np.rint((b_all - b_lo) / lut_b_step).astype(np.int64), 0, nb - 1)
    flat = iL * nb + ib
    cnt = np.bincount(flat, minlength=nL * nb).astype(np.float64).reshape(nL, nb)
    asum = np.bincount(flat, weights=a_all, minlength=nL * nb).reshape(nL, nb)
    bsum = np.bincount(flat, weights=b_all, minlength=nL * nb).reshape(nL, nb)
    occ = cnt > 0
    a_tab = np.where(occ, asum / np.maximum(cnt, 1.0), np.nan)
    b_tab = np.where(occ, bsum / np.maximum(cnt, 1.0), np.nan)
    a_tab3 = _nearest_valid_fill(np.stack([a_tab, b_tab], axis=-1), occ)[..., 0]
    # spread of a within occupied cells -> surface-table resolution error
    asum2 = np.bincount(flat, weights=a_all ** 2, minlength=nL * nb).reshape(nL, nb)
    var = np.clip(asum2 / np.maximum(cnt, 1e-12) - (asum / np.maximum(cnt, 1e-12)) ** 2, 0, None)
    spread = float(np.sqrt(var[occ]).mean())

    # --- Huang22-style parabola a=c1 b^2+c2 b+c3 (cross-check print only)
    edges = np.arange(np.floor(b_all.min()), np.ceil(b_all.max()) + 1.0001, 1.0)
    ib_all = np.clip(np.searchsorted(edges, b_all, side='right') - 1, 0, len(edges) - 2)
    nbins = len(edges) - 1
    cntb = np.bincount(ib_all, minlength=nbins).astype(np.float64)
    asum_b = np.bincount(ib_all, weights=a_all, minlength=nbins)
    okb = cntb >= 50
    bc = 0.5 * (edges[:-1] + edges[1:])[okb]
    am = (asum_b / np.maximum(cntb, 1.0))[okb]
    c1, c2, c3 = np.polyfit(bc, am, 2)
    fit_resid = float(np.abs(am - (c1 * bc ** 2 + c2 * bc + c3)).mean())

    return dict(axis=axis, lightdomain=lightdomain, step=int(step),
                lut_L_step=float(lut_L_step), lut_b_step=float(lut_b_step),
                c1=float(c1), c2=float(c2), c3=float(c3), fit_resid=fit_resid,
                b_lo=b_lo, b_hi=b_hi, nL=nL, nb=nb,
                a_tab=torch.from_numpy(a_tab3),
                coverage=float(occ.mean()), spread=spread, band_dist=band_dist)


def _get_gamut(axis, lightdomain, step, lut_L_step, lut_b_step) -> dict:
    key = (axis, lightdomain, int(step), float(lut_L_step), float(lut_b_step))
    if key not in _GAMUT_CACHE:
        _GAMUT_CACHE[key] = _build_gamut(axis, lightdomain, int(step),
                                         float(lut_L_step), float(lut_b_step))
    return _GAMUT_CACHE[key]


def _surface_a(g: dict, L: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Gamut-surface lookup a(L~, b~) per lut_mode ('interp' bilinear / 'nn')."""
    nL, nb = g['nL'], g['nb']
    L = L.clamp(0.0, (nL - 1) * g['lut_L_step'])
    b = b.clamp(g['b_lo'], g['b_lo'] + (nb - 1) * g['lut_b_step'])
    tab = g['a_tab']
    if g['lut_mode_cached'] == 'nn':
        iL = (L / g['lut_L_step']).round().long().clamp(0, nL - 1)
        ib = ((b - g['b_lo']) / g['lut_b_step']).round().long().clamp(0, nb - 1)
        return tab[iL, ib]
    fL = (L / g['lut_L_step']).clamp(0, nL - 1.0000001)
    fb = ((b - g['b_lo']) / g['lut_b_step']).clamp(0, nb - 1.0000001)
    i0 = fL.long()
    j0 = fb.long()
    i1 = (i0 + 1).clamp(max=nL - 1)
    j1 = (j0 + 1).clamp(max=nb - 1)
    wL = (fL - i0.to(fL.dtype))[:, None] if fL.ndim > 1 else (fL - i0.to(fL.dtype))
    wB = (fb - j0.to(fb.dtype))[:, None] if fb.ndim > 1 else (fb - j0.to(fb.dtype))
    top = tab[i0, j0] * (1 - wB) + tab[i0, j1] * wB
    bot = tab[i1, j0] * (1 - wB) + tab[i1, j1] * wB
    return (top * (1 - wL) + bot * wL).squeeze(-1) if top.ndim > 1 else (top * (1 - wL) + bot * wL)


# ------------------------------------------------------------------------- per image

def _neighbour_partner(H: int, W: int, radius: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random valid Manhattan-radius neighbour per pixel; isolated
    pixels pair with themselves. Returns (N,) int64."""
    N = H * W
    yy, xx = np.divmod(np.arange(N), W)
    offs = [(dy, dx) for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if 0 < abs(dy) + abs(dx) <= radius]
    if not offs:
        return np.arange(N)
    js = np.zeros((len(offs), N), dtype=np.int64)
    valid = np.zeros((len(offs), N), dtype=bool)
    for k, (dy, dx) in enumerate(offs):
        y2, x2 = yy + dy, xx + dx
        v = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
        valid[k] = v
        js[k] = np.where(v, y2 * W + x2, 0)
    counts = valid.sum(0)
    u = rng.random(N) * counts                     # uniform in [0, count)
    cum = np.cumsum(valid, axis=0)                 # (K,N)
    selk = np.minimum((cum <= u[None, :]).sum(0), len(offs) - 1)
    pick = js[selk, np.arange(N)]
    return np.where(counts > 0, pick, np.arange(N))


def _pair_direction(lab: np.ndarray, lab_s: np.ndarray, j: np.ndarray):
    """Eq.7 unit direction for every pixel against partner map j; also returns
    the informative mask (False = zero-chroma-distance pair, e.g. grey-grey or
    within a flat bar: no contrast information, excluded from averaging).
    Signs per Huang22: b_i by a* category (a>=0 -> +1), a_i away from partner."""
    a, b = lab[:, 1], lab[:, 2]
    ja, jb = a[j], b[j]
    orig = np.hypot(a - ja, b - jb)
    ok = orig > 1e-6
    simd = np.hypot(lab_s[:, 1] - lab_s[j, 1], lab_s[:, 2] - lab_s[j, 2])
    ratio = np.clip(simd / np.maximum(orig, 1e-12), 0.0, 1.0)
    bmag = np.sqrt(np.maximum(0.0, 1.0 - ratio ** 2))
    sa = np.where(a >= ja, 1.0, -1.0)
    sb = np.where(a >= 0.0, 1.0, -1.0)
    d = np.stack([sa * ratio, sb * bmag], axis=1)
    return d, ok


def _kmeans_dirs(xy: np.ndarray, d: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Fig.4: k-means on (a*,b*) positions; every pixel gets the unit-normalised
    mean direction of its cluster (raw pair-direction noise averaged out)."""
    N = xy.shape[0]
    k = max(1, min(int(k), N))
    cent = xy[rng.permutation(N)[:k]].copy()
    labels = None
    for _ in range(_KMEANS_ITERS):
        dist2 = ((xy[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        new_labels = dist2.argmin(1)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            m = labels == c
            if m.any():
                cent[c] = xy[m].mean(0)
    labels = new_labels
    out = d.copy()
    for c in range(k):
        m = labels == c
        if not m.any():
            continue
        v = d[m].mean(0)
        n = np.linalg.norm(v)
        if n > 1e-8:
            out[m] = v / n
        else:
            out[m, 0] = 0.0                                   # fallback: pure +-b*
            out[m, 1] = np.where(xy[m, 1] >= 0.0, 1.0, -1.0)
    return out


def _solve_single(img: torch.Tensor, axis: str, params: dict, g: dict) -> dict:
    """Full per-image pipeline up to the final perceived target; returns the
    target Lab (N,3), the alpha*/E curve and diagnostics."""
    beta = float(params['beta_protan'] if axis == 'protan' else params['beta_deutan'])
    eta = float(params['eta'])
    H, W = img.shape[1], img.shape[2]
    linearize = params['lightdomain'] == 'linearized'
    with torch.no_grad():
        lab = _lab_of(img.reshape(1, 3, H, W))
        sim = simulate_machado_cvd(img.reshape(1, 3, H, W), axis, 1.0, linearize=linearize)
        lab_s = _lab_of(sim)
    N = lab.shape[0]
    R = np.linalg.norm(lab - lab_s, axis=1)                   # Eq.1
    CR = beta * R                                             # Eq.4

    rng = np.random.default_rng(_PAIR_SEED)
    jn = _neighbour_partner(H, W, int(params['neighbor_radius']), rng)
    jg = rng.integers(0, N, size=N)
    d1 = _pair_direction(lab, lab_s, jn)
    d2 = _pair_direction(lab, lab_s, jg)
    d = d1 + d2
    n = np.linalg.norm(d, axis=1, keepdims=True)
    d = np.where(n > 1e-8, d / np.maximum(n, 1e-12), d1)      # degenerate -> nbr dir
    d = _kmeans_dirs(lab[:, 1:], d, int(params['k']), np.random.default_rng(_KMEANS_SEED))

    # Eq.8 pair table P, |P| = 2N
    ia = np.tile(np.arange(N), 2)
    ja = np.concatenate([jn, jg])
    target = eta * ((lab[ia, 0] - lab[ja, 0]) ** 2
                    + (lab[ia, 1] - lab[ja, 1]) ** 2
                    + (lab[ia, 2] - lab[ja, 2]) ** 2)
    L0 = torch.from_numpy(lab_s[:, 0])
    a0 = lab_s[:, 1]
    b0c = np.clip(lab_s[:, 2], g['b_lo'], g['b_hi'])          # centre: simulated chroma
    u = CR * d[:, 0]
    w = CR * d[:, 1]
    b_lo_t = torch.tensor(g['b_lo'])
    b_hi_t = torch.tensor(g['b_hi'])

    alpha = params['alpha']
    if isinstance(alpha, str) and alpha.strip() != 'auto':
        alpha = float(alpha)
    alphas = _ALPHA_GRID if isinstance(alpha, str) else np.array([float(alpha)])

    gg = {**g, 'lut_mode_cached': params['lut_mode']}
    E_all, best = [], None
    for a_ in alphas:
        b_raw = b0c + a_ * w
        clamp_b = float(((b_raw < g['b_lo']) | (b_raw > g['b_hi'])).mean())
        b_t = np.clip(b_raw, g['b_lo'], g['b_hi'])            # band-boundary pullback
        # resit on the fitted endpoint surface: off-surface delta-a is not
        # perceivable at severity 1.0 (see module docstring)
        a_t = _surface_a(gg, L0, torch.from_numpy(b_t)).numpy()
        ach = ((lab_s[ia, 0] - lab_s[ja, 0]) ** 2
               + (a_t[ia] - a_t[ja]) ** 2
               + (b_t[ia] - b_t[ja]) ** 2)
        E_all.append(float(np.mean((ach - target) ** 2)))
        if best is None or E_all[-1] < best[0]:
            best = (E_all[-1], float(a_), a_t, b_t, clamp_b, float(np.abs(a_ * u).mean()))
    _, a_star, a_t, b_t, clamp_b, da_cancel = best
    t_lab = np.stack([lab_s[:, 0], a_t, b_t], axis=1)
    return dict(t_lab=t_lab, alpha=a_star, E_all=E_all, alphas=alphas,
                clamp_frac=clamp_b, da_cancelled=da_cancel,
                R_mean=float(R.mean()), CR_mean=float(CR.mean()))


# -------------------------------------------------------------------------------- API

def recolor(inp: torch.Tensor, axis: str, **params) -> torch.Tensor:
    """Zhou24 recoloring. inp: sRGB [0,1] (B,3,H,W) or (3,H,W); output same
    dtype/device. Severity fixed to 1.0 by protocol (docs/BASELINES_zh.md
    Sec.2). Pure CPU numerics; per-image global alpha, pairing and k-means
    exactly as the paper processes single images."""
    if axis not in ('protan', 'deutan'):
        raise ValueError(f'axis must be protan/deutan, got {axis!r}')
    p = {**DEFAULTS, **params}
    if p['lut_mode'] not in ('interp', 'nn'):
        raise ValueError(f"lut_mode must be 'interp' or 'nn', got {p['lut_mode']!r}")
    squeeze = inp.ndim == 3
    x = inp.unsqueeze(0) if squeeze else inp
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f'expected (B,3,H,W) or (3,H,W), got {tuple(inp.shape)}')
    g = _get_gamut(axis, p['lightdomain'], int(p['gamut_grid_step']),
                   float(p['lut_L_step']), float(p['lut_b_step']))
    cpu = x.detach().to('cpu', dtype=torch.float32)
    outs = []
    for b in range(cpu.shape[0]):
        info = _solve_single(cpu[b], axis, p, g)
        t = torch.from_numpy(info['t_lab'])                  # (H*W,3) float64
        rgb13n = t.t().reshape(3, 1, -1).unsqueeze(0)         # (1,3,1,N) channels L,a,b
        with torch.no_grad():
            rgb = lab_to_rgb(rgb13n, clip=False)             # exact inverse of rgb_to_lab
        rgb = rgb.reshape(3, -1).clamp(0.0, 1.0)             # (3,N) RGB, Sec.3 output clip
        outs.append(rgb.reshape(3, cpu.shape[2], cpu.shape[3]))
    out = torch.stack(outs).to(dtype=inp.dtype, device=inp.device)
    return out.squeeze(0) if squeeze else out


# ---------------------------------------------------------------------------- selftest

def _synthetic_bar(size=(128, 162)) -> torch.Tensor:
    """Artificial colour bars: red/green confusion pairs + blue/yellow/etc."""
    cols = [(0.85, 0.10, 0.10), (0.25, 0.55, 0.15), (0.95, 0.55, 0.10),
            (0.10, 0.30, 0.80), (0.90, 0.90, 0.90), (0.55, 0.15, 0.60),
            (0.95, 0.95, 0.20), (0.15, 0.15, 0.15), (0.20, 0.70, 0.75)]
    h, w = size
    img = torch.zeros(3, h, w)
    bw = w // len(cols)
    for i, c in enumerate(cols):
        img[0, :, i * bw:(i + 1) * bw] = c[0]
        img[1, :, i * bw:(i + 1) * bw] = c[1]
        img[2, :, i * bw:(i + 1) * bw] = c[2]
    return img


def selftest() -> None:
    from metrics import tcc

    torch.manual_seed(0)
    checks = []

    def report(name, ok, detail):
        checks.append((name, bool(ok)))
        print(f'  [{"ok" if ok else "FAIL"}] {name}: {detail}')

    inp = torch.rand(1, 3, 64, 64)
    t0 = time.perf_counter()
    _ = recolor(inp, 'protan')
    cold_ms = 1000 * (time.perf_counter() - t0)

    for axis in ('protan', 'deutan'):
        g = _get_gamut(axis, DEFAULTS['lightdomain'], DEFAULTS['gamut_grid_step'],
                       DEFAULTS['lut_L_step'], DEFAULTS['lut_b_step'])
        ref = {'protan': (0.0034, -0.1224), 'deutan': (0.0033, -0.1275)}[axis]
        print(f'gamut[{axis}] own surface table: cell a-spread={g["spread"]:.3f}, '
              f'node coverage={g["coverage"]:.3f}, mean grid self-sim gap={g["band_dist"]:.3f}; '
              f'parabola c1={g["c1"]:.4f} (Huang22 {ref[0]:.4f}) '
              f'c2={g["c2"]:.4f} (Huang22 {ref[1]:.4f}) c3={g["c3"]:.3f} '
              f'resid={g["fit_resid"]:.3f}; b-range=[{g["b_lo"]:.1f},{g["b_hi"]:.1f}]')

        # (a) alpha=0 projection idempotence: simulate(output) ~ output
        #     (BASELINES_zh Sec.3: "alpha=0 => output colours all inside the
        #      dichromatic projection plane")
        out0 = recolor(inp, axis, alpha=0.0)
        sim_out0 = simulate_machado_cvd(out0, axis, 1.0, linearize=True)
        r1 = (rgb_to_lab(sim_out0.clamp(0, 1).double())
              - rgb_to_lab(out0.double())).norm(dim=1).flatten(1)
        sim_in = simulate_machado_cvd(inp, axis, 1.0, linearize=True)
        r2 = (rgb_to_lab(sim_out0.double()) - rgb_to_lab(sim_in.double())).norm(dim=1).flatten(1)
        report(f'{axis} invariant(a) sim(out@a=0) ~ out', float(r1.mean()) < 1.5,
               f'mean|Lab(sim(out))-Lab(out)| = {float(r1.mean()):.3f} '
               f'(p95 {float(r1.quantile(0.95)):.3f}); perceived drift vs sim(input) = '
               f'{float(r2.mean()):.3f} Lab units')

        # (b) alpha* is the argmin of the 21-point E curve
        info = _solve_single(inp[0], axis, dict(DEFAULTS), g)
        amin = int(np.argmin(info['E_all']))
        ok_b = (abs(info['alphas'][amin] - info['alpha']) < 1e-9
                and all(info['E_all'][amin] <= e + 1e-12 for e in info['E_all']))
        report(f'{axis} invariant(b) alpha* = argmin E', ok_b,
               f'alpha*={info["alpha"]:+.1f}  E(alpha*)={info["E_all"][amin]:.2f}  '
               f'E(alpha=0)={info["E_all"][10]:.2f}  curve[min,max]='
               f'[{min(info["E_all"]):.2f},{max(info["E_all"]):.2f}]  '
               f'b-clamp frac={info["clamp_frac"]:.3f}  mean|delta-a| absorbed by '
               f'surface resit={info["da_cancelled"]:.2f}  R mean={info["R_mean"]:.2f}  '
               f'CR mean={info["CR_mean"]:.2f}')

        # (c) output in [0,1], finite, batch/plumbing intact
        t0 = time.perf_counter()
        out2 = recolor(torch.rand(2, 3, 32, 48), axis)
        ms64 = 1000 * (time.perf_counter() - t0)
        ok_c = bool(torch.isfinite(out2).all() and out2.min() >= 0.0
                    and out2.max() <= 1.0 and out2.shape == (2, 3, 32, 48)
                    and out2.dtype == torch.float32)
        report(f'{axis} invariant(c) range/batch', ok_c,
               f'range [{float(out2.min()):.3f},{float(out2.max()):.3f}] dtype {out2.dtype}')

        # (d) B-tier trend probe: TCC gain > 0 on artificial colour bars
        bar = _synthetic_bar().unsqueeze(0)
        obar = recolor(bar, axis)
        lin_bar = rgb_to_lab(simulate_machado_cvd(bar, axis, 1.0).double())[0]
        lin_obar = rgb_to_lab(simulate_machado_cvd(obar, axis, 1.0).double())[0]
        t_in, t_out = float(tcc(lin_bar, seed=7)), float(tcc(lin_obar, seed=7))
        report(f'{axis} trend(d) colour-bar TCC gain', t_out > t_in,
               f'TCC sim(input)={t_in:.2f} -> sim(output)={t_out:.2f} (gain {t_out - t_in:+.2f})')

        print(f'Zhou24 {axis}: 64x64-class single image (2x32x48 batch) CPU = {ms64:.0f} ms')

    # (e) protocol-size timing
    big = torch.rand(1, 3, 256, 256)
    t0 = time.perf_counter()
    _ = recolor(big, 'protan')
    t1 = time.perf_counter()
    _ = recolor(big, 'deutan')
    t2 = time.perf_counter()
    print(f'Zhou24 timing: 256x256 protan {1000 * (t1 - t0):.0f} ms, deutan '
          f'{1000 * (t2 - t1):.0f} ms (warm caches); cold first call incl. gamut '
          f'build {cold_ms:.0f} ms; paper self-report: <6 s per 200k-px image on a '
          f'3.8 GHz AMD CPU (scale reference only)')

    fails = [n for n, ok in checks if not ok]
    if fails:
        print(f'zhou24 selftest: FAIL ({len(fails)}/{len(checks)}): {fails}')
        raise AssertionError(f'zhou24 selftest failed checks: {fails}')
    print('zhou24 selftest: PASS')


if __name__ == '__main__':
    selftest()
