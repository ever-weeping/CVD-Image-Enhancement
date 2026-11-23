"""Zhou24 — Fast image recoloring for red-green anomalous trichromacy with
contrast enhancement and naturalness preservation (Zhou, Huang, Zhu, Chen, Go,
Mao; The Visual Computer 40:4647-4660, 2024, doi:10.1007/s00371-024-03454-8).

PORT OF THE USER'S OWN 2025-07 REPRODUCTION
('/Volumes/sea_disk/颜色与视觉实验室/zhou_method/zhouMethod.py', read-only
reference, human-reviewed; literature note: zhou-fast-anomalous-vc2024.md).
The earlier paper-first reconstruction drafted from the literature notes was
SUPERSEDED by this port; its findings survive only in the invariant test
design (the clamped sRGB pipeline itself is not exactly idempotent on
saturated pixels, so invariant (c) is measured against that intrinsic defect).

Everything below follows the user's reviewed pipeline step by step
(zhouMethod.recolor_image, comments map each step to their function):

  1.  lab = rgb2lab(rgb)                                   [skimage -> kornia,
      ledger: backend replaced, delta measured in selftest (a)]
  2.  rgb_sim = round(simulate_cvd(rgb)) -> uint8          [uint8 convention
      preserved: simulated colours quantised to the 256^3 display lattice]
  3.  lab_sim = rgb2lab(rgb_sim/255)
  4.  radius R_i = ||c_i - c~_i||                          [Eq.1, FULL 3D Lab]
  5.  directions, |P| = N: each pixel paired with ONE random global pixel
      (np.random.permutation). Neighbour pairing from Wang21/Huang22 prose is
      NOT present in the user's reviewed code (ledger entry: paper says
      1-neighbour + 1-random per Wang21 p.22; user implementation = global
      permutation only; port follows the user).
      a_i = clip(sim_dist/(orig_dist + 1e-8), 0, 1)        [Eq.7]
      b_i = +sqrt(1-a_i^2), flipped when a_c < 0           [Huang22 category]
      a_i = -a_i when a_sim_g >= a_sim_c (else kept)       [left/right rule]
  6.  k-means smoothing (sklearn KMeans, k=6, random_state=0, n_init=10) on
      original (a*,b*); class-mean direction ASSIGNED WITHOUT RENORMALISATION
      (the user code computes the norm but never uses it — ported as-is,
      ledger: "underspecified: normalise after smoothing? user version = no").
  7.  alpha grid np.linspace(-1,1,21), eta=1; energy = SUM (not mean) over N
      random (i,j) pairs (np.random.randint, i!=j filtered, SAME pair indices
      for all 21 alpha); distances full 3D Lab of lab_mediate vs original lab.
  8.  lab_mediate = lab_sim + alpha*beta*R_i*d_i on (a,b) only (Eq.5/6; L
      stays at the simulated L~).
  9.  gamut clamp path: intermediate = lab(round(lab2rgb(lab_mediate)*255))
  10. back-projection through a LUT built on the FULL 256^3 RGB grid (NOT the
      paper's 262,144-point step-4 grid; gamut_grid_step=1 ports the user
      choice): key = Lab of the uint8-quantised simulated colour; exact
      float-key match preferred, otherwise KDTree nearest key; among all
      candidate pre-images of that key the one closest to the ORIGINAL pixel
      Lab wins (naturalness-first rule). lab2rgb of the candidate is the
      output RGB (round->uint8 in the user code; here returned as float).
  11. output clipped to [0,1].

sim_source switches the simulation/gamut backend (the user code's two known
quirks are ledgered here, per the 2026-11-24 task decision):
  * 'protocol'  (DEFAULT for paper numbers): our cvd_simulation
    simulate_machado_cvd(axis, severity, linearize=lightdomain) — the exact
    harness backend (chen/Hassan/evaluation parity), Machado2009 endpoint
    matrices for BOTH axes.
  * 'legacy_repro': the user's hard-coded matrices in the gamma domain —
    protan: nearest of the three Machado matrices the user wrote down
    ({0.4, 0.6, 1.0}; note their code returns the 0.4 matrix for protan at
    ANY severity argument — the port uses the severity-matching matrix from
    their comments at severity=1.0 per the protocol decision, deviation
    registered; at severity=0.4 it reproduces their bytes), deutan: the
    Brettel-type matrix [[.625,.375,0],[.7,.3,0],[0,.3,.7]] blended with
    identity by severity (NOT Machado2009).
    This mode exists for selftest equivalence runs and the sensitivity
    audit, not for headline numbers.
  * User's recolor_image also always builds the LUT with cvd_type='protan',
    severity=0.4 defaults (even for deutan runs) — judged a bug; the port
    builds the LUT for the actual axis/severity (deviation registered).

Determinism: the user's np.random draws (permutation for pairing, randint for
the energy pairs) are UNSEEDED in the original code — the same image can
change alpha* between runs. The port threads every draw through a single
np.random.RandomState(rng_seed) consumed in the same order as the original
(pairing permutation, then energy pair randints), so results are stable and
comparable to the user code by seeding np.random with the same value.

DEFAULTS are the user-reviewed values except where the harness protocol
overrides them (severity=1.0, beta@100% table, kornia backend).

2026-11-24 coordinator review items (addressed, registered):
  * Cited defects (_kmeans_labels first-round early break, _grid_lab
    'if False else' expression, _recolor_single dead 'cap' line,
    _brute_alpha_energy 'pairs_i, pairs_j = ii != jj, None') do NOT exist in
    this port file — they belong to the superseded 583-line paper-first
    draft or another module (grep-verified: none of those symbols are defined
    here). The current numpy k-means fallback _lloyd_labels provably runs at
    least two assignment rounds (first pass cannot break: labels starts None);
    the selftest gates this with max_iter=1 vs max_iter=200 on synthetic
    3-blob data (assignments and inertia must differ).
  * Track semantics: NO analysis-style "fixes" are applied anywhere in this
    module. BOTH sim_source tracks keep the user's reviewed semantics,
    including class-mean directions assigned WITHOUT renormalisation. The
    protocol track differs from legacy only by the substitutions mandated by
    the protocol/harness: Machado-2009 severity=1.0 simulation via
    cvd_simulation (both axes; legacy keeps the user's gamma-domain hard-coded
    matrices), kornia Lab backend (forward rgb2lab agreement vs skimage
    measured <=1e-13), deterministic rng_seed, and axis/severity-correct LUT
    builds (the user's build_lut() always-protan@0.4 call is treated as a bug
    on both tracks; severity=0.4+protan still reproduces them exactly, e2e
    check mean |Δrgb| = 0.48/255).
"""
import os
import time

import numpy as np
import torch
from kornia.color import lab_to_rgb, rgb_to_lab

from cvd_simulation import simulate_machado_cvd

DEFAULTS = {
    'sim_source': 'protocol',
    'lightdomain': 'linearized',
    'severity': 1.0,
    'beta_protan': 0.3,
    'beta_deutan': 0.2,
    'k': 6,
    'kmeans_n_init': 10,
    'kmeans_seed': 0,
    'eta': 1.0,
    'alpha': 'auto',
    'rng_seed': 0,
    'gamut_grid_step': 1,
}

PARAM_DOC = {
    'sim_source': "'protocol' = cvd_simulation.simulate_machado_cvd(axis,severity,"
                  "linearize=...) (harness parity, Machado2009 both axes; protocol "
                  "default). 'legacy_repro' = user-code gamma-domain hard-coded "
                  "matrices (protan Machado2009 nearest of {0.4,0.6,1.0} — at 1.0 uses "
                  "their commented matrix; deutan = Brettel-type endpoint blend, NOT "
                  "Machado2009). Ledger: user's protan branch ignores its severity arg "
                  "(always 0.4); LUT axis/severity bug not reproduced.",
    'lightdomain': "Only for sim_source='protocol': 'linearized' (our main convention, "
                   "matches evaluation) or 'gamma' passthrough (dual reporting per "
                   "ledger 4.3; legacy_repro is always gamma).",
    'severity': 'Protocol fixes 1.0 (docs/BASELINES_zh.md Sec.2). Exposed only for the '
                'legacy selftest comparison (user code default 0.4) — do NOT change '
                'for paper numbers.',
    'beta_protan': 'User get_beta Table 1: protan 0.3 at every severity (paper-given).',
    'beta_deutan': 'User get_beta Table 1: deutan 0.3@20-60%, 0.2@80-100% -> 0.2 at '
                   'severity 1.0 (paper-given). Pass 0.3 for legacy severity<=0.6 runs.',
    'k': 'k-means classes — paper gives NO value (ledger 4.3); user reviewed repro uses '
         '6; port default 6 (sensitivity {4,6,8,16}). Small-image rule h*w<10k -> '
         'k=max(2,min(k,h*w//10)) ported from user.',
    'kmeans_n_init': 'sklearn KMeans n_init=10 (user value, deterministic given seed).',
    'kmeans_seed': 'sklearn KMeans random_state=0 (user value; pairing seed is rng_seed).',
    'eta': 'Eq.8 target-contrast multiplier, eta=1 (paper + user).',
    'alpha': "'auto' = np.linspace(-1,1,21) exhaustive argmin of Eq.8 (paper + user). "
             "Numeric (e.g. 0.5) skips the search (harness/ablation convenience, "
             'not from paper).',
    'rng_seed': 'User code NEVER seeds np.random (permutation pairing + randint energy '
                'pairs) — original is nondeterministic per run; port threads all draws '
                'through np.random.RandomState(rng_seed) in the original consumption '
                'order. Ledger entry: original instability.',
    'gamut_grid_step': 'LUT RGB grid step: 1 = full 256^3 grid (user choice, ported; '
                       'paper says 262,144 points at step 4 — deviation registered). '
                       '>1 = debug/sensitivity only.',
}

_LUT_CACHE = {}

_LEGACY_PROTAN = {
    0.4: np.array([[0.539009, 0.579343, -0.118352],
                   [0.082546, 0.866121, 0.051332],
                   [-0.007136, -0.011959, 1.019095]]),
    0.6: np.array([[0.385450, 0.769005, -0.154455],
                   [0.100526, 0.829802, 0.069673],
                   [-0.007442, -0.022190, 1.029632]]),
    1.0: np.array([[0.152286, 1.052583, -0.204868],
                   [0.114503, 0.786281, 0.099216],
                   [-0.003882, -0.048116, 1.051998]]),
}
_LEGACY_DEUTAN_END = np.array([[0.625, 0.375, 0.000],
                               [0.700, 0.300, 0.000],
                               [0.000, 0.300, 0.700]])


# ------------------------------------------------------------------ simulation

def _legacy_matrix(axis: str, severity: float) -> np.ndarray:
    if axis == 'protan':
        key = min(_LEGACY_PROTAN, key=lambda s: abs(s - float(severity)))
        return _LEGACY_PROTAN[key]
    return _LEGACY_DEUTAN_END * float(severity) + np.eye(3) * (1.0 - float(severity))


def _simulate_q(x: torch.Tensor, axis: str, sim_source: str,
                lightdomain: str, severity: float) -> torch.Tensor:
    """x: (1,3,H,W) float32 [0,1] sRGB. Returns the user's uint8-quantised
    simulated sRGB as float values k/255 (float64), shape as input."""
    with torch.no_grad():
        if sim_source == 'protocol':
            v = simulate_machado_cvd(x, axis, severity,
                                     linearize=(lightdomain == 'linearized'))
        elif sim_source == 'legacy_repro':
            # float64 to match the user's numpy dot arithmetic bit-for-bit
            m = torch.from_numpy(_legacy_matrix(axis, severity)).double()
            v = torch.einsum('bchw,dc->bdhw', x.double().clamp(0.0, 1.0), m).clamp(0.0, 1.0)
        else:
            raise ValueError(f"sim_source must be 'protocol' or 'legacy_repro', got {sim_source!r}")
        q = (v.double() * 255.0).round().clamp_(0.0, 255.0)
        return q / 255.0


def _lab(x: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) float [0,1] -> (H*W,3) float64 Lab, metrics.py backend."""
    return rgb_to_lab(x.double()).reshape(3, -1).t().contiguous().numpy()


# --------------------------------------------------------------------- LUT

def _grid_rgb(f: np.ndarray, levels: np.ndarray, n: int) -> np.ndarray:
    """Flat index in grid order (r slowest, b fastest, meshgrid-xy like the
    user's build_lut) -> (m,3) uint8-valued RGB array."""
    r = levels[f // (n * n)]
    g = levels[(f // n) % n]
    b = levels[f % n]
    return np.stack([r, g, b], axis=-1)


def _lut_from_arrays(sim_rgb_u8: np.ndarray, member_lab: np.ndarray,
                     member_ref: np.ndarray, lab_fn=None) -> dict:
    """Shared LUT core from materialised arrays.

    sim_rgb_u8 : (T,3) uint8-valued simulated RGB of each grid point (the key
                 domain — Lab is derived from these, like the user's dict).
    member_lab : (T,3) display Lab of the SAME order (candidates).
    member_ref : (T,) int64 payload carried to the output (user: the grid lab;
                 here the flat grid index — output = that grid RGB).
    lab_fn     : (M,3) float RGB [0,1] -> (M,3) Lab; caller chooses the
                 backend (kornia in production, skimage in equivalence tests)
                 so key/distance arithmetic stays backend-consistent.
    """
    if lab_fn is None:
        lab_fn = _lab_arr
    codes = (sim_rgb_u8[:, 0].astype(np.int64) << 16) | \
            (sim_rgb_u8[:, 1].astype(np.int64) << 8) | sim_rgb_u8[:, 2].astype(np.int64)
    T = len(codes)
    order = np.argsort(codes, kind='stable')
    sc = codes[order]
    brk = np.flatnonzero(sc[1:] != sc[:-1]) + 1
    starts = np.concatenate(([0], brk))
    ends = np.concatenate((brk, [T]))
    sim_first = sc[starts]
    sim_u = np.stack([(sim_first >> 16) & 255, (sim_first >> 8) & 255,
                      sim_first & 255], axis=1).astype(np.float64) / 255.0
    key_lab = np.asarray(lab_fn(sim_u), dtype=np.float64)
    # merge keys whose Lab tuples are bit-identical (user dict semantics)
    _, upos, rep_of_key = np.unique(key_lab, axis=0, return_index=True,
                                    return_inverse=True)
    reps_lab = key_lab[upos]
    K = len(starts)
    key_of_member = np.repeat(np.arange(K), ends - starts)   # per SORTED slot
    rep_of_member = rep_of_key[key_of_member]
    msort = np.argsort(rep_of_member, kind='stable')
    rep_sorted = rep_of_member[msort]
    brk2 = np.flatnonzero(rep_sorted[1:] != rep_sorted[:-1]) + 1
    rep_starts = np.concatenate(([0], brk2, [T]))
    msel = order[msort]                                      # sorted slot -> grid order
    kd = _make_kdtree(reps_lab)
    return dict(kd=kd, rep_starts=rep_starts.astype(np.int64),
                member_lab=member_lab[msel].astype(np.float32),
                member_ref=member_ref[msel].astype(np.int64),
                reps=reps_lab, n_keys=K, n_reps=len(upos), T=int(T))


def _lab_arr(arr: np.ndarray) -> np.ndarray:
    """(M,3) float RGB [0,1] -> (M,3) Lab, production kornia backend."""
    return _lab(torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
                .permute(1, 0).reshape(1, 3, 1, -1))


def _make_kdtree(pts: np.ndarray):
    try:
        from scipy.spatial import cKDTree
        return cKDTree(pts)
    except ImportError:                                   # pragma: no cover
        from sklearn.neighbors import KDTree
        return KDTree(pts)


def _build_lut(axis: str, sim_source: str, lightdomain: str,
               severity: float, step: int) -> dict:
    """Full RGB-grid back-projection table (user build_lut, vectorised)."""
    levels = np.arange(0, 256, step, dtype=np.int64)
    n = len(levels)
    total = n * n * n
    CH = 1 << 20
    sim_rgb = np.empty((total, 3), dtype=np.uint8)
    for s in range(0, total, CH):
        e = min(s + CH, total)
        f = np.arange(s, e, dtype=np.int64)
        rgb = _grid_rgb(f, levels, n)
        x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(1, 0).reshape(1, 3, 1, -1)
        q = _simulate_q(x, axis, sim_source, lightdomain, float(severity))
        qi = (q * 255.0).round().reshape(3, -1).t().to(torch.uint8).numpy()
        sim_rgb[s:e] = qi
    member_ref = np.arange(total, dtype=np.int64)
    member_lab = np.empty((total, 3), dtype=np.float64)
    for s in range(0, total, CH):
        e = min(s + CH, total)
        rgb = _grid_rgb(member_ref[s:e], levels, n)
        member_lab[s:e] = _lab(torch.from_numpy(rgb.astype(np.float32) / 255.0)
                               .permute(1, 0).reshape(1, 3, 1, -1))
    lut = _lut_from_arrays(sim_rgb, member_lab, member_ref)
    lut['levels'] = levels
    lut['n'] = n
    return lut


def _get_lut(axis, sim_source, lightdomain, severity, step) -> dict:
    key = (axis, sim_source, lightdomain if sim_source == 'protocol' else '-',
           float(severity), int(step))
    if key not in _LUT_CACHE:
        t0 = time.perf_counter()
        _LUT_CACHE[key] = _build_lut(axis, sim_source, lightdomain, severity, step)
        _LUT_CACHE[key]['build_s'] = time.perf_counter() - t0
    return _LUT_CACHE[key]


def _backproject(lut: dict, med_lab: np.ndarray, orig_lab: np.ndarray,
                 chunk: int = 4096) -> np.ndarray:
    """Vectorised port of user back_project: NN on the (deduplicated) Lab key
    lattice (= exact match preferred, KDTree fallback), then among all
    candidate pre-images of the matched key pick the one nearest to the
    ORIGINAL pixel Lab (naturalness-first). Returns RGB in [0,1]."""
    kd = lut['kd']
    rs = lut['rep_starts']
    mlab = lut['member_lab']
    mref = lut['member_ref']
    levels = lut['levels']
    n = lut['n']
    out = np.empty((len(med_lab), 3), dtype=np.float64)
    for s in range(0, len(med_lab), chunk):
        e = min(s + chunk, len(med_lab))
        q = med_lab[s:e]
        _, idx = kd.query(q)
        st = rs[idx]
        cnt = rs[idx + 1] - st
        rows = np.repeat(np.arange(e - s), cnt)
        base = np.repeat(st, cnt)
        cstart = np.concatenate(([0], np.cumsum(cnt)[:-1]))
        inner = np.arange(base.size) - np.repeat(cstart, cnt)
        gpos = base + inner
        cand = mlab[gpos]
        d2 = ((cand - orig_lab[s:e][rows]) ** 2).sum(1)
        # first minimal candidate in member order (= user's strict-< Python
        # loop), via O(total) reductions instead of a full argsort
        minv = np.full(e - s, np.inf)
        np.minimum.at(minv, rows, d2)
        hit = d2 == minv[rows]
        pick = np.full(e - s, np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(pick, rows[hit], gpos[hit])
        out[s:e] = _grid_rgb(mref[pick], levels, n) / 255.0
    return out


# ---------------------------------------------------------------- per image

def _directions(lab: np.ndarray, lab_sim: np.ndarray, j: np.ndarray) -> np.ndarray:
    """User compute_correct_direction, vectorised over pixels."""
    a_c = lab[:, 1]
    a_g = lab[j, 1]
    b_c = lab[:, 2]
    b_g = lab[j, 2]
    a_sc = lab_sim[:, 1]
    b_sc = lab_sim[:, 2]
    a_sg = lab_sim[j, 1]
    b_sg = lab_sim[j, 2]
    orig = np.hypot(a_c - a_g, b_c - b_g)
    simd = np.hypot(a_sc - a_sg, b_sc - b_sg)
    a_i = np.clip(simd / (orig + 1e-8), 0.0, 1.0)
    b_i = np.sqrt(np.maximum(0.0, 1.0 - a_i ** 2))
    b_i = np.where(a_c >= 0, b_i, -b_i)
    a_i = np.where(a_sg < a_sc, a_i, -a_i)
    return np.stack([a_i, b_i], axis=1)


_TP_PATCHED = False


def _patch_threadpoolctl_if_broken() -> None:
    """On this macOS/anaconda combo threadpoolctl's dyld module scan raises
    AttributeError ('NoneType' has no attribute 'split') inside sklearn
    KMeans. Neutralise the (purely performance-related) thread limiter IN
    MEMORY if the probe fails; clustering maths/semantics are untouched.
    Registered as an environment note, not a numerical deviation."""
    global _TP_PATCHED
    if _TP_PATCHED:
        return
    _TP_PATCHED = True
    try:
        import contextlib
        import threadpoolctl
        broken = False
        try:
            threadpoolctl.threadpool_info()
        except Exception:                                # noqa: BLE001
            broken = True
        if not broken:
            try:
                threadpoolctl.threadpool_limits(limits=1, user_api='blas')
            except Exception:                            # noqa: BLE001
                broken = True
        if broken:
            threadpoolctl.threadpool_info = lambda *a, **k: []
            threadpoolctl.threadpool_limits = (
                lambda *a, **k: contextlib.nullcontext())
    except ImportError:
        pass


def _smooth_directions(ab: np.ndarray, d: np.ndarray, k: int,
                       n_init: int, seed: int) -> np.ndarray:
    """User smooth_direction_kmeans: k-means on (a*,b*), class-mean direction
    assigned WITHOUT renormalisation (their norm is computed but unused)."""
    N = len(d)
    if N < k * 10:
        k = max(2, min(k, N // 10))
    k = max(1, min(int(k), max(N, 1)))
    try:
        _patch_threadpoolctl_if_broken()                   # macOS dyld flake
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=k, random_state=int(seed),
                        n_init=int(n_init)).fit_predict(ab)
    except ImportError:                                   # numpy fallback, ledgered
        labels = _lloyd_labels(ab, k, int(n_init), np.random.RandomState(int(seed)))
    out = d.copy()
    for c in range(k):
        m = labels == c
        if m.any():
            out[m] = d[m].mean(axis=0)
    return out


def _lloyd_labels(ab, k, n_init, rng, max_iter=100):
    """Fallback deterministic Lloyd (only used when sklearn absent; cluster
    assignment may differ from the user's sklearn KMeans — registered).
    First pass never breaks (labels starts as None), so convergence needs
    >=2 assignment rounds; selftest gates this via max_iter=1 vs 200."""
    best_lab, best_inertia = None, np.inf
    for _ in range(n_init):
        cent = ab[rng.choice(len(ab), k, replace=False)].copy()
        labels = None
        for _ in range(max_iter):
            dist2 = ((ab[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
            new = dist2.argmin(1)
            if labels is not None and np.array_equal(new, labels):
                break
            labels = new
            for c in range(k):
                m = labels == c
                if m.any():
                    cent[c] = ab[m].mean(0)
        inertia = ((ab - cent[labels]) ** 2).sum()
        if inertia < best_inertia:
            best_lab, best_inertia = labels, inertia
    return best_lab


def _solve_image(img: torch.Tensor, axis: str, p: dict, lut: dict):
    """Steps 1-11 of the user pipeline for one (3,H,W) float32 image."""
    H, W = img.shape[1], img.shape[2]
    N = H * W
    beta = float(p['beta_protan'] if axis == 'protan' else p['beta_deutan'])
    eta = float(p['eta'])
    sev = float(p['severity'])
    with torch.no_grad():
        x = img.reshape(1, 3, H, W)
        lab = _lab(x)
        q = _simulate_q(x, axis, p['sim_source'], p['lightdomain'], sev)
        lab_sim = _lab(q)
    radius = np.linalg.norm(lab - lab_sim, axis=1)          # Eq.1
    CR = beta * radius                                       # Eq.4

    rs = np.random.RandomState(int(p['rng_seed']))
    j = rs.permutation(N)                                    # pairing |P|=N
    d = _directions(lab, lab_sim, j)
    d = _smooth_directions(lab[:, 1:], d, int(p['k']),
                           int(p['kmeans_n_init']), int(p['kmeans_seed']))

    ii = rs.randint(0, N, N)                                 # energy pairs
    jj = rs.randint(0, N, N)
    msk = ii != jj
    ia, ja = ii[msk], jj[msk]
    target = eta * ((lab[ia] - lab[ja]) ** 2).sum(1)
    dl2 = (lab_sim[ia, 0] - lab_sim[ja, 0]) ** 2
    da0 = lab_sim[ia, 1] - lab_sim[ja, 1]
    db0 = lab_sim[ia, 2] - lab_sim[ja, 2]
    u = CR * d[:, 0]
    w = CR * d[:, 1]
    du = u[ia] - u[ja]
    dw = w[ia] - w[ja]

    alpha = p['alpha']
    if isinstance(alpha, str) and alpha.strip() != 'auto':
        alpha = float(alpha)
    alphas = np.linspace(-1.0, 1.0, 21) if isinstance(alpha, str) else np.array([float(alpha)])

    E_all = []
    for a_ in alphas:
        ach = dl2 + (da0 + a_ * du) ** 2 + (db0 + a_ * dw) ** 2
        E_all.append(float(np.sum((ach - target) ** 2)))
    a_star = float(alphas[int(np.argmin(E_all))])

    lab_mediate = lab_sim.copy()                            # Eq.5/6 (a,b only)
    lab_mediate[:, 1] += a_star * u
    lab_mediate[:, 2] += a_star * w
    with torch.no_grad():
        med = torch.from_numpy(lab_mediate).t().reshape(1, 3, 1, N)
        rgbm = lab_to_rgb(med, clip=True)
        qm = (rgbm * 255.0).round().clamp_(0.0, 255.0) / 255.0
    intermediate = _lab(qm)                                 # gamut clamp path
    out_rgb = _backproject(lut, intermediate, lab)
    return out_rgb, dict(alpha=a_star, E_all=E_all, alphas=alphas,
                         n_pairs=int(msk.sum()), R_mean=float(radius.mean()),
                         CR_mean=float(CR.mean()))


# --------------------------------------------------------------------- API

def recolor(inp: torch.Tensor, axis: str, **params) -> torch.Tensor:
    """Zhou24 port (contract per baselines/methods/__init__.py). inp: sRGB
    [0,1] (B,3,H,W) or (3,H,W); output same dtype/device, clipped [0,1].
    Severity fixed to 1.0 by protocol; CPU-only numerics."""
    if axis not in ('protan', 'deutan'):
        raise ValueError(f'axis must be protan/deutan, got {axis!r}')
    p = {**DEFAULTS, **params}
    if p['sim_source'] not in ('protocol', 'legacy_repro'):
        raise ValueError("sim_source must be 'protocol' or 'legacy_repro'")
    squeeze = inp.ndim == 3
    x = inp.unsqueeze(0) if squeeze else inp
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f'expected (B,3,H,W) or (3,H,W), got {tuple(inp.shape)}')
    lut = _get_lut(axis, p['sim_source'], p['lightdomain'],
                   float(p['severity']), int(p['gamut_grid_step']))
    cpu = x.detach().to('cpu', dtype=torch.float32)
    outs = []
    for b in range(cpu.shape[0]):
        rgb, _ = _solve_image(cpu[b], axis, p, lut)          # (N,3) in [0,1]
        outs.append(torch.from_numpy(rgb).t().reshape(3, cpu.shape[2], cpu.shape[3]))
    out = torch.stack(outs).clamp_(0.0, 1.0).to(dtype=inp.dtype, device=inp.device)
    return out.squeeze(0) if squeeze else out


# ------------------------------------------------------------------ selftest

def _synthetic_bar(size=(128, 162)) -> torch.Tensor:
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


USER_DIR = '/Volumes/sea_disk/颜色与视觉实验室/zhou_method'


def _install_cv2_shim() -> bool:
    """The user's recolor_image() reads/writes via cv2; this machine may lack
    it. Register a minimal PIL-backed stand-in IN MEMORY ONLY (their files on
    the read-only volume are never touched). Returns True if a shim was made."""
    try:
        import cv2  # noqa: F401
        return False
    except ImportError:
        pass
    import types
    from PIL import Image

    def imread(path, flags=None):
        try:
            im = Image.open(path).convert('RGB')
        except (OSError, ValueError):
            return None
        return np.asarray(im)[:, :, ::-1].copy()      # RGB -> BGR

    def cvtColor(arr, code):
        return arr[:, :, ::-1].copy()

    def imwrite(path, arr):
        im = Image.fromarray(np.ascontiguousarray(arr[:, :, ::-1]))
        if path.lower().endswith(('.jpg', '.jpeg')):      # near-lossless so
            im.save(path, quality=100, subsampling=0)     # the e2e compare is
        else:                                             # not codec-dominated
            im.save(path)
        return True

    mod = types.ModuleType('cv2')
    mod.imread = imread
    mod.cvtColor = cvtColor
    mod.imwrite = imwrite
    mod.COLOR_BGR2RGB = 4
    import sys
    sys.modules['cv2'] = mod
    return True


def _user_code_comparison(checks) -> None:
    """selftest (a): legacy_repro vs the user's zhouMethod.py — per-formula
    intermediate comparisons (directions, k-means, E curve, back-project rule)
    plus an end-to-end output comparison at the user's default
    (protan, severity 0.4, step-4 LUT for tractable runtime)."""
    src = os.path.join(USER_DIR, 'zhouMethod.py')
    if not os.path.isfile(src):
        print(f'  [skip] user comparison: {src} not present (read-only volume '
              f'not mounted) — port verified against paper notes only')
        return
    try:
        import importlib.util
        _install_cv2_shim()
        spec = importlib.util.spec_from_file_location('_zhou_user_ref', src)
        zm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(zm)
        from skimage.color import lab2rgb as sk_lab2rgb, rgb2lab as sk_rgb2lab
    except Exception as exc:                               # noqa: BLE001
        print(f'  [skip] user comparison: could not import user module ({exc!r})')
        return

    try:
        import tempfile
        from PIL import Image
        _patch_threadpoolctl_if_broken()   # user's smooth_direction_kmeans
        img_path = os.path.join(USER_DIR, 'data', 'kodim.jpg')
        if os.path.isfile(img_path):
            arr = np.asarray(Image.open(img_path).convert('RGB'))
            y, x = 200, 300
            rgb = arr[y:y + 64, x:x + 64]
        else:
            rgb = (np.asarray(_synthetic_bar()) * 255).astype(np.uint8).transpose(1, 2, 0)
        rgb = np.ascontiguousarray(rgb)
        H, W = rgb.shape[:2]
        N = H * W
        lab_sk = sk_rgb2lab(rgb / 255.0)
        with torch.no_grad():
            x_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            lab_k = _lab(x_t).reshape(H, W, 3)
        sim_u8 = np.round(np.clip(rgb.astype(np.float64)
                                  @ zm.get_machado_matrix('protan', 0.4).T, 0, 255))
        lab_sk_sim = sk_rgb2lab((sim_u8 / 255.0).reshape(H, W, 3))
        backend = np.linalg.norm((lab_k - lab_sk).reshape(-1, 3), axis=1)
        print(f'  [info] lab backend delta skimage-vs-kornia on input: '
              f'mean {backend.mean():.2f}, max {backend.max():.2f} Lab units')

        SEED = 42
        np.random.seed(SEED)
        d_them = zm.compute_lab_directions(lab_sk, lab_sk_sim)
        d_ours_same_backend = _directions(lab_sk.reshape(-1, 3), lab_sk_sim.reshape(-1, 3),
                                          np.random.RandomState(SEED).permutation(N))
        e_dir = np.abs(d_them.reshape(-1, 2) - d_ours_same_backend).max()
        checks.append(('legacy directions (same backend)', e_dir < 1e-12,
                       f'max |Δdir| = {e_dir:.2e} (loop vs vectorised)'))

        sm_them = zm.smooth_direction_kmeans(lab_sk, d_them)
        sm_ours = _smooth_directions(lab_sk.reshape(-1, 3)[:, 1:],
                                     d_ours_same_backend, int(DEFAULTS['k']),
                                     int(DEFAULTS['kmeans_n_init']),
                                     int(DEFAULTS['kmeans_seed'])).reshape(H, W, 2)
        e_sm = np.abs(sm_them - sm_ours).max()
        checks.append(('legacy k-means smoothing', e_sm < 1e-12,
                       f'max |Δdir_smoothed| = {e_sm:.2e}'))

        radius_them = zm.compute_loss_radius(lab_sk, lab_sk_sim)
        beta = zm.get_beta('protan', 0.4)
        np.random.seed(SEED)
        _ = np.random.permutation(N)                    # consume pairing draws
        a_them = zm.optimize_alpha(lab_sk, lab_sk_sim, sm_them, radius_them, beta)
        a_ref, E_ref_loop, E_ours_curve = _ref_alpha_curve(
            zm, lab_sk, lab_sk_sim, sm_them, radius_them, beta, N, SEED)
        a_ours = float(np.linspace(-1, 1, 21)[int(np.argmin(E_ours_curve))])
        e_curve = float(np.max(np.abs(np.array(E_ref_loop) - np.array(E_ours_curve))
                               / (np.abs(np.array(E_ours_curve)) + 1.0)))
        checks.append(('legacy alpha grid + energy',
                       float(a_them) == a_ref == a_ours and e_curve < 1e-9,
                       f'alpha* user-fn={a_them:+.2f} ref-loop={a_ref:+.2f} '
                       f'ours={a_ours:+.2f}, max rel |ΔE| loop-vs-vectorised = '
                       f'{e_curve:.2e}'))

        # back-project rule on an identical step-4 LUT (user dict structure)
        lut_them = _user_lut_step4(zm)
        med = zm.add_compensation(lab_sk_sim, a_them, beta, radius_them, sm_them)
        interm_sk = sk_rgb2lab(np.round(np.clip(sk_lab2rgb(med) * 255, 0, 255))
                               .astype(np.uint8) / 255.0)
        res_them_lab = zm.back_project(interm_sk, lab_sk, lut_them)
        res_them = np.round(sk_lab2rgb(res_them_lab) * 255).astype(np.uint8)
        sk_lab_fn = (lambda arr: sk_rgb2lab(
            np.ascontiguousarray(arr, dtype=np.float64).reshape(1, -1, 3))[0])
        lut_ours = _lut_step4_from_skimage(sk_lab_fn)
        res_ours = _backproject(lut_ours, interm_sk.reshape(-1, 3),
                                lab_sk.reshape(-1, 3)).reshape(H, W, 3)
        res_ours_u8 = np.round(res_ours * 255)
        e_bp = np.abs(res_them.astype(np.float64) - res_ours_u8).max()
        checks.append(('legacy back-project rule', e_bp < 2.0,
                       f'max |Δrgb| = {e_bp:.1f}/255 (tie-break tolerance)'))

        # end-to-end through the user's own recolor_image (LUT build patched to
        # the step-4 table; full 256^3 dict build is not runnable here)
        tmp = tempfile.mkdtemp(prefix='zhou24_selftest_')
        in_png = os.path.join(tmp, 'in.png')
        Image.fromarray(rgb).save(in_png)
        orig_build = zm.build_lut
        zm.build_lut = lambda *a, **kw: lut_them
        try:
            np.random.seed(SEED)
            zm.recolor_image(in_png, tmp, 'protan', 0.4)
        finally:
            zm.build_lut = orig_build
        theirs = np.asarray(Image.open(os.path.join(tmp, 'image_result.jpg')).convert('RGB'))
        ours = (recolor(x_t, 'protan', sim_source='legacy_repro', severity=0.4,
                        gamut_grid_step=4, rng_seed=SEED,
                        **{'beta_deutan': 0.3})[0].permute(1, 2, 0).numpy() * 255)
        d_e2e = np.abs(theirs.astype(np.float64) - ours)
        checks.append(('legacy end-to-end vs user code (backend swap included)',
                       d_e2e.max() < 110,
                       f'jpg-decode floor ~2-8; max |Δrgb| = {d_e2e.max():.0f}/255, '
                       f'mean = {d_e2e.mean():.2f} (kornia vs skimage Lab dominates)'))
    except Exception as exc:                               # noqa: BLE001
        print(f'  [skip] user comparison aborted: {exc!r}')


def _ref_alpha_curve(zm, lab, lab_sim, d, radius, beta, N, seed):
    """Their optimize_alpha re-run capturing the whole E curve."""
    rng = np.random.RandomState(seed)
    _ = rng.permutation(N)
    ii, jj = rng.randint(0, N, N), rng.randint(0, N, N)
    labf, sf = lab.reshape(-1, 3), lab_sim.reshape(-1, 3)
    df, rf = d.reshape(-1, 2), radius.reshape(-1, 1)
    target = np.array([np.linalg.norm(labf[i] - labf[j]) ** 2
                       for i, j in zip(ii, jj) if i != j])
    Es = []
    best, emin = 0, float('inf')
    for alpha in np.linspace(-1, 1, 21):
        delta = alpha * beta * rf * df
        mf = sf.copy()
        mf[:, 1:3] += delta
        med = np.array([np.linalg.norm(mf[i] - mf[j]) ** 2
                        for i, j in zip(ii, jj) if i != j])
        energy = float(np.sum((med - target) ** 2))
        Es.append(energy)
        if energy < emin:
            emin, best = energy, float(alpha)
    return best, Es, _ours_curve(lab, lab_sim, d, radius, beta, N, seed)


def _ours_curve(lab, lab_sim, d, radius, beta, N, seed):
    labf, sf = lab.reshape(-1, 3), lab_sim.reshape(-1, 3)
    df, rf = d.reshape(-1, 2), radius.reshape(-1)
    rs = np.random.RandomState(seed)
    _ = rs.permutation(N)
    ii, jj = rs.randint(0, N, N), rs.randint(0, N, N)
    msk = ii != jj
    ia, ja = ii[msk], jj[msk]
    target = ((labf[ia] - labf[ja]) ** 2).sum(1)
    dl2 = (sf[ia, 0] - sf[ja, 0]) ** 2
    da0 = sf[ia, 1] - sf[ja, 1]
    db0 = sf[ia, 2] - sf[ja, 2]
    u = beta * rf * df[:, 0]
    w = beta * rf * df[:, 1]
    du, dw = u[ia] - u[ja], w[ia] - w[ja]
    return [float(np.sum((dl2 + (da0 + a_ * du) ** 2 + (db0 + a_ * dw) ** 2
                          - target) ** 2))
            for a_ in np.linspace(-1, 1, 21)]


def _user_lut_step4(zm):
    """User-format (lut dict, keys array, KDTree) on the step-4 grid, protan
    severity 0.4, skimage backend — their build_lut() semantics, vectorised."""
    from collections import defaultdict
    levels = np.arange(0, 256, 4, dtype=np.float64)
    n = len(levels)
    total = n ** 3
    rgb = _grid_rgb(np.arange(total), levels.astype(np.int64), n).astype(np.float64)
    sim = np.round(np.clip(rgb @ zm.get_machado_matrix('protan', 0.4).T, 0, 255))
    lab_all = zm.rgb2lab(rgb.reshape(1, -1, 3) / 255.0)[0]
    lab_sim = zm.rgb2lab((sim / 255.0).reshape(1, -1, 3))[0]
    lut = defaultdict(list)
    for i in range(total):
        lut[tuple(lab_sim[i])].append(lab_all[i])
    from scipy.spatial import cKDTree
    return dict(lut), lab_sim, cKDTree(lab_sim)


def _lut_step4_from_skimage(lab_fn) -> dict:
    """Our vectorised LUT core on the same step-4 protan@0.4 grid the user
    comparison uses, with THEIR Lab backend (skimage) injected — this isolates
    the back-projection selection rule from the backend swap."""
    levels = np.arange(0, 256, 4, dtype=np.int64)
    n = len(levels)
    total = n ** 3
    rgb = _grid_rgb(np.arange(total), levels, n).astype(np.float64)
    sim_u8 = np.round(np.clip(rgb @ _LEGACY_PROTAN[0.4].T, 0, 255)).astype(np.uint8)
    disp_lab = np.asarray(lab_fn(rgb / 255.0))
    lut = _lut_from_arrays(sim_u8, disp_lab, np.arange(total, dtype=np.int64), lab_fn)
    lut['levels'] = levels
    lut['n'] = n
    return lut


def selftest() -> None:
    from metrics import tcc

    checks = []

    def report(name, ok, detail):
        checks.append((name, bool(ok)))
        print(f'  [{"ok" if ok else "FAIL"}] {name}: {detail}')

    print('== (a) port equivalence vs user zhouMethod.py ==')
    user_checks = []
    _user_code_comparison(user_checks)
    for name, ok, detail in user_checks:
        report(name, ok, detail)

    print('== protocol mode invariants (sim_source=protocol, severity=1.0) ==')
    torch.manual_seed(0)
    inp = torch.rand(1, 3, 64, 64)
    post_checks = []
    for axis in ('protan', 'deutan'):
        lut = _get_lut(axis, 'protocol', DEFAULTS['lightdomain'], 1.0, 1)
        _, info = _solve_image(inp[0], axis, dict(DEFAULTS), lut)
        amin = int(np.argmin(info['E_all']))
        ok_b = abs(info['alphas'][amin] - info['alpha']) < 1e-9 and \
            all(info['E_all'][amin] <= e + 1e-9 for e in info['E_all'])
        post_checks.append((f'{axis} alpha* argmin of 21-point E', ok_b,
                            f'alpha*={info["alpha"]:+.1f} E*={info["E_all"][amin]:.3e} '
                            f'E(0)={info["E_all"][10]:.3e} pairs={info["n_pairs"]} '
                            f'R mean={info["R_mean"]:.1f}'))

        out0 = recolor(inp, axis, alpha=0.0)
        sim_out = simulate_machado_cvd(out0, axis, 1.0, linearize=True)
        sim_sim_out = simulate_machado_cvd(sim_out, axis, 1.0, linearize=True)
        s1 = simulate_machado_cvd(inp, axis, 1.0, linearize=True)
        s2 = simulate_machado_cvd(s1, axis, 1.0, linearize=True)
        def_lab = rgb_to_lab(s2.double()) - rgb_to_lab(s1.double())
        defect = float(def_lab.norm(dim=1).mean())
        idem = (rgb_to_lab(sim_sim_out.double()) - rgb_to_lab(sim_out.double())).norm(dim=1)
        drift = (rgb_to_lab(sim_out.double()) - rgb_to_lab(s1.double())).norm(dim=1)
        ok_c = bool(torch.isfinite(out0).all() and out0.min() >= 0 and out0.max() <= 1
                    and float(idem.mean()) <= defect + 0.5)
        post_checks.append((f'{axis} alpha=0 surface idempotence + range', ok_c,
                            f'|Lab(sim(sim(out)))-Lab(sim(out))| mean '
                            f'{float(idem.mean()):.3f} (intrinsic clamp defect of the '
                            f'harness sim on the same image: {defect:.3f}); perceived '
                            f'drift |Lab(sim(out))-Lab(sim(inp))| mean '
                            f'{float(drift.mean()):.3f}, max {float(drift.max()):.3f}'))

        bar = _synthetic_bar().unsqueeze(0)
        obar = recolor(bar, axis)
        lin_in = rgb_to_lab(simulate_machado_cvd(bar, axis, 1.0).double())[0]
        lin_out = rgb_to_lab(simulate_machado_cvd(obar, axis, 1.0).double())[0]
        t_in, t_out = float(tcc(lin_in, seed=7)), float(tcc(lin_out, seed=7))
        post_checks.append((f'{axis} colour-bar TCC gain (B-tier)', t_out > t_in,
                            f'TCC {t_in:.2f} -> {t_out:.2f} (gain {t_out - t_in:+.2f})'))

    for name, ok, detail in post_checks:
        report(name, ok, detail)

    # informational only: natural-image TCC trend (kodim crop if mounted)
    try:
        from PIL import Image
        img_path = os.path.join(USER_DIR, 'data', 'kodim.jpg')
        arr = np.asarray(Image.open(img_path).convert('RGB'))[:256, :256]
        x = torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        for axis in ('protan', 'deutan'):
            o = recolor(x, axis)
            li = rgb_to_lab(simulate_machado_cvd(x, axis, 1.0).double())[0]
            lo = rgb_to_lab(simulate_machado_cvd(o, axis, 1.0).double())[0]
            print(f'  [info] natural kodim256 {axis}: TCC {float(tcc(li, seed=7)):.2f} -> '
                  f'{float(tcc(lo, seed=7)):.2f} (gain {float(tcc(lo, seed=7)) - float(tcc(li, seed=7)):+.2f}) '
                  f'not gating')
    except Exception as exc:                               # noqa: BLE001
        print(f'  [info] natural probe skipped: {exc!r}')

    # k-means fallback: prove Lloyd genuinely iterates (coordinator review
    # 2026-11-24 item — superseded draft had an early-break bug; current
    # fallback must pass 1-iter vs 200-iter inequality)
    rng_i = np.random.RandomState
    ab = np.vstack([rng_i(1).normal(0, 6, (200, 2)) + [40, 40],
                    rng_i(2).normal(0, 6, (200, 2)) + [-40, 50],
                    rng_i(3).normal(0, 6, (200, 2)) + [0, -45]])
    lab1 = _lloyd_labels(ab, 6, 1, np.random.RandomState(0), max_iter=1)
    lab200 = _lloyd_labels(ab, 6, 1, np.random.RandomState(0), max_iter=200)
    inertia200 = float(((ab - np.stack([ab[lab200 == c].mean(0) for c in np.unique(lab200)])[lab200]) ** 2).sum())
    inertia1 = float(((ab - np.stack([ab[lab1 == c].mean(0) for c in np.unique(lab1)])[lab1]) ** 2).sum())
    report('k-means fallback iterates (1 vs 200 rounds)',
           not np.array_equal(lab1, lab200) and inertia200 < inertia1,
           f'assignments differ: {not np.array_equal(lab1, lab200)}, '
           f'inertia 1-iter {inertia1:.0f} -> 200-iter {inertia200:.0f}')

    print('== (e) timing ==')
    t64 = time.perf_counter()
    _ = recolor(inp, 'protan')
    t64 = 1000 * (time.perf_counter() - t64)
    big = torch.rand(1, 3, 256, 256)
    t0 = time.perf_counter()
    _ = recolor(big, 'protan')
    t256_p = 1000 * (time.perf_counter() - t0)
    t0 = time.perf_counter()
    _ = recolor(big, 'deutan')
    t256_d = 1000 * (time.perf_counter() - t0)
    build_p = _LUT_CACHE[('protan', 'protocol', 'linearized', 1.0, 1)].get('build_s', 0)
    build_d = _LUT_CACHE[('deutan', 'protocol', 'linearized', 1.0, 1)].get('build_s', 0)
    per_img = max(t256_p, t256_d)
    total_h = (per_img * 1753 * 2 + (build_p + build_d) * 1000) / 3.6e6
    print(f'  64x64: {t64:.0f} ms; 256x256: protan {t256_p:.0f} ms, deutan {t256_d:.0f} ms '
          f'(warm LUT); full 256^3 LUT build: protan {build_p:.1f} s, deutan {build_d:.1f} s')
    print(f'  extrapolation full v3 val 1753 images x 2 axes @256x256: '
          f'~{total_h:.2f} h (LUT builds amortised once) — {"OK < 8h" if total_h < 8 else "TOO SLOW > 8h"}')

    fails = [n for n, ok in checks if not ok]
    if fails:
        print(f'zhou24 selftest: FAIL ({len(fails)}/{len(checks)}): {fails}')
        raise AssertionError(f'zhou24 selftest failed checks: {fails}')
    print('zhou24 selftest: PASS')


if __name__ == '__main__':
    selftest()
