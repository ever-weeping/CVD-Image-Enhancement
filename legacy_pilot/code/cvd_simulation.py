"""Differentiable red–green CVD simulation utilities.

The public API uses a deficiency *axis* (``protan`` or ``deutan``) plus a
continuous simulator parameter.  This avoids the common but incorrect practice
of calling every non-zero severity "protanopia" or "deuteranopia": values in
``0 < severity < 1`` are labelled simulated anomalous trichromacy, while
``severity == 1`` is the simulated dichromat endpoint.  The continuous number
is an algorithm parameter, not a clinical diagnosis or measured clinical grade.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import torch

CVDAxis = Literal["protan", "deutan"]
CVDType = Literal["protanopia", "deuteranopia", "protanomaly", "deuteranomaly"]
SimulatorName = Literal["machado", "vienot", "brettel", "fast"]


def validate_severity(severity: float) -> float:
    value = float(severity)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"CVD simulator severity must be in [0, 1], got {severity}")
    return value


def normalise_cvd_axis(cvd_type: str) -> CVDAxis:
    """Map diagnostic/axis aliases to the protan or deutan axis."""
    key = cvd_type.lower().replace("-", "_")
    aliases = {
        "protan": "protan",
        "protanopia": "protan",
        "protanomaly": "protan",
        "protanomalous": "protan",
        "deutan": "deutan",
        "deuteran": "deutan",
        "deuteranopia": "deutan",
        "deuteranomaly": "deutan",
        "deuteranomalous": "deutan",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported red–green CVD axis/type: {cvd_type}")
    return aliases[key]  # type: ignore[return-value]


def cvd_condition_label(cvd_type: str, severity: float) -> str:
    """Return a terminology-safe label for a simulated condition."""
    axis = normalise_cvd_axis(cvd_type)
    value = validate_severity(severity)
    if value == 0.0:
        return "normal_trichromacy"
    if value == 1.0:
        return "protanopia" if axis == "protan" else "deuteranopia"
    return "protanomaly" if axis == "protan" else "deuteranomaly"


def condition_vector(
    cvd_type: str,
    severity: float,
    batch_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Encode ``[is_protan, is_deutan, simulator_severity]`` for conditioning."""
    axis = normalise_cvd_axis(cvd_type)
    severity = validate_severity(severity)
    row = [1.0, 0.0, severity] if axis == "protan" else [0.0, 1.0, severity]
    return torch.tensor(row, dtype=dtype, device=device).view(1, 3).repeat(batch_size, 1)


def _machado_colour_name(cvd_type: str) -> str:
    return "Protanomaly" if normalise_cvd_axis(cvd_type) == "protan" else "Deuteranomaly"


@lru_cache(maxsize=64)
def _machado_matrix_tuple(cvd_type: str, severity: float) -> tuple[tuple[float, ...], ...]:
    try:
        from colour.blindness import matrix_cvd_Machado2009
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Machado simulation requires colour-science; install requirements.txt."
        ) from exc
    severity = validate_severity(severity)
    matrix = matrix_cvd_Machado2009(_machado_colour_name(cvd_type), severity)
    return tuple(tuple(float(v) for v in row) for row in matrix)


def _matrix(rows: list[list[float]] | tuple[tuple[float, ...], ...], x: torch.Tensor) -> torch.Tensor:
    return torch.tensor(rows, dtype=x.dtype, device=x.device)


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    nonlinear = torch.exp(2.4 * torch.log(((x + 0.055) / 1.055).clamp_min(1e-8)))
    selector = (x > 0.04045).to(x.dtype)
    return (1.0 - selector) * (x / 12.92) + selector * nonlinear


def linear_to_srgb(x: torch.Tensor, *, clamp: bool = True) -> torch.Tensor:
    # Negative values cannot be exponentiated by the sRGB transfer function.
    safe = x.clamp_min(0.0)
    nonlinear = 1.055 * torch.exp((1.0 / 2.4) * torch.log(safe.clamp_min(1e-8))) - 0.055
    selector = (safe > 0.0031308).to(safe.dtype)
    out = (1.0 - selector) * (12.92 * safe) + selector * nonlinear
    return out.clamp(0.0, 1.0) if clamp else out


def _prepare(rgb: torch.Tensor) -> tuple[torch.Tensor, bool]:
    squeeze = rgb.ndim == 3
    if squeeze:
        rgb = rgb.unsqueeze(0)
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"Expected RGB tensor shaped (B, 3, H, W), got {tuple(rgb.shape)}")
    return rgb, squeeze


def _apply_linear_matrix(rgb: torch.Tensor, rows: list[list[float]], severity: float) -> torch.Tensor:
    rgb, squeeze = _prepare(rgb)
    severity = validate_severity(severity)
    linear = srgb_to_linear(rgb)
    endpoint = torch.einsum("bchw,dc->bdhw", linear, _matrix(rows, rgb))
    simulated = (1.0 - severity) * linear + severity * endpoint
    out = linear_to_srgb(simulated)
    return out.squeeze(0) if squeeze else out


def simulate_machado_cvd(
    rgb: torch.Tensor,
    cvd_type: str = "deutan",
    severity: float = 1.0,
    linearize: bool = True,
) -> torch.Tensor:
    """Machado et al. (2009) model for anomalous trichromacy through endpoints."""
    rgb, squeeze = _prepare(rgb)
    severity = validate_severity(severity)
    mat = _matrix(_machado_matrix_tuple(cvd_type, severity), rgb)
    working = srgb_to_linear(rgb) if linearize else rgb.clamp(0.0, 1.0)
    simulated = torch.einsum("bchw,dc->bdhw", working, mat)
    out = linear_to_srgb(simulated) if linearize else simulated.clamp(0.0, 1.0)
    return out.squeeze(0) if squeeze else out


_FAST_ENDPOINTS = {
    "protan": [[0.567, 0.433, 0.000], [0.558, 0.442, 0.000], [0.000, 0.242, 0.758]],
    "deutan": [[0.625, 0.375, 0.000], [0.700, 0.300, 0.000], [0.000, 0.300, 0.700]],
}


def simulate_fast_cvd(
    rgb: torch.Tensor,
    cvd_type: str = "deutan",
    severity: float = 1.0,
) -> torch.Tensor:
    """Severity-aware heuristic for smoke tests; not used for paper results."""
    return _apply_linear_matrix(rgb, _FAST_ENDPOINTS[normalise_cvd_axis(cvd_type)], severity)


# Precomputed sRGB/Smith–Pokorny matrices from the public-domain libDaltonLens
# implementation of Viénot, Brettel & Mollon (1999).
_VIENOT_ENDPOINTS = {
    "protan": [[0.11238, 0.88762, 0.0], [0.11238, 0.88762, 0.0], [0.00401, -0.00401, 1.0]],
    "deutan": [[0.29275, 0.70725, 0.0], [0.29275, 0.70725, 0.0], [-0.02234, 0.02234, 1.0]],
}


def simulate_vienot_cvd(
    rgb: torch.Tensor,
    cvd_type: str = "deutan",
    severity: float = 1.0,
) -> torch.Tensor:
    """Viénot et al. (1999) endpoint projection with linear interpolation.

    Intermediate values are robustness-control interpolations, not a clinical
    anomalous-trichromacy model. Formal second-model sensitivity analysis in
    this project therefore uses this backend only at severity 1.
    """
    return _apply_linear_matrix(rgb, _VIENOT_ENDPOINTS[normalise_cvd_axis(cvd_type)], severity)


# Precomputed two-plane transforms from the public-domain libDaltonLens
# implementation of Brettel, Viénot & Mollon (1997).
_BRETTEL = {
    "protan": {
        "m1": [[0.14980, 1.19548, -0.34528], [0.10764, 0.84864, 0.04372], [0.00384, -0.00540, 1.00156]],
        "m2": [[0.14570, 1.16172, -0.30742], [0.10816, 0.85291, 0.03892], [0.00386, -0.00524, 1.00139]],
        "n": [0.00048, 0.00393, -0.00441],
    },
    "deutan": {
        "m1": [[0.36477, 0.86381, -0.22858], [0.26294, 0.64245, 0.09462], [-0.02006, 0.02728, 0.99278]],
        "m2": [[0.37298, 0.88166, -0.25464], [0.25954, 0.63506, 0.10540], [-0.01980, 0.02784, 0.99196]],
        "n": [-0.00281, -0.00611, 0.00892],
    },
}


def simulate_brettel_cvd(
    rgb: torch.Tensor,
    cvd_type: str = "deutan",
    severity: float = 1.0,
) -> torch.Tensor:
    """Brettel et al. (1997) two-plane dichromacy projection."""
    rgb, squeeze = _prepare(rgb)
    severity = validate_severity(severity)
    params = _BRETTEL[normalise_cvd_axis(cvd_type)]
    linear = srgb_to_linear(rgb)
    m1, m2 = _matrix(params["m1"], rgb), _matrix(params["m2"], rgb)
    out1 = torch.einsum("bchw,dc->bdhw", linear, m1)
    out2 = torch.einsum("bchw,dc->bdhw", linear, m2)
    n = torch.tensor(params["n"], dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    choose_first = ((linear * n).sum(dim=1, keepdim=True) >= 0).to(linear.dtype)
    endpoint = choose_first * out1 + (1.0 - choose_first) * out2
    simulated = (1.0 - severity) * linear + severity * endpoint
    out = linear_to_srgb(simulated)
    return out.squeeze(0) if squeeze else out


def simulate_cvd(
    rgb: torch.Tensor,
    cvd_type: str = "deutan",
    severity: float = 1.0,
    simulator: SimulatorName = "machado",
) -> torch.Tensor:
    """Dispatch a differentiable CVD simulator."""
    if simulator == "machado":
        return simulate_machado_cvd(rgb, cvd_type=cvd_type, severity=severity)
    if simulator == "vienot":
        return simulate_vienot_cvd(rgb, cvd_type=cvd_type, severity=severity)
    if simulator == "brettel":
        return simulate_brettel_cvd(rgb, cvd_type=cvd_type, severity=severity)
    if simulator == "fast":
        return simulate_fast_cvd(rgb, cvd_type=cvd_type, severity=severity)
    raise ValueError(f"Unknown CVD simulator: {simulator}")
