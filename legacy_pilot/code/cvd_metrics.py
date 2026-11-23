"""Losses and per-image metrics for input-preserving CVD recoloring."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import deltaE_ciede2000, rgb2lab

from cvd_simulation import simulate_cvd, srgb_to_linear


def rgb_to_xyz(rgb: torch.Tensor) -> torch.Tensor:
    matrix = torch.tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=rgb.dtype,
        device=rgb.device,
    )
    return torch.einsum("bchw,dc->bdhw", srgb_to_linear(rgb), matrix)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB to CIELAB (D65) in conventional L*, a*, b* units."""
    xyz = rgb_to_xyz(rgb)
    white = torch.tensor([0.95047, 1.0, 1.08883], dtype=rgb.dtype, device=rgb.device)
    xyz = xyz / white.view(1, 3, 1, 1)
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    cube_root = torch.exp((1.0 / 3.0) * torch.log(xyz.clamp_min(1e-8)))
    selector = (xyz > epsilon).to(xyz.dtype)
    f = selector * cube_root + (1.0 - selector) * ((kappa * xyz + 16.0) / 116.0)
    lightness = 116.0 * f[:, 1:2] - 16.0
    a = 500.0 * (f[:, 0:1] - f[:, 1:2])
    b = 200.0 * (f[:, 1:2] - f[:, 2:3])
    return torch.cat([lightness, a, b], dim=1)


def local_contrast(x: torch.Tensor) -> torch.Tensor:
    """Four-neighbour vector-gradient magnitude with stable derivatives."""
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    gx = F.pad((dx.square().sum(1, keepdim=True) + 1e-6).sqrt(), (0, 1, 0, 0))
    gy = F.pad((dy.square().sum(1, keepdim=True) + 1e-6).sqrt(), (0, 0, 0, 1))
    return 0.5 * (gx + gy)


def confusion_mask(
    rgb: torch.Tensor,
    cvd_axis: str,
    severity: float,
    simulator: str = "machado",
    relative_threshold: float = 0.12,
) -> torch.Tensor:
    """Soft, input-fixed mask of local chroma contrast lost under simulation."""
    normal_ab = rgb_to_lab(rgb)[:, 1:3]
    simulated_ab = rgb_to_lab(
        simulate_cvd(rgb, cvd_type=cvd_axis, severity=severity, simulator=simulator)
    )[:, 1:3]
    normal = local_contrast(normal_ab)
    deficient = local_contrast(simulated_ab)
    relative_loss = (normal - deficient) / (normal + 2.0)
    mask = torch.sigmoid((relative_loss - relative_threshold) * 12.0)
    return F.avg_pool2d(mask, 5, stride=1, padding=2).clamp(0.0, 1.0)


def ssim_luma_per_image(x_l: torch.Tensor, y_l: torch.Tensor) -> torch.Tensor:
    """Windowed SSIM on L*/100, returned per image."""
    x_l, y_l = x_l / 100.0, y_l / 100.0
    c1, c2 = 0.01**2, 0.03**2
    mu_x = F.avg_pool2d(x_l, 7, 1, 3)
    mu_y = F.avg_pool2d(y_l, 7, 1, 3)
    sig_x = F.avg_pool2d(x_l.square(), 7, 1, 3) - mu_x.square()
    sig_y = F.avg_pool2d(y_l.square(), 7, 1, 3) - mu_y.square()
    sig_xy = F.avg_pool2d(x_l * y_l, 7, 1, 3) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sig_x + sig_y + c2)
    )
    return score.clamp(-1.0, 1.0).flatten(1).mean(1)


def differentiable_losses(
    inp: torch.Tensor,
    out: torch.Tensor,
    mask: torch.Tensor,
    cvd_axis: str,
    severity: float,
    simulator: str,
    decouple_loss: torch.Tensor,
    *,
    w_structure: float = 1.0,
    w_fidelity: float = 1.5,
    w_cvd: float = 1.8,
    w_decouple: float = 0.03,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Training objective; every advertised term contributes a gradient."""
    lab_i, lab_o = rgb_to_lab(inp), rgb_to_lab(out)
    l_i, l_o = lab_i[:, 0:1], lab_o[:, 0:1]
    ab_i, ab_o = lab_i[:, 1:3], lab_o[:, 1:3]

    structure = 0.7 * (1.0 - ssim_luma_per_image(l_i, l_o).mean())
    structure = structure + 0.3 * F.l1_loss(local_contrast(l_i), local_contrast(l_o)) / 20.0

    nonconf = (1.0 - mask).clamp_min(0.05)
    local_fidelity = (nonconf * (ab_i - ab_o).abs()).sum() / (2.0 * nonconf.sum() + 1e-6)
    # Differentiable global statistics replace the previous detached histogram.
    dims = (0, 2, 3)
    stats = F.l1_loss(out.mean(dims), inp.mean(dims)) + F.l1_loss(
        out.std(dims, unbiased=False), inp.std(dims, unbiased=False)
    )
    fidelity = local_fidelity / 35.0 + 0.25 * stats

    cvd_i = rgb_to_lab(simulate_cvd(inp, cvd_axis, severity, simulator))[:, 1:3]
    cvd_o = rgb_to_lab(simulate_cvd(out, cvd_axis, severity, simulator))[:, 1:3]
    contrast_i, contrast_o = local_contrast(cvd_i), local_contrast(cvd_o)
    target_margin = 0.04 * mask * (contrast_i + 2.0)
    cvd = (mask * F.relu(contrast_i + target_margin - contrast_o)).sum() / (
        mask.sum() + 1e-6
    )
    cvd = cvd / 20.0

    total = (
        w_structure * structure
        + w_fidelity * fidelity
        + w_cvd * cvd
        + w_decouple * decouple_loss
    )
    logs = {
        "loss": float(total.detach().cpu()),
        "L_structure": float(structure.detach().cpu()),
        "L_fidelity": float(fidelity.detach().cpu()),
        "L_cvd": float(cvd.detach().cpu()),
        "L_decouple": float(decouple_loss.detach().cpu()),
    }
    return total, logs


def _weighted_mean_per_image(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (value * weight).flatten(1).sum(1) / (weight.flatten(1).sum(1) + 1e-6)


def _delta_e00_per_image(
    inp: torch.Tensor, out: torch.Tensor, mask: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    inp_np = inp.detach().cpu().permute(0, 2, 3, 1).numpy()
    out_np = out.detach().cpu().permute(0, 2, 3, 1).numpy()
    mask_np = mask.detach().cpu().squeeze(1).numpy()
    all_values, nonconf_values = [], []
    for x, y, m in zip(inp_np, out_np, mask_np):
        delta = deltaE_ciede2000(rgb2lab(x), rgb2lab(y))
        all_values.append(float(delta.mean()))
        weights = np.clip(1.0 - m, 0.05, 1.0)
        nonconf_values.append(float(np.sum(delta * weights) / np.sum(weights)))
    return np.asarray(all_values), np.asarray(nonconf_values)


@torch.no_grad()
def batch_metrics(
    inp: torch.Tensor,
    out: torch.Tensor,
    raw_out: torch.Tensor,
    mask: torch.Tensor,
    cvd_axis: str,
    severity: float,
    simulator: str,
) -> Dict[str, np.ndarray]:
    """Compute explicitly named, per-image outcomes for paired statistics."""
    lab_i, lab_o = rgb_to_lab(inp), rgb_to_lab(out)
    cvd_i = rgb_to_lab(simulate_cvd(inp, cvd_axis, severity, simulator))[:, 1:3]
    cvd_o = rgb_to_lab(simulate_cvd(out, cvd_axis, severity, simulator))[:, 1:3]
    contrast_i, contrast_o = local_contrast(cvd_i), local_contrast(cvd_o)
    roi_input = _weighted_mean_per_image(contrast_i, mask)
    roi_output = _weighted_mean_per_image(contrast_o, mask)
    delta_e, delta_e_nonconf = _delta_e00_per_image(inp, out, mask)
    ssim = ssim_luma_per_image(lab_i[:, 0:1], lab_o[:, 0:1])
    gamut = ((raw_out < 0.0) | (raw_out > 1.0)).float().flatten(1).mean(1)
    change = (out - inp).abs().flatten(1).mean(1)
    return {
        "cvd_contrast_input": roi_input.cpu().numpy(),
        "cvd_contrast_output": roi_output.cpu().numpy(),
        "cvd_contrast_gain": (roi_output - roi_input).cpu().numpy(),
        "cvd_contrast_ratio": (roi_output / (roi_input + 1e-6)).cpu().numpy(),
        "delta_e00_mean": delta_e,
        "delta_e00_nonconf": delta_e_nonconf,
        "ssim_luma": ssim.cpu().numpy(),
        "gamut_preclip_rate": gamut.cpu().numpy(),
        "mean_abs_change": change.cpu().numpy(),
        "mask_ratio": mask.flatten(1).mean(1).cpu().numpy(),
    }
