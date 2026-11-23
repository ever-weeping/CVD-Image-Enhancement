"""Compact condition-aware recoloring models and ablation definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = 4 if out_channels % 4 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLM(nn.Module):
    """Small axis–simulator-control conditioner initialized as identity."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.affine = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 2 * channels))
        nn.init.zeros_(self.affine[-1].weight)
        nn.init.zeros_(self.affine[-1].bias)

    def forward(self, feature: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(condition).chunk(2, dim=1)
        gamma = 0.25 * torch.tanh(gamma).view(feature.shape[0], -1, 1, 1)
        beta = 0.10 * torch.tanh(beta).view(feature.shape[0], -1, 1, 1)
        return feature * (1.0 + gamma) + beta


class BaselineRecolorNet(nn.Module):
    """Single-stream conditional U-Net used as a parameter-matched baseline."""

    def __init__(self, width: int = 20) -> None:
        super().__init__()
        self.enc1 = ConvBlock(4, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.condition = FiLM(width * 2)
        self.mid = ConvBlock(width * 2, width * 2)
        self.dec = ConvBlock(width * 3, width)
        self.out = nn.Conv2d(width, 3, 3, padding=1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        e1 = self.enc1(torch.cat([x, mask], dim=1))
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        mid = self.mid(self.condition(e2, condition))
        up = F.interpolate(mid, size=x.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.dec(torch.cat([up, e1], dim=1))
        delta = 0.18 * torch.tanh(self.out(decoded)) * (0.15 + 0.85 * mask)
        raw = x + delta
        return raw.clamp(0.0, 1.0), {"shared": mid}, raw


class DisentangledRecolorNet(nn.Module):
    """Shared encoder plus structure, fidelity and confusion-aware branches."""

    def __init__(
        self,
        width: int = 16,
        use_structure: bool = True,
        use_fidelity: bool = True,
        use_confusion: bool = True,
        use_mask: bool = True,
    ) -> None:
        super().__init__()
        if not any((use_structure, use_fidelity, use_confusion)):
            raise ValueError("At least one branch must remain active")
        self.use_structure = use_structure
        self.use_fidelity = use_fidelity
        self.use_confusion = use_confusion
        self.use_mask = use_mask
        self.shared1 = ConvBlock(4, width)
        self.shared2 = ConvBlock(width, width * 2)
        self.condition = FiLM(width * 2)
        self.structure = ConvBlock(width * 2, width)
        self.fidelity = ConvBlock(width * 2, width)
        self.confusion = ConvBlock(width * 2, width)
        self.fuse = ConvBlock(width, width)
        self.out = nn.Conv2d(width, 3, 3, padding=1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        routing_mask = mask if self.use_mask else torch.ones_like(mask)
        shared1 = self.shared1(torch.cat([x, routing_mask], dim=1))
        shared2 = self.shared2(F.avg_pool2d(shared1, 2))
        shared2 = self.condition(shared2, condition)
        mask_low = F.interpolate(routing_mask, size=shared2.shape[-2:], mode="bilinear", align_corners=False)

        weighted_features = []
        weights = []
        active: Dict[str, torch.Tensor] = {}
        if self.use_structure:
            feature = self.structure(shared2)
            active["F_structure"] = feature
            weighted_features.append(feature)
            weights.append(torch.ones_like(mask_low))
        if self.use_fidelity:
            feature = self.fidelity(shared2)
            active["F_fidelity"] = feature
            weighted_features.append((1.0 - mask_low) * feature)
            weights.append(1.0 - mask_low)
        if self.use_confusion:
            feature = self.confusion(shared2 * (0.25 + 0.75 * mask_low))
            active["F_confusion"] = feature
            weighted_features.append(mask_low * feature)
            weights.append(mask_low)

        numerator = torch.stack(weighted_features, dim=0).sum(dim=0)
        denominator = torch.stack(weights, dim=0).sum(dim=0).clamp_min(1e-4)
        fused = self.fuse(numerator / denominator)
        up = F.interpolate(fused, size=x.shape[-2:], mode="bilinear", align_corners=False)
        delta = 0.18 * torch.tanh(self.out(up)) * (0.15 + 0.85 * routing_mask)
        raw = x + delta
        return raw.clamp(0.0, 1.0), active, raw


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str
    use_structure: bool = True
    use_fidelity: bool = True
    use_confusion: bool = True
    use_decouple: bool = True
    use_mask: bool = True


VARIANTS: Dict[str, Variant] = {
    "baseline": Variant("baseline", "baseline", use_decouple=False),
    "ours_full": Variant("ours_full", "disentangled"),
    "ours_no_structure": Variant("ours_no_structure", "disentangled", use_structure=False),
    "ours_no_fidelity": Variant("ours_no_fidelity", "disentangled", use_fidelity=False),
    "ours_no_confusion": Variant("ours_no_confusion", "disentangled", use_confusion=False),
    "ours_no_decouple": Variant("ours_no_decouple", "disentangled", use_decouple=False),
    "ours_no_mask": Variant("ours_no_mask", "disentangled", use_mask=False),
}


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def matched_baseline_width(full_width: int) -> int:
    """Select a valid baseline width closest to the full model parameter count."""
    target = parameter_count(DisentangledRecolorNet(full_width))
    candidates = range(4, max(64, full_width * 4) + 1, 4)
    return min(candidates, key=lambda width: abs(parameter_count(BaselineRecolorNet(width)) - target))


def make_model(variant: Variant, width: int, baseline_width: Optional[int] = None) -> nn.Module:
    if variant.kind == "baseline":
        return BaselineRecolorNet(baseline_width or matched_baseline_width(width))
    return DisentangledRecolorNet(
        width,
        use_structure=variant.use_structure,
        use_fidelity=variant.use_fidelity,
        use_confusion=variant.use_confusion,
        use_mask=variant.use_mask,
    )


def feature_decouple_loss(features: Dict[str, torch.Tensor]) -> torch.Tensor:
    keys = sorted(key for key in features if key.startswith("F_"))
    if len(keys) < 2:
        reference = next(iter(features.values()))
        return reference.new_tensor(0.0)
    total = next(iter(features.values())).new_tensor(0.0)
    pairs = 0
    for i, key_a in enumerate(keys):
        for key_b in keys[i + 1 :]:
            a, b = features[key_a].flatten(2), features[key_b].flatten(2)
            a = a - a.mean(dim=2, keepdim=True)
            b = b - b.mean(dim=2, keepdim=True)
            # A relatively large explicit floor avoids exploding derivatives for
            # nearly constant branch channels on MPS and in early training.
            a = a / (a.square().sum(dim=2, keepdim=True) + 1e-4).sqrt()
            b = b / (b.square().sum(dim=2, keepdim=True) + 1e-4).sqrt()
            total = total + (a * b).sum(dim=2).abs().mean()
            pairs += 1
    return total / pairs
