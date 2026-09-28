#!/usr/bin/env python3
"""Reproducible pilot experiments for conditional CVD-aware recoloring.

The script deliberately separates simulated anomalous trichromacy
(``0 < severity < 1``) from simulated dichromat endpoints (``severity == 1``),
uses the same split/initial seed/condition schedule for every learned variant,
and writes per-image outcomes before any aggregation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import wilcoxon

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from cvd_metrics import batch_metrics, confusion_mask, differentiable_losses
from cvd_models import VARIANTS, Variant, feature_decouple_loss, make_model, parameter_count
from cvd_simulation import condition_vector, cvd_condition_label, normalise_cvd_axis, simulate_cvd
from runtime import resolve_device, should_pin_memory


METRIC_NAMES = [
    "cvd_contrast_input",
    "cvd_contrast_output",
    "cvd_contrast_gain",
    "cvd_contrast_ratio",
    "delta_e00_mean",
    "delta_e00_nonconf",
    "ssim_luma",
    "gamut_preclip_rate",
    "mean_abs_change",
    "mask_ratio",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ImagePathDataset(Dataset):
    def __init__(self, root: Path, size: int, paths: Optional[Sequence[Path]] = None) -> None:
        self.root = root.resolve()
        self.size = size
        self.paths = list(paths) if paths is not None else sorted(
            path for path in self.root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            crop = min(width, height)
            left, top = (width - crop) // 2, (height - crop) // 2
            image = image.crop((left, top, left + crop, top + crop)).resize(
                (self.size, self.size), Image.Resampling.BICUBIC
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
        try:
            name = str(path.resolve().relative_to(self.root))
        except ValueError:
            name = path.name
        return torch.from_numpy(array).permute(2, 0, 1), name


def make_split(paths: Sequence[Path], train_images: int, eval_images: int, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(len(paths)))
    random.Random(seed).shuffle(indices)
    required = train_images + eval_images
    if len(indices) < required:
        raise ValueError(f"Need {required} images for disjoint train/eval splits, found {len(indices)}")
    return indices[:train_images], indices[train_images:required]


def make_loader(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    device: torch.device,
    workers: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=should_pin_memory(device),
        generator=generator,
    )


def write_dict_rows(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def identity_transform(
    inp: torch.Tensor, mask: torch.Tensor, axis: str, severity: float, simulator: str
) -> tuple[torch.Tensor, torch.Tensor]:
    return inp, inp


def error_compensation_transform(
    inp: torch.Tensor, mask: torch.Tensor, axis: str, severity: float, simulator: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transparent engineering daltonization baseline, not a paper reproduction."""
    simulated = simulate_cvd(inp, axis, severity, simulator)
    error = inp - simulated
    correction = torch.zeros_like(error)
    if normalise_cvd_axis(axis) == "protan":
        correction[:, 1] = 0.70 * error[:, 0]
        correction[:, 2] = 0.70 * error[:, 0]
    else:
        correction[:, 0] = 0.70 * error[:, 1]
        correction[:, 2] = 0.70 * error[:, 1]
    raw = inp + 0.65 * correction * (0.20 + 0.80 * mask)
    return raw.clamp(0.0, 1.0), raw


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def save_panel(
    path: Path,
    inp: torch.Tensor,
    out: torch.Tensor,
    mask: torch.Tensor,
    axis: str,
    severity: float,
    simulator: str,
) -> None:
    panels = [
        ("input", tensor_to_pil(inp)),
        ("input simulated", tensor_to_pil(simulate_cvd(inp, axis, severity, simulator))),
        ("mask", tensor_to_pil(mask.repeat(3, 1, 1))),
        ("recolored", tensor_to_pil(out)),
        ("output simulated", tensor_to_pil(simulate_cvd(out, axis, severity, simulator))),
    ]
    width, height = panels[0][1].size
    canvas = Image.new("RGB", (width * len(panels), height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        canvas.paste(image, (index * width, 24))
        draw.text((index * width + 4, 5), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def train_model(
    model: nn.Module,
    variant: Variant,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
) -> tuple[list[dict], float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = [(axis, severity) for axis in args.axes for severity in args.train_severities]
    condition_rng = random.Random(seed + 173)
    iterator = iter(loader)
    history: list[dict] = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        try:
            inp, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            inp, _ = next(iterator)
        inp = inp.to(device)
        axis, severity = schedule[condition_rng.randrange(len(schedule))]
        mask = confusion_mask(inp, axis, severity, args.train_simulator).detach()
        condition = condition_vector(axis, severity, inp.shape[0], dtype=inp.dtype, device=device)
        out, features, _ = model(inp, mask, condition)
        decouple = feature_decouple_loss(features) if variant.use_decouple else inp.new_tensor(0.0)
        loss, logs = differentiable_losses(
            inp,
            out,
            mask,
            axis,
            severity,
            args.train_simulator,
            decouple,
            w_structure=args.w_structure,
            w_fidelity=args.w_fidelity,
            w_cvd=args.w_cvd,
            w_decouple=args.w_decouple,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at seed={seed}, variant={variant.name}, step={step}: {logs}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        non_finite_gradients = [
            name for name, parameter in model.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        if non_finite_gradients:
            raise FloatingPointError(
                f"Non-finite gradients at seed={seed}, variant={variant.name}, step={step}: "
                + ", ".join(non_finite_gradients[:8])
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if step == 1 or step == args.steps or step % args.log_every == 0:
            row = {"seed": seed, "variant": variant.name, "step": step, "axis": axis, "severity": severity, **logs}
            history.append(row)
            print(
                f"[{seed}:{variant.name}] {step:04d}/{args.steps} {axis} s={severity:.3f} "
                f"loss={logs['loss']:.4f} cvd={logs['L_cvd']:.4f}",
                flush=True,
            )
    synchronize(device)
    return history, time.perf_counter() - started


@torch.no_grad()
def evaluate_variant(
    *,
    name: str,
    seed: int,
    loader: DataLoader,
    dataset_name: str,
    device: torch.device,
    axes: Sequence[str],
    severities: Sequence[float],
    metric_simulator: str,
    mask_simulator: str,
    model: Optional[nn.Module] = None,
    transform: Optional[Callable] = None,
    params: int = 0,
    train_seconds: float = 0.0,
    sample_root: Optional[Path] = None,
    sample_images: int = 0,
    condition_policy: str = "correct",
) -> list[dict]:
    if (model is None) == (transform is None):
        raise ValueError("Pass exactly one of model or transform")
    if model is not None:
        model.eval()
        if condition_policy not in {"correct", "axis_swap", "fixed_s05"}:
            raise ValueError(f"Unknown condition policy: {condition_policy}")
    else:
        condition_policy = "not_applicable"
    rows: list[dict] = []
    for axis in axes:
        for severity in severities:
            saved = 0
            for inp, image_names in loader:
                inp = inp.to(device)
                mask = confusion_mask(inp, axis, severity, mask_simulator).detach()
                synchronize(device)
                started = time.perf_counter()
                if model is not None:
                    condition_axis = (
                        "deutan" if axis == "protan" else "protan"
                    ) if condition_policy == "axis_swap" else axis
                    condition_severity = 0.5 if condition_policy == "fixed_s05" else severity
                    condition = condition_vector(
                        condition_axis,
                        condition_severity,
                        inp.shape[0],
                        dtype=inp.dtype,
                        device=device,
                    )
                    out, _, raw = model(inp, mask, condition)
                else:
                    condition_axis = "not_applicable"
                    condition_severity = -1.0
                    out, raw = transform(inp, mask, axis, severity, mask_simulator)
                synchronize(device)
                elapsed_ms = 1000.0 * (time.perf_counter() - started) / inp.shape[0]
                values = batch_metrics(inp, out, raw, mask, axis, severity, metric_simulator)
                for index, image_name in enumerate(image_names):
                    row = {
                        "dataset": dataset_name,
                        "seed": seed,
                        "variant": name,
                        "axis": axis,
                        "simulated_condition": cvd_condition_label(axis, severity),
                        "simulator_severity": float(severity),
                        "metric_simulator": metric_simulator,
                        "mask_simulator": mask_simulator,
                        "condition_policy": condition_policy,
                        "condition_axis": condition_axis,
                        "condition_severity": float(condition_severity),
                        "image": image_name,
                        "parameters": params,
                        "train_seconds": train_seconds,
                        "runtime_ms_per_image": elapsed_ms,
                    }
                    row.update({metric: float(values[metric][index]) for metric in METRIC_NAMES})
                    rows.append(row)
                    if sample_root is not None and saved < sample_images:
                        filename = f"{axis}_s{severity:.3f}_{saved:02d}.png"
                        save_panel(sample_root / name / filename, inp[index], out[index], mask[index], axis, severity, metric_simulator)
                        saved += 1
    return rows


def mean_rows(rows: Sequence[dict], group_fields: Sequence[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)
    output = []
    numeric = ["runtime_ms_per_image", *METRIC_NAMES]
    for key, members in groups.items():
        summary = dict(zip(group_fields, key))
        summary["n_rows_aggregated"] = len(members)
        summary["n_image_seed_observations"] = int(
            sum(int(member.get("n_image_seed_observations", 1)) for member in members)
        )
        for field in numeric:
            values = np.asarray([float(member[field]) for member in members], dtype=float)
            summary[field] = float(values.mean())
            summary[f"{field}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary["parameters"] = int(float(members[0]["parameters"]))
        summary["train_seconds"] = float(members[0]["train_seconds"])
        output.append(summary)
    return sorted(output, key=lambda row: tuple(str(row[field]) for field in group_fields))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def paired_statistics(rows: Sequence[dict], bootstrap_samples: int, seed: int) -> list[dict]:
    filtered = [
        row for row in rows
        if row["dataset"] == "coco_val2017" and row["metric_simulator"] == "machado"
    ]
    comparisons = ["baseline", "error_compensation"]
    metrics = ["cvd_contrast_gain", "delta_e00_mean", "ssim_luma"]
    raw_results = []
    rng = np.random.default_rng(seed)
    for comparator in comparisons:
        for axis in sorted({str(row["axis"]) for row in filtered}):
            for severity in sorted({float(row["simulator_severity"]) for row in filtered}):
                subset = [row for row in filtered if row["axis"] == axis and float(row["simulator_severity"]) == severity]
                for metric in metrics:
                    by_variant_image: dict[tuple[str, str], list[float]] = defaultdict(list)
                    for row in subset:
                        by_variant_image[(str(row["variant"]), str(row["image"]))].append(float(row[metric]))
                    images = sorted(
                        image for image in {key[1] for key in by_variant_image}
                        if ("ours_full", image) in by_variant_image and (comparator, image) in by_variant_image
                    )
                    if not images:
                        continue
                    ours = np.asarray([np.mean(by_variant_image[("ours_full", image)]) for image in images])
                    other = np.asarray([np.mean(by_variant_image[(comparator, image)]) for image in images])
                    differences = ours - other
                    if np.allclose(differences, 0.0):
                        p_value = 1.0
                    else:
                        p_value = float(wilcoxon(differences, zero_method="wilcox", alternative="two-sided").pvalue)
                    draws = rng.integers(0, len(differences), size=(bootstrap_samples, len(differences)))
                    boot = differences[draws].mean(axis=1)
                    raw_results.append({
                        "comparison": f"ours_full-minus-{comparator}",
                        "axis": axis,
                        "simulator_severity": severity,
                        "metric": metric,
                        "n_images": len(images),
                        "paired_mean_difference": float(differences.mean()),
                        "bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
                        "bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
                        "wilcoxon_p": p_value,
                    })
    adjusted = holm_adjust([row["wilcoxon_p"] for row in raw_results]) if raw_results else []
    for row, value in zip(raw_results, adjusted):
        row["holm_adjusted_p"] = value
    return raw_results


def stress_paths(root: Path, axis: str, limit: int, seed: int) -> list[Path]:
    folder = root / ("Color_cvd_P_experiment_generated" if axis == "protan" else "Color_cvd_D_experiment_generated")
    paths = sorted(folder.glob("*.png"))
    random.Random(seed + (0 if axis == "protan" else 1)).shuffle(paths)
    return paths[:limit]


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("val2017"))
    parser.add_argument("--stress-data", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/pilot_v2"))
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--train-images", type=int, default=256)
    parser.add_argument("--eval-images", type=int, default=48)
    parser.add_argument("--stress-images-per-axis", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19])
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--axes", nargs="+", choices=["protan", "deutan"], default=["protan", "deutan"])
    parser.add_argument("--train-severities", type=float, nargs="+", default=[0.25, 0.50, 0.75, 1.0])
    parser.add_argument("--eval-severities", type=float, nargs="+", default=[0.25, 0.50, 0.75, 1.0])
    parser.add_argument("--interpolation-severities", type=float, nargs="+", default=[0.375, 0.625, 0.875])
    parser.add_argument("--stress-severities", type=float, nargs="+", default=[0.50, 1.0])
    parser.add_argument("--train-simulator", choices=["machado", "fast"], default="machado")
    parser.add_argument("--robustness-simulator", choices=["brettel", "vienot"], default="brettel")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sample-images", type=int, default=1)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument(
        "--condition-probes",
        action="store_true",
        help="For ours_full, evaluate swapped-axis and fixed-s=0.5 conditioning without retraining.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--w-structure", type=float, default=1.0)
    parser.add_argument("--w-fidelity", type=float, default=1.5)
    parser.add_argument("--w-cvd", type=float, default=1.8)
    parser.add_argument("--w-decouple", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for severity in [*args.train_severities, *args.eval_severities, *args.interpolation_severities, *args.stress_severities]:
        if not 0.0 < severity <= 1.0:
            raise ValueError(f"All experiment severities must be in (0, 1], got {severity}")
    for name in args.variants:
        if name not in VARIANTS:
            raise KeyError(f"Unknown variant {name}; choose from {sorted(VARIANTS)}")

    device = resolve_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = ImagePathDataset(args.data, args.size)
    train_indices, eval_indices = make_split(dataset.paths, args.train_images, args.eval_images, args.split_seed)
    split_rows = [
        {"split": split, "index": index, "image": dataset.paths[index].name}
        for split, indices in (("train", train_indices), ("eval", eval_indices)) for index in indices
    ]
    write_dict_rows(args.out / "split_manifest.csv", split_rows)

    config = vars(args).copy()
    config.update({
        "data": str(args.data.resolve()),
        "stress_data": str(args.stress_data.resolve()) if args.stress_data else None,
        "out": str(args.out.resolve()),
        "device_resolved": str(device),
        "git_commit_at_start": git_commit(),
        "torch_version": torch.__version__,
        "terminology_note": "0<s<1 is a simulated anomaly condition; s=1 is a simulated dichromat endpoint; s is not a clinical grade.",
    })
    (args.out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"device={device} train={len(train_indices)} eval={len(eval_indices)} steps={args.steps} "
        f"seeds={args.seeds} variants={args.variants}",
        flush=True,
    )

    all_rows: list[dict] = []
    history_rows: list[dict] = []
    full_eval_severities = sorted(set(args.eval_severities + args.interpolation_severities))
    for seed_number, seed in enumerate(args.seeds):
        # Deterministic baselines are repeated across seeds so the paired aggregation
        # has the same table topology as learned variants.
        eval_loader = make_loader(dataset, eval_indices, args.batch_size, device, args.num_workers, shuffle=False, seed=seed)
        for baseline_name, transform in (("identity", identity_transform), ("error_compensation", error_compensation_transform)):
            all_rows.extend(evaluate_variant(
                name=baseline_name,
                seed=seed,
                loader=eval_loader,
                dataset_name="coco_val2017",
                device=device,
                axes=args.axes,
                severities=full_eval_severities,
                metric_simulator=args.train_simulator,
                mask_simulator=args.train_simulator,
                transform=transform,
                sample_root=args.out / "samples" if seed_number == 0 else None,
                sample_images=args.sample_images if seed_number == 0 and baseline_name in {"identity", "error_compensation"} else 0,
            ))
            all_rows.extend(evaluate_variant(
                name=baseline_name,
                seed=seed,
                loader=eval_loader,
                dataset_name="coco_val2017",
                device=device,
                axes=args.axes,
                severities=[1.0],
                metric_simulator=args.robustness_simulator,
                mask_simulator=args.train_simulator,
                transform=transform,
            ))
            if args.stress_data:
                for axis in args.axes:
                    paths = stress_paths(args.stress_data, axis, args.stress_images_per_axis, args.split_seed)
                    stress_dataset = ImagePathDataset(args.stress_data, args.size, paths=paths)
                    stress_loader = make_loader(
                        stress_dataset,
                        list(range(len(stress_dataset))),
                        args.batch_size,
                        device,
                        args.num_workers,
                        shuffle=False,
                        seed=seed,
                    )
                    all_rows.extend(evaluate_variant(
                        name=baseline_name,
                        seed=seed,
                        loader=stress_loader,
                        dataset_name=f"synthetic_stress_{axis}",
                        device=device,
                        axes=[axis],
                        severities=args.stress_severities,
                        metric_simulator=args.train_simulator,
                        mask_simulator=args.train_simulator,
                        transform=transform,
                    ))

        for variant_name in args.variants:
            seed_everything(seed)
            variant = VARIANTS[variant_name]
            train_loader = make_loader(dataset, train_indices, args.batch_size, device, args.num_workers, shuffle=True, seed=seed)
            eval_loader = make_loader(dataset, eval_indices, args.batch_size, device, args.num_workers, shuffle=False, seed=seed)
            model = make_model(variant, args.width).to(device)
            history, train_seconds = train_model(model, variant, train_loader, device, args, seed)
            history_rows.extend(history)
            params = parameter_count(model)
            all_rows.extend(evaluate_variant(
                name=variant_name,
                seed=seed,
                loader=eval_loader,
                dataset_name="coco_val2017",
                device=device,
                axes=args.axes,
                severities=full_eval_severities,
                metric_simulator=args.train_simulator,
                mask_simulator=args.train_simulator,
                model=model,
                params=params,
                train_seconds=train_seconds,
                sample_root=args.out / "samples" if seed_number == 0 else None,
                sample_images=args.sample_images if seed_number == 0 and variant_name in {"baseline", "ours_full"} else 0,
            ))
            if args.condition_probes and variant_name == "ours_full":
                for probe_name, policy in (
                    ("ours_full_axis_swap", "axis_swap"),
                    ("ours_full_fixed_s05", "fixed_s05"),
                ):
                    all_rows.extend(evaluate_variant(
                        name=probe_name,
                        seed=seed,
                        loader=eval_loader,
                        dataset_name="coco_val2017",
                        device=device,
                        axes=args.axes,
                        severities=full_eval_severities,
                        metric_simulator=args.train_simulator,
                        mask_simulator=args.train_simulator,
                        model=model,
                        params=params,
                        train_seconds=train_seconds,
                        condition_policy=policy,
                    ))
            # Second-model endpoint sensitivity is intentionally endpoint-only.
            all_rows.extend(evaluate_variant(
                name=variant_name,
                seed=seed,
                loader=eval_loader,
                dataset_name="coco_val2017",
                device=device,
                axes=args.axes,
                severities=[1.0],
                metric_simulator=args.robustness_simulator,
                mask_simulator=args.train_simulator,
                model=model,
                params=params,
                train_seconds=train_seconds,
            ))
            if args.stress_data:
                for axis in args.axes:
                    paths = stress_paths(args.stress_data, axis, args.stress_images_per_axis, args.split_seed)
                    stress_dataset = ImagePathDataset(args.stress_data, args.size, paths=paths)
                    stress_loader = make_loader(stress_dataset, list(range(len(stress_dataset))), args.batch_size, device, args.num_workers, shuffle=False, seed=seed)
                    all_rows.extend(evaluate_variant(
                        name=variant_name,
                        seed=seed,
                        loader=stress_loader,
                        dataset_name=f"synthetic_stress_{axis}",
                        device=device,
                        axes=[axis],
                        severities=args.stress_severities,
                        metric_simulator=args.train_simulator,
                        mask_simulator=args.train_simulator,
                        model=model,
                        params=params,
                        train_seconds=train_seconds,
                    ))
            if args.save_checkpoints:
                checkpoint_dir = args.out / "checkpoints"
                checkpoint_dir.mkdir(exist_ok=True)
                torch.save({"variant": variant_name, "seed": seed, "model": model.state_dict()}, checkpoint_dir / f"{variant_name}_seed{seed}.pt")
            del model

    write_dict_rows(args.out / "per_image_metrics.csv", all_rows)
    write_dict_rows(args.out / "train_history.csv", history_rows)
    by_seed = mean_rows(
        all_rows,
        ["dataset", "seed", "variant", "axis", "simulated_condition", "simulator_severity", "metric_simulator", "mask_simulator"],
    )
    write_dict_rows(args.out / "summary_by_seed.csv", by_seed)
    conditions = mean_rows(
        by_seed,
        ["dataset", "variant", "axis", "simulated_condition", "simulator_severity", "metric_simulator", "mask_simulator"],
    )
    write_dict_rows(args.out / "summary_conditions.csv", conditions)
    overall_source = [
        row for row in by_seed if row["dataset"] == "coco_val2017" and row["metric_simulator"] == args.train_simulator
    ]
    overall = mean_rows(overall_source, ["variant"])
    write_dict_rows(args.out / "summary_overall.csv", overall)
    statistics = paired_statistics(all_rows, args.bootstrap_samples, args.split_seed)
    write_dict_rows(args.out / "paired_statistics.csv", statistics)
    print(f"Wrote reproducible results to {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
