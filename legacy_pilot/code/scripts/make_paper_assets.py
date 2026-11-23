#!/usr/bin/env python3
"""Generate paper figures/tables from one immutable experiment result folder."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


COLORS = {
    "identity": "#999999",
    "error_compensation": "#009E73",
    "baseline": "#E69F00",
    "ours_full": "#0072B2",
    "ours_no_structure": "#CC79A7",
    "ours_no_fidelity": "#56B4E9",
    "ours_no_confusion": "#D55E00",
    "ours_no_decouple": "#F0E442",
}


def primary(per_image: pd.DataFrame) -> pd.DataFrame:
    return per_image[
        (per_image["dataset"] == "coco_val2017")
        & (per_image["metric_simulator"] == "machado")
    ].copy()


def save_method_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str, color: str) -> None:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.4, edgecolor=color, facecolor="white"
        ))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.2, color="#333333"))

    box(0.2, 2.4, 1.3, 1.0, "RGB image", "#444444")
    box(0.2, 0.7, 1.3, 1.0, "axis + s", "#444444")
    box(2.0, 3.6, 1.5, 0.9, "Machado\nsimulation", "#009E73")
    box(2.0, 2.2, 1.5, 0.9, "confusion\nmask", "#009E73")
    box(2.0, 0.7, 1.5, 0.9, "FiLM\ncondition", "#009E73")
    box(4.0, 2.2, 1.5, 1.0, "shared\nencoder", "#0072B2")
    box(6.0, 4.2, 1.7, 0.9, "structure\nbranch", "#CC79A7")
    box(6.0, 2.5, 1.7, 0.9, "fidelity\nbranch", "#56B4E9")
    box(6.0, 0.8, 1.7, 0.9, "confusion\nbranch", "#D55E00")
    box(8.3, 2.2, 1.8, 1.1, "mask-normalized\nfusion", "#0072B2")
    box(10.6, 2.2, 1.2, 1.1, "residual\noutput", "#444444")
    box(8.3, 0.3, 1.8, 0.9, "four-part\nobjective", "#444444")

    arrow(1.5, 2.9, 2.0, 4.0)
    arrow(1.5, 2.9, 2.0, 2.65)
    arrow(1.5, 1.2, 2.0, 1.15)
    arrow(3.5, 2.65, 4.0, 2.7)
    arrow(3.5, 1.15, 4.0, 2.45)
    arrow(5.5, 2.7, 6.0, 4.65)
    arrow(5.5, 2.7, 6.0, 2.95)
    arrow(5.5, 2.7, 6.0, 1.25)
    arrow(7.7, 4.65, 8.3, 2.95)
    arrow(7.7, 2.95, 8.3, 2.75)
    arrow(7.7, 1.25, 8.3, 2.45)
    arrow(10.1, 2.75, 10.6, 2.75)
    arrow(11.2, 2.2, 9.2, 1.2)
    ax.text(6.85, 5.45, "candidate mask-guided multi-branch recoloring architecture", ha="center", fontsize=12, weight="bold")
    ax.text(6.85, 0.08, "s is a simulator control parameter; s=1 is the simulated dichromat endpoint", ha="center", fontsize=9, color="#555555")
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_tradeoff(data: pd.DataFrame, path: Path) -> None:
    grouped = data[data["variant"].isin(["identity", "error_compensation", "baseline", "ours_full"])].groupby("variant")[["delta_e00_mean", "cvd_contrast_gain"]].mean()
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for variant, row in grouped.iterrows():
        ax.scatter(row["delta_e00_mean"], row["cvd_contrast_gain"], s=70, color=COLORS[variant], edgecolor="black", linewidth=0.5)
        label = "candidate" if variant == "ours_full" else variant.replace("_", " ")
        ax.annotate(label, (row["delta_e00_mean"], row["cvd_contrast_gain"]), xytext=(6, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Mean CIEDE2000 fidelity cost (lower is better)")
    ax.set_ylabel("Masked simulated-CVD contrast gain (higher is better)")
    ax.grid(alpha=0.25)
    ax.set_title("Pilot trade-off on COCO val2017")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def save_severity_curves(data: pd.DataFrame, path: Path) -> None:
    seed_image = data.groupby(["variant", "axis", "simulator_severity", "seed"])["cvd_contrast_gain"].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, panel in zip(["protan", "deutan"], axes):
        subset = seed_image[seed_image["axis"] == axis]
        for variant in ["baseline", "ours_full"]:
            item = subset[subset["variant"] == variant]
            summary = item.groupby("simulator_severity")["cvd_contrast_gain"].agg(["mean", "min", "max"]).reset_index()
            panel.plot(summary["simulator_severity"], summary["mean"], marker="o", label=variant, color=COLORS[variant])
            panel.fill_between(summary["simulator_severity"], summary["min"], summary["max"], color=COLORS[variant], alpha=0.15)
        for severity in [0.375, 0.625, 0.875]:
            panel.axvline(severity, color="#BBBBBB", lw=0.7, ls=":")
        panel.set_title(f"{axis} axis")
        panel.set_xlabel("simulated degree s")
        panel.grid(alpha=0.22)
    axes[0].set_ylabel("masked simulated-CVD contrast gain")
    axes[1].legend(frameon=False)
    fig.suptitle("Protocol response across simulated-degree settings (band: two-seed range)")
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def save_qualitative_samples(run: Path, path: Path) -> None:
    """Show deterministic first-saved endpoint samples without cherry-picking."""

    columns = [
        ("Input", "baseline", 0),
        ("Baseline", "baseline", 3),
        ("Candidate", "ours_full", 3),
        ("Baseline simulated", "baseline", 4),
        ("Candidate simulated", "ours_full", 4),
        ("Input-fixed mask", "ours_full", 2),
    ]
    rows = [("protan endpoint", "protan_s1.000_00.png"), ("deutan endpoint", "deutan_s1.000_00.png")]
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(11.5, 4.2))
    for row_index, (row_label, filename) in enumerate(rows):
        for column_index, (title, variant, tile_index) in enumerate(columns):
            panel = plt.imread(run / "samples" / variant / filename)
            height, width = panel.shape[:2]
            tile = width // 5
            crop = panel[height - tile : height, tile_index * tile : (tile_index + 1) * tile]
            ax = axes[row_index, column_index]
            ax.imshow(crop)
            ax.axis("off")
            if row_index == 0:
                ax.set_title(title, fontsize=9)
            if column_index == 0:
                ax.set_ylabel(row_label, fontsize=9)
    fig.suptitle("Deterministic first-saved 64 x 64 endpoint examples (illustrative only)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def tables(per_image: pd.DataFrame, output: Path, condition_per_image: pd.DataFrame | None = None) -> None:
    data = primary(per_image)
    fields = ["cvd_contrast_gain", "cvd_contrast_ratio", "delta_e00_mean", "delta_e00_nonconf", "ssim_luma", "gamut_preclip_rate", "runtime_ms_per_image"]
    main_names = ["identity", "error_compensation", "baseline", "ours_full"]
    main_data = data[data["variant"].isin(main_names)]
    main = main_data.groupby("variant")[fields].mean().reset_index()
    params = main_data.groupby("variant")["parameters"].first().reset_index()
    main = main.merge(params, on="variant")
    main.to_csv(output / "table_main.csv", index=False, float_format="%.6f")

    endpoint = per_image[
        (per_image["dataset"] == "coco_val2017")
        & (per_image["simulator_severity"] == 1.0)
    ].groupby(["variant", "axis", "metric_simulator"])[["cvd_contrast_gain", "cvd_contrast_ratio"]].mean().reset_index()
    endpoint.to_csv(output / "table_endpoint_robustness.csv", index=False, float_format="%.6f")

    stress = per_image[per_image["dataset"].str.startswith("synthetic")].groupby(
        ["variant", "dataset", "simulator_severity"]
    )[["cvd_contrast_gain", "cvd_contrast_ratio", "delta_e00_mean", "ssim_luma"]].mean().reset_index()
    stress.to_csv(output / "table_synthetic_stress.csv", index=False, float_format="%.6f")

    all_summary = data.groupby("variant")[fields].mean().reset_index()
    all_params = data.groupby("variant")["parameters"].first().reset_index()
    all_summary = all_summary.merge(all_params, on="variant")
    full = all_summary[all_summary["variant"] == "ours_full"].iloc[0]
    ablation_names = [
        "ours_full", "ours_no_structure", "ours_no_fidelity",
        "ours_no_confusion", "ours_no_decouple", "ours_no_mask",
    ]
    ablation = all_summary[all_summary["variant"].isin(ablation_names)].copy()
    ablation["gain_difference_vs_full"] = ablation["cvd_contrast_gain"] - full["cvd_contrast_gain"]
    ablation["delta_e_difference_vs_full"] = ablation["delta_e00_mean"] - full["delta_e00_mean"]
    ablation.to_csv(output / "table_ablation.csv", index=False, float_format="%.6f")

    condition_names = ["ours_full", "ours_full_axis_swap", "ours_full_fixed_s05"]
    condition_source = primary(condition_per_image) if condition_per_image is not None else data
    condition = condition_source[condition_source["variant"].isin(condition_names)].groupby("variant")[fields].mean().reset_index()
    condition.to_csv(output / "table_condition_probes.csv", index=False, float_format="%.6f")

    by_seed = main_data[main_data["variant"].isin(["baseline", "ours_full"])].groupby(
        ["variant", "seed"]
    )[["cvd_contrast_gain", "delta_e00_mean", "ssim_luma", "gamut_preclip_rate"]].mean().reset_index()
    by_seed.to_csv(output / "table_by_seed.csv", index=False, float_format="%.6f")


def paired_summary(run: Path, output: Path) -> None:
    paired = pd.read_csv(run / "paired_statistics.csv")
    paired = paired[paired["comparison"] == "ours_full-minus-baseline"]
    rows = []
    for metric in ["cvd_contrast_gain", "delta_e00_mean", "ssim_luma"]:
        item = paired[paired["metric"] == metric]
        rows.append({
            "metric": metric,
            "conditions": len(item),
            "paired_difference_min": item["paired_mean_difference"].min(),
            "paired_difference_max": item["paired_mean_difference"].max(),
            "bootstrap_ci95_low_min": item["bootstrap_ci95_low"].min(),
            "bootstrap_ci95_low_max": item["bootstrap_ci95_low"].max(),
            "bootstrap_ci95_high_min": item["bootstrap_ci95_high"].min(),
            "bootstrap_ci95_high_max": item["bootstrap_ci95_high"].max(),
            "ci_excluding_zero": int(((item["bootstrap_ci95_low"] > 0) | (item["bootstrap_ci95_high"] < 0)).sum()),
            "holm_p_min": item["holm_adjusted_p"].min(),
            "holm_p_max": item["holm_adjusted_p"].max(),
        })
    pd.DataFrame(rows).to_csv(output / "table_paired_summary.csv", index=False, float_format="%.8f")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--supplemental-run", type=Path, nargs="*", default=[])
    parser.add_argument("--condition-run", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    per_image = pd.read_csv(args.run / "per_image_metrics.csv")
    known_variants = set(per_image["variant"].unique())
    for run in args.supplemental_run:
        extra = pd.read_csv(run / "per_image_metrics.csv")
        extra = extra[~extra["variant"].isin(known_variants)]
        known_variants.update(extra["variant"].unique())
        per_image = pd.concat([per_image, extra], ignore_index=True)
    data = primary(per_image)
    save_method_figure(args.out / "figure_method.png")
    save_tradeoff(data, args.out / "figure_tradeoff.png")
    save_severity_curves(data, args.out / "figure_severity.png")
    save_qualitative_samples(args.run, args.out / "figure_qualitative.png")
    condition_per_image = pd.read_csv(args.condition_run / "per_image_metrics.csv") if args.condition_run else None
    tables(per_image, args.out, condition_per_image)
    paired_summary(args.run, args.out)
    print(f"Wrote paper assets to {args.out.resolve()}")


if __name__ == "__main__":
    main()
