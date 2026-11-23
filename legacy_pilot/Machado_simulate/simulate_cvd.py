#!/usr/bin/env python3
"""Simulate red-green color vision deficiency with Machado et al. matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_MATRIX_FILE = ROOT / "machado_matrices.json" # 这里也可以用colour.blindness.matrix_cvd_Machado2009()来生成


ALIASES = {
    "protan": "protanomaly",
    "protanomaly": "protanomaly",
    "protanopia": "protanomaly",
    "red": "protanomaly",
    "deutan": "deuteranomaly",
    "deuteranomaly": "deuteranomaly",
    "deuteranopia": "deuteranomaly",
    "green": "deuteranomaly",
}


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert gamma-encoded sRGB values in [0, 1] to linear sRGB."""
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Convert linear sRGB values in [0, 1] to gamma-encoded sRGB."""
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055)


def load_matrix(deficiency: str, severity: float, matrix_file: Path = DEFAULT_MATRIX_FILE) -> np.ndarray:
    """Load or interpolate a Machado 3x3 matrix for a deficiency and severity."""
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be between 0.0 and 1.0")

    key = ALIASES.get(deficiency.lower())
    if key is None:
        choices = ", ".join(sorted(ALIASES))
        raise ValueError(f"unknown deficiency {deficiency!r}; choose one of: {choices}")

    with matrix_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    matrices = data["types"][key]["matrices"]
    exact_key = f"{severity:.1f}"
    if exact_key in matrices:
        return np.array(matrices[exact_key], dtype=np.float64)

    lower = np.floor(severity * 10.0) / 10.0
    upper = np.ceil(severity * 10.0) / 10.0
    lower_key = f"{lower:.1f}"
    upper_key = f"{upper:.1f}"
    t = (severity - lower) / (upper - lower)

    lower_matrix = np.array(matrices[lower_key], dtype=np.float64)
    upper_matrix = np.array(matrices[upper_key], dtype=np.float64)
    return (1.0 - t) * lower_matrix + t * upper_matrix


def simulate_array(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3x3 matrix to a uint8 RGB image array."""
    rgb_float = rgb.astype(np.float64) / 255.0
    linear_rgb = srgb_to_linear(rgb_float)
    simulated = linear_rgb @ matrix.T
    srgb = linear_to_srgb(simulated)
    return np.rint(srgb * 255.0).astype(np.uint8)


def simulate_image(input_path: Path, output_path: Path, deficiency: str, severity: float) -> None:
    matrix = load_matrix(deficiency, severity)
    with Image.open(input_path) as image:
        rgba = image.convert("RGBA")
        arr = np.array(rgba)
        arr[..., :3] = simulate_array(arr[..., :3], matrix)
        Image.fromarray(arr, mode="RGBA").save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate red-green color vision deficiency using Machado et al. 2009 matrices."
    )
    parser.add_argument("input", type=Path, help="input image path")
    parser.add_argument("output", type=Path, help="output image path")
    parser.add_argument(
        "-t",
        "--type",
        default="protanomaly",
        choices=sorted(ALIASES),
        help="deficiency type; protanopia/deuteranopia are severity 1.0 aliases",
    )
    parser.add_argument(
        "-s",
        "--severity",
        type=float,
        default=1.0,
        help="severity from 0.0 to 1.0; 1.0 is protanopia/deuteranopia",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    simulate_image(args.input, args.output, args.type, args.severity)


if __name__ == "__main__":
    main()
