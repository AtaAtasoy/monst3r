#!/usr/bin/env python3
"""Convert intrinsics text file to NPZ with shape (N, 3, 3)."""

import argparse
from pathlib import Path

import numpy as np


def load_intrinsics_txt(path: Path) -> np.ndarray:
    """Load intrinsics from txt and return array shaped (N, 3, 3)."""
    data = np.loadtxt(str(path), dtype=np.float64)

    if data.ndim == 0:
        raise ValueError(f"Expected intrinsics data in {path}, got scalar.")

    if data.ndim == 1:
        if data.size != 9:
            raise ValueError(f"Expected 9 values for single-frame intrinsics, got {data.size}.")
        data = data.reshape(1, 9)

    if data.ndim != 2:
        raise ValueError(f"Expected 2D intrinsics table, got shape {data.shape}.")

    if data.shape[1] != 9:
        raise ValueError(f"Expected 9 columns per row, got shape {data.shape}.")

    return data.reshape(-1, 3, 3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert pred_intrinsics.txt to pred_intrinsics.npz with shape (N, 3, 3)."
    )
    parser.add_argument("input_txt", type=Path, help="Path to input intrinsics txt file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path. Defaults to <input_dir>/pred_intrinsics.npz",
    )
    args = parser.parse_args()

    input_txt = args.input_txt
    if not input_txt.exists():
        raise FileNotFoundError(f"Input file not found: {input_txt}")

    output_path = args.output if args.output is not None else input_txt.with_suffix(".npz")

    intrinsics = load_intrinsics_txt(input_txt)
    np.savez(output_path, data=intrinsics)

    print(f"Saved: {output_path}")
    print("Key: data")
    print(f"Shape: {intrinsics.shape}")


if __name__ == "__main__":
    main()
