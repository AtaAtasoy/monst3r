#!/usr/bin/env python3
"""
Photometric render quality evaluation script.
Calculates LPIPS, SSIM, and PSNR between rendered images and ground truth.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from natsort import natsorted
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import json


def load_image(path: Path) -> np.ndarray:
    """Load image as numpy array in RGB format, normalized to [0, 1]."""
    img = Image.open(path).convert("RGB")
    return np.array(img).astype(np.float32) / 255.0


def load_image_torch(path: Path, device: torch.device) -> torch.Tensor:
    """Load image as torch tensor for LPIPS, normalized to [-1, 1]."""
    img = Image.open(path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0
    # Convert to [-1, 1] range for LPIPS
    img = (img * 2.0) - 1.0
    # Convert to [B, C, H, W] format
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return img


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate PSNR between two images."""
    return psnr(img1, img2, data_range=1.0)


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate SSIM between two images."""
    return ssim(img1, img2, data_range=1.0, channel_axis=2)


def calculate_lpips(
    img1: torch.Tensor, img2: torch.Tensor, lpips_fn: lpips.LPIPS
) -> float:
    """Calculate LPIPS between two images."""
    with torch.no_grad():
        return lpips_fn(img1, img2).item()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate photometric render quality metrics (LPIPS, SSIM, PSNR)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to data directory containing ground truth frames and reconstruction_renders",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for LPIPS calculation",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    render_dir = data_dir / "reconstruction_renders"
    device = torch.device(args.device)

    # Validate directories exist
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not render_dir.exists():
        raise FileNotFoundError(f"Render directory not found: {render_dir}")

    # Get sorted list of render files
    render_paths = natsorted(list(render_dir.glob("render_*.png")))
    if not render_paths:
        raise FileNotFoundError(f"No render files found in {render_dir}")

    # Get sorted list of ground truth files
    gt_paths = natsorted(list(data_dir.glob("frame_*.png")))
    if not gt_paths:
        raise FileNotFoundError(f"No ground truth files found in {data_dir}")

    # Verify matching number of files
    if len(render_paths) != len(gt_paths):
        print(
            f"Warning: Number of renders ({len(render_paths)}) != number of GTs ({len(gt_paths)})"
        )
        # Use the minimum of both
        num_frames = min(len(render_paths), len(gt_paths))
        render_paths = render_paths[:num_frames]
        gt_paths = gt_paths[:num_frames]
    else:
        num_frames = len(render_paths)

    print(f"Evaluating {num_frames} frames...")
    print(f"Render directory: {render_dir}")
    print(f"Ground truth directory: {data_dir}")
    print(f"Device: {device}")
    print("-" * 60)

    # Initialize LPIPS model
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    # Metrics storage
    all_psnr = []
    all_ssim = []
    all_lpips = []

    for i, (render_path, gt_path) in enumerate(zip(render_paths, gt_paths)):
        # Load images as numpy arrays for PSNR and SSIM
        render_np = load_image(render_path)
        gt_np = load_image(gt_path)

        # Load images as torch tensors for LPIPS
        render_torch = load_image_torch(render_path, device)
        gt_torch = load_image_torch(gt_path, device)

        # Calculate metrics
        psnr_val = calculate_psnr(gt_np, render_np)
        ssim_val = calculate_ssim(gt_np, render_np)
        lpips_val = calculate_lpips(gt_torch, render_torch, lpips_fn)

        all_psnr.append(psnr_val)
        all_ssim.append(ssim_val)
        all_lpips.append(lpips_val)

        print(
            f"Frame {i:04d}: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}, LPIPS={lpips_val:.4f}"
        )

    # Calculate and print statistics
    print("-" * 60)
    print("Summary Statistics:")
    print(f"  PSNR  - Mean: {np.mean(all_psnr):.4f}, Std: {np.std(all_psnr):.4f}")
    print(f"  SSIM  - Mean: {np.mean(all_ssim):.4f}, Std: {np.std(all_ssim):.4f}")
    print(f"  LPIPS - Mean: {np.mean(all_lpips):.4f}, Std: {np.std(all_lpips):.4f}")

    # Return metrics as dict for potential programmatic use
    metrics = {
        "psnr": {
            "mean": np.mean(all_psnr),
            "std": np.std(all_psnr),
            "values": all_psnr,
        },
        "ssim": {
            "mean": np.mean(all_ssim),
            "std": np.std(all_ssim),
            "values": all_ssim,
        },
        "lpips": {
            "mean": np.mean(all_lpips),
            "std": np.std(all_lpips),
            "values": all_lpips,
        },
    }

    # Save metrics to file
    metrics_file = data_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f)

    return metrics


if __name__ == "__main__":
    main()
