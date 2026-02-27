#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import enlarge_seg_masks, load_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MonST3R preprocessing to produce dynamic masks and split frames "
            "into foreground/background regions."
        )
    )
    parser.add_argument("--input", required=True, help="Input video path or image directory")
    parser.add_argument("--output_dir", default="demo_tmp/dynamic_preprocess")
    parser.add_argument("--seq_name", default=None, help="Output sequence name")
    parser.add_argument(
        "--weights",
        default="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth",
        help="Model weights path or HF model name",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image_size", type=int, default=512, choices=[224, 512])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--fps", type=int, default=0, help="0 keeps every frame")
    parser.add_argument("--num_frames", type=int, default=200)

    parser.add_argument("--scenegraph_type", default="swinstride")
    parser.add_argument("--winsize", type=int, default=5)
    parser.add_argument("--refid", type=int, default=0)

    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--schedule", default="linear", choices=["linear", "cosine"])
    parser.add_argument("--shared_focal", action="store_true", default=True)
    parser.add_argument("--not_shared_focal", action="store_false", dest="shared_focal")
    parser.add_argument("--translation_weight", type=float, default=1.0)
    parser.add_argument("--temporal_smoothing_weight", type=float, default=0.01)
    parser.add_argument("--flow_loss_weight", type=float, default=0.01)
    parser.add_argument("--flow_loss_start_iter", type=float, default=0.1)
    parser.add_argument("--flow_loss_threshold", type=float, default=25.0)
    parser.add_argument("--motion_mask_thre", type=float, default=0.35)
    parser.add_argument("--not_batchify", action="store_true", default=False)

    parser.add_argument("--window_wise", action="store_true", default=True)
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--window_overlap_ratio", type=float, default=0.5)
    parser.add_argument(
        "--sam2_mask_refine",
        action="store_true",
        default=True,
        help="Run SAM2 refinement over the initial motion mask seeds",
    )
    parser.add_argument(
        "--no_sam2_mask_refine",
        action="store_false",
        dest="sam2_mask_refine",
    )
    parser.add_argument(
        "--require_sam2_masks",
        action="store_true",
        default=True,
        help="Fail if SAM2 masks are not produced (matches demo DAVIS behavior)",
    )
    parser.add_argument(
        "--allow_non_sam2_masks",
        action="store_false",
        dest="require_sam2_masks",
    )

    parser.add_argument(
        "--dilate_kernel_size",
        type=int,
        default=3,
        help="Kernel size used to create enlarged_dynamic_mask_* from dynamic_mask_*",
    )
    parser.add_argument(
        "--separation_mask_prefix",
        default="enlarged_dynamic_mask",
        choices=["dynamic_mask", "enlarged_dynamic_mask"],
        help="Mask type used for foreground/background frame split",
    )
    parser.add_argument(
        "--split_regions",
        action="store_true",
        default=False,
        help="Also export foreground/background RGB regions using the selected mask prefix",
    )
    parser.add_argument("--silent", action="store_true", default=False)
    return parser.parse_args()


def _resolve_seq_name(input_path: Path, seq_name: str | None) -> str:
    if seq_name:
        return seq_name
    return input_path.stem if input_path.is_file() else input_path.name


def _canonical_scenegraph(scenegraph_type: str, winsize: int, refid: int) -> str:
    if scenegraph_type in {"swin", "swinstride", "swin2stride"}:
        return f"{scenegraph_type}-{winsize}-noncyclic"
    if scenegraph_type == "oneref":
        return f"oneref-{refid}"
    return scenegraph_type


def _split_foreground_background(
    seq_dir: Path, mask_prefix: str, silent: bool = False
) -> None:
    fg_dir = seq_dir / "foreground_rgb"
    bg_dir = seq_dir / "background_rgb"
    fg_mask_dir = seq_dir / "foreground_masks"
    bg_mask_dir = seq_dir / "background_masks"
    for out_dir in (fg_dir, bg_dir, fg_mask_dir, bg_mask_dir):
        out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(seq_dir.glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {seq_dir}")

    for frame_path in frame_paths:
        frame_idx = int(frame_path.stem.split("_")[-1])
        mask_path = seq_dir / f"{mask_prefix}_{frame_idx}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None:
            raise RuntimeError(f"Failed to read frame or mask for index {frame_idx}")
        if frame.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Shape mismatch at {frame_idx}: frame {frame.shape[:2]} vs mask {mask.shape[:2]}"
            )

        dynamic = mask > 127
        static = ~dynamic

        fg = np.zeros_like(frame)
        bg = np.zeros_like(frame)
        fg[dynamic] = frame[dynamic]
        bg[static] = frame[static]

        cv2.imwrite(str(fg_dir / f"frame_{frame_idx:04d}.png"), fg)
        cv2.imwrite(str(bg_dir / f"frame_{frame_idx:04d}.png"), bg)
        cv2.imwrite(str(fg_mask_dir / f"mask_{frame_idx:04d}.png"), dynamic.astype(np.uint8) * 255)
        cv2.imwrite(str(bg_mask_dir / f"mask_{frame_idx:04d}.png"), static.astype(np.uint8) * 255)

    if not silent:
        print(f"[split] wrote foreground/background regions to: {seq_dir}")


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    seq_name = _resolve_seq_name(input_path, args.seq_name)
    seq_dir = Path(args.output_dir) / seq_name
    seq_dir.mkdir(parents=True, exist_ok=True)

    if not args.silent:
        print(f"[load] input={input_path}")
    if input_path.is_dir():
        filelist = str(input_path)
    else:
        filelist = [str(input_path)]
    imgs = load_images(
        filelist,
        size=args.image_size,
        verbose=not args.silent,
        fps=args.fps,
        num_frames=args.num_frames,
    )
    if len(imgs) == 1:
        raise RuntimeError("Need at least 2 frames to compute motion masks.")

    if not args.silent:
        print("[model] loading model")
    weights_path = args.weights if os.path.exists(args.weights) else args.weights
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)
    model.eval()

    scenegraph = _canonical_scenegraph(args.scenegraph_type, args.winsize, args.refid)
    if not args.silent:
        print(f"[pairs] scenegraph={scenegraph}")
    pairs = make_pairs(imgs, scene_graph=scenegraph, prefilter=None, symmetrize=True)

    if not args.silent:
        print("[infer] running pair inference")
    output = inference(pairs, model, args.device, batch_size=args.batch_size, verbose=not args.silent)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    if not args.silent:
        print(f"[align] mode={mode}")
    scene = global_aligner(
        output,
        device=args.device,
        mode=mode,
        verbose=not args.silent,
        shared_focal=args.shared_focal,
        temporal_smoothing_weight=args.temporal_smoothing_weight,
        translation_weight=args.translation_weight,
        flow_loss_weight=args.flow_loss_weight,
        flow_loss_start_epoch=args.flow_loss_start_iter,
        flow_loss_thre=args.flow_loss_threshold,
        motion_mask_thre=args.motion_mask_thre,
        sam2_mask_refine=args.sam2_mask_refine,
        use_self_mask=True,
        num_total_iter=args.niter,
        empty_cache=len(imgs) > 72,
        batchify=not (args.not_batchify or args.window_wise),
        window_wise=args.window_wise,
        window_size=args.window_size,
        window_overlap_ratio=args.window_overlap_ratio,
        prev_video_results=None,
    )

    if not args.silent:
        print("[save] writing masks")
    if args.require_sam2_masks and getattr(scene, "sam2_dynamic_masks", None) is None:
        raise RuntimeError(
            "SAM2 masks were not produced. Ensure SAM2 checkpoint exists at "
            "third_party/sam2/checkpoints/sam2.1_hiera_large.pt and run with "
            "--sam2_mask_refine (enabled by default)."
        )
    scene.save_dynamic_masks(str(seq_dir))
    enlarge_seg_masks(str(seq_dir), kernel_size=args.dilate_kernel_size)

    if args.split_regions:
        # Region splitting needs the resized/cropped RGB frames in MonST3R preprocessing space.
        scene.save_rgb_imgs(str(seq_dir))
        _split_foreground_background(
            seq_dir=seq_dir, mask_prefix=args.separation_mask_prefix, silent=args.silent
        )
    if not args.silent:
        print(f"[done] output={seq_dir}")


if __name__ == "__main__":
    main()


"""
/home/atasoy/miniconda3/bin/conda run -n monst3r python preprocess_dynamic_regions.py \
  --input /mnt/hdd/davis_subset/car-roundabout \
  --output_dir demo_tmp/davis \
  --seq_name car-roundabout \
  --window_wise \
  --window_size 30 \
  --window_overlap_ratio 0.5 \
  --niter 300 \
  --scenegraph_type swinstride \
  --winsize 5 \
  --image_size 512 \
  --batch_size 16 \
  --fps 0 \
  --num_frames 200 \
  --flow_loss_weight 0.01 \
  --flow_loss_start_iter 0.1 \
  --flow_loss_threshold 25 \
  --motion_mask_thre 0.35 \
  --sam2_mask_refine \
  --require_sam2_masks \
  --dilate_kernel_size 3 \
  --separation_mask_prefix enlarged_dynamic_mask

"""
