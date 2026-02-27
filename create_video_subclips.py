import argparse
from pathlib import Path
import imageio
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="Create video subclips from rendered images and masks"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to sequence directory"
    )
    parser.add_argument(
        "--start", type=int, required=True, help="Start index (inclusive)"
    )
    parser.add_argument("--end", type=int, required=True, help="End index (exclusive)")
    parser.add_argument(
        "--radius_multiplier",
        type=float,
        default=1.0,
        help="Radius multiplier used for rendering",
    )
    parser.add_argument("--fps", type=int, default=30, help="Video framerate")

    args = parser.parse_args()
    data_path = Path(args.data_dir)

    # Define directories based on radius multiplier (matching v2v_render.py)
    render_dir = data_path / f"gray_background_orbit_renders_{args.radius_multiplier}"
    mask_dir = data_path / f"mask_orbit_renders_{args.radius_multiplier}"

    if not render_dir.exists():
        print(f"Error: Render directory does not exist: {render_dir}")
        return
    if not mask_dir.exists():
        print(f"Error: Mask directory does not exist: {mask_dir}")
        return

    print(f"Processing frames {args.start} to {args.end}...")

    render_frames = []
    mask_frames = []

    # Load frames
    for i in range(args.start, args.end):
        # Load render
        render_path = render_dir / f"render_{i:04d}.png"
        if not render_path.exists():
            print(f"Warning: Frame {i} not found at {render_path}")
            continue

        img = Image.open(render_path)
        img_np = np.array(img)
        render_frames.append(img_np)

        # Load mask
        mask_path = mask_dir / f"mask_{i:04d}.png"
        if not mask_path.exists():
            print(f"Warning: Mask {i} not found at {mask_path}")
            # If mask missing, maybe append black/dummy? Or just fail?
            # Ideally consistent with renders. Let's just warn and skip for now or consistency check.
            continue

        mask = Image.open(mask_path)
        mask_np = np.array(mask)

        # Ensure mask is RGB for video if it's grayscale
        if mask_np.ndim == 2:
            mask_np = np.stack([mask_np] * 3, axis=-1)
        elif mask_np.shape[2] == 4:  # RGBA
            mask_np = mask_np[..., :3]

        mask_frames.append(mask_np)

    if not render_frames:
        print("No frames loaded. Exiting.")
        return

    # Save videos
    output_suffix = f"{args.start}_{args.end}"
    render_video_path = render_dir / f"input_video_{output_suffix}.mp4"
    mask_video_path = mask_dir / f"input_mask_{output_suffix}.mp4"

    print(f"Saving render video to {render_video_path}")
    imageio.mimwrite(
        str(render_video_path),
        render_frames,
        fps=args.fps,
        codec="libx264",
        format="FFMPEG",
    )  # ty:ignore[no-matching-overload]

    print(f"Saving mask video to {mask_video_path}")
    imageio.mimwrite(
        str(mask_video_path),
        mask_frames,
        fps=args.fps,
        codec="libx264",
        format="FFMPEG",
    )  # ty:ignore[no-matching-overload]

    print("Done.")


if __name__ == "__main__":
    main()
