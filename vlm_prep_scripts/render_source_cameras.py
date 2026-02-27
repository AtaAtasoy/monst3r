"""
Renders the source cameras using top-down orthographic view. 
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from render_top_down_semantic import (
    _make_orthographic_camera,
    load_intrinsics,
    load_tum_poses,
    make_topdown_pose,
    src_colormap_color,
)

from dreifus.render import project, draw_onto_image

def _list_frame_plys(ply_dir: Path) -> list[Path]:
    return sorted([f for f in ply_dir.glob("frame_*.ply") if "_masked" not in f.name])


def main():
    parser = argparse.ArgumentParser(description="Render only source cameras in top-down orthographic view.")
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_poses = load_tum_poses(data_dir / "pred_traj.txt")

    src_positions_np = np.array([pose.get_translation() for pose in source_poses], dtype=np.float32)
    look_at = src_positions_np.mean(axis=0)
    topdown_pose = make_topdown_pose(height=0.5, look_at=look_at)

    img_w = 512
    img_h = 512

    scene_paths = _list_frame_plys(data_dir / "all_points")
    camera, _ = _make_orthographic_camera(
        topdown_pose=topdown_pose,
        img_w=img_w,
        img_h=img_h,
        scene_paths=scene_paths,
        margin=1.0,
        device=device,
    )

    src_positions = torch.tensor(src_positions_np, dtype=torch.float32, device=device)
    image_size = torch.tensor([[img_h, img_w]], dtype=torch.float32, device=device)
    src_screen = camera.transform_points_screen(src_positions[None, ...], image_size=image_size)[0, :, :2]
    src_screen_pts = [(int(round(x.item())), int(round(y.item()))) for x, y in src_screen]

    image = Image.new("RGB", (img_w, img_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(image)

    for i, (x, y) in enumerate(src_screen_pts):
        t = i / max(1, len(src_screen_pts) - 1)
        color = src_colormap_color(t)
        draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=color)

    out_path = data_dir / "source_cameras_topdown.png"
    image.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
