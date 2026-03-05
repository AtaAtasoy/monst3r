import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from dreifus.matrix import CameraCoordinateConvention, Intrinsics, Pose, PoseType
from pytorch3d.io import IO
from pytorch3d.renderer import (
    AlphaCompositor,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)
from pytorch3d.structures import Pointclouds

"""
python render_scene_from_vlm_cameras.py \
--data_dir demo_tmp/davis/tennis/normalized_nofilter \
--traj_path lifted_pred_traj.txt \
--output_dir renders_vlm_lifted
"""


def load_tum_poses(path: Path) -> list[Pose]:
    poses = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(f"Invalid TUM line in {path}: {line}")
            _, tx, ty, tz, qw, qx, qy, qz = map(float, parts)
            poses.append(
                Pose.from_quaternion(
                    quaternion=[qx, qy, qz, qw],
                    translation=[tx, ty, tz],
                    pose_type=PoseType.CAM_2_WORLD,
                    camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
                )
            )
    if not poses:
        raise ValueError(f"No valid poses found in {path}")
    return poses


def load_intrinsics(path: Path) -> list[Intrinsics]:
    data = np.loadtxt(str(path))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    intr = []
    for row in data:
        if row.size != 9:
            raise ValueError(f"Expected 9 values per intrinsics row, got {row.size}")
        intr.append(Intrinsics(matrix_or_fx=row.reshape(3, 3).astype(np.float64)))
    if not intr:
        raise ValueError(f"No intrinsics found in {path}")
    return intr


def get_hw_from_intrinsics(intr: Intrinsics) -> tuple[int, int]:
    h = int(round(float(intr.cy) * 2.0))
    w = int(round(float(intr.cx) * 2.0))
    return max(h, 1), max(w, 1)


def list_frame_plys(ply_dir: Path) -> list[Path]:
    return sorted([p for p in ply_dir.glob("frame_*.ply") if "_masked" not in p.name])


def create_video_from_frames(frame_pattern: str, output_video: Path, fps: int = 25) -> None:
    cmd = [
        "ffmpeg",
        "-framerate",
        str(fps),
        "-i",
        frame_pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(output_video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {output_video}: {result.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render scene point clouds using lifted VLM camera trajectory."
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--traj_path", type=str, required=True, help="Lifted TUM trajectory txt.")
    parser.add_argument("--intrinsics_path", type=str, default=None)
    parser.add_argument("--ply_dir", type=str, default=None, help="Defaults to <data_dir>/all_points")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    traj_path = Path(args.traj_path)
    intrinsics_path = Path(args.intrinsics_path) if args.intrinsics_path else data_dir / "pred_intrinsics.txt"
    ply_dir = Path(args.ply_dir) if args.ply_dir else data_dir / "all_points"
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "renders_vlm_lifted"
    frames_dir = output_dir / "frames"
    video_dir = output_dir / "vis-videos"
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    if not traj_path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {traj_path}")
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Missing intrinsics file: {intrinsics_path}")
    if not ply_dir.exists():
        raise FileNotFoundError(f"Missing point-cloud directory: {ply_dir}")

    poses = load_tum_poses(traj_path)
    intrinsics = load_intrinsics(intrinsics_path)
    ply_files = list_frame_plys(ply_dir)
    if not ply_files:
        raise ValueError(f"No frame_*.ply files found in {ply_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    io = IO()

    h, w = get_hw_from_intrinsics(intrinsics[0])
    raster_settings = PointsRasterizationSettings(
        image_size=(h, w),
    )

    n = len(poses)
    print(f"Rendering {n} frames")
    for i in range(n):
        ply_path = ply_files[min(i, len(ply_files) - 1)]
        intr = intrinsics[min(i, len(intrinsics) - 1)]
        pose = poses[i]

        cloud = io.load_pointcloud(str(ply_path), device=device)
        points = cloud.points_list()[0]
        if points.numel() == 0:
            img_np = np.zeros((h, w, 3), dtype=np.uint8)
            Image.fromarray(img_np).save(frames_dir / f"{i:04d}.png")
            continue

        features = cloud.features_list()[0]
        if features is None or features.numel() == 0:
            features = torch.ones_like(points)
        pc = Pointclouds(points=[points], features=[features])

        p3d_c2w = pose.change_camera_coordinate_convention(
            new_camera_coordinate_convention=CameraCoordinateConvention.PYTORCH_3D
        )
        r = p3d_c2w.get_rotation_matrix()
        p3d_w2c = p3d_c2w.change_pose_type(
            new_pose_type=PoseType.WORLD_2_CAM, inplace=False
        )
        t = p3d_w2c.get_translation()

        r_t = torch.tensor(r, dtype=torch.float32, device=device)[None, :, :]
        t_t = torch.tensor(t, dtype=torch.float32, device=device)[None, :]
        image_size = torch.tensor([[h, w]], dtype=torch.float32, device=device)

        k = torch.tensor(
            [
                [intr.fx, 0.0, intr.cx, 0.0],
                [0.0, intr.fy, intr.cy, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )[None, :, :]

        camera = PerspectiveCameras(
            in_ndc=False,
            R=r_t,
            T=t_t,
            K=k,
            image_size=image_size,
            device=device,
        )
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
            compositor=AlphaCompositor(),
        )
        image = renderer(pc)[0, ..., :3].detach().cpu().numpy().clip(0, 1)
        Image.fromarray((image * 255).astype(np.uint8)).save(frames_dir / f"{i:04d}.png")
        print(f"Rendered {i + 1}/{n}", end="\r")

    print()
    create_video_from_frames(str(frames_dir / "%04d.png"), video_dir / "render.mp4", fps=args.fps)
    print(f"Saved frames to: {frames_dir}")
    print(f"Saved video to: {video_dir / 'render.mp4'}")


if __name__ == "__main__":
    main()
