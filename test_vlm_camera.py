import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from dreifus.matrix import CameraCoordinateConvention, Intrinsics, Pose, PoseType
from pytorch3d.io import IO
from pytorch3d.renderer import (
    AlphaCompositor,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)
from pytorch3d.structures import Pointclouds


def load_intrinsics(intr_path: Path) -> list[Intrinsics]:
    data = np.loadtxt(str(intr_path))
    if data.ndim == 1:
        data = data.reshape(1, -1)

    mats: list[Intrinsics] = []
    for row in data:
        if row.size != 9:
            raise ValueError(f"Expected 9 values per intrinsics row, got {row.size}")
        K = row.reshape(3, 3).astype(np.float64)
        mats.append(Intrinsics(matrix_or_fx=K))
    return mats


def pose_to_euler_yxz(pose: Pose) -> tuple[float, float, float]:
    euler = pose.get_euler_angles("YXZ")
    return float(euler[0]), float(euler[1]), float(euler[2])


def get_hw_from_intrinsics(intr: Intrinsics) -> tuple[int, int]:
    K = intr.numpy()
    cx, cy = K[0, 2], K[1, 2]
    width = int(round(cx * 2))
    height = int(round(cy * 2))
    return height, width


def load_vlm_poses(vlm_json_path: Path) -> list[Pose]:
    with vlm_json_path.open("r", encoding="utf-8") as f:
        text = f.read().strip()

    if text.startswith("```"):
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
        text = "\n".join(lines)

    data = json.loads(text)
    poses_dict = data.get("poses", {})
    if not poses_dict:
        raise ValueError(f"No 'poses' field found in {vlm_json_path}")

    def frame_key_to_idx(k: str) -> int:
        if not k.startswith("t_"):
            raise ValueError(f"Unexpected pose key '{k}', expected 't_<index>'")
        return int(k.split("_", 1)[1])

    sorted_items = sorted(poses_dict.items(), key=lambda kv: frame_key_to_idx(kv[0]))
    poses: list[Pose] = []

    for key, entry in sorted_items:
        if "rotation" not in entry or "translation" not in entry:
            raise ValueError(f"Pose {key} must contain 'rotation' and 'translation'")

        yaw, pitch, roll = [float(v) for v in entry["rotation"]]
        tx, ty, tz = [float(v) for v in entry["translation"]]

        pose = Pose.from_euler(
            euler_angles=[yaw, pitch, roll],
            translation=[tx, ty, tz],
            euler_mode="YXZ",
            camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
            pose_type=PoseType.CAM_2_WORLD,
        )
        poses.append(pose)

    return poses


def create_video_from_pngs(frames_dir: Path, output_video_path: Path, fps: int = 25) -> None:
    png_files = sorted(frames_dir.glob("*.png"))
    if not png_files:
        print(f"Warning: No PNG frames found in {frames_dir}. Skipping video export.")
        return

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        str(frames_dir / "*.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-y",
        str(output_video_path),
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {output_video_path}:\n{result.stderr}")


def select_ply_dir(scene_dir: Path, region: str) -> Path:
    if region == "all":
        return scene_dir / "filtered_scene"
    if region == "static":
        return scene_dir / "background_points"
    if region == "dynamic":
        return scene_dir / "foreground_points"
    raise ValueError(f"Unknown region: {region}")


def read_user_demand(prompt_txt_path: Path) -> str:
    with prompt_txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("User Demand:"):
                value = line.split(":", 1)[1].strip()
                if value:
                    return value
                break
    raise ValueError(f"Could not find non-empty 'User Demand:' line in {prompt_txt_path}")


def demand_to_filename(user_demand: str) -> str:
    slug = user_demand.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    if not slug:
        slug = "camera-action"
    return f"{slug}.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a normalized scene using a VLM-generated camera trajectory."
    )
    parser.add_argument(
        "scene_dir",
        type=str,
        help="Path like demo_tmp/davis/tennis/normalized_pct100_conf0",
    )
    parser.add_argument(
        "vlm_json",
        type=str,
        help="Path to VLM output JSON file containing poses[t_i].",
    )
    parser.add_argument(
        "prompt_txt",
        type=str,
        help="Path to prompt text file containing a line starting with 'User Demand:'.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="all",
        choices=["all", "static", "dynamic"],
        help="Which scene subset to render (same semantics as render_normalized_scene.py).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: <scene_dir>/test_vlm_camera_<region>",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Output MP4 frame rate.",
    )
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    vlm_json_path = Path(args.vlm_json)
    prompt_txt_path = Path(args.prompt_txt)

    intr_path = scene_dir / "pred_intrinsics.txt"
    if not intr_path.exists():
        raise FileNotFoundError(f"Missing intrinsics file: {intr_path}")
    if not vlm_json_path.exists():
        raise FileNotFoundError(f"Missing VLM JSON file: {vlm_json_path}")
    if not prompt_txt_path.exists():
        raise FileNotFoundError(f"Missing prompt text file: {prompt_txt_path}")

    ply_dir = select_ply_dir(scene_dir, args.region)
    if not ply_dir.exists():
        raise FileNotFoundError(f"Missing PLY directory: {ply_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else scene_dir / f"test_vlm_camera_{args.region}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    io = IO()

    intrinsics = load_intrinsics(intr_path)
    vlm_poses = load_vlm_poses(vlm_json_path)
    user_demand = read_user_demand(prompt_txt_path)

    if len(intrinsics) == 0:
        raise ValueError("No intrinsics found")

    H, W = get_hw_from_intrinsics(intrinsics[0])
    raster_settings = PointsRasterizationSettings(image_size=(H, W))

    frame_plys = sorted([p for p in ply_dir.glob("frame_*.ply") if "_masked" not in p.name])
    if not frame_plys:
        raise FileNotFoundError(f"No frame_*.ply files found in {ply_dir}")

    n_render = min(len(vlm_poses), len(frame_plys))
    if n_render == 0:
        raise ValueError("No frames to render")

    print(f"Rendering {n_render} frames from VLM trajectory...")

    for i in range(n_render):
        ply_path = frame_plys[i]
        pc = io.load_pointcloud(str(ply_path), device=device)
        points = pc.points_list()[0]
        features = pc.features_list()[0]

        if points.numel() == 0:
            print(f"Warning: Empty point cloud at {ply_path}, skipping.")
            continue
        if features is None or features.numel() == 0:
            features = torch.ones_like(points)

        frame_pc = Pointclouds(points=[points], features=[features])

        c2w_pose = vlm_poses[i]
        yaw, pitch, roll = pose_to_euler_yxz(c2w_pose)

        p3d_c2w_pose = c2w_pose.change_camera_coordinate_convention(
            new_camera_coordinate_convention=CameraCoordinateConvention.PYTORCH_3D
        )
        R = p3d_c2w_pose.get_rotation_matrix()
        p3d_w2c_pose = p3d_c2w_pose.change_pose_type(
            new_pose_type=PoseType.WORLD_2_CAM, inplace=False
        )
        T = p3d_w2c_pose.get_translation()

        R_c2w = torch.from_numpy(R).float().to(device)
        T_w2c = torch.from_numpy(T).float().to(device)

        intr = intrinsics[min(i, len(intrinsics) - 1)]
        fov_y = 2.0 * np.arctan(H / (2.0 * float(intr.fy)))
        fov_y_deg = float(np.degrees(fov_y))

        camera = FoVPerspectiveCameras(
            R=R_c2w[None, :, :],
            T=T_w2c[None, :],
            fov=fov_y_deg,
            degrees=True,
            device=device,
        )

        rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings)
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())
        images = renderer(frame_pc)

        img_np = images[0, ..., :3].detach().cpu().numpy().clip(0, 1)
        save_path = output_dir / f"{i:04d}.png"
        Image.fromarray((img_np * 255).astype(np.uint8)).save(save_path)

        if i % 10 == 0 or i == n_render - 1:
            print(
                f"  frame {i:04d}/{n_render - 1:04d} "
                f"| yaw={yaw:.4f} pitch={pitch:.4f} roll={roll:.4f}",
                end="\r",
            )

    print()

    video_dir = output_dir / "vis-videos"
    video_path = video_dir / demand_to_filename(user_demand)
    create_video_from_pngs(output_dir, video_path, fps=args.fps)

    print(f"Saved renders to {output_dir}")
    print(f"Saved video to {video_path}")


if __name__ == "__main__":
    main()
