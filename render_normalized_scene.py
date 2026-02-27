import argparse
import numpy as np
from pathlib import Path
import subprocess
from PIL import Image
from dreifus.matrix import Pose, CameraCoordinateConvention, PoseType
from dreifus.matrix import Intrinsics
from pytorch3d.structures import Pointclouds
from pytorch3d.io import IO
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
import torch
import cv2


def load_tum_poses(path) -> list[Pose]:
    """
    Load poses from a TUM format file: timestamp tx ty tz qw qx qy qz
    Returns a list of 4x4 numpy arrays (c2w poses)
    """
    poses = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = list(map(float, line.strip().split()))
            tx, ty, tz = parts[1:4]
            qw, qx, qy, qz = parts[4:8]  # scalar first

            poses.append(
                Pose.from_quaternion(
                    quaternion=[qx, qy, qz, qw],
                    translation=[tx, ty, tz],
                    pose_type=PoseType.CAM_2_WORLD,
                    camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
                )
            )
    return poses


def load_intrinsics(path) -> list[Intrinsics]:
    """
    Load intrinsics from a file.
    Assumes generic format, one K per line (flattened 9 items or 3x3)
    """
    intrinsics = []
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    for row in data:
        K = row.reshape(3, 3)
        intrinsics.append(Intrinsics(matrix_or_fx=K))
    return intrinsics

def get_hw_from_intrinsics(intrinsics: list[Intrinsics]) -> tuple[int, int]:
    """
    Infer image height and width from intrinsics, assuming cx,cy are at the center.
    Returns (height, width) as integers.
    """
    # Use the first intrinsics as reference
    K = intrinsics[0].numpy()
    cx, cy = K[0, 2], K[1, 2]
    width = int(round(cx * 2))
    height = int(round(cy * 2))
    return height, width


def load_dynamic_mask(data_path: Path, frame_idx: int, mask_type: str) -> np.ndarray:
    """
    Load a dynamic mask image and return as a boolean array.
    White (255) = dynamic, Black (0) = static.
    """
    mask_path = data_path / f"{mask_type}_{frame_idx}.png"
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Mask not found: {mask_path}\n"
            f"Available mask types: 'dynamic_mask' or 'enlarged_dynamic_mask'"
        )
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    dynamic_mask = mask > 127
    return dynamic_mask


def filter_points_by_mask(
    points: torch.Tensor,
    features: torch.Tensor,
    dynamic_mask: np.ndarray,
    region: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Filter points and features using a dynamic mask.
    """
    flat_mask = torch.from_numpy(dynamic_mask.flatten()).to(points.device)

    if len(flat_mask) != len(points):
        print(
            f"Warning: Mask has {len(flat_mask)} pixels but point cloud has {len(points)} points. "
            f"Skipping mask filtering."
        )
        return points, features

    if region == "dynamic":
        keep = flat_mask
    elif region == "static":
        keep = ~flat_mask
    else:
        raise ValueError(f"Unknown region: {region}")

    return points[keep], features[keep]


def create_video_from_pngs(
    frames_dir: Path,
    output_video_path: Path,
    fps: int = 25,
) -> None:
    png_files = sorted(frames_dir.glob("*.png"))
    if not png_files:
        print(f"Warning: No PNG frames found in {frames_dir}. Skipping video export.")
        return

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    try:
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
        if result.returncode == 0:
            return
        print(f"Warning: ffmpeg failed, falling back to OpenCV writer.\n{result.stderr}")
    except FileNotFoundError:
        print("Warning: ffmpeg not found, falling back to OpenCV writer.")

    # Fallback path
    first = cv2.imread(str(png_files[0]), cv2.IMREAD_COLOR)
    if first is None:
        print(f"Warning: Could not read first frame {png_files[0]}. Skipping video export.")
        return

    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    try:
        for frame_path in png_files:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"Warning: Could not read frame {frame_path}. Skipping frame.")
                continue
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


def main():

    parser = argparse.ArgumentParser(
        description="Render already-normalized, filtered scenes from filter_points.py output."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to normalized, filtered sequence directory (from filter_points.py)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="all",
        choices=["all", "static", "dynamic"],
        help="Which region to render",
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        default="enlarged_dynamic_mask",
        choices=["dynamic_mask", "enlarged_dynamic_mask"],
        help="Which mask to use for filtering (only for static/dynamic regions)",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    data_path = Path(args.data_dir)

    # Load Data
    traj_path = data_path / "pred_traj.txt"
    poses = load_tum_poses(traj_path)
    intrinsics_path = data_path / "pred_intrinsics.txt"
    intrinsics = load_intrinsics(intrinsics_path)
    num_frames = len(poses)

    # Infer image size from intrinsics
    H, W = get_hw_from_intrinsics(intrinsics)

    # Output directory
    output_dir = data_path / f"renders_{args.region}"
    output_dir.mkdir(parents=True, exist_ok=True)

    raster_settings = PointsRasterizationSettings(
        image_size=(H, W),
    )

    # Choose which PLY directory to use
    if args.region == "all":
        ply_dir = data_path / "all_points"
    elif args.region == "static":
        ply_dir = data_path / "background_points"
    elif args.region == "dynamic":
        ply_dir = data_path / "foreground_points"
    else:
        raise ValueError(f"Unknown region: {args.region}")

    for i in range(num_frames):
        print(f"Rendering frame {i}/{num_frames}...", end="\r")

        ply_path = ply_dir / f"frame_{i:04d}.ply"
        if not ply_path.exists():
            print(f"Warning: {ply_path} does not exist. Skipping.")
            continue
        pc = IO().load_pointcloud(str(ply_path), device=device)
        points = pc.points_list()[0]
        features = pc.features_list()[0]

        frame_pc = Pointclouds(points=[points], features=[features])

        # Pose
        opencv_c2w_pose: Pose = poses[i]
        pytorch3d_c2w_pose = opencv_c2w_pose.change_camera_coordinate_convention(
            new_camera_coordinate_convention=CameraCoordinateConvention.PYTORCH_3D
        )

        R = pytorch3d_c2w_pose.get_rotation_matrix()
        pytorch3d_w2c_pose = pytorch3d_c2w_pose.change_pose_type(
            new_pose_type=PoseType.WORLD_2_CAM, inplace=False
        )
        T = pytorch3d_w2c_pose.get_translation()
        
        # convert to tensor
        R_c2w = torch.from_numpy(R).float().to(device)
        T_w2c = torch.from_numpy(T).float().to(device)
        
        # Intrinsics
        intr = intrinsics[i]
        fov_y = 2 * np.arctan(H / (2 * intr.fy))
        fov_y_deg = np.degrees(fov_y)

        camera = FoVPerspectiveCameras(
            R=R_c2w[None, :, :],
            T=T_w2c[None, :],
            fov=fov_y_deg,
            degrees=fov_y_deg,
            device=device,
        )

        rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(),
        )
        images = renderer(frame_pc)

        img_np = images[0, ..., :3].cpu().numpy().clip(0, 1)
        save_path = output_dir / f"{i:04d}.png"
        Image.fromarray((img_np * 255).astype(np.uint8)).save(save_path)

    video_dir = output_dir / "vis-videos"
    capture_video_path = video_dir / "capture-cameras.mp4"
    create_video_from_pngs(output_dir, capture_video_path, fps=25)

    print(f"\nSaved {num_frames} renders to {output_dir}")
    print(f"Saved video to {capture_video_path}")


if __name__ == "__main__":
    main()
