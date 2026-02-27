import argparse
import numpy as np
from pathlib import Path
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

    Args:
        points: (N, 3) tensor
        features: (N, 3) tensor (RGB)
        dynamic_mask: (H, W) boolean array, True = dynamic
        region: 'static' or 'dynamic'

    Returns:
        Filtered (points, features) tuple
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


def main():
    parser = argparse.ArgumentParser(
        description="Render all exported point clouds with dynamic/static region filtering"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to exported sequence directory",
    )
    parser.add_argument("--H", type=int, default=288, help="Output image height")
    parser.add_argument("--W", type=int, default=512, help="Output image width")
    parser.add_argument(
        "--region",
        type=str,
        default="all",
        choices=["all", "static", "dynamic"],
        help="Which region to render: 'all' (default), 'static' (background only), or 'dynamic' (moving objects only)",
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        default="enlarged_dynamic_mask",
        choices=["dynamic_mask", "enlarged_dynamic_mask"],
        help="Which mask to use for region filtering (default: enlarged_dynamic_mask)",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    data_path = Path(args.data_dir)

    # Load Poses
    traj_path = data_path / "pred_traj.txt"
    poses = load_tum_poses(traj_path)
    print(f"Loaded {len(poses)} poses.")

    # Load Intrinsics
    intrinsics_path = data_path / "pred_intrinsics.txt"
    intrinsics = load_intrinsics(intrinsics_path)
    print(f"Loaded {len(intrinsics)} intrinsics.")

    assert len(poses) == len(intrinsics), "Number of poses and intrinsics must match."

    num_frames = len(poses)
    print(f"Processing {num_frames} frames (region={args.region})...")

    # Output directory
    output_dir = data_path / f"{args.region}_renders"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Since filtered point clouds have different sizes per frame,
    # we cannot batch them into a single Pointclouds object.
    # Render frame-by-frame instead.
    raster_settings = PointsRasterizationSettings(
        image_size=(args.H, args.W),
        radius=0.01,
    )

    for i in range(num_frames):
        print(f"Rendering frame {i}/{num_frames}...", end="\r")

        # 1. Load Point Cloud
        ply_path = data_path / f"frame_{i:04d}.ply"
        pc = IO().load_pointcloud(str(ply_path), device=device)
        points = pc.points_list()[0]
        features = pc.features_list()[0]

        # 2. Apply mask filtering if needed
        if args.region != "all":
            dynamic_mask = load_dynamic_mask(data_path, i, args.mask_type)
            points, features = filter_points_by_mask(
                points, features, dynamic_mask, args.region
            )
            if points.shape[0] == 0:
                # Save a black frame and skip rendering
                Image.fromarray(np.zeros((args.H, args.W, 3), dtype=np.uint8)).save(
                    output_dir / f"{i:04d}.png"
                )
                continue

        frame_pc = Pointclouds(points=[points], features=[features])

        # 3. Process Pose
        opencv_c2w_pose: Pose = poses[i]
        pytorch3d_c2w_pose = opencv_c2w_pose.change_camera_coordinate_convention(
            new_camera_coordinate_convention=CameraCoordinateConvention.PYTORCH_3D
        )

        R = pytorch3d_c2w_pose.get_rotation_matrix()
        pytorch3d_w2c_pose = pytorch3d_c2w_pose.change_pose_type(
            new_pose_type=PoseType.WORLD_2_CAM, inplace=False
        )
        T = pytorch3d_w2c_pose.get_translation()

        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R).float().to(device)
        if isinstance(T, np.ndarray):
            T = torch.from_numpy(T).float().to(device)

        # 4. Process Intrinsics & Camera
        intr = intrinsics[i]
        fov_y = 2 * np.arctan(args.H / (2 * intr.fy))
        fov_y_deg = np.degrees(fov_y)

        camera = FoVPerspectiveCameras(
            R=R[None, :, :],
            T=T[None, :],
            fov=fov_y_deg,
            degrees=fov_y_deg,
            device=device,
        )

        # 5. Render (AlphaCompositor works for both full and filtered point clouds)
        rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(),
        )
        images = renderer(frame_pc)

        # 6. Save
        img_np = images[0, ..., :3].cpu().numpy().clip(0, 1)
        save_path = output_dir / f"{i:04d}.png"
        Image.fromarray((img_np * 255).astype(np.uint8)).save(save_path)

    print(f"\nSaved {num_frames} renders to {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
