import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from dreifus.matrix import Pose, CameraCoordinateConvention, PoseType
from dreifus.matrix import Intrinsics
from pytorch3d.io import IO
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)
import torch


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
            # timestamp = parts[0]
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


def main():
    parser = argparse.ArgumentParser(
        description="Render exported point clouds with poses"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to exported sequence directory",
    )
    parser.add_argument(
        "--output", type=str, default="render.png", help="Output image path"
    )
    parser.add_argument(
        "--frame_idx", type=int, default=0, help="Frame index to render"
    )
    parser.add_argument("--H", type=int, default=288, help="Output image height")
    parser.add_argument("--W", type=int, default=512, help="Output image width")
    parser.add_argument(
        "--masked", action="store_true", help="Render masked point clouds"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    data_path = Path(args.data_dir)

    # Load Poses
    traj_path = data_path / "pred_traj.txt"
    poses = load_tum_poses(traj_path)
    print(f"Loaded {len(poses)} poses.")

    # Add Cameras
    intrinsics_path = data_path / "pred_intrinsics.txt"
    intrinsics = load_intrinsics(intrinsics_path)
    print(f"Loaded {len(intrinsics)} intrinsics.")

    assert len(poses) == len(intrinsics), "Number of poses and intrinsics must match."

    suffix = "_masked" if args.masked else ""
    point_cloud = IO().load_pointcloud(
        str(data_path / f"frame_{args.frame_idx:04d}{suffix}.ply"), device=device
    )

    # Select camera for rendering
    opencv_c2w_pose: Pose = poses[args.frame_idx]
    pytorch3d_c2w_pose = opencv_c2w_pose.change_camera_coordinate_convention(
        new_camera_coordinate_convention=CameraCoordinateConvention.PYTORCH_3D
    )

    R = pytorch3d_c2w_pose.get_rotation_matrix()
    pytorch3d_w2c_pose = pytorch3d_c2w_pose.change_pose_type(
        new_pose_type=PoseType.WORLD_2_CAM, inplace=False
    )
    T = pytorch3d_w2c_pose.get_translation()

    intrinsics = intrinsics[args.frame_idx]

    # Check if principal point is centered (within a small tolerance)
    # cx_centered = abs(intrinsics.cx - args.W / 2) < 1e-3
    # cy_centered = abs(intrinsics.cy - args.H / 2) < 1e-3
    # use_pulsar = cx_centered and cy_centered
    use_pulsar = True

    raster_settings = PointsRasterizationSettings(
        image_size=(args.H, args.W),  # H, W in pixels
        radius=0.01,
    )

    if use_pulsar:
        # Use FoVPerspectiveCameras with PulsarPointsRenderer (faster)
        print(
            "Principal point is centered. Using FoVPerspectiveCameras with Pulsar backend."
        )

        # Convert focal length (pixels) to FoV (radians), then to degrees
        fov_y = 2 * np.arctan(args.H / (2 * intrinsics.fy))
        fov_y_deg = np.degrees(fov_y)

        camera = FoVPerspectiveCameras(
            R=R[None, :, :],
            T=T[None, :],
            fov=fov_y_deg,
            degrees=fov_y_deg,
            device=device,
        )

        rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings)
        renderer = PulsarPointsRenderer(
            rasterizer=rasterizer,
            n_channels=3,
        ).to(device)

        images = renderer(
            point_cloud,
            gamma=(1e-4,),
            bg_col=torch.tensor(
                [0.0, 0.0, 0.0],
                dtype=torch.float32,
                device=device,
            ),
        )
    else:
        # Use PerspectiveCameras with PointsRenderer (supports off-center principal point)
        print(
            "Principal point is off-center. Using PerspectiveCameras with PointsRenderer."
        )

        K = torch.tensor(
            [
                [intrinsics.fx, 0, intrinsics.cx, 0],
                [0, intrinsics.fy, intrinsics.cy, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=torch.float32,
        )

        image_size = torch.tensor([[args.H, args.W]], device=device)

        camera = PerspectiveCameras(
            in_ndc=False,
            R=R[None, :, :],
            T=T[None, :],
            K=K[None, :, :],
            device=device,
            image_size=image_size,
        )

        rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(),
        )

        images = renderer(point_cloud)

    image = images[0, ..., :3].cpu().numpy().clip(0, 1)
    Image.fromarray((image * 255).astype(np.uint8)).save(args.output)
    print(f"Saved rendered image to {args.output}")


if __name__ == "__main__":
    main()
