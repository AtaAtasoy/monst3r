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
        description="Render all exported point clouds with poses"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to exported sequence directory",
    )
    parser.add_argument("--H", type=int, default=288, help="Output image height")
    parser.add_argument("--W", type=int, default=512, help="Output image width")

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

    num_frames = len(poses)
    print(f"Processing {num_frames} frames...")

    points_list = []
    features_list = []

    R_list = []
    T_list = []
    K_list = []

    # Pre-loading and processing loop
    for i in range(num_frames):
        if i % 10 == 0:
            print(f"Loading frame {i}/{num_frames}...", end="\r")

        # 1. Load Point Cloud
        ply_path = data_path / f"frame_{i:04d}.ply"
        # IO().load_pointcloud returns a batch of size 1
        pc = IO().load_pointcloud(str(ply_path), device=device)
        points_list.append(pc.points_list()[0])
        features_list.append(pc.features_list()[0])  # ty:ignore[not-subscriptable]

        # 2. Process Pose
        opencv_c2w_pose: Pose = poses[i]
        pytorch3d_c2w_pose = opencv_c2w_pose.change_camera_coordinate_convention(
            new_camera_coordinate_convention=CameraCoordinateConvention.PYTORCH_3D
        )

        R = pytorch3d_c2w_pose.get_rotation_matrix()
        pytorch3d_w2c_pose = pytorch3d_c2w_pose.change_pose_type(
            new_pose_type=PoseType.WORLD_2_CAM, inplace=False
        )
        T = pytorch3d_w2c_pose.get_translation()

        # Ensure R and T are tensors
        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R).float().to(device)
        if isinstance(T, np.ndarray):
            T = torch.from_numpy(T).float().to(device)

        R_list.append(R)
        T_list.append(T)

        # 3. Process Intrinsics
        intr = intrinsics[i]
        K = torch.tensor(
            [
                [intr.fx, 0, intr.cx, 0],
                [0, intr.fy, intr.cy, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=torch.float32,
            device=device,
        )
        K_list.append(K)

    print("\nConstructing batch pointclouds...")
    batch_pc = Pointclouds(points=points_list, features=features_list)

    R_batch = torch.stack(R_list)
    T_batch = torch.stack(T_list)
    K_batch = torch.stack(K_list)

    # use_pulsar logic preserved but defaulting to False
    use_pulsar = False

    raster_settings = PointsRasterizationSettings(
        image_size=(args.H, args.W),  # H, W in pixels
        radius=0.01,
    )

    print("Rendering...")
    if use_pulsar:
        # Use FoVPerspectiveCameras with PulsarPointsRenderer (faster)
        print("Using FoVPerspectiveCameras with Pulsar backend.")

        # Calculate FOV for all frames
        fov_list = []
        for i in range(num_frames):
            intr = intrinsics[i]
            fov_y = 2 * np.arctan(args.H / (2 * intr.fy))
            fov_list.append(np.degrees(fov_y))

        fov_tensor = torch.tensor(fov_list, dtype=torch.float32, device=device)

        camera = FoVPerspectiveCameras(
            R=R_batch,
            T=T_batch,
            fov=fov_tensor,
            degrees=True,
            device=device,
        )

        rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings)
        renderer = PulsarPointsRenderer(
            rasterizer=rasterizer,
            n_channels=3,
        ).to(device)

        images = renderer(
            batch_pc,
            gamma=(1e-4,),
            bg_col=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device),
        )
        # Pulsar typically returns RGBA or RGB depending on n_channels settings, here n_channels=3
        # Reference: image = images[0, ..., :3]

    else:
        # Use PerspectiveCameras with PointsRenderer (supports off-center principal point)
        print(
            "Principal point is off-center. Using PerspectiveCameras with PointsRenderer."
        )

        image_size = torch.tensor([[args.H, args.W]], device=device).expand(
            num_frames, -1
        )

        camera = PerspectiveCameras(
            in_ndc=False,
            R=R_batch,
            T=T_batch,
            K=K_batch,
            device=device,
            image_size=image_size,
        )

        rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(),
        )

        images = renderer(batch_pc)

    # Save images
    output_dir = data_path / "reconstruction_renders"
    output_dir.mkdir(parents=True, exist_ok=True)

    images_np = images[..., :3].cpu().numpy()

    print(f"Saving images to {output_dir}")
    for i in range(num_frames):
        img_np = images_np[i].clip(0, 1)
        save_path = output_dir / f"render_{i:04d}.png"
        Image.fromarray((img_np * 255).astype(np.uint8)).save(save_path)

    print("Done.")


if __name__ == "__main__":
    main()
