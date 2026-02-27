import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import imageio
from dreifus.matrix import Pose, CameraCoordinateConvention, PoseType
from dreifus.matrix import Intrinsics
from pytorch3d.structures import Pointclouds
from pytorch3d.io import IO
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
)
import torch
import open3d as o3d


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


def generate_orbit_poses(
    center: np.ndarray,
    radius: float,
    n_views: int,
    up: np.ndarray = np.array(
        [0, -1, 0]
    ),  # The world up hint. Defines which way is up in the world. Should match the up vector of the camera coordinate system.
    height: float = 0.0,
) -> list[Pose]:
    """Generate a turntable/orbit path around center at given radius and height."""
    angles = np.linspace(-np.pi, 0, n_views, endpoint=False)
    poses = []
    # cam_pos = center + [r.cos(theta), height, r.sin(theta)] where theta is in [0, max(angles)], position on the circle
    for theta in angles:
        cam_pos = center + np.array(
            [radius * np.cos(theta), height, radius * np.sin(theta)]
        )  # this is the translation part of the pose

        forward = (
            center - cam_pos
        )  # this is the forward direction of the camera, -z axis in opengl
        forward /= np.linalg.norm(forward)

        right = np.cross(
            forward, up
        )  # this is the right direction of the camera, x axis in opengl, perpendicular to forward and up
        right /= np.linalg.norm(right)

        true_up = np.cross(
            right, forward
        )  # this is the up direction of the camera, y axis in opengl, perpendicular to forward and right
        true_up /= np.linalg.norm(true_up)

        R_cam = np.stack([right, true_up, -forward], axis=1)
        poses.append(
            Pose(
                np.block(
                    [
                        [R_cam, cam_pos.reshape(3, 1)],
                        [np.zeros((1, 3)), np.ones((1, 1))],
                    ]
                ),
                pose_type=PoseType.CAM_2_WORLD,
                camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL,
            )
        )
    return poses


def fuse_scene_centroid(point_clouds: list[o3d.geometry.PointCloud]) -> np.ndarray:  # ty:ignore[possibly-missing-attribute]
    all_pts = np.concatenate(
        [np.asarray(pcd.points) for pcd in point_clouds if len(pcd.points) > 0], axis=0
    )
    return all_pts.mean(axis=0)


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
    parser.add_argument("--radius_multiplier", type=float, default=1.0, help="Radius multiplier")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    data_path = Path(args.data_dir)

    # Add Cameras
    intrinsics_path = data_path / "pred_intrinsics.txt"
    intrinsics = load_intrinsics(intrinsics_path)
    print(f"Loaded {len(intrinsics)} intrinsics.")

    # Load Poses
    traj_path = data_path / "pred_traj.txt"
    poses = load_tum_poses(traj_path)
    print(f"Loaded {len(poses)} poses.")

    point_clouds = [
        o3d.io.read_point_cloud(str(data_path / f"frame_{i:04d}.ply"))
        for i in range(len(poses))
    ]

    center = fuse_scene_centroid(point_clouds)
    radius_multiplier = args.radius_multiplier
    radius = radius_multiplier * np.linalg.norm(center - poses[0].get_translation())
    orbit_poses = generate_orbit_poses(
        center, radius=float(radius), n_views=len(poses), height=0.0
    )
    print(f"Created {len(orbit_poses)} orbit poses.")

    assert len(orbit_poses) == len(intrinsics), (
        "Number of orbit poses and intrinsics must match."
    )

    num_frames = len(orbit_poses)
    print(f"Processing {num_frames} orbit poses...")

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
        opencv_c2w_pose: Pose = orbit_poses[i]
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

    raster_settings = PointsRasterizationSettings(
        image_size=(args.H, args.W),  # H, W in pixels
        radius=0.01,
    )

    print("Rendering...")
    image_size = torch.tensor([[args.H, args.W]], device=device).expand(num_frames, -1)

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
        compositor=AlphaCompositor(
            background_color=torch.tensor([0.5, 0.5, 0.5], device=device)
        ),  # gray background
    )

    images = renderer(batch_pc)

    # Get fragments for mask generation (idx >= 0 means a point was rendered at that pixel)
    fragments = rasterizer(batch_pc)
    # fragments.idx has shape (N, H, W, points_per_pixel), check if any point was rendered
    # idx == -1 means no point, idx >= 0 means point rendered
    point_rendered = fragments.idx[..., 0] >= 0  # Shape: (N, H, W)
    # Create mask: white background (1.0), black foreground (0.0)
    masks = (~point_rendered).float()  # Invert: True where NO point -> white background

    # Save images
    output_dir = data_path / f"gray_background_orbit_renders_{radius_multiplier}"
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_output_dir = data_path / f"mask_orbit_renders_{radius_multiplier}"
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    images_np = images[..., :3].cpu().numpy()
    masks_np = masks.cpu().numpy()

    print(f"Saving images to {output_dir}")
    print(f"Saving masks to {mask_output_dir}")
    render_frames = []
    mask_frames = []

    for i in range(num_frames):
        img_np = images_np[i].clip(0, 1)
        img_uint8 = (img_np * 255).astype(np.uint8)
        save_path = output_dir / f"render_{i:04d}.png"
        Image.fromarray(img_uint8).save(save_path)
        render_frames.append(img_uint8)

        mask_np = masks_np[i]
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        mask_path = mask_output_dir / f"mask_{i:04d}.png"
        Image.fromarray(mask_uint8).save(mask_path)
        # Convert grayscale mask to RGB for video
        mask_frames.append(np.stack([mask_uint8] * 3, axis=-1))

    # Save videos
    render_video_path = output_dir / "input_video.mp4"
    mask_video_path = mask_output_dir / "input_mask.mp4"

    video_fps = 30
    print(f"Saving render video to {render_video_path}")
    imageio.mimwrite(
        str(render_video_path),
        render_frames,
        fps=video_fps,
        codec="libx264",
        format="FFMPEG",
    )  # ty:ignore[no-matching-overload]

    print(f"Saving mask video to {mask_video_path}")
    imageio.mimwrite(
        str(mask_video_path),
        mask_frames,
        fps=video_fps,
        codec="libx264",
        format="FFMPEG",
    )  # ty:ignore[no-matching-overload]

    print("Done.")


if __name__ == "__main__":
    main()
