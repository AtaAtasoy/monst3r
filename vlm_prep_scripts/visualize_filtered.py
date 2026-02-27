"""
Visualize filtered foreground and background point clouds.
"""

import pyvista as pv
from dreifus.pyvista import add_coordinate_axes, add_camera_frustum
from dreifus.matrix import Pose, CameraCoordinateConvention, PoseType, Intrinsics
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import open3d as o3d


def load_tum_poses(path) -> list[Pose]:
    poses = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            _, tx, ty, tz, qw, qx, qy, qz = map(float, parts)
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
    intrinsics = []
    with open(path, "r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 9:
                K = np.array(vals).reshape(3, 3)
            else:
                continue
            intrinsics.append(Intrinsics(matrix_or_fx=K))
    return intrinsics


def load_point_clouds(directory: Path) -> list:
    if not directory.exists():
        return []

    ply_files = sorted(directory.glob("*.ply"))
    print(f"Found {len(ply_files)} point clouds in {directory}")

    pcds = []
    for ply_path in ply_files:
        o3d_pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(o3d_pcd.points)
        colors = np.asarray(o3d_pcd.colors)

        if len(points) == 0:
            continue

        pv_pcd = pv.PolyData(points)
        if colors.shape[0] > 0:
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            pv_pcd["RGB"] = colors
        pcds.append(pv_pcd)
    return pcds


def main(args):
    data_dir = Path(args.base_dir)
    fg_dir = data_dir / "foreground_points"
    bg_dir = data_dir / "background_points"

    # Load scene intrinsics and poses (optional, for context)
    intrinsics_path = data_dir / "pred_intrinsics.txt"
    traj_path = data_dir / "pred_traj.txt"

    source_poses = []
    intrinsics_list = []

    if args.show_cameras:
        if intrinsics_path.exists():
            intrinsics_list = load_intrinsics(intrinsics_path)

        if traj_path.exists():
            source_poses = load_tum_poses(traj_path)

    # Load point clouds
    fg_pcds = load_point_clouds(fg_dir) if args.show_foreground else []
    bg_pcds = load_point_clouds(bg_dir) if args.show_background else []

    # --- Interactive visualization ---
    p = pv.Plotter()
    add_coordinate_axes(p)

    # Add unit cube (centered at origin) as a spatial reference
    unit_cube = pv.Cube(
        center=(0.0, 0.0, 0.0), x_length=1.0, y_length=1.0, z_length=1.0
    )
    p.add_mesh(
        unit_cube,
        color="gray",
        opacity=0.1,
        style="surface",
        show_edges=True,
        edge_color="gray",
    )

    # Add Foreground Points
    for i, pcd in enumerate(fg_pcds):
        # We can just add them.
        # If we want to distinguish, maybe add scalar? But they have RGB.
        p.add_mesh(pcd, rgb=True, point_size=3)

    # Add Background Points
    for i, pcd in enumerate(bg_pcds):
        if args.dim_background:
            p.add_mesh(pcd, color="gray", opacity=0.3, point_size=2)
        else:
            p.add_mesh(pcd, rgb=True, point_size=2)

    # Show source camera frustums
    if args.show_cameras and source_poses and intrinsics_list:
        # Use first intrinsic if not enough
        intr = intrinsics_list[0]

        # Downsample cameras if too many?
        display_poses = list(enumerate(source_poses))

        for i, pose in display_poses:
            curr_intr = intrinsics_list[i] if i < len(intrinsics_list) else intr
            add_camera_frustum(p, pose, curr_intr, color="green")

    print("Starting visualization...")
    p.show()


if __name__ == "__main__":
    parser = ArgumentParser(description="Visualize filtered point clouds.")
    parser.add_argument(
        "--base_dir",
        type=str,
        help="Path to the base directory containing filtered subdirectories",
    )
    parser.add_argument(
        "--show_cameras",
        action="store_true",
        help="Show source camera frustums",
    )
    parser.add_argument(
        "--hide_foreground",
        dest="show_foreground",
        action="store_false",
        default=True,
        help="Hide foreground points",
    )
    parser.add_argument(
        "--hide_background",
        dest="show_background",
        action="store_false",
        default=True,
        help="Hide background points",
    )
    parser.add_argument(
        "--dim_background",
        action="store_true",
        help="Show background points in gray/transparent to highlight foreground",
    )

    args = parser.parse_args()
    main(args)
