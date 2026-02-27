"""
Visualize a normalized filtered scene from a top-down camera using orthographic projection.

Supports TWO directory layouts:

1. Joint normalization (--normalize):

    normalized/
    ├── pred_traj.txt
    ├── pred_intrinsics.txt
    ├── normalization_params.json
    ├── filtered_scene/       (all surviving points)
    ├── foreground_points/    (dynamic / masked objects)
    └── background_points/    (static scene)

2. Independent normalization (--normalize_separate):
   Two *flat* directories with PLYs at root:

    normalized_foreground_points/
    ├── pred_traj.txt
    ├── pred_intrinsics.txt
    ├── normalization_params.json
    └── frame_*.ply

    normalized_background_points/
    └── (same layout)

   Point --data_dir at the flat directory, or use --fg_dir / --bg_dir.

Usage:
    # Joint layout
    python visualize_filtered_topdown_orthographic.py --data_dir .../normalized --show both
    python visualize_filtered_topdown_orthographic.py --data_dir .../normalized --show fg --render

    # Independent layout (two flat dirs)
    python visualize_filtered_topdown_orthographic.py --fg_dir .../normalized_foreground_points --render
    python visualize_filtered_topdown_orthographic.py --bg_dir .../normalized_background_points --render
    python visualize_filtered_topdown_orthographic.py --fg_dir .../normalized_foreground_points --bg_dir .../normalized_background_points --show both --render
"""

import pyvista as pv
from dreifus.pyvista import add_coordinate_axes, add_camera_frustum, render_from_camera
from dreifus.matrix import Pose, CameraCoordinateConvention, PoseType, Intrinsics
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import open3d as o3d


# ──────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────

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


def load_ply_dir(ply_dir: Path) -> list[pv.PolyData]:
    """Load all frame_*.ply files from a directory as PyVista PolyData."""
    ply_files = sorted(
        [f for f in ply_dir.glob("frame_*.ply") if "_masked" not in f.name]
    )
    pcds = []
    for ply_path in ply_files:
        o3d_pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(o3d_pcd.points)
        colors = np.asarray(o3d_pcd.colors)

        if len(points) == 0:
            pcds.append(None)
            continue

        pv_pcd = pv.PolyData(points)
        if colors.shape[0] > 0:
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            pv_pcd["RGB"] = colors
        pcds.append(pv_pcd)
    return pcds


# ──────────────────────────────────────────────────────────────
# Top-down camera
# ──────────────────────────────────────────────────────────────

def make_topdown_pose(
    height: float = 1.5,
    look_at: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> Pose:
    """
    Top-down camera in OpenCV convention (X right, Y down, Z forward).
    Camera is placed at (look_at[0], look_at[1] - height, look_at[2])
    looking straight down along +Y world.
    """
    cam_pos = np.array([look_at[0], look_at[1] - height, look_at[2]])

    z_cam = look_at - cam_pos
    z_cam = z_cam / np.linalg.norm(z_cam)

    up_hint = np.array([0.0, 0.0, 1.0])
    x_cam = np.cross(z_cam, up_hint)
    x_cam = x_cam / np.linalg.norm(x_cam)

    y_cam = np.cross(z_cam, x_cam)
    y_cam = y_cam / np.linalg.norm(y_cam)

    R_c2w = np.stack([x_cam, y_cam, z_cam], axis=1)

    mat = np.eye(4)
    mat[:3, :3] = R_c2w
    mat[:3, 3] = cam_pos

    return Pose(
        mat,
        pose_type=PoseType.CAM_2_WORLD,
        camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
    )


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def _resolve_dirs(args):
    """
    Resolve FG / BG / filtered directories and a root for traj / intrinsics.

    Supports:
      --data_dir  (joint layout with sub-dirs)
      --fg_dir / --bg_dir  (independent flat layouts)
    """
    fg_dir = bg_dir = filt_dir = root_dir = None

    if args.fg_dir:
        fg_dir = Path(args.fg_dir)
    if args.bg_dir:
        bg_dir = Path(args.bg_dir)

    if args.data_dir:
        data_dir = Path(args.data_dir)
        # Detect layout: does this dir contain sub-dirs or PLYs at root?
        has_subdirs = (data_dir / "foreground_points").is_dir() or \
                      (data_dir / "background_points").is_dir() or \
                      (data_dir / "filtered_scene").is_dir()
        has_root_plys = any(data_dir.glob("frame_*.ply"))

        if has_subdirs:
            # Joint layout
            root_dir = data_dir
            if fg_dir is None:
                fg_dir = data_dir / "foreground_points"
            if bg_dir is None:
                bg_dir = data_dir / "background_points"
            if filt_dir is None:
                filt_dir = data_dir / "filtered_scene"
        elif has_root_plys:
            # Flat layout – treat data_dir itself as the PLY source
            root_dir = data_dir
            # Guess category from dir name
            name = data_dir.name.lower()
            if "foreground" in name and fg_dir is None:
                fg_dir = data_dir
            elif "background" in name and bg_dir is None:
                bg_dir = data_dir
            else:
                # Generic – treat as "all"
                filt_dir = data_dir
        else:
            root_dir = data_dir

    # Determine root_dir for trajectory / intrinsics
    if root_dir is None:
        root_dir = fg_dir or bg_dir  # fall back

    return fg_dir, bg_dir, filt_dir, root_dir


def main(args):
    fg_dir, bg_dir, filt_dir, root_dir = _resolve_dirs(args)

    # ── Load trajectory & intrinsics from the root ───────────
    intrinsics_path = root_dir / "pred_intrinsics.txt"
    intrinsics_list = load_intrinsics(intrinsics_path)
    intr = intrinsics_list[0]

    traj_path = root_dir / "pred_traj.txt"
    source_poses = load_tum_poses(traj_path) if traj_path.exists() else []

    # ── Determine what to show ───────────────────────────────
    show = args.show  # "fg", "bg", "both", or "all"

    # Auto-detect show mode when using --fg_dir / --bg_dir
    if show == "all" and (args.fg_dir or args.bg_dir):
        if args.fg_dir and args.bg_dir:
            show = "both"
        elif args.fg_dir:
            show = "fg"
        else:
            show = "bg"
        print(f"[auto] --show set to '{show}' based on provided dirs")

    # ── Load point clouds ────────────────────────────────────
    fg_pcds, bg_pcds, filtered_pcds = [], [], []

    if show in ("fg", "both"):
        if fg_dir and fg_dir.is_dir():
            fg_pcds = load_ply_dir(fg_dir)
            print(f"Loaded {len(fg_pcds)} foreground point clouds from {fg_dir}")
        else:
            print(f"Warning: foreground dir not found ({fg_dir})")

    if show in ("bg", "both"):
        if bg_dir and bg_dir.is_dir():
            bg_pcds = load_ply_dir(bg_dir)
            print(f"Loaded {len(bg_pcds)} background point clouds from {bg_dir}")
        else:
            print(f"Warning: background dir not found ({bg_dir})")

    if show == "all":
        if filt_dir and filt_dir.is_dir():
            filtered_pcds = load_ply_dir(filt_dir)
            print(f"Loaded {len(filtered_pcds)} filtered-scene point clouds from {filt_dir}")
        else:
            print(f"Warning: filtered_scene dir not found ({filt_dir})")

    # ── Top-down camera pose ─────────────────────────────────
    height = args.height if args.height is not None else 0.5

    if args.adaptive and source_poses:
        cam_positions = np.array([p.get_translation() for p in source_poses])
        look_at = cam_positions.mean(axis=0)
        print(f"[adaptive] look_at: {look_at},  height: {height:.3f}")
    else:
        look_at = np.array([0.0, 0.0, 0.0])
        print(f"[fixed] look_at: {look_at},  height: {height:.3f}")

    topdown_pose = make_topdown_pose(height=height, look_at=look_at)
    print(f"Top-down camera position: {topdown_pose.get_translation()}")

    # ── Colours for FG / BG distinction ──────────────────────
    # FG uses original RGB; BG can optionally be tinted for visibility
    fg_color_label = "foreground (RGB)"
    bg_color_label = "background (RGB)"

    # ── Interactive plotter ──────────────────────────────────
    p = pv.Plotter()
    add_coordinate_axes(p)

    # Unit cube reference
    unit_cube = pv.Cube(center=(0, 0, 0), x_length=1, y_length=1, z_length=1)
    p.add_mesh(
        unit_cube, color="gray", opacity=0.15,
        style="surface", show_edges=True, edge_color="gray",
    )

    # Add point clouds to interactive view
    def _add_pcds(pcds, label):
        for pcd in pcds:
            if pcd is not None:
                p.add_mesh(pcd, rgb=True)

    if show == "all":
        _add_pcds(filtered_pcds, "filtered")
    else:
        if show in ("fg", "both"):
            _add_pcds(fg_pcds, fg_color_label)
        if show in ("bg", "both"):
            _add_pcds(bg_pcds, bg_color_label)

    # Source camera frustums
    if args.show_source_cameras and source_poses:
        src_intr = (
            intrinsics_list
            if len(intrinsics_list) == len(source_poses)
            else [intr] * len(source_poses)
        )
        display_poses = (
            list(enumerate(source_poses)) if args.render else [(0, source_poses[0])]
        )
        for i, pose in display_poses:
            add_camera_frustum(p, pose, src_intr[i], color="green")

    # Top-down frustum
    add_camera_frustum(p, topdown_pose, intr, color="red")

    projection = args.projection

    # ── Render to images ─────────────────────────────────────
    if args.render:
        if args.resolution is not None:
            img_w = int(args.resolution)
            img_h = int(args.resolution)
            if img_w <= 0:
                img_w, img_h = 1024, 1024
                print("[warn] Invalid --resolution; falling back to 1024x1024")
            else:
                print(f"[render] Using overridden square resolution: {img_w}x{img_h}")
        else:
            img_w = int(round(2.0 * float(intr.cx)))
            img_h = int(round(2.0 * float(intr.cy)))
            if img_w <= 0 or img_h <= 0:
                img_w, img_h = 1024, 1024
                print("[warn] Invalid intrinsics principal point; falling back to 1024x1024")
            else:
                print(
                    f"[render] Using approximated source resolution from intrinsics: "
                    f"{img_w}x{img_h} (W≈2*cx, H≈2*cy)"
                )

        suffix = show  # "fg", "bg", "both", "all"
        render_dir = root_dir / f"top-down-renders-{projection}-{suffix}"
        render_dir.mkdir(parents=True, exist_ok=True)

        # Build per-frame list of meshes to render
        if show == "all":
            frame_pcds = filtered_pcds
        elif show == "both":
            # Merge fg + bg per frame (pad shorter list with None)
            n = max(len(fg_pcds), len(bg_pcds))
            frame_pcds = []
            for i in range(n):
                meshes = []
                if i < len(fg_pcds) and fg_pcds[i] is not None:
                    meshes.append(fg_pcds[i])
                if i < len(bg_pcds) and bg_pcds[i] is not None:
                    meshes.append(bg_pcds[i])
                frame_pcds.append(meshes if meshes else None)
        elif show == "fg":
            frame_pcds = fg_pcds
        else:  # bg
            frame_pcds = bg_pcds

        for idx, item in enumerate(frame_pcds):
            print(f"Rendering top-down ({suffix}) {idx}/{len(frame_pcds)} …", end="\r")
            rp = pv.Plotter(window_size=[img_w, img_h], off_screen=True)
            rp.background_color = (0, 0, 0, 0)

            if item is None:
                pass
            elif isinstance(item, list):
                for m in item:
                    rp.add_mesh(m, rgb=True, point_size=3)
            else:
                rp.add_mesh(item, rgb=True, point_size=3)

            # Camera setup
            cam_pos = topdown_pose.get_translation()
            rp.camera.position = (cam_pos[0], cam_pos[1], cam_pos[2])
            rp.camera.focal_point = look_at
            rp.camera.up = (0, 0, 1)

            if projection == "ortho":
                rp.camera.enable_parallel_projection()
                cam_distance = float(np.linalg.norm(np.asarray(cam_pos) - np.asarray(look_at)))
                rp.camera.parallel_scale = cam_distance * img_h / (2.0 * float(intr.fy))
            else:
                rp.camera.disable_parallel_projection()
                fov_y = 2.0 * np.arctan(img_h / (2.0 * float(intr.fy)))
                rp.camera.view_angle = np.degrees(fov_y)

            rendered = rp.screenshot(transparent_background=True)
            rp.close()

            out_path = render_dir / f"top-down-render_{idx:04d}.png"
            Image.fromarray(rendered).save(out_path)

        print(f"\nSaved {len(frame_pcds)} top-down renders to {render_dir}")

    p.show()


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Top-down visualization of normalized filtered scene"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Path to normalized/ (joint layout) or a flat normalized dir",
    )
    parser.add_argument(
        "--fg_dir", type=str, default=None,
        help="Path to the independently normalized foreground directory",
    )
    parser.add_argument(
        "--bg_dir", type=str, default=None,
        help="Path to the independently normalized background directory",
    )
    parser.add_argument(
        "--show", type=str, default="all",
        choices=["fg", "bg", "both", "all"],
        help="What to display: fg (foreground), bg (background), both (fg+bg side by side), "
             "all (filtered_scene combined). Default: all",
    )
    parser.add_argument(
        "--height", type=float, default=None,
        help="Height of top-down camera above scene centre",
    )
    parser.add_argument(
        "--show_source_cameras", action="store_true",
        help="Show the source camera frustums",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render per-frame images from the top-down camera and save to disk",
    )
    parser.add_argument(
        "--adaptive", action="store_true",
        help="Position top-down camera based on camera centroid instead of origin",
    )
    parser.add_argument(
        "--projection", type=str, default="ortho", choices=["ortho", "perspective"],
        help="Projection type for rendering: ortho or perspective. Default: ortho",
    )
    parser.add_argument(
        "--resolution", type=int, default=None,
        help="Square render resolution override. If set, output is RESOLUTION x RESOLUTION. "
             "If not set, resolution is approximated from intrinsics (W≈2*cx, H≈2*cy).",
    )
    args = parser.parse_args()
    main(args)
