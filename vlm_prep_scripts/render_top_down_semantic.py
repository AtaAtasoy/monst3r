import argparse
from pathlib import Path
import json
import shutil
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw
from dreifus.matrix import CameraCoordinateConvention, Pose, PoseType, Intrinsics
from pytorch3d.io import IO
from pytorch3d.renderer import (
    AlphaCompositor,
    OrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)
from pytorch3d.structures import Pointclouds


import matplotlib.cm as cm
_colormap = cm.get_cmap('viridis')


def colormap_color(t):
    rgb = _colormap(t)[:3]
    return tuple(int(255 * c) for c in rgb)


def load_intrinsics(intr_path: Path) -> list[Intrinsics]:
    data = np.loadtxt(str(intr_path))
    if data.ndim == 1:
        data = data.reshape(1, -1)

    mats: list[Intrinsics] = []
    for row in data:
        if row.size != 9:
            raise ValueError(
                f"Expected 9 values per intrinsics row, got {row.size}")
        K = row.reshape(3, 3).astype(np.float64)
        mats.append(Intrinsics(matrix_or_fx=K))
    return mats


def load_tum_poses(path: Path) -> list[Pose]:
    poses = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
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


def make_topdown_pose(
    height: float = 1.5,
    look_at: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> Pose:
    cam_pos = np.array([look_at[0], look_at[1] - height, look_at[2]])

    z_cam = look_at - cam_pos
    z_cam = z_cam / np.linalg.norm(z_cam)

    up_hint = np.array([0.0, 0.0, 1.0])
    x_cam = np.cross(z_cam, up_hint)
    x_cam = x_cam / np.linalg.norm(x_cam)

    y_cam = np.cross(z_cam, x_cam)
    y_cam = y_cam / np.linalg.norm(y_cam)

    r_c2w = np.stack([x_cam, y_cam, z_cam], axis=1)

    mat = np.eye(4)
    mat[:3, :3] = r_c2w
    mat[:3, 3] = cam_pos

    return Pose(
        mat,
        pose_type=PoseType.CAM_2_WORLD,
        camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
    )


def _resolve_dirs(args):
    fg_dir = bg_dir = all_dir = root_dir = None

    if args.fg_dir:
        fg_dir = Path(args.fg_dir)
    if args.bg_dir:
        bg_dir = Path(args.bg_dir)

    if args.data_dir:
        data_dir = Path(args.data_dir)
        has_subdirs = (
            (data_dir / "foreground_points").is_dir()
            or (data_dir / "background_points").is_dir()
            or (data_dir / "all_points").is_dir()
        )
        has_root_plys = any(data_dir.glob("frame_*.ply"))

        if has_subdirs:
            root_dir = data_dir
            if fg_dir is None:
                fg_dir = data_dir / "foreground_points"
            if bg_dir is None:
                bg_dir = data_dir / "background_points"
            if all_dir is None:
                all_dir = data_dir / "all_points"
        elif has_root_plys:
            root_dir = data_dir
            name = data_dir.name.lower()
            if "foreground" in name and fg_dir is None:
                fg_dir = data_dir
            elif "background" in name and bg_dir is None:
                bg_dir = data_dir
            else:
                all_dir = data_dir
        else:
            root_dir = data_dir

    if root_dir is None:
        root_dir = fg_dir or bg_dir

    return fg_dir, bg_dir, all_dir, root_dir


def _list_frame_plys(ply_dir: Path) -> list[Path]:
    return sorted([f for f in ply_dir.glob("frame_*.ply") if "_masked" not in f.name])


def _load_points_and_features_and_normals(ply_path: Path, io: IO, device: torch.device):
    if ply_path is None or not ply_path.exists():
        return None, None, None

    cloud = io.load_pointcloud(str(ply_path), device=device)
    points = cloud.points_list()[0]
    if points.numel() == 0:
        return None, None, None

    features = cloud.features_list()[0]
    if features is None or features.numel() == 0:
        features = torch.ones_like(points)

    normals = cloud.normals_list()[0] if cloud.normals_list() is not None else cloud.estimate_normals(assign_to_self=True)
    return points, features, normals


def _compute_look_at(look_at_mode: str, source_poses: list[Pose], fg_files: list[Path], bg_files: list[Path], all_files: list[Path]) -> tuple[np.ndarray, str]:
    io_cpu = IO()

    def _bbox_center(paths: list[Path]):
        if not paths:
            return None
        mins = []
        maxs = []
        for pth in paths:
            cloud = io_cpu.load_pointcloud(str(pth), device="cpu")
            pts = cloud.points_list()[0]
            if pts.numel() == 0:
                continue
            mins.append(pts.min(dim=0)[0])
            maxs.append(pts.max(dim=0)[0])
        if not mins:
            return None
        pts_min = torch.stack(mins).min(dim=0)[0]
        pts_max = torch.stack(maxs).max(dim=0)[0]
        return ((pts_min + pts_max) / 2).numpy()

    if look_at_mode == "camera_centroid" and source_poses:
        cam_positions = np.array([p.get_translation() for p in source_poses])
        return cam_positions.mean(axis=0), "camera_centroid"

    if look_at_mode == "scene":
        candidates = []
        if fg_files:
            candidates += fg_files
        if bg_files:
            candidates += bg_files
        if not candidates and all_files:
            candidates = all_files
        center = _bbox_center(candidates)
        if center is not None:
            return center, "scene_bbox"

    if look_at_mode == "fg":
        center = _bbox_center(fg_files)
        if center is not None:
            return center, "fg_bbox"

    return np.array([0.0, 0.0, 0.0]), "origin"


def _collect_scene_bounds_in_view(scene_paths: list[Path], topdown_pose: Pose) -> tuple[float, float]:
    if not scene_paths:
        return 1.0, 1.0

    io_cpu = IO()

    pytorch3d_c2w_pose = topdown_pose.change_camera_coordinate_convention(
        new_camera_coordinate_convention=CameraCoordinateConvention.PYTORCH_3D, inplace=False
    )
    r = np.asarray(pytorch3d_c2w_pose.get_rotation_matrix(), dtype=np.float64)
    pytorch3d_w2c_pose = pytorch3d_c2w_pose.change_pose_type(
        new_pose_type=PoseType.WORLD_2_CAM, inplace=False
    )
    t = np.asarray(pytorch3d_w2c_pose.get_translation(), dtype=np.float64)

    min_x, max_x = float("inf"), float("-inf")
    min_y, max_y = float("inf"), float("-inf")

    for path in scene_paths:
        cloud = io_cpu.load_pointcloud(str(path), device="cpu")
        pts = cloud.points_list()[0]
        if pts.numel() == 0:
            continue
        pts_np = pts.detach().cpu().numpy().astype(np.float64, copy=False)
        pts_cam = pts_np @ r.T + t[None, :]
        min_x = min(min_x, float(np.min(pts_cam[:, 0])))
        max_x = max(max_x, float(np.max(pts_cam[:, 0])))
        min_y = min(min_y, float(np.min(pts_cam[:, 1])))
        max_y = max(max_y, float(np.max(pts_cam[:, 1])))

    if not np.isfinite([min_x, max_x, min_y, max_y]).all():
        return 1.0, 1.0

    half_w = max(abs(min_x), abs(max_x))
    half_h = max(abs(min_y), abs(max_y))
    half_w = max(half_w, 1e-4)
    half_h = max(half_h, 1e-4)
    return half_w, half_h


def _collect_xyz_bounds_in_view(
    xyz_world: np.ndarray,
    topdown_pose: Pose,

) -> tuple[float, float]:
    """Compute orthographic half extents (x/y) in the top-down *camera view frame* from world-space XYZ points."""
    if xyz_world is None:
        return 1.0, 1.0

    xyz_world = np.asarray(xyz_world, dtype=np.float64).reshape(-1, 3)
    if xyz_world.size == 0:
        return 1.0, 1.0

    pytorch3d_c2w_pose = topdown_pose.change_camera_coordinate_convention(
        new_camera_coordinate_convention=CameraCoordinateConvention.PYTORCH_3D, inplace=False
    )
    r = pytorch3d_c2w_pose.get_rotation_matrix()
    pytorch3d_w2c_pose = pytorch3d_c2w_pose.change_pose_type(
        new_pose_type=PoseType.WORLD_2_CAM, inplace=False
    )
    t = pytorch3d_w2c_pose.get_translation()

    # Match the convention used by _collect_scene_bounds_in_view: pts_cam = pts @ r.T + t
    xyz_cam = xyz_world @ r.T + t[None, :]
    xs = xyz_cam[:, 0]
    ys = xyz_cam[:, 1]

    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)

    half_w = float(max(abs(x_lo), abs(x_hi), 1e-4))
    half_h = float(max(abs(y_lo), abs(y_hi), 1e-4))
    return half_w, half_h


def _make_orthographic_camera(
    topdown_pose: Pose,
    img_w: int,
    img_h: int,
    scene_paths: list[Path],
    margin: float,
    device: torch.device,
    *,
    camera_positions_world: np.ndarray | None = None,
):
    pytorch3d_c2w_pose = topdown_pose.change_camera_coordinate_convention(
        new_camera_coordinate_convention=CameraCoordinateConvention.PYTORCH_3D, inplace=False
    )
    r = pytorch3d_c2w_pose.get_rotation_matrix()
    pytorch3d_w2c_pose = pytorch3d_c2w_pose.change_pose_type(
        new_pose_type=PoseType.WORLD_2_CAM, inplace=False
    )
    t = pytorch3d_w2c_pose.get_translation()

    half_w_scene, half_h_scene = _collect_scene_bounds_in_view(
        scene_paths, topdown_pose)
    if camera_positions_world is None:
        raise ValueError(
            "camera_positions_world is required (trajectory must be present)")
    cam_xyz = np.asarray(camera_positions_world,
                         dtype=np.float64).reshape(-1, 3)
    if cam_xyz.size == 0:
        raise ValueError(
            "camera_positions_world is empty (trajectory must be present)")

    half_w_cam, half_h_cam = _collect_xyz_bounds_in_view(cam_xyz, topdown_pose)

    # scene_and_cameras: cover both, but don't zoom out more than needed.
    half_w = max(float(half_w_scene), float(half_w_cam))
    half_h = max(float(half_h_scene), float(half_h_cam))

    half_w *= margin
    half_h *= margin

    fx = (img_w * 0.5) / half_w
    fy = (img_h * 0.5) / half_h
    cx = img_w * 0.5
    cy = img_h * 0.5

    camera = OrthographicCameras(
        focal_length=((fx, fy),),
        principal_point=((cx, cy),),
        R=r[None, :, :],
        T=t[None, :],
        in_ndc=False,
        image_size=((img_h, img_w),),
        device=device,
    )

    params = {
        "focal_length_xy": [float(fx), float(fy)],
        "principal_point_xy": [float(cx), float(cy)],
        "view_half_extent_xy": [float(half_w), float(half_h)],
        "margin": float(margin),
    }
    return camera, params


def _to_uint8_rgb(img: torch.Tensor) -> np.ndarray:
    image = img[..., :3].detach().cpu().numpy().clip(0, 1)
    return (image * 255).astype(np.uint8)


def _compute_mean_bbox_and_center_screen(
    camera,
    points: torch.Tensor,
    img_w: int,
    img_h: int,
    centered_bbox: bool = False,
):
    if points is None or points.numel() == 0:
        return None, None

    image_size = torch.tensor(
        [[img_h, img_w]], dtype=torch.float32, device=points.device)
    screen_pts = camera.transform_points_screen(
        points[None, ...], image_size=image_size)[0, :, :2]

    xs = screen_pts[:, 0]
    ys = screen_pts[:, 1]
    center_xy = None

    if centered_bbox:
        lower_q = 0.02
        upper_q = 0.98
        x_lo = torch.quantile(xs, lower_q)
        x_hi = torch.quantile(xs, upper_q)
        y_lo = torch.quantile(ys, lower_q)
        y_hi = torch.quantile(ys, upper_q)
        inlier = (xs >= x_lo) & (xs <= x_hi) & (ys >= y_lo) & (ys <= y_hi)

        xin = xs[inlier] if inlier.any() else xs
        yin = ys[inlier] if inlier.any() else ys

        mean_x = xin.mean()
        mean_y = yin.mean()
        center_xy = torch.stack([mean_x, mean_y])
        half_w = torch.quantile(torch.abs(xin - mean_x), 0.95)
        half_h = torch.quantile(torch.abs(yin - mean_y), 0.95)

        x0 = int(torch.floor(mean_x - half_w).item())
        y0 = int(torch.floor(mean_y - half_h).item())
        x1 = int(torch.ceil(mean_x + half_w).item())
        y1 = int(torch.ceil(mean_y + half_h).item())
    else:
        # Robust bbox: ignore extreme outliers using central quantiles in screen space.
        lower_q = 0.02
        upper_q = 0.98
        x0 = int(torch.floor(torch.quantile(xs, lower_q)).item())
        y0 = int(torch.floor(torch.quantile(ys, lower_q)).item())
        x1 = int(torch.ceil(torch.quantile(xs, upper_q)).item())
        y1 = int(torch.ceil(torch.quantile(ys, upper_q)).item())

    x0 = max(0, min(img_w - 1, x0))
    y0 = max(0, min(img_h - 1, y0))
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))

    if x1 <= x0 or y1 <= y0:
        return None, None

    mean_xy = center_xy if center_xy is not None else screen_pts.mean(dim=0)
    cx = int(round(mean_xy[0].item()))
    cy = int(round(mean_xy[1].item()))
    cx = max(0, min(img_w - 1, cx))
    cy = max(0, min(img_h - 1, cy))

    return (x0, y0, x1, y1), (cx, cy)


def _draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color=(0, 255, 255), width: int = 4, head_len: float = 14.0):
    x0, y0 = start
    x1, y1 = end
    draw.line([x0, y0, x1, y1], fill=color, width=width)

    dx = x1 - x0
    dy = y1 - y0
    norm = np.hypot(dx, dy)
    if norm < 1e-6:
        return

    ux = dx / norm
    uy = dy / norm
    left = (
        int(round(x1 - head_len * ux + 0.5 * head_len * uy)),
        int(round(y1 - head_len * uy - 0.5 * head_len * ux)),
    )
    right = (
        int(round(x1 - head_len * ux - 0.5 * head_len * uy)),
        int(round(y1 - head_len * uy + 0.5 * head_len * ux)),
    )
    draw.polygon([end, left, right], fill=color)


def _build_camera_frustum_world_points(
    pose_cam2world: Pose,
    intr: Intrinsics,
    *,
    depth_scale: float,
) -> np.ndarray:
    
    fx = intr.fx
    fy = intr.fy
    cx = intr.cx
    cy = intr.cy
    
    
    img_w, img_h = 2.0 * intr.cx, 2.0 * intr.cy
    img_w = max(img_w, 1.0)
    img_h = max(img_h, 1.0)

    # OpenCV camera coordinates: +X right, +Y down, +Z forward.
    corners_px = np.array(
        [
            [0.0, 0.0],
            [img_w - 1.0, 0.0],
            [img_w - 1.0, img_h - 1.0],
            [0.0, img_h - 1.0],
        ],
        dtype=np.float64,
    )
    z = float(depth_scale)
    x = (corners_px[:, 0] - cx) / max(fx, 1e-8) * z
    y = (corners_px[:, 1] - cy) / max(fy, 1e-8) * z
    corners_cam = np.stack([x, y, np.full_like(x, z)], axis=1)

    frustum_cam = np.concatenate(
        [np.zeros((1, 3), dtype=np.float64), corners_cam], axis=0
    )
    pose_mat = np.asarray(pose_cam2world.numpy(), dtype=np.float64)
    rot = pose_mat[:3, :3]
    trans = pose_mat[:3, 3]
    frustum_world = frustum_cam @ rot.T + trans[None, :]
    return frustum_world.astype(np.float32)


def _draw_frustum_wireframe(
    draw: ImageDraw.ImageDraw,
    points_xy: list[tuple[int, int]],
    *,
    color: tuple[int, int, int],
    width: int,
) -> None:
    if len(points_xy) != 5:
        return
    o, c0, c1, c2, c3 = points_xy
    draw.line([o, c0], fill=color, width=width)
    draw.line([o, c1], fill=color, width=width)
    draw.line([o, c2], fill=color, width=width)
    draw.line([o, c3], fill=color, width=width)
    draw.line([c0, c1], fill=color, width=width)
    draw.line([c1, c2], fill=color, width=width)
    draw.line([c2, c3], fill=color, width=width)
    draw.line([c3, c0], fill=color, width=width)


def _project_world_points_screen(
    camera: OrthographicCameras,
    world_xyz: np.ndarray,
    *,
    device: torch.device,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    world_xyz = np.asarray(world_xyz, dtype=np.float32).reshape(-1, 3)
    if world_xyz.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    points = torch.tensor(world_xyz, dtype=torch.float32, device=device)
    image_size = torch.tensor(
        [[img_h, img_w]], dtype=torch.float32, device=device
    )
    screen_pts = camera.transform_points_screen(
        points[None, ...], image_size=image_size
    )[0, :, :2]
    return screen_pts.detach().cpu().numpy()


def _create_video_from_frames(frame_pattern: str, output_video: Path, fps: int = 25) -> None:
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
        raise RuntimeError(
            f"ffmpeg failed ({output_video.name}): {result.stderr}")


def _trajectory_dict_from_points(points_xy: list[tuple[int, int] | None]) -> dict:
    traj = {}
    for idx, pt in enumerate(points_xy):
        key = f"t_{idx}"
        traj[key] = {"position": list(pt)} if pt is not None else {"position": None}
    return traj


def main():
    parser = argparse.ArgumentParser(
        description="Semantic top-down renderer with OrthographicCameras, foreground bbox, and motion arrows"
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--fg_dir", type=str, default=None)
    parser.add_argument("--bg_dir", type=str, default=None)
    parser.add_argument(
        "--render_context",
        type=str,
        default="all",
        choices=["all", "bg", "fg"],
        help="Point-cloud content for rendering canvas",
    )
    parser.add_argument("--height", type=float, default=0.5)
    parser.add_argument(
        "--look_at_mode",
        type=str,
        default="scene",
        choices=["origin", "camera_centroid", "scene", "fg"],
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Square render resolution (default: 512)",
    )
    parser.add_argument(
        "--ortho_margin",
        type=float,
        default=1,
        help="Margin multiplier for orthographic view extents",
    )
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fg_dir, bg_dir, all_dir, root_dir = _resolve_dirs(args)
    if root_dir is None:
        raise ValueError(
            "Could not resolve root directory. Provide --data_dir or --fg_dir/--bg_dir")

    traj_path = root_dir / "pred_traj.txt"
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {traj_path}")
    source_poses = load_tum_poses(traj_path)
    if not source_poses:
        raise ValueError(f"Trajectory file is empty or invalid: {traj_path}")
    intr_path = root_dir / "pred_intrinsics.txt"
    if not intr_path.exists():
        raise FileNotFoundError(f"Missing intrinsics file: {intr_path}")
    source_intrinsics = load_intrinsics(intr_path)
    if not source_intrinsics:
        raise ValueError(f"Intrinsics file is empty or invalid: {intr_path}")
    print(f"Loaded {len(source_intrinsics)} intrinsics from {intr_path}")

    fg_files = _list_frame_plys(fg_dir) if fg_dir and fg_dir.is_dir() else []
    bg_files = _list_frame_plys(bg_dir) if bg_dir and bg_dir.is_dir() else []
    all_files = _list_frame_plys(
        all_dir) if all_dir and all_dir.is_dir() else []

    if not fg_files:
        raise ValueError(
            "Foreground point clouds are required for bbox/motion visualization")

    look_at, source_tag = _compute_look_at(
        args.look_at_mode, source_poses, fg_files, bg_files, all_files)
    print(
        f"[look_at] mode={args.look_at_mode}, source={source_tag}, value={look_at}, height={args.height:.3f}")

    topdown_pose = make_topdown_pose(height=args.height, look_at=look_at)

    if args.resolution > 0:
        img_w = img_h = int(args.resolution)
    else:
        img_w = img_h = 512

    if args.render_context == "fg":
        n_frames = len(fg_files)
    elif args.render_context == "bg":
        n_frames = len(bg_files)
    else:
        n_frames = len(all_files)

    if n_frames == 0:
        raise ValueError("No frames found for selected render context")

    scene_paths_for_scale = []
    if args.render_context == "fg":
        scene_paths_for_scale.extend(fg_files)
    elif args.render_context == "bg":
        scene_paths_for_scale.extend(bg_files)
    else:
        scene_paths_for_scale.extend(all_files)

    src_cam_positions_np = np.array(
        [p.get_translation() for p in source_poses], dtype=np.float64)

    camera, camera_params = _make_orthographic_camera(
        topdown_pose=topdown_pose,
        img_w=img_w,
        img_h=img_h,
        scene_paths=scene_paths_for_scale,
        margin=args.ortho_margin,
        device=device,
        camera_positions_world=src_cam_positions_np,
    )

    raster_settings = PointsRasterizationSettings(
        image_size=(img_h, img_w),
        bin_size=0,
        radius=0.003,  # default value
        points_per_pixel=10,  # default value
    )

    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(
            cameras=camera, raster_settings=raster_settings),
        compositor=AlphaCompositor(),
    )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = root_dir / \
            f"top-down-semantic-ortho-{args.render_context}-{args.look_at_mode}"

    render_dir = output_dir / "render"
    bbox_dir = output_dir / "bbox_overlay"
    motion_dir = output_dir / "motion_overlay"
    render_dir.mkdir(parents=True, exist_ok=True)
    bbox_dir.mkdir(parents=True, exist_ok=True)
    motion_dir.mkdir(parents=True, exist_ok=True)

    io = IO()
    centers = []
    per_frame = []

    # Always calculate source camera trajectory in screen coordinates
    src_cam_positions = np.array([p.get_translation()
                                 for p in source_poses], dtype=np.float32)
    src_cam_positions = torch.tensor(
        src_cam_positions, dtype=torch.float32, device=device)
    image_size = torch.tensor(
        [[img_h, img_w]], dtype=torch.float32, device=device)
    src_screen_pts = camera.transform_points_screen(
        src_cam_positions[None, ...], image_size=image_size)[0, :, :2]
    src_traj_screen = [(int(round(x.item())), int(round(y.item())))
                       for x, y in src_screen_pts]

    frustum_depth_world = 0.08 * max(camera_params["view_half_extent_xy"])

    for idx in range(n_frames):
        print(
            f"Rendering semantic top-down frame {idx + 1}/{n_frames}...", end="\r")

        if args.render_context == "fg":
            context_paths = [fg_files[idx]] if idx < len(fg_files) else []
        elif args.render_context == "bg":
            context_paths = [bg_files[idx]] if idx < len(bg_files) else []
        else:
            context_paths = [all_files[idx]] if idx < len(all_files) else []

        fg_path = fg_files[idx] if idx < len(fg_files) else None
        fg_points, _, _ = _load_points_and_features_and_normals(
            fg_path, io=io, device=device
        )

        all_points = []
        all_features = []
        # all_normals = []
        for pth in context_paths:
            pts, feat, _ = _load_points_and_features_and_normals(
                pth, io=io, device=device
            )
            if pts is None:
                continue
            all_points.append(pts)
            all_features.append(feat)
            # all_normals.append(normals)

        if all_points:
            frame_pc = Pointclouds(
                points=[torch.cat(all_points, dim=0)],
                features=[torch.cat(all_features, dim=0)],
                # normals=[torch.cat(all_normals, dim=0)],
            )
            frame_pc.estimate_normals(assign_to_self=True)
            image = renderer(frame_pc)[0]
            img_np = _to_uint8_rgb(image)
        else:
            img_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        base_img = Image.fromarray(img_np)
        base_img.save(render_dir / f"top-down-render_{idx:04d}.png")

        # bbox, center = _compute_bbox_and_center_screen(camera, fg_points, img_w, img_h)
        bbox, center = _compute_mean_bbox_and_center_screen(
            camera,
            fg_points,
            img_w,
            img_h,
            centered_bbox=True,
        )
        centers.append(center)

        bbox_img = base_img.copy()
        bbox_draw = ImageDraw.Draw(bbox_img)
        if bbox is not None:
            bbox_draw.rectangle(bbox, outline=(255, 0, 0), width=3)
            bbox_draw.ellipse(
                [center[0] - 4, center[1] - 4, center[0] + 4, center[1] + 4],
                fill=(255, 255, 0),
            )

        bbox_img.save(bbox_dir / f"bbox-overlay_{idx:04d}.png")

        motion_img = bbox_img.copy()
        motion_draw = ImageDraw.Draw(motion_img)
        valid_centers = [c for c in centers if c is not None]

        # Draw the motion center trajectory as a polyline with color gradient, and an arrowhead at the end
        if len(valid_centers) >= 2:
            for i in range(len(valid_centers) - 1):
                t = i / max(1, len(valid_centers) -
                            2) if len(valid_centers) > 2 else 0
                color = colormap_color(t)
                motion_draw.line(
                    [valid_centers[i], valid_centers[i + 1]], fill=color, width=3)
            # Draw an arrowhead at the end of the trajectory
            _draw_arrow(
                motion_draw,
                valid_centers[-2],
                valid_centers[-1],
                color=colormap_color(1.0),
                width=4
            )

        if src_traj_screen:
            cam_idx = min(idx, len(src_traj_screen) - 1)
            cam_t = cam_idx / max(1, len(src_traj_screen) - 1) if len(src_traj_screen) > 1 else 0.0
            cam_color = colormap_color(cam_t)
            curr_pose = source_poses[cam_idx]
            curr_intr = (
                source_intrinsics[cam_idx]
                if cam_idx < len(source_intrinsics)
                else source_intrinsics[-1]
            )
            frustum_world = _build_camera_frustum_world_points(
                curr_pose,
                curr_intr,
                depth_scale=frustum_depth_world,
            )
            frustum_screen = _project_world_points_screen(
                camera,
                frustum_world,
                device=device,
                img_w=img_w,
                img_h=img_h,
            )
            frustum_px = [
                (int(round(float(x))), int(round(float(y))))
                for x, y in frustum_screen
            ]
            _draw_frustum_wireframe(
                motion_draw,
                frustum_px,
                color=cam_color,
                width=2,
            )

        motion_img.save(motion_dir / f"motion-overlay_{idx:04d}.png")

        frame_info = {
            "frame_index": idx,
            "bbox_xyxy": list(bbox) if bbox is not None else None,
            "center_xy": list(center) if center is not None else None,
        }
        per_frame.append(frame_info)

    print(f"\nSaved semantic renders to {output_dir}")

    moving_object_centers = [
        tuple(int(v) for v in frame["center_xy"]) if frame["center_xy"] is not None else None
        for frame in per_frame
    ]
    moving_object_trajectory = _trajectory_dict_from_points(moving_object_centers)
    capture_camera_trajectory = _trajectory_dict_from_points(src_traj_screen)
    topdown_camera_pixels = {
        "position": [
            int(round(camera_params["principal_point_xy"][0])),
            int(round(camera_params["principal_point_xy"][1])),
        ]
    }

    capture_camera_json = {
        "image_size": [img_h, img_w],
        "camera_trajectory_pixels": capture_camera_trajectory,
    }
    topdown_camera_json = {
        "image_size": [img_h, img_w],
        "topdown_camera_pixel": topdown_camera_pixels,
        "orthographic_params": camera_params,
    }
    moving_bbox_json = {
        "image_size": [img_h, img_w],
        "foreground_bbox_pixels": {
            f"t_{frame['frame_index']}": {
                "bbox_xyxy": frame["bbox_xyxy"],
                "center_xy": frame["center_xy"],
            }
            for frame in per_frame
        },
    }

    semantic_meta = {
        "projection_type": "orthographic",
        "image_size": [img_h, img_w],
        "camera_height": args.height,
        "orthographic_camera_to_world_4x4": topdown_pose.numpy().tolist(),
        "orthographic_cam_parameters": camera_params,
        "capture_camera_trajectory_pixels": [list(p) for p in src_traj_screen],
        # "foreground_motion_path_pixels": moving_object_trajectory,
        "per_frame_foreground_metadata": per_frame,
    }

    meta_path = output_dir / "semantic_topdown_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(semantic_meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    with open(output_dir / "capture_cameras_pixels.json", "w") as f:
        json.dump(capture_camera_json, f, indent=2)
    with open(output_dir / "topdown_camera_pixels.json", "w") as f:
        json.dump(topdown_camera_json, f, indent=2)
    with open(output_dir / "moving_bbox_pixels.json", "w") as f:
        json.dump(moving_bbox_json, f, indent=2)

    video_dir = output_dir / "vis-videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    _create_video_from_frames(
        str(render_dir / "top-down-render_%04d.png"),
        video_dir / "render.mp4",
    )
    _create_video_from_frames(
        str(bbox_dir / "bbox-overlay_%04d.png"),
        video_dir / "bbox_overlay.mp4",
    )
    _create_video_from_frames(
        str(motion_dir / "motion-overlay_%04d.png"),
        video_dir / "motion_overlay.mp4",
    )
    print(f"Saved videos to {video_dir}")

    temp_dir = output_dir / ".temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
