import os
import json
import numpy as np
import pyvista as pv
import open3d as o3d
from pathlib import Path
from PIL import Image
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from dreifus.pyvista import add_coordinate_axes, add_camera_frustum
from dreifus.matrix import Pose, CameraCoordinateConvention, PoseType, Intrinsics


def _to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def export_ortho_camera_params(
    out_path: Path,
    look_at: np.ndarray,
    cam_pos: np.ndarray,
    view_up_for_pyvista: np.ndarray,
    parallel_scale: float,
    use_novel_topdown: bool,
    up_world: np.ndarray,
    x_cam: np.ndarray,
    y_cam: np.ndarray,
):
    """Export orthographic top-down camera parameters to JSON.

    Includes both PyVista camera parameters and an OpenCV CAM_2_WORLD pose matrix
    for the novel top-down camera (when available).
    """
    z_cam = _normalize(-up_world) if use_novel_topdown else _normalize(np.cross(x_cam, y_cam))
    if use_novel_topdown:
        R_c2w = np.stack([_normalize(x_cam), _normalize(y_cam), z_cam], axis=1)
    else:
        # best-effort: keep provided axes
        R_c2w = np.stack([_normalize(x_cam), _normalize(y_cam), _normalize(z_cam)], axis=1)

    T_c2w = np.eye(4, dtype=np.float64)
    T_c2w[:3, :3] = R_c2w
    T_c2w[:3, 3] = cam_pos

    payload = {
        "camera_type": "orthographic",
        "convention": {
            "opencv_camera_axes": {"x": "right", "y": "down", "z": "forward"},
            "pose_type": "CAM_2_WORLD",
        },
        "pyvista": {
            "position": _to_list(np.asarray(cam_pos, dtype=float)),
            "focal_point": _to_list(np.asarray(look_at, dtype=float)),
            "up": _to_list(np.asarray(view_up_for_pyvista, dtype=float)),
            "parallel_scale": float(parallel_scale),
            "parallel_projection": True,
        },
        "novel_topdown": {
            "enabled": bool(use_novel_topdown),
            "look_at": _to_list(np.asarray(look_at, dtype=float)),
            "up_world_est": _to_list(np.asarray(up_world, dtype=float)),
            "x_cam_world": _to_list(np.asarray(x_cam, dtype=float)),
            "y_cam_world": _to_list(np.asarray(y_cam, dtype=float)),
            "z_cam_world": _to_list(np.asarray(z_cam, dtype=float)),
            "T_c2w": _to_list(T_c2w.astype(float)),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))


def quat_wxyz_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Unit quaternion (w,x,y,z) -> 3x3 rotation."""
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_pred_traj_c2w_opencv(path: Path):
    """Load TUM-style trajectory: t tx ty tz qw qx qy qz.

    Returns:
      - ts: (N,) float
      - C: (N,3) camera centers in world
      - R_c2w: (N,3,3) camera-to-world rotation

    Convention: OpenCV camera axes (+X right, +Y down, +Z forward).
    Pose type: CAM_2_WORLD.
    """
    data = np.loadtxt(str(path), dtype=np.float64)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != 8:
        raise ValueError(f"Expected 8 columns in pred_traj, got {data.shape[1]}")

    ts = data[:, 0]
    C = data[:, 1:4]
    qwxyz = data[:, 4:8]
    R_c2w = np.stack([quat_wxyz_to_rotmat(*row) for row in qwxyz], axis=0)
    return ts, C, R_c2w


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / (n + eps)


def estimate_world_up_from_opencv_c2w(R_c2w: np.ndarray) -> np.ndarray:
    """Estimate world 'up' from OpenCV CAM_2_WORLD rotations.

    In OpenCV camera coordinates, +Y is down, so camera up is -Y_cam.
    With CAM_2_WORLD, down_world is the 2nd column of R.
    Therefore up_world_i = -R[:,1].
    """
    down_world = R_c2w[:, :, 1]
    ups = -down_world
    return _normalize(np.mean(ups, axis=0))


def make_topdown_basis_from_up(up_world: np.ndarray, right_hint_world: np.ndarray | None = None):
    """Build OpenCV-style camera axes in world for a top-down view.

    We want a novel top-down camera that looks "down" along -up_world.
      z_cam_world = forward = -up_world
    Choose x_cam_world (right) from a hint projected onto the view plane,
    and y_cam_world (down) to complete a right-handed basis.

    Returns (x_cam_world, y_cam_world, z_cam_world).
    """
    up_world = _normalize(up_world)
    z_cam = _normalize(-up_world)

    if right_hint_world is None:
        right_hint_world = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(right_hint_world, z_cam))) > 0.95:
            right_hint_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    right_hint_world = _normalize(right_hint_world)
    x_cam = right_hint_world - z_cam * float(np.dot(right_hint_world, z_cam))
    x_cam = _normalize(x_cam)

    # OpenCV: y is down; ensure right-handed basis: x x y = z => y = z x x
    y_cam = _normalize(np.cross(z_cam, x_cam))

    return x_cam, y_cam, z_cam


def stream_scene_stats(ply_paths: list[Path], max_points: int, seed: int = 0):
    """Compute centroid and an unbiased subsample of points from many PLYs."""
    rng = np.random.default_rng(seed)

    total_points = 0
    sum_xyz = np.zeros(3, dtype=np.float64)

    if max_points > 0:
        reservoir = np.empty((max_points, 3), dtype=np.float64)
        res_count = 0
        seen = 0
    else:
        reservoir = np.zeros((0, 3), dtype=np.float64)
        res_count = 0
        seen = 0

    for p in ply_paths:
        if not p.exists():
            continue
        pcd = o3d.io.read_point_cloud(str(p))
        pts = np.asarray(pcd.points)
        if pts.size == 0:
            continue

        pts = pts.astype(np.float64, copy=False)
        n = pts.shape[0]
        total_points += n
        sum_xyz += pts.sum(axis=0)

        if max_points <= 0:
            continue

        for idx in range(n):
            seen += 1
            pt = pts[idx]
            if res_count < max_points:
                reservoir[res_count] = pt
                res_count += 1
            else:
                j = int(rng.integers(0, seen))
                if j < max_points:
                    reservoir[j] = pt

    centroid = (sum_xyz / max(1, total_points)).astype(np.float64)
    return centroid, reservoir[:res_count]


def compute_fit_extents(points_world: np.ndarray, origin: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, up_axis: np.ndarray):
    rel = points_world - origin[None, :]
    dx = rel @ x_axis
    dy = rel @ y_axis
    du = rel @ up_axis

    max_abs_x = float(np.max(np.abs(dx))) if dx.size else 0.0
    max_abs_y = float(np.max(np.abs(dy))) if dy.size else 0.0
    u_min = float(np.min(du)) if du.size else 0.0
    u_max = float(np.max(du)) if du.size else 0.0

    return max_abs_x, max_abs_y, u_min, u_max


def load_tum_poses(path: Path) -> list[Pose]:
    """Load TUM-style poses as dreifus Pose objects (OpenCV convention, CAM_2_WORLD)."""
    poses: list[Pose] = []
    if not path.exists():
        return poses
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


def load_intrinsics(path: Path) -> list[Intrinsics]:
    """Load per-frame intrinsics as dreifus Intrinsics objects from pred_intrinsics.txt."""
    intrinsics: list[Intrinsics] = []
    if not path.exists():
        return intrinsics
    with open(path, "r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) != 9:
                continue
            K = np.array(vals, dtype=np.float64).reshape(3, 3)
            intrinsics.append(Intrinsics(matrix_or_fx=K))
    return intrinsics


def visualize_scene(
    root: Path,
    fg_files: list[Path],
    bg_files: list[Path],
    look_at: np.ndarray,
    cam_pos: np.ndarray,
    view_up_for_pyvista: np.ndarray,
    parallel_scale: float,
    x_cam: np.ndarray,
    y_cam: np.ndarray,
    up_world: np.ndarray,
    use_novel_topdown: bool,
    max_frames: int,
    show_source_cameras: bool,
    show_topdown_camera: bool,
    frustum_depth: float,
):
    """Interactive PyVista scene viewer for sanity-checking camera/axes."""
    p = pv.Plotter()
    add_coordinate_axes(p)

    # light unit cube reference
    unit_cube = pv.Cube(center=(0.0, 0.0, 0.0), x_length=1.0, y_length=1.0, z_length=1.0)
    p.add_mesh(unit_cube, color="gray", opacity=0.15, style="surface", show_edges=True, edge_color="gray")

    # add a few pointcloud frames
    n = min(max_frames, len(fg_files))
    for i in range(n):
        if i < len(bg_files):
            bg = load_ply_as_pv(bg_files[i])
            if bg is not None:
                p.add_mesh(bg, rgb=True, opacity=0.25, point_size=2)
        fg = load_ply_as_pv(fg_files[i])
        if fg is not None:
            p.add_mesh(fg, rgb=True, opacity=1.0, point_size=3)

    # camera frustums
    traj_path = root / "pred_traj.txt"
    intr_path = root / "pred_intrinsics.txt"
    if show_source_cameras and traj_path.exists() and intr_path.exists():
        source_poses = load_tum_poses(traj_path)
        intrinsics_list = load_intrinsics(intr_path)
        if source_poses and intrinsics_list:
            intr0 = intrinsics_list[0]
            for i in range(min(n, len(source_poses))):
                curr_intr = intrinsics_list[i] if i < len(intrinsics_list) else intr0
                add_camera_frustum(p, source_poses[i], curr_intr, color="green")

    if show_topdown_camera and use_novel_topdown:
        # Visualize the novel top-down camera using dreifus Pose + Intrinsics.
        intrinsics_list = load_intrinsics(intr_path)
        if intrinsics_list:
            intr0 = intrinsics_list[0]
            z_cam = _normalize(-up_world)
            R_td = np.stack([_normalize(x_cam), _normalize(y_cam), z_cam], axis=1)

            mat = np.eye(4, dtype=np.float64)
            mat[:3, :3] = R_td
            mat[:3, 3] = cam_pos
            topdown_pose = Pose(
                mat,
                pose_type=PoseType.CAM_2_WORLD,
                camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
            )
            add_camera_frustum(p, topdown_pose, intr0, color="red")

    # set interactive camera to match the novel top-down orthographic view
    p.camera.enable_parallel_projection()
    p.camera.position = tuple(cam_pos)
    p.camera.focal_point = tuple(look_at)
    p.camera.up = tuple(view_up_for_pyvista)
    p.camera.parallel_scale = float(parallel_scale)

    p.show()

# ──────────────────────────────────────────────────────────────
# Loaders & Helpers
# ──────────────────────────────────────────────────────────────

def load_ply_as_pv(path):
    """Load PLY file and convert to PyVista PolyData with RGB support."""
    if not path.exists():
        return None
    o3d_pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(o3d_pcd.points)
    cols = np.asarray(o3d_pcd.colors)
    if len(pts) == 0:
        return None
    pv_pcd = pv.PolyData(pts)
    if cols.shape[0] > 0:
        pv_pcd["RGB"] = (cols * 255).astype(np.uint8) if cols.max() <= 1.0 else cols
    return pv_pcd

def get_semantic_color(frame_idx, total_frames, cmap_name="plasma"):
    """Generate an RGB color based on temporal progress."""
    cmap = plt.get_cmap(cmap_name)
    return np.array(cmap(frame_idx / max(1, total_frames))[:3])

# ──────────────────────────────────────────────────────────────
# Core Visualization Logic
# ──────────────────────────────────────────────────────────────

def create_bev_map(pts, res=512):
    """Creates a 2D Height + Density map (BEV Occupancy Grid)."""
    # Mapping points from unit cube [-0.5, 0.5] to image grid [0, res]
    u = ((pts[:, 0] + 0.5) * (res - 1)).astype(int)
    v = ((pts[:, 2] + 0.5) * (res - 1)).astype(int) # Z is depth in OpenCV, becomes V
    
    mask = (u >= 0) & (u < res) & (v >= 0) & (v < res)
    u, v, y = u[mask], v[mask], pts[mask][:, 1] # Y is height

    bev = np.zeros((res, res, 3), dtype=np.uint8)
    for i in range(len(u)):
        # Channel R: Max Height
        h_val = int(np.clip((y[i] + 0.5) * 255, 0, 255))
        bev[v[i], u[i], 0] = max(bev[v[i], u[i], 0], h_val)
        # Channel G: Density (Saturation)
        bev[v[i], u[i], 1] = min(255, bev[v[i], u[i], 1] + 30)
        # Channel B: Occupancy Flag
        bev[v[i], u[i], 2] = 255
    return bev


def create_bev_map_topdown(
    pts_world: np.ndarray,
    look_at: np.ndarray,
    x_cam: np.ndarray,
    y_cam: np.ndarray,
    up_world: np.ndarray,
    extent: float,
    u_min: float,
    u_max: float,
    res: int = 512,
):
    """BEV aligned with the novel top-down camera basis.

    Image axes:
      - u increases along +x_cam (OpenCV +X right)
      - v increases along +y_cam (OpenCV +Y down)
    Height channel uses projection onto up_world.
    """
    rel = pts_world - look_at[None, :]
    dx = rel @ x_cam
    dy = rel @ y_cam
    du = rel @ up_world

    extent = float(max(extent, 1e-6))
    u = ((dx / (2.0 * extent) + 0.5) * (res - 1)).astype(np.int32)
    v = ((dy / (2.0 * extent) + 0.5) * (res - 1)).astype(np.int32)

    mask = (u >= 0) & (u < res) & (v >= 0) & (v < res)
    u = u[mask]
    v = v[mask]
    du = du[mask]

    bev = np.zeros((res, res, 3), dtype=np.uint8)

    denom = (u_max - u_min) if (u_max - u_min) > 1e-9 else 1.0
    h = (du - u_min) / denom
    h_val = np.clip((h * 255.0).astype(np.uint8), 0, 255)

    bev[v, u, 2] = 255
    np.add.at(bev[:, :, 1], (v, u), 30)
    np.clip(bev[:, :, 1], 0, 255, out=bev[:, :, 1])

    for k in range(u.shape[0]):
        if h_val[k] > bev[v[k], u[k], 0]:
            bev[v[k], u[k], 0] = h_val[k]

    return bev

def main(args):
    root = Path(args.data_dir)
    fg_dir = root / "foreground_points"
    bg_dir = root / "background_points"
    
    # Setup Output Directories
    out_root = root / "master_visualizations"
    ghost_dir = out_root / "temporal_ghosting"
    bev_dir = out_root / "bev_occupancy"
    rgb_dir = out_root / "rgb_renders"
    rgb_fg_dir = rgb_dir / "foreground"
    rgb_bg_dir = rgb_dir / "background"

    for d in [ghost_dir, bev_dir]:
        d.mkdir(parents=True, exist_ok=True)
    if args.save_rgb_renders:
        for d in [rgb_dir, rgb_fg_dir, rgb_bg_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # File Discovery
    fg_files = sorted(list(fg_dir.glob("frame_*.ply")))
    bg_files = sorted(list(bg_dir.glob("frame_*.ply")))
    num_frames = len(fg_files)

    print(f"Starting Master Render: {num_frames} frames found.")

    # 1. Load Normalization (optional; informational)
    norm_path = root / "normalization_params.json"
    if norm_path.exists():
        with open(norm_path, 'r') as f:
            norm_params = json.load(f)
            if 'scale' in norm_params:
                print(f"Loaded normalization: Scale {norm_params['scale']:.4f}")

    # 2. Build a novel top-down camera aligned to OpenCV CAM_2_WORLD poses
    traj_path = root / "pred_traj.txt"
    use_novel_topdown = traj_path.exists()
    if use_novel_topdown:
        _, C_all, R_c2w_all = load_pred_traj_c2w_opencv(traj_path)
        up_world = estimate_world_up_from_opencv_c2w(R_c2w_all)
        right_hint = _normalize(np.mean(R_c2w_all[:, :, 0], axis=0))
        x_cam, y_cam, _ = make_topdown_basis_from_up(up_world=up_world, right_hint_world=right_hint)

        if args.fit_mode == "all":
            fit_paths = bg_files + fg_files
        elif args.fit_mode == "background":
            fit_paths = bg_files if bg_files else fg_files
        else:
            fit_paths = fg_files

        look_at, sample_pts = stream_scene_stats(fit_paths, max_points=args.fit_max_points, seed=args.seed)
        max_abs_x, max_abs_y, u_min, u_max = compute_fit_extents(sample_pts, look_at, x_cam, y_cam, up_world)
        extent = args.margin * max(max_abs_x, max_abs_y, 1e-3)

        if args.height is not None:
            cam_height = float(args.height)
        else:
            cam_height = float(max(u_max + extent, extent, 0.5))

        cam_pos = look_at + up_world * cam_height
        view_up_for_pyvista = -y_cam
    else:
        # Fallback to old behavior if pred_traj is missing
        look_at = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        cam_pos = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        view_up_for_pyvista = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        extent = 0.5
        up_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_cam = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        y_cam = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        u_min, u_max = -0.5, 0.5

    if args.export_camera_params:
        export_ortho_camera_params(
            out_path=out_root / "orthographic_camera.json",
            look_at=look_at,
            cam_pos=cam_pos,
            view_up_for_pyvista=view_up_for_pyvista,
            parallel_scale=extent,
            use_novel_topdown=use_novel_topdown,
            up_world=up_world,
            x_cam=x_cam,
            y_cam=y_cam,
        )

    # Optional interactive viewer
    if args.visualize:
        frustum_depth = float(args.frustum_depth)
        if frustum_depth <= 0.0:
            frustum_depth = float(max(0.05, 0.25 * extent))
        visualize_scene(
            root=root,
            fg_files=fg_files,
            bg_files=bg_files,
            look_at=look_at,
            cam_pos=cam_pos,
            view_up_for_pyvista=view_up_for_pyvista,
            parallel_scale=extent,
            x_cam=x_cam,
            y_cam=y_cam,
            up_world=up_world,
            use_novel_topdown=use_novel_topdown,
            max_frames=int(args.vis_max_frames),
            show_source_cameras=bool(args.show_source_cameras),
            show_topdown_camera=bool(args.show_topdown_camera),
            frustum_depth=frustum_depth,
        )

        if args.skip_render:
            print("Visualization complete (skipping render loop due to --skip_render).")
            return

    # ──────────────────────────────────────────────────────────────
    # Render Loop
    # ──────────────────────────────────────────────────────────────
    for i in range(num_frames):
        print(f"Processing frame {i}/{num_frames}...", end="\r")
        
        # --- PHASE 1: Temporal Ghosting & Semantic Time-Coding ---
        p = pv.Plotter(off_screen=True, window_size=[args.res, args.res])
        p.background_color = "black"
        
        # Add static background reference (desaturated for contrast)
        bg = load_ply_as_pv(bg_files[i] if i < len(bg_files) else bg_files[0])
        if bg:
            p.add_mesh(bg, rgb=True, opacity=0.3, point_size=2)

        # Accumulation window for Ghosting
        window = int(args.window)
        all_pts_for_bev = []

        start = max(0, i - window)
        denom = max(1, i - start)

        for j in range(start, i + 1):
            fg_path = fg_files[j]
            fg = load_ply_as_pv(fg_path)
            if fg is None: continue
            
            # Semantic Time Coding: Color by "age" relative to current frame
            color = get_semantic_color(j, num_frames)
            # Opacity decay: Older frames fade out
            alpha = (j - start) / denom
            alpha = float(np.clip(alpha, 0.05, 1.0))
            
            p.add_mesh(fg, color=color, opacity=alpha, point_size=5)
            all_pts_for_bev.append(np.asarray(fg.points))

        # Orthographic novel top-down camera (aligned to OpenCV source camera poses when available)
        p.camera.enable_parallel_projection()
        p.camera.position = tuple(cam_pos)
        p.camera.focal_point = tuple(look_at)
        p.camera.up = tuple(view_up_for_pyvista)
        p.camera.parallel_scale = float(extent)
        
        p.screenshot(ghost_dir / f"ghost_{i:04d}.png")
        p.close()

        # --- PHASE 1b: Plain RGB renders (current frame only) ---
        if args.save_rgb_renders:
            # Background RGB render
            if bg_files:
                p_bg = pv.Plotter(off_screen=True, window_size=[args.res, args.res])
                p_bg.background_color = "black"
                bg_curr = load_ply_as_pv(bg_files[i] if i < len(bg_files) else bg_files[0])
                if bg_curr is not None:
                    p_bg.add_mesh(bg_curr, rgb=True, opacity=1.0, point_size=2)
                p_bg.camera.enable_parallel_projection()
                p_bg.camera.position = tuple(cam_pos)
                p_bg.camera.focal_point = tuple(look_at)
                p_bg.camera.up = tuple(view_up_for_pyvista)
                p_bg.camera.parallel_scale = float(extent)
                p_bg.screenshot(rgb_bg_dir / f"bg_{i:04d}.png")
                p_bg.close()

            # Foreground RGB render
            p_fg = pv.Plotter(off_screen=True, window_size=[args.res, args.res])
            p_fg.background_color = "black"
            fg_curr = load_ply_as_pv(fg_files[i])
            if fg_curr is not None:
                p_fg.add_mesh(fg_curr, rgb=True, opacity=1.0, point_size=4)
            p_fg.camera.enable_parallel_projection()
            p_fg.camera.position = tuple(cam_pos)
            p_fg.camera.focal_point = tuple(look_at)
            p_fg.camera.up = tuple(view_up_for_pyvista)
            p_fg.camera.parallel_scale = float(extent)
            p_fg.screenshot(rgb_fg_dir / f"fg_{i:04d}.png")
            p_fg.close()

        # --- PHASE 2: BEV Occupancy Grid ---
        if all_pts_for_bev:
            combined_pts = np.concatenate(all_pts_for_bev, axis=0)
            if use_novel_topdown:
                bev_img = create_bev_map_topdown(
                    combined_pts,
                    look_at=look_at,
                    x_cam=x_cam,
                    y_cam=y_cam,
                    up_world=up_world,
                    extent=extent,
                    u_min=u_min,
                    u_max=u_max,
                    res=args.res,
                )
            else:
                bev_img = create_bev_map(combined_pts, res=args.res)
            Image.fromarray(bev_img).save(bev_dir / f"bev_{i:04d}.png")

    print(f"\nSuccess! Visualizations saved to: {out_root}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Master Visualization Script for MoNST3R Scenes")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to normalized/ directory")
    parser.add_argument("--window", type=int, default=15, help="Number of frames for temporal accumulation")
    parser.add_argument("--res", type=int, default=512, help="Output resolution")
    parser.add_argument(
        "--fit_mode",
        type=str,
        default="all",
        choices=["all", "background", "foreground"],
        help="Which points to use to fit the novel top-down camera (requires pred_traj.txt)",
    )
    parser.add_argument(
        "--fit_max_points",
        type=int,
        default=300_000,
        help="Max points sampled to fit camera (reservoir sampling; requires pred_traj.txt)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=None,
        help="Camera height along estimated world-up (defaults to auto; requires pred_traj.txt)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.10,
        help="Fit margin for orthographic parallel scale (requires pred_traj.txt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for camera-fit sampling (requires pred_traj.txt)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open an interactive PyVista viewer showing point clouds and camera frustums",
    )
    parser.add_argument(
        "--vis_max_frames",
        type=int,
        default=3,
        help="Max frames to load into the interactive viewer (kept small for performance)",
    )
    parser.add_argument(
        "--show_source_cameras",
        action="store_true",
        help="In the interactive viewer, draw the source camera frustums (requires pred_traj/pred_intrinsics)",
    )
    parser.add_argument(
        "--show_topdown_camera",
        action="store_true",
        help="In the interactive viewer, draw the novel top-down camera frustum (requires pred_intrinsics)",
    )
    parser.add_argument(
        "--frustum_depth",
        type=float,
        default=0.0,
        help="Frustum depth used for visualization (0 = auto from fitted extent)",
    )
    parser.add_argument(
        "--skip_render",
        action="store_true",
        help="When using --visualize, skip the off-screen render loop and only open the interactive viewer",
    )
    parser.add_argument(
        "--save_rgb_renders",
        action="store_true",
        help="Also save plain RGB renders (current frame foreground/background) from the novel top-down camera",
    )
    parser.add_argument(
        "--export_camera_params",
        action="store_true",
        help="Export the computed orthographic camera parameters to master_visualizations/orthographic_camera.json",
    )
    
    args = parser.parse_args()
    main(args)