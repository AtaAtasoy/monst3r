import argparse
import json
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from dreifus.matrix import CameraCoordinateConvention, Intrinsics, Pose, PoseType
from pytorch3d.io import IO
from pytorch3d.renderer import (
    AlphaCompositor,
    OrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)
from pytorch3d.structures import Pointclouds


def load_tum_traj(path: Path) -> list[Pose]:
    poses = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(f"Invalid TUM line in {path}: {line}")
            _, tx, ty, tz, qw, qx, qy, qz = map(float, parts)
            poses.append(
                Pose.from_quaternion(
                    quaternion=[qx, qy, qz, qw],
                    translation=[tx, ty, tz],
                    pose_type=PoseType.CAM_2_WORLD,
                    camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
                )
            )
    if not poses:
        raise ValueError(f"No valid poses found in {path}")
    return poses


def load_npz_traj(path: Path) -> list[Pose]:
    payload = np.load(path)
    if "data" not in payload.files:
        raise ValueError(f"Missing `data` in NPZ file: {path}")
    data = np.asarray(payload["data"], dtype=np.float64)
    inds = np.asarray(payload["inds"], dtype=np.int64) if "inds" in payload.files else np.arange(len(data), dtype=np.int64)

    if data.ndim != 3 or data.shape[1:] != (4, 4):
        raise ValueError(
            f"Expected data shape (N, 4, 4) in {path}, got {data.shape}"
        )

    if len(inds) != len(data):
        raise ValueError(f"Mismatch: len(inds)={len(inds)} does not match len(data)={len(data)}")

    sorted_pairs = sorted(
        zip(inds.tolist(), data.tolist()),
        key=lambda x: int(x[0]),
    )
    poses = []
    for _, mat in sorted_pairs:
        pose = Pose(
            np.asarray(mat, dtype=np.float64),
            pose_type=PoseType.CAM_2_WORLD,
            camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
        )
        poses.append(pose)
    if not poses:
        raise ValueError(f"No valid poses found in {path}")
    return poses


def load_npz_matrix4x4(path: Path) -> Tuple[list[np.ndarray], list[int]]:
    payload = np.load(path)
    if "data" not in payload.files:
        raise ValueError(f"Missing `data` in NPZ file: {path}")
    data = np.asarray(payload["data"], dtype=np.float64)
    inds = np.asarray(payload["inds"], dtype=np.int64) if "inds" in payload.files else np.arange(len(data), dtype=np.int64)

    if data.ndim != 3 or data.shape[1:] != (4, 4):
        raise ValueError(f"Expected data shape (N, 4, 4) in {path}, got {data.shape}")
    if len(inds) != len(data):
        raise ValueError(f"Mismatch: len(inds)={len(inds)} does not match len(data)={len(data)} in {path}")

    entries = sorted(zip(inds.tolist(), [np.asarray(m, dtype=np.float64) for m in data]))
    sorted_data = [m for _, m in entries]
    sorted_inds = [int(i) for i, _ in entries]
    return sorted_data, sorted_inds


def load_npz_intrinsics(path: Path) -> Tuple[list[np.ndarray], list[int]]:
    payload = np.load(path)
    if "data" not in payload.files:
        raise ValueError(f"Missing `data` in intrinsics NPZ file: {path}")
    data = np.asarray(payload["data"], dtype=np.float64)
    inds = np.asarray(payload["inds"], dtype=np.int64) if "inds" in payload.files else np.arange(len(data), dtype=np.int64)

    if data.ndim != 2 or data.shape[1] != 4:
        if data.ndim == 3 and data.shape[1:] == (3, 3):
            entries = sorted(zip(inds.tolist(), [np.asarray(m, dtype=np.float64) for m in data]))
            sorted_data = [m for _, m in entries]
            sorted_inds = [int(i) for i, _ in entries]
            return sorted_data, sorted_inds
        raise ValueError(f"Expected intrinsics data shape (N, 4) or (N, 3, 3) in {path}, got {data.shape}")

    if len(inds) != len(data):
        raise ValueError(f"Mismatch: len(inds)={len(inds)} does not match len(data)={len(data)} in {path}")

    entries = sorted(zip(inds.tolist(), [np.asarray(v, dtype=np.float64) for v in data]))
    sorted_data = [v for _, v in entries]
    sorted_inds = [int(i) for i, _ in entries]
    return sorted_data, sorted_inds


def load_gen3c_intrinsics(data_seq: list[np.ndarray]) -> list[Intrinsics]:
    intrinsics = []
    for v in data_seq:
        arr = np.asarray(v, dtype=np.float64).reshape(-1)
        if arr.size == 4:
            fx, fy, cx, cy = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
            mat = np.array(
                [[fx, 0.0, cx, 0.0], [0.0, fy, cy, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
                dtype=np.float64,
            )
            # Convert to Intrinsics matrix form by keeping only upper-left 3x3 projection part.
            intrinsics.append(Intrinsics(matrix_or_fx=mat[:3, :3].astype(np.float64)))
        elif v.shape == (3, 3):
            intrinsics.append(Intrinsics(matrix_or_fx=v.astype(np.float64)))
        elif v.shape == (4, 4):
            intrinsics.append(Intrinsics(matrix_or_fx=v[:3, :3].astype(np.float64)))
        else:
            raise ValueError(f"Unsupported intrinsics shape: {v.shape}")
    return intrinsics


def resolve_gen3c_file(root_dir: Path, subdir: str, explicit: Optional[str] = None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        return p
    candidates = sorted((root_dir / subdir).glob("*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No .npz files found in {root_dir / subdir}")
    if len(candidates) > 1:
        raise ValueError(f"Multiple candidates in {root_dir / subdir}: {[str(c) for c in candidates]}")
    return candidates[0]


def load_gen3c_output(
    gen3c_output_dir: Path,
    pose_npz: Optional[str] = None,
    intrinsics_npz: Optional[str] = None,
) -> Tuple[list[Pose], Optional[list[Intrinsics]]]:
    pose_path = resolve_gen3c_file(gen3c_output_dir, "pose", pose_npz)
    pose_data, pose_inds = load_npz_matrix4x4(pose_path)
    pose_map = dict(zip(pose_inds, pose_data))

    poses = [Pose(m, pose_type=PoseType.CAM_2_WORLD, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV) for m in pose_data]
    intrinsics = None
    if pose_npz is None and (gen3c_output_dir / "intrinsics").exists():
        intr_path = resolve_gen3c_file(gen3c_output_dir, "intrinsics", intrinsics_npz)
        intr_data, intr_inds = load_npz_intrinsics(intr_path)
        intr_map = dict(zip(intr_inds, intr_data))
        common_inds = [i for i in pose_map.keys() if i in intr_map]
        if not common_inds:
            if len(intr_inds) != len(pose_inds):
                raise ValueError("No matching frame indices between pose.npz and intrinsics.npz")
            common_inds = sorted(pose_inds)
        common_inds = sorted(common_inds)
        poses = [
            Pose(
                pose_map[i],
                pose_type=PoseType.CAM_2_WORLD,
                camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
            )
            for i in common_inds
        ]
        intrinsics = load_gen3c_intrinsics([intr_map[i] for i in common_inds])
    elif pose_npz and intrinsics_npz:
        intr_data, intr_inds = load_npz_intrinsics(Path(intrinsics_npz))
        intr_map = dict(zip(intr_inds, intr_data))
        common_inds = [i for i in pose_map.keys() if i in intr_map]
        if not common_inds:
            raise ValueError("No matching frame indices between provided pose npz and intrinsics npz")
        common_inds = sorted(common_inds)
        poses = [
            Pose(
                pose_map[i],
                pose_type=PoseType.CAM_2_WORLD,
                camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
            )
            for i in common_inds
        ]
        intrinsics = load_gen3c_intrinsics([intr_map[i] for i in common_inds])
    elif not pose_npz and pose_path is not None:
        # Keep pose-only mode so caller can request source comparison without frustums.
        poses = [Pose(m, pose_type=PoseType.CAM_2_WORLD, camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV) for m in pose_map.values()]
    elif intrinsics_npz:
        # Explicit intrinsics without auto pose path should not happen.
        raise ValueError("Provide --gen3c-pose-npz together with --gen3c-intrinsics-npz when overriding pose path.")

    return poses, intrinsics


def load_intrinsics(path: Path) -> list[Intrinsics]:
    data = np.loadtxt(str(path))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    intr = []
    for row in data:
        if row.size != 9:
            raise ValueError(f"Expected 9 intrinsics values per row, got {row.size}")
        intr.append(Intrinsics(matrix_or_fx=row.reshape(3, 3).astype(np.float64)))
    if not intr:
        raise ValueError(f"No intrinsics found in {path}")
    return intr


def build_topdown_camera(
    cam2world: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_h: int,
    img_w: int,
    device: torch.device,
) -> OrthographicCameras:
    opencv_pose = Pose(
        cam2world,
        pose_type=PoseType.CAM_2_WORLD,
        camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
    )
    p3d_c2w = opencv_pose.change_camera_coordinate_convention(
        new_camera_coordinate_convention=CameraCoordinateConvention.PYTORCH_3D
    )
    r = p3d_c2w.get_rotation_matrix()
    p3d_w2c = p3d_c2w.change_pose_type(
        new_pose_type=PoseType.WORLD_2_CAM, inplace=False
    )
    t = p3d_w2c.get_translation()
    return OrthographicCameras(
        focal_length=((fx, fy),),
        principal_point=((cx, cy),),
        R=torch.tensor(r, dtype=torch.float32, device=device)[None, :, :],
        T=torch.tensor(t, dtype=torch.float32, device=device)[None, :],
        in_ndc=False,
        image_size=((img_h, img_w),),
        device=device,
    )


def project_topdown(
    world_xyz: np.ndarray,
    camera: OrthographicCameras,
    device: torch.device,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    world_xyz = np.asarray(world_xyz, dtype=np.float32).reshape(-1, 3)
    if world_xyz.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    points = torch.tensor(world_xyz, dtype=torch.float32, device=device)
    image_size = torch.tensor([[img_h, img_w]], dtype=torch.float32, device=device)
    screen = camera.transform_points_screen(points[None, ...], image_size=image_size)[0, :, :2]
    return screen.detach().cpu().numpy()


def render_scene_frame_topdown(
    ply_path: Path,
    camera: OrthographicCameras,
    img_h: int,
    img_w: int,
    device: torch.device,
) -> np.ndarray:
    io = IO()
    cloud = io.load_pointcloud(str(ply_path), device=device)
    points = cloud.points_list()[0]
    if points.numel() == 0:
        return np.zeros((img_h, img_w, 3), dtype=np.uint8)
    features = cloud.features_list()[0]
    if features is None or features.numel() == 0:
        features = torch.ones_like(points)
    pc = Pointclouds(points=[points], features=[features])
    raster_settings = PointsRasterizationSettings(
        image_size=(img_h, img_w),
        bin_size=0,
        radius=0.003,
        points_per_pixel=10,
    )
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
        compositor=AlphaCompositor(),
    )
    image = renderer(pc)[0, ..., :3].detach().cpu().numpy().clip(0, 1)
    return (image * 255).astype(np.uint8)


def list_frame_plys(ply_dir: Path) -> list[Path]:
    return sorted([p for p in ply_dir.glob("frame_*.ply") if "_masked" not in p.name])

def build_frustum_world_points(pose_cam2world: Pose, intr: Intrinsics, depth_scale: float) -> np.ndarray:
    fx = intr.fx
    fy = intr.fy
    cx = intr.cx
    cy = intr.cy
    img_w, img_h = max(2.0 * cx, 1.0), max(2.0 * cy, 1.0)
    corners_px = np.array(
        [[0.0, 0.0], [img_w - 1.0, 0.0], [img_w - 1.0, img_h - 1.0], [0.0, img_h - 1.0]],
        dtype=np.float64,
    )
    z = float(depth_scale)
    x = (corners_px[:, 0] - cx) / max(fx, 1e-8) * z
    y = (corners_px[:, 1] - cy) / max(fy, 1e-8) * z
    corners_cam = np.stack([x, y, np.full_like(x, z)], axis=1)
    frustum_cam = np.concatenate([np.zeros((1, 3), dtype=np.float64), corners_cam], axis=0)

    pose_mat = np.asarray(pose_cam2world.numpy(), dtype=np.float64)
    r = pose_mat[:3, :3]
    t = pose_mat[:3, 3]
    frustum_world = frustum_cam @ r.T + t[None, :]
    return frustum_world


def draw_frustum_wireframe(draw: ImageDraw.ImageDraw, pts_xy: list[tuple[int, int]], color: tuple[int, int, int], width: int = 2) -> None:
    if len(pts_xy) != 5:
        return
    o, c0, c1, c2, c3 = pts_xy
    draw.line([o, c0], fill=color, width=width)
    draw.line([o, c1], fill=color, width=width)
    draw.line([o, c2], fill=color, width=width)
    draw.line([o, c3], fill=color, width=width)
    draw.line([c0, c1], fill=color, width=width)
    draw.line([c1, c2], fill=color, width=width)
    draw.line([c2, c3], fill=color, width=width)
    draw.line([c3, c0], fill=color, width=width)


def create_video_from_frames(frame_pattern: str, output_video: Path, fps: int) -> None:
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
        print(f"[warn] ffmpeg failed for {output_video}:\n{result.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize generated 3D camera trajectory from top-down orthographic view."
    )
    parser.add_argument("--semantic_metadata_json", type=str, required=True)
    parser.add_argument(
        "--generated_traj_txt",
        type=str,
        default=None,
        help="Lifted/generated TUM trajectory.",
    )
    parser.add_argument(
        "--generated_traj_npz",
        type=str,
        default=None,
        help="Lifted/generated NPZ trajectory (data, inds).",
    )
    parser.add_argument(
        "--gen3c-output-dir",
        type=str,
        default=None,
        help="Gen3C output directory containing pose/ and intrinsics/ .npz files.",
    )
    parser.add_argument(
        "--gen3c-pose-npz",
        type=str,
        default=None,
        help="Optional explicit pose NPZ inside gen3c-output (pose/*.npz).",
    )
    parser.add_argument(
        "--gen3c-intrinsics-npz",
        type=str,
        default=None,
        help="Optional explicit intrinsics NPZ inside gen3c-output (intrinsics/*.npz).",
    )
    parser.add_argument("--source_traj_txt", type=str, default=None, help="Optional source trajectory for comparison.")
    parser.add_argument("--intrinsics_path", type=str, default=None, help="Required to render frustum overlays.")
    parser.add_argument("--render_dir", type=str, default=None, help="Optional background frames folder containing top-down-render_XXXX.png.")
    parser.add_argument("--scene_ply_dir", type=str, default=None, help="Optional: render top-down background with PyTorch3D from frame_*.ply.")
    parser.add_argument("--output_dir", type=str, default="topdown_generated_trajectory_viz")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    semantic = json.loads(Path(args.semantic_metadata_json).read_text(encoding="utf-8"))
    img_h, img_w = semantic["image_size"]
    cam2world = np.asarray(semantic["orthographic_camera_to_world_4x4"], dtype=np.float64)
    fx, fy = map(float, semantic["orthographic_cam_parameters"]["focal_length_xy"])
    cx, cy = map(float, semantic["orthographic_cam_parameters"]["principal_point_xy"])
    view_half_extent = semantic["orthographic_cam_parameters"]["view_half_extent_xy"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    topdown_camera = build_topdown_camera(
        cam2world=cam2world,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        img_h=int(img_h),
        img_w=int(img_w),
        device=device,
    )

    has_tum = args.generated_traj_txt is not None
    has_npz = args.generated_traj_npz is not None
    has_gen3c_dir = args.gen3c_output_dir is not None
    has_gen3c_pose = args.gen3c_pose_npz is not None or args.gen3c_intrinsics_npz is not None
    if args.gen3c_pose_npz is None and args.gen3c_intrinsics_npz is not None:
        raise ValueError("--gen3c-pose-npz is required when --gen3c-intrinsics-npz is provided.")
    if has_tum and (has_npz or has_gen3c_dir):
        raise ValueError("Specify only one trajectory input mode.")
    if has_npz and (has_gen3c_dir or has_gen3c_pose):
        raise ValueError("Specify only one trajectory input mode.")
    if has_gen3c_dir and has_gen3c_pose:
        raise ValueError("Specify --gen3c-output-dir or explicit --gen3c-pose-npz/--gen3c-intrinsics-npz, not both.")
    if not (has_tum or has_npz or has_gen3c_dir or has_gen3c_pose):
        raise ValueError("Specify one of --generated_traj_txt, --generated_traj_npz, --gen3c-output-dir, --gen3c-pose-npz, or --gen3c-intrinsics-npz.")
    if has_tum:
        gen_poses = load_tum_traj(Path(args.generated_traj_txt))
        gen_intrinsics = None
    elif has_npz:
        gen_poses = load_npz_traj(Path(args.generated_traj_npz))
        gen_intrinsics = None
    else:
        gen3c_dir = Path(args.gen3c_output_dir or Path(args.gen3c_pose_npz).parent)
        gen_poses, gen_intrinsics = load_gen3c_output(
            gen3c_output_dir=gen3c_dir,
            pose_npz=args.gen3c_pose_npz,
            intrinsics_npz=args.gen3c_intrinsics_npz,
        )

    src_poses = load_tum_traj(Path(args.source_traj_txt)) if args.source_traj_txt else None

    intrinsics = None
    if gen_intrinsics is not None:
        intrinsics = gen_intrinsics
    elif args.intrinsics_path:
        intrinsics = load_intrinsics(Path(args.intrinsics_path))
    else:
        raise ValueError("Missing intrinsics for frustum overlays.")

    scene_ply_files = None
    if args.scene_ply_dir:
        scene_ply_files = list_frame_plys(Path(args.scene_ply_dir))
        if not scene_ply_files:
            raise ValueError(f"No frame_*.ply files found in {args.scene_ply_dir}")

    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    n = len(gen_poses)
    depth_scale = 0.08 * max(float(view_half_extent[0]), float(view_half_extent[1]))

    for i in range(n):
        if scene_ply_files is not None:
            ply_path = scene_ply_files[min(i, len(scene_ply_files) - 1)]
            bg_np = render_scene_frame_topdown(
                ply_path=ply_path,
                camera=topdown_camera,
                img_h=int(img_h),
                img_w=int(img_w),
                device=device,
            )
            base = Image.fromarray(bg_np)
        elif args.render_dir:
            bg_path = Path(args.render_dir) / f"top-down-render_{i:04d}.png"
            if bg_path.exists():
                base = Image.open(bg_path).convert("RGB")
            else:
                base = Image.new("RGB", (img_w, img_h), color=(0, 0, 0))
        else:
            base = Image.new("RGB", (img_w, img_h), color=(0, 0, 0))
        draw = ImageDraw.Draw(base)

        # Frustum overlays.
        if intrinsics is not None:
            intr = intrinsics[min(i, len(intrinsics) - 1)]
            g_pose = gen_poses[i]
            g_frustum_world = build_frustum_world_points(g_pose, intr, depth_scale=depth_scale)
            g_frustum_screen = project_topdown(
                g_frustum_world, topdown_camera, device, int(img_h), int(img_w)
            )
            g_frustum_px = [(int(round(x)), int(round(y))) for x, y in g_frustum_screen]
            draw_frustum_wireframe(draw, g_frustum_px, color=(0, 255, 0), width=2)

            if src_poses is not None:
                s_pose = src_poses[min(i, len(src_poses) - 1)]
                s_frustum_world = build_frustum_world_points(s_pose, intr, depth_scale=depth_scale)
                s_frustum_screen = project_topdown(
                    s_frustum_world, topdown_camera, device, int(img_h), int(img_w)
                )
                s_frustum_px = [(int(round(x)), int(round(y))) for x, y in s_frustum_screen]
                draw_frustum_wireframe(draw, s_frustum_px, color=(0, 255, 255), width=2)

        base.save(frames_dir / f"{i:04d}.png")
        print(f"Rendered overlay frame {i + 1}/{n}", end="\r")

    print()
    create_video_from_frames(str(frames_dir / "%04d.png"), output_dir / "generated_topdown_trajectory.mp4", fps=args.fps)
    print(f"Saved frames to: {frames_dir}")
    print(f"Saved video to: {output_dir / 'generated_topdown_trajectory.mp4'}")


if __name__ == "__main__":
    main()
