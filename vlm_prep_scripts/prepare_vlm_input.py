import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
from dreifus.matrix import CameraCoordinateConvention, Intrinsics, Pose, PoseType


def load_tum_traj(traj_path: Path) -> list[dict]:
    """
    Load trajectory file in TUM format.
    Expected line format: timestamp tx ty tz qw qx qy qz
    """
    frames = []
    with traj_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            ts = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qw, qx, qy, qz = map(float, parts[4:8])
            pose = Pose.from_quaternion(
                quaternion=[qx, qy, qz, qw],
                translation=[tx, ty, tz],
                pose_type=PoseType.CAM_2_WORLD,
                camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
            )
            frames.append(
                {
                    "timestamp": ts,
                    "pose": pose,
                }
            )

    return frames


def pose_to_euler_yxz(pose: Pose) -> tuple[float, float, float]:
    """
    Convert rotation matrix to intrinsic Y-X-Z Euler angles [yaw, pitch, roll] in radians
    using dreifus pose/euler utilities.
    """
    euler = pose.get_euler_angles("YXZ")
    return float(euler[0]), float(euler[1]), float(euler[2])  # yaw, pitch, roll order


def load_intrinsics(intr_path: Path) -> list[Intrinsics]:
    data = np.loadtxt(str(intr_path))
    if data.ndim == 1:
        data = data.reshape(1, -1)

    mats: list[Intrinsics] = []
    for row in data:
        if row.size != 9:
            raise ValueError(f"Expected 9 values per intrinsics row, got {row.size}")
        K = row.reshape(3, 3).astype(np.float64)
        mats.append(Intrinsics(matrix_or_fx=K))
    return mats


def resolve_capture_intrinsics(intr_mats: list[Intrinsics], frame_count: int) -> list[Intrinsics]:
    if len(intr_mats) == frame_count:
        return intr_mats
    if len(intr_mats) == 1:
        return [intr_mats[0] for _ in range(frame_count)]
    raise ValueError(
        f"Intrinsics count ({len(intr_mats)}) does not match frame count ({frame_count}) and is not 1"
    )


def build_capture_cameras(traj_path: Path, intr_path: Path) -> list[dict]:
    traj = load_tum_traj(traj_path)
    intr_mats = load_intrinsics(intr_path)
    intr_per_frame = resolve_capture_intrinsics(intr_mats, len(traj))

    cameras = []
    for i, (frame, K) in enumerate(zip(traj, intr_per_frame)):
        pose: Pose = frame["pose"]
        c2w = np.array(pose.numpy(), dtype=np.float64)
        t = np.array(pose.get_translation(), dtype=np.float64)

        yaw, pitch, roll = pose_to_euler_yxz(pose)  # intrinsic Y-X-Z order for consistency with top-down camera

        cameras.append(
            {
                "frame_index": i,
                "timestamp": frame["timestamp"],
                "coordinate_convention": "opencv_rdf",
                "pose_type": "camera_to_world",
                "rotation": {
                    "euler_order": "intrinsic_yxz",
                    "yaw_pitch_roll_rad": [yaw, pitch, roll],
                },
                "translation_xyz": [float(t[0]), float(t[1]), float(t[2])],
                "camera_to_world_4x4": c2w.tolist(),
                "intrinsics": {
                    "fx": float(K.fx),
                    "fy": float(K.fy),
                    "cx": float(K.cx),
                    "cy": float(K.cy),
                    "matrix_3x3": K.numpy().tolist(),
                },
            }
        )

    return cameras


def build_topdown_camera_from_semantic_metadata(semantic_meta_path: Path) -> dict:
    with semantic_meta_path.open("r", encoding="utf-8") as f:
        td = json.load(f)

    c2w = np.array(td["camera_to_world_4x4"], dtype=np.float64)
    pose = Pose(
        c2w,
        pose_type=PoseType.CAM_2_WORLD,
        camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
    )
    t = np.array(pose.get_translation(), dtype=np.float64)
    yaw, pitch, roll = pose_to_euler_yxz(pose)  # intrinsic Y-X-Z order for consistency with capture cameras

    ortho = td.get("orthographic_params", {})
    focal = ortho.get("focal_length_xy", [None, None])
    principal = ortho.get("principal_point_xy", [None, None])
    fx, fy = focal[0], focal[1]
    cx, cy = principal[0], principal[1]
    intrinsics_3x3 = None
    if fx is not None and fy is not None and cx is not None and cy is not None:
        intrinsics_3x3 = [
            [float(fx), 0.0, float(cx)],
            [0.0, float(fy), float(cy)],
            [0.0, 0.0, 1.0],
        ]

    output = {
        "coordinate_convention": "opencv_rdf",
        "pose_type": "camera_to_world",
        "rotation": {
            "euler_order": "intrinsic_yxz",
            "yaw_pitch_roll_rad": [yaw, pitch, roll],
        },
        "translation_xyz": [float(t[0]), float(t[1]), float(t[2])],
        "camera_to_world_4x4": c2w.tolist(),
        "intrinsics": {
            "fx": None if fx is None else float(fx),
            "fy": None if fy is None else float(fy),
            "cx": None if cx is None else float(cx),
            "cy": None if cy is None else float(cy),
            "matrix_3x3": intrinsics_3x3,
        },
        "projection_type": td.get("projection_type", "orthographic"),
        "projection_params": ortho,
        "look_at_mode": td.get("look_at_mode"),
        "look_at_point": td.get("look_at_point"),
        "camera_height": td.get("camera_height"),
        "image_size_hw": td.get("image_size"),
        "render_context": td.get("render_context"),
        "camera_trajectory_pixels": td.get("camera_trajectory_pixels", []),
        "camera_trajectory_pixels_local": td.get("camera_trajectory_pixels_local", []),
        "semantic_frame_annotations": td.get("frames", []),
    }

    return output


def _normalize_pixel_trajectory(points: list) -> list[list[int]]:
    out: list[list[int]] = []
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        x, y = p[0], p[1]
        out.append([int(round(float(x))), int(round(float(y)))])
    return out


def attach_topdown_pixel_trajectory_to_capture_cameras(capture_cameras: list[dict], topdown_camera: dict) -> None:
    global_traj = _normalize_pixel_trajectory(topdown_camera.get("camera_trajectory_pixels", []))
    local_traj = _normalize_pixel_trajectory(topdown_camera.get("camera_trajectory_pixels_local", []))
    n_capture = len(capture_cameras)

    if len(global_traj) != n_capture:
        global_traj = []
    if len(local_traj) != n_capture:
        local_traj = []

    for i, camera in enumerate(capture_cameras):
        camera["topdown_pixel_xy"] = global_traj[i] if global_traj else None
        camera["topdown_pixel_xy_local"] = local_traj[i] if local_traj else None


def list_pngs(dir_path: Path, pattern: str) -> list[Path]:
    return sorted(dir_path.glob(pattern))


def list_mp4s(dir_path: Path) -> list[Path]:
    return sorted(dir_path.glob("*.mp4")) if dir_path.exists() else []


def add_files_to_zip(
    zf: zipfile.ZipFile,
    files: list[Path],
    zip_prefix: str,
) -> list[str]:
    rels = []
    for f in files:
        arcname = f"{zip_prefix}/{f.name}"
        zf.write(f, arcname=arcname)
        rels.append(arcname)
    return rels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare zipped VLM input: capture renders + top-down renders + unified camera JSON."
    )
    parser.add_argument(
        "scene_dir",
        type=str,
        help="Directory containing pred_traj.txt, pred_intrinsics.txt, and renders_all/",
    )
    parser.add_argument(
        "topdown_dir",
        type=str,
        help="Directory containing semantic_topdown_metadata.json and motion_overlay/motion-overlay_*.png",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="demo_tmp/vlm_inputs",
        help="Directory to write VLM zip package(s). Default: demo_tmp/vlm_inputs",
    )

    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    topdown_dir = Path(args.topdown_dir)
    capture_render_dir = scene_dir / "renders_all"
    output_dir = Path(args.output_dir)
    base_dir = scene_dir.parent.name if scene_dir.parent.name else scene_dir.name
    output_zip = output_dir / f"{base_dir}.{scene_dir.stem}.zip"

    traj_path = scene_dir / "pred_traj.txt"
    intr_path = scene_dir / "pred_intrinsics.txt"
    semantic_meta_path = topdown_dir / "semantic_topdown_metadata.json"
    semantic_render_dir = topdown_dir / "motion_overlay"
    semantic_video_dir = topdown_dir / "vis-videos"

    if not traj_path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {traj_path}")
    if not intr_path.exists():
        raise FileNotFoundError(f"Missing intrinsics file: {intr_path}")
    if not semantic_meta_path.exists():
        raise FileNotFoundError(f"Missing semantic top-down metadata file: {semantic_meta_path}")
    if not semantic_render_dir.exists():
        raise FileNotFoundError(f"Missing semantic top-down render directory: {semantic_render_dir}")
    if not capture_render_dir.exists():
        raise FileNotFoundError(f"Missing capture render directory: {capture_render_dir}")
    if not topdown_dir.exists():
        raise FileNotFoundError(f"Missing top-down render directory: {topdown_dir}")

    capture_cameras = build_capture_cameras(traj_path, intr_path)
    topdown_camera = build_topdown_camera_from_semantic_metadata(semantic_meta_path)
    attach_topdown_pixel_trajectory_to_capture_cameras(capture_cameras, topdown_camera)

    capture_pngs = list_pngs(capture_render_dir, "*.png")
    topdown_pngs = list_pngs(semantic_render_dir, "motion-overlay_*.png")
    semantic_mp4s = list_mp4s(semantic_video_dir)

    capture_video_candidates = []
    capture_video_candidates.extend(list_mp4s(capture_render_dir / "vis-videos"))
    capture_video_candidates.extend(list_mp4s(capture_render_dir))
    if scene_dir != capture_render_dir:
        capture_video_candidates.extend(list_mp4s(scene_dir / "vis-videos"))
        capture_video_candidates.extend(list_mp4s(scene_dir))
    capture_mp4s = sorted(set(capture_video_candidates))

    if not capture_pngs:
        raise FileNotFoundError(f"No capture PNGs found in {capture_render_dir}")
    if not topdown_pngs:
        raise FileNotFoundError(f"No top-down PNGs found in {topdown_dir}")

    combined = {
        "metadata": {
            "scene_dir": str(scene_dir),
            "capture_render_dir": str(capture_render_dir),
            "topdown_dir": str(topdown_dir),
            "topdown_render_dir": str(semantic_render_dir),
            "topdown_metadata_file": str(semantic_meta_path),
            "coordinate_convention": "opencv_rdf",
            "pose_type": "camera_to_world",
            "euler_order": "intrinsic_yxz",
            "capture_frame_count": len(capture_cameras),
            "capture_render_count": len(capture_pngs),
            "topdown_render_count": len(topdown_pngs),
            "capture_video_count": len(capture_mp4s),
            "topdown_video_count": len(semantic_mp4s),
        },
        "capture_cameras": capture_cameras,
        "topdown_camera": topdown_camera,
        "render_files": {
            "capture": [f"renders/capture/{p.name}" for p in capture_pngs],
            "top_down": [f"renders/top_down/{p.name}" for p in topdown_pngs],
        },
        "video_files": {
            "capture": [f"videos/capture/{p.name}" for p in capture_mp4s],
            "top_down": [f"videos/top_down/{p.name}" for p in semantic_mp4s],
        },
    }

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("camera_parameters.json", json.dumps(combined, indent=2))
        add_files_to_zip(zf, capture_pngs, "renders/capture")
        add_files_to_zip(zf, topdown_pngs, "renders/top_down")
        add_files_to_zip(zf, capture_mp4s, "videos/capture")
        add_files_to_zip(zf, semantic_mp4s, "videos/top_down")

    print(f"Wrote VLM package: {output_zip}")
    print("Included files:")
    print(f"- camera_parameters.json")
    print(f"- renders/capture/*.png ({len(capture_pngs)} files)")
    print(f"- renders/top_down/*.png ({len(topdown_pngs)} files)")
    print(f"- videos/capture/*.mp4 ({len(capture_mp4s)} files)")
    print(f"- videos/top_down/*.mp4 ({len(semantic_mp4s)} files)")


if __name__ == "__main__":
    main()
