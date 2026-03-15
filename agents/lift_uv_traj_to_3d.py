import argparse
import json
from pathlib import Path

import numpy as np

"""
python agents/lift_uv_traj_to_3d.py \
  --trajectory_json trajectory.json \
  --semantic_metadata_json demo_tmp/davis/tennis/normalized_nofilter/top-down-semantic-ortho-all-scene/semantic_topdown_metadata.json \
  --source_traj_txt demo_tmp/davis/tennis/normalized_nofilter/pred_traj.txt \
  --output_json lifted_camera_trajectory_3d.json \
  --output_tum lifted_pred_traj.txt
"""


def load_tum_traj(path: Path) -> list[dict]:
    poses = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(f"Invalid TUM line in {path}: {line}")
            ts, tx, ty, tz, qw, qx, qy, qz = map(float, parts)
            poses.append(
                {
                    "timestamp": ts,
                    "t": np.array([tx, ty, tz], dtype=np.float64),
                    "q": np.array([qw, qx, qy, qz], dtype=np.float64),
                }
            )
    if not poses:
        raise ValueError(f"No valid poses found in {path}")
    return poses


def load_uv_trajectory(path: Path) -> list[tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    traj = data.get("camera_trajectory_pixels")
    if not isinstance(traj, dict):
        raise ValueError("`camera_trajectory_pixels` must be a JSON object")

    indexed = []
    for k, v in traj.items():
        if not k.startswith("t_"):
            continue
        idx = int(k[2:])
        if not isinstance(v, dict) or "position" not in v:
            continue
        pos = v["position"]
        if not (isinstance(pos, list) and len(pos) == 2):
            continue
        u, vv = float(pos[0]), float(pos[1])
        indexed.append((idx, (u, vv)))

    if not indexed:
        raise ValueError(f"No valid `t_i.position` entries found in {path}")
    indexed.sort(key=lambda x: x[0])

    max_idx = indexed[-1][0]
    uv = [None] * (max_idx + 1)
    for idx, p in indexed:
        uv[idx] = p

    # Fill missing indices by nearest-neighbor hold for robustness.
    last = None
    for i in range(len(uv)):
        if uv[i] is not None:
            last = uv[i]
        elif last is not None:
            uv[i] = last
    next_valid = None
    for i in range(len(uv) - 1, -1, -1):
        if uv[i] is not None:
            next_valid = uv[i]
        elif next_valid is not None:
            uv[i] = next_valid

    if any(p is None for p in uv):
        raise ValueError("Could not fill missing UV trajectory entries")
    return [p for p in uv if p is not None]


def world_to_cam(points_world: np.ndarray, cam2world: np.ndarray) -> np.ndarray:
    r_c2w = cam2world[:3, :3]
    t_c2w = cam2world[:3, 3]
    r_w2c = r_c2w.T
    t_w2c = -r_w2c @ t_c2w
    return points_world @ r_w2c.T + t_w2c[None, :]


def cam_to_world(points_cam: np.ndarray, cam2world: np.ndarray) -> np.ndarray:
    r_c2w = cam2world[:3, :3]
    t_c2w = cam2world[:3, 3]
    return points_cam @ r_c2w.T + t_c2w[None, :]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lift VLM UV camera trajectory (top-down orthographic pixels) into world-space 3D camera centers."
    )
    parser.add_argument("--trajectory_json", type=str, required=True)
    parser.add_argument("--semantic_metadata_json", type=str, required=True)
    parser.add_argument("--source_traj_txt", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="lifted_camera_trajectory_3d.json")
    parser.add_argument("--output_tum", type=str, default="lifted_pred_traj.txt")
    parser.add_argument(
        "--depth_mode",
        type=str,
        default="source",
        choices=["source", "mean", "constant"],
        help="How to set orthographic camera-frame depth (z) for each lifted UV point.",
    )
    parser.add_argument(
        "--constant_depth",
        type=float,
        default=0.0,
        help="Depth used when --depth_mode=constant.",
    )
    args = parser.parse_args()

    traj_path = Path(args.trajectory_json)
    meta_path = Path(args.semantic_metadata_json)
    source_path = Path(args.source_traj_txt)
    output_json = Path(args.output_json)
    output_tum = Path(args.output_tum)

    uv_traj = load_uv_trajectory(traj_path)
    source_poses = load_tum_traj(source_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    cam2world = np.asarray(meta["orthographic_camera_to_world_4x4"], dtype=np.float64)
    params = meta["orthographic_cam_parameters"]
    fx, fy = map(float, params["focal_length_xy"])
    cx, cy = map(float, params["principal_point_xy"])

    source_world = np.stack([p["t"] for p in source_poses], axis=0)
    source_cam = world_to_cam(source_world, cam2world)
    source_depths = source_cam[:, 2]

    n = len(uv_traj)
    if args.depth_mode == "source":
        z_vals = np.array([source_depths[min(i, len(source_depths) - 1)] for i in range(n)], dtype=np.float64)
    elif args.depth_mode == "mean":
        z_vals = np.full((n,), float(np.mean(source_depths)), dtype=np.float64)
    else:
        z_vals = np.full((n,), float(args.constant_depth), dtype=np.float64)

    cam_pts = np.zeros((n, 3), dtype=np.float64)
    for i, (u, v) in enumerate(uv_traj):
        x_cam = (u - cx) / max(fx, 1e-8)
        y_cam = (v - cy) / max(fy, 1e-8)
        cam_pts[i] = np.array([x_cam, y_cam, z_vals[i]], dtype=np.float64)

    world_pts = cam_to_world(cam_pts, cam2world)

    lifted = []
    for i in range(n):
        src = source_poses[min(i, len(source_poses) - 1)]
        lifted.append(
            {
                "frame_index": i,
                "uv": [float(uv_traj[i][0]), float(uv_traj[i][1])],
                "camera_position_world_xyz": world_pts[i].astype(float).tolist(),
                "source_quaternion_wxyz": src["q"].astype(float).tolist(),
                "lift_depth_in_topdown_cam": float(z_vals[i]),
            }
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "source_trajectory_json": str(traj_path),
                "source_semantic_metadata_json": str(meta_path),
                "source_pred_traj_txt": str(source_path),
                "depth_mode": args.depth_mode,
                "orthographic_intrinsics": {
                    "focal_length_xy": [fx, fy],
                    "principal_point_xy": [cx, cy],
                },
                "orthographic_camera_to_world_4x4": cam2world.tolist(),
                "frames": lifted,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    output_tum.parent.mkdir(parents=True, exist_ok=True)
    with open(output_tum, "w", encoding="utf-8") as f:
        for i in range(n):
            src = source_poses[min(i, len(source_poses) - 1)]
            tx, ty, tz = world_pts[i].tolist()
            qw, qx, qy, qz = src["q"].tolist()
            f.write(f"{float(i):.6f} {tx:.9f} {ty:.9f} {tz:.9f} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f}\n")

    print(f"Loaded {len(uv_traj)} UV points from: {traj_path}")
    print(f"Loaded {len(source_poses)} source poses from: {source_path}")
    print(f"Saved lifted 3D trajectory JSON to: {output_json}")
    print(f"Saved lifted TUM trajectory to: {output_tum}")


if __name__ == "__main__":
    main()
