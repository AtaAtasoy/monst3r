#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def quat_wxyz_to_matrix(q: list[float]) -> np.ndarray:
    if len(q) != 4:
        raise ValueError(f"Expected quaternion of length 4, got {len(q)}")

    qw, qx, qy, qz = [float(v) for v in q]
    norm = np.linalg.norm([qw, qx, qy, qz])
    if norm <= 0:
        raise ValueError("Quaternion has zero norm")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    ww = qw * qw
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    rot = np.array(
        [
            [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float64,
    )
    return rot


def load_lifted_frames(lifted_json_path: Path) -> list[dict]:
    loaded = Path(lifted_json_path).read_text(encoding="utf-8")
    payload = json.loads(loaded)
    frames = payload.get("frames", [])
    if not isinstance(frames, list) or len(frames) == 0:
        raise ValueError("Input lifted trajectory JSON must contain a non-empty `frames` list.")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert lifted 3D trajectory JSON into GEN3C-compatible NPZ pose data."
    )
    parser.add_argument(
        "--lifted-json",
        type=str,
        required=True,
        help="Path to lifted_camera_trajectory_3d.json",
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        default="lifted_pred_traj.npz",
        help="Path to save generated NPZ.",
    )
    args = parser.parse_args()

    lifted_json = Path(args.lifted_json).resolve()
    output_npz = Path(args.output_npz).resolve()

    frames = load_lifted_frames(lifted_json)
    frames = sorted(frames, key=lambda x: int(x.get("frame_index", 0)))
    n = len(frames)
    poses = np.zeros((n, 4, 4), dtype=np.float64)
    inds = np.zeros((n,), dtype=np.int64)

    for i, frame in enumerate(frames):
        frame_index = int(frame["frame_index"])
        pos = frame["camera_position_world_xyz"]
        quat = frame["source_quaternion_wxyz"]
        if not (isinstance(pos, list) and len(pos) == 3):
            raise ValueError(f"Frame {frame_index}: invalid camera_position_world_xyz")
        if not (isinstance(quat, list) and len(quat) == 4):
            raise ValueError(f"Frame {frame_index}: invalid source_quaternion_wxyz")

        rot = quat_wxyz_to_matrix(quat)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rot
        pose[:3, 3] = np.asarray(pos, dtype=np.float64)

        poses[i] = pose
        inds[i] = frame_index

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, data=poses, inds=inds)
    print(f"Saved lifted trajectory npz to: {output_npz}")


if __name__ == "__main__":
    main()
