import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from render_top_down_semantic import (
    load_tum_poses,
)

from dreifus.matrix import Pose

def calculate_avg_distance_between_poses(poses: list[Pose]) -> float:
    if len(poses) < 2:
        return 0.0
    positions = np.array([pose.get_translation() for pose in poses])
    dists = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    return dists.mean()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate average distance between source camera poses.")
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    source_poses = load_tum_poses(data_dir / "pred_traj.txt")
    print(f"Average distance between source camera positions: {calculate_avg_distance_between_poses(source_poses):.4f} units")



if __name__ == "__main__":
    main()
