import numpy as np
from pathlib import Path


from argparse import ArgumentParser


def create_custom_trajectory(
    pose_path,
    mode="zoom",
    zoom_enhancement=0.15,
    amplitude=0.05,
    frequency=1.0,
    radius=0.5,
    angle=360,
):
    pose_path = Path(pose_path)
    resolved_path = pose_path.resolve()

    # Generate descriptive filename based on parameters
    if mode == "zoom":
        suffix = f"-zoom_z{zoom_enhancement:.2f}"
    elif mode == "global-zoom":
        suffix = f"-global_zoom_z{zoom_enhancement:.2f}"
    elif mode == "s-curve":
        suffix = f"-s_z{zoom_enhancement:.2f}_a{amplitude:.2f}_f{frequency:.2f}"
    elif mode == "spiral":
        suffix = f"-spiral_r{radius:.2f}_deg{angle}"
    elif mode == "anchored-spiral":
        suffix = f"-anchored_spiral_r{radius:.2f}_deg{angle}"
    elif mode == "half-spiral":
        suffix = f"-half_spiral_r{radius:.2f}_deg180"
    elif mode == "reverse-zoom":
        suffix = f"-reverse_zoom_z{zoom_enhancement:.2f}"
    else:
        suffix = f"-{mode}"

    output_path = pose_path.with_name(f"{pose_path.stem}{suffix}{pose_path.suffix}")

    print(f"Loading {resolved_path}...")
    data_obj = np.load(resolved_path)
    poses = data_obj["data"].copy()
    inds = data_obj["inds"]

    n_frames = len(poses)

    print(f"Applying mode: {mode}")

    # Store initial state for modes that anchor to the first frame
    initial_pose = poses[0].copy()
    initial_pos = initial_pose[:3, 3]
    initial_rot = initial_pose[:3, :3]

    # Initial local axes
    initial_right = initial_pose[:3, 0]
    initial_up = initial_pose[:3, 1]  # Down in OpenCV
    initial_fwd = initial_pose[:3, 2]

    # Middle index for half-spiral parameterization
    mid_idx = n_frames // 2
    world_x = np.array([1.0, 0.0, 0.0], dtype=poses.dtype)

    for i in range(n_frames):
        t = i / (max(1, n_frames - 1))

        if mode == "global-zoom":
            # Constant orientation from frame 0
            poses[i, :3, :3] = initial_rot
            
            # Move along initial forward direction (z-axis)
            offset = t * zoom_enhancement * initial_fwd
            poses[i, :3, 3] = initial_pos + offset

        elif mode == "s-curve":
            # 1. Constant orientation from frame 0
            poses[i, :3, :3] = initial_rot

            # 2. Synthetic path starting from frame 0 position
            # Forward movement + Lateral S-sway
            sway = amplitude * np.sin(2 * np.pi * frequency * t)
            offset = (t * zoom_enhancement * initial_fwd) + (sway * initial_right)
            poses[i, :3, 3] = initial_pos + offset

        elif mode == "zoom":
            # Original mode: augment the existing trajectory/orientation
            fwd = poses[i, :3, 2]
            offset = t * zoom_enhancement * fwd
            poses[i, :3, 3] += offset

        elif mode == "spiral":
            # 1. Define Orbit center (Point at 'radius' distance in front of initial camera)
            center = initial_pos + radius * initial_fwd

            # 2. Calculate New Position
            # We rotate around the 'initial_up' axis (OpenCV Y)
            theta = (angle * np.pi / 180.0) * t

            # Rotation around center
            # relative_pos at t=0 is -radius * initial_fwd
            # We rotate this vector
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            # Vector from center to camera
            # Initially (t=0, theta=0): -radius * initial_fwd
            # As theta increases, we sway towards 'initial_right'
            rel_pos = (-radius * cos_t) * initial_fwd + (radius * sin_t) * initial_right

            new_pos = center + rel_pos
            poses[i, :3, 3] = new_pos

            # 3. Update Orientation (Look at center)
            new_fwd = center - new_pos
            new_fwd /= np.linalg.norm(new_fwd)

            # Reconstruct orthogonal basis using initial_up as reference
            new_right = np.cross(initial_up, new_fwd)
            new_right /= np.linalg.norm(new_right)

            new_down = np.cross(new_fwd, new_right)

            poses[i, :3, 0] = new_right
            poses[i, :3, 1] = new_down
            poses[i, :3, 2] = new_fwd
        elif mode == "anchored-spiral":
            # Spiral anchored at the initial pose:
            # theta spans [0, angle], so frame 0 is exactly initial_pos.
            theta = (angle * np.pi / 180.0) * t

            center = initial_pos + radius * initial_fwd
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rel_pos = (-radius * cos_t) * initial_fwd + (radius * sin_t) * initial_right
            new_pos = center + rel_pos
            poses[i, :3, 3] = new_pos

            new_fwd = center - new_pos
            new_fwd /= np.linalg.norm(new_fwd)

            new_right = np.cross(initial_up, new_fwd)
            new_right /= np.linalg.norm(new_right)

            new_down = np.cross(new_fwd, new_right)

            poses[i, :3, 0] = new_right
            poses[i, :3, 1] = new_down
            poses[i, :3, 2] = new_fwd
        elif mode == "half-spiral":
            # Half arc anchored to the original first pose:
            # - middle frame stays at initial_pos (theta=0)
            # - sweep left->right along global X
            # - depth bend follows initial forward direction

            if n_frames <= 1:
                theta = 0.0
            elif i <= mid_idx:
                left_denom = max(1, mid_idx)
                theta = -0.5 * np.pi + 0.5 * np.pi * (i / left_denom)
            else:
                right_denom = max(1, n_frames - 1 - mid_idx)
                theta = 0.5 * np.pi * ((i - mid_idx) / right_denom)

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            offset = (radius * sin_t) * world_x + (radius * (1.0 - cos_t)) * initial_fwd
            new_pos = initial_pos + offset
            poses[i, :3, 3] = new_pos

            # Look towards the arc center derived from the initial pose.
            look_center = initial_pos + radius * initial_fwd
            new_fwd = look_center - new_pos
            new_fwd /= np.linalg.norm(new_fwd)

            # Preserve roll by reusing initial "down" axis as reference.
            ref_down = initial_up
            new_right = np.cross(ref_down, new_fwd)
            right_norm = np.linalg.norm(new_right)
            if right_norm < 1e-8:
                # Fallback in degenerate case where ref_down || new_fwd.
                ref_down = np.array([0.0, 1.0, 0.0], dtype=poses.dtype)
                new_right = np.cross(ref_down, new_fwd)
                right_norm = np.linalg.norm(new_right)
            new_right /= right_norm

            new_down = np.cross(new_fwd, new_right)
            new_down /= np.linalg.norm(new_down)

            poses[i, :3, 0] = new_right
            poses[i, :3, 1] = new_down
            poses[i, :3, 2] = new_fwd
        elif mode == "reverse-zoom":
            # the initial frames zoom in more, then it gradually reduces the zoom effect towards the end
            fwd = poses[i, :3, 2]
            offset = (1 - t) * zoom_enhancement * fwd
            poses[i, :3, 3] += offset

    print(f"Saving custom trajectory to {output_path}...")
    np.savez_compressed(output_path, data=poses, inds=inds)
    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--pose_path", type=str, required=True, help="Path to the input .npz pose file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="zoom",
        choices=[
            "zoom",
            "global-zoom",
            "s-curve",
            "spiral",
            "anchored-spiral",
            "half-spiral",
            "reverse-zoom",
        ],
        help="Type of trajectory modification to apply",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=0.15,
        help="Strength of the zoom-in effect (forward movement)",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.05,
        help="Amplitude of the lateral sway for s-curve",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=1.0,
        help="Number of S-cycles over the sequence",
    )
    parser.add_argument(
        "--radius", type=float, default=0.5, help="Radius of the orbit for spiral mode"
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=360,
        help="Total angle range for spiral mode (e.g. 180 or 360)",
    )
    args = parser.parse_args()

    create_custom_trajectory(
        args.pose_path,
        mode=args.mode,
        zoom_enhancement=args.zoom,
        amplitude=args.amplitude,
        frequency=args.frequency,
        radius=args.radius,
        angle=args.angle,
    )
