import os
os.environ["PYVISTA_OFF_SCREEN"] = "true"

import numpy as np
import pyvista as pv
from pathlib import Path
import colorsys
from argparse import ArgumentParser


def get_color(frame_idx, total_frames):
    """Red -> Blue gradient over the sequence."""
    hue = 0.66 * (frame_idx / max(1, total_frames - 1))
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return (r, g, b)


def swap_yz(pt):
    """Swap Y and Z so that Z is forward and Y is up in the visualization."""
    pt = np.asarray(pt)
    return np.array([pt[0], pt[2], pt[1]])


def add_coordinate_axes(plotter, origin, length=0.1):
    """Draw RGB coordinate axes (X=red, Y=green, Z=blue) at origin."""
    axes = [
        (np.array([length, 0, 0]), "red",   "X"),
        (np.array([0, length, 0]), "blue",  "Z (fwd)"),
        (np.array([0, 0, length]), "green", "Y (up)"),
    ]
    for direction, color, label in axes:
        end = origin + direction
        line = pv.Line(origin, end)
        plotter.add_mesh(line, color=color, line_width=4)
        plotter.add_point_labels(
            [end], [label], font_size=14, text_color=color,
            point_size=0, shape=None, render_points_as_spheres=False,
        )


def build_frustum_mesh(pose_c2w, size=0.03):
    """Build a frustum mesh (lines + face) from a c2w pose matrix.

    Returns (lines_mesh, face_mesh, center_display, corners_display, down_display).
    """
    C = swap_yz(pose_c2w[:3, 3])
    right = swap_yz(pose_c2w[:3, 0])
    down = swap_yz(pose_c2w[:3, 1])
    fwd = swap_yz(pose_c2w[:3, 2])

    hw = size * 0.8
    hh = size * 0.5
    d = size

    corners = [
        C + d * fwd + hw * right + hh * down,
        C + d * fwd - hw * right + hh * down,
        C + d * fwd - hw * right - hh * down,
        C + d * fwd + hw * right - hh * down,
    ]

    # Edges: apex -> 4 corners + 4 rectangle edges
    points = [C] + corners  # indices: 0=apex, 1-4=corners
    lines = []
    # Apex to corners
    for i in range(1, 5):
        lines.extend([2, 0, i])
    # Rectangle edges
    for i in range(4):
        lines.extend([2, i + 1, (i % 4) + 1 + (1 if i < 3 else -3)])
    # Fix rectangle: 1-2, 2-3, 3-4, 4-1
    lines = []
    for i in range(1, 5):
        lines.extend([2, 0, i])
    lines.extend([2, 1, 2])
    lines.extend([2, 2, 3])
    lines.extend([2, 3, 4])
    lines.extend([2, 4, 1])

    line_mesh = pv.PolyData(np.array(points), lines=np.array(lines))

    # Face (image plane quad)
    face_points = np.array(corners)
    face = pv.PolyData(face_points, faces=[4, 0, 1, 2, 3])

    return line_mesh, face, C, corners, down


def main():
    parser = ArgumentParser(
        description="Visualize camera trajectories from .npz pose files using PyVista (headless)."
    )
    parser.add_argument(
        "--pose_path", type=str, required=True, help="Path to the .npz pose file"
    )
    parser.add_argument(
        "--step_size", type=int, default=10, help="Step size for frames"
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Output image path. Defaults to <pose_stem>_trajectory.png next to the pose file.",
    )
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=[512, 512],
        metavar=("W", "H"), help="Output image resolution (default: 1920 1080)",
    )
    parser.add_argument(
        "--frustum_size", type=float, default=0.03,
        help="Size of camera frustums (default: 0.03)",
    )
    args = parser.parse_args()

    pose_path = Path(args.pose_path)
    resolved_path = pose_path.resolve()
    if not resolved_path.exists():
        print(f"Error: Pose file not found: {resolved_path}")
        return

    # ── Load pose data ───────────────────────────────────────────────
    print(f"Visualizing Pose: {resolved_path}")
    pose_data = np.load(resolved_path)
    poses_raw = pose_data["data"]
    inds = pose_data["inds"]
    n_frames = len(inds)

    indices = list(range(0, n_frames, args.step_size))

    # ── Setup PyVista plotter ────────────────────────────────────────
    width, height = args.resolution
    p = pv.Plotter(off_screen=True, window_size=[width, height])
    p.set_background("white")

    # ── Draw frustums ────────────────────────────────────────────────
    all_cam_centers = []
    for idx in indices:
        frame_id = inds[idx]
        color = get_color(idx, n_frames)

        line_mesh, face_mesh, C, corners, down = build_frustum_mesh(
            poses_raw[idx], size=args.frustum_size
        )
        p.add_mesh(line_mesh, color=color, line_width=2, opacity=0.85)
        p.add_mesh(face_mesh, color=color, opacity=0.15)

        # Frame label
        label_pos = corners[2] + 0.3 * args.frustum_size * (-np.array(down))
        p.add_point_labels(
            [label_pos], [str(frame_id)], font_size=12, text_color=color,
            point_size=0, shape=None, render_points_as_spheres=False,
        )
        all_cam_centers.append(C)

    # ── Trajectory line ──────────────────────────────────────────────
    all_positions = np.array([swap_yz(poses_raw[i][:3, 3]) for i in range(n_frames)])
    trajectory = pv.Spline(all_positions, n_points=n_frames)
    p.add_mesh(trajectory, color="gray", line_width=2, opacity=0.5)

    # Start / End markers
    p.add_mesh(pv.Sphere(radius=args.frustum_size * 0.3, center=all_positions[0]),
               color="red", label="Start")
    p.add_mesh(pv.Sphere(radius=args.frustum_size * 0.3, center=all_positions[-1]),
               color="blue", label="End")

    # ── Coordinate axes ──────────────────────────────────────────────
    # Use bbox center instead of arithmetic mean.
    # Mean is biased for partial arcs (e.g., 180 deg), which makes trajectories
    # appear visually off-center despite correct pose generation.
    min_pos = all_positions.min(axis=0)
    max_pos = all_positions.max(axis=0)
    center = 0.5 * (min_pos + max_pos)
    extent = max_pos - min_pos
    max_extent = max(extent.max(), 0.01)
    axis_len = max_extent * 0.3
    pad = max_extent * 0.6
    axis_origin = center - np.array([pad * 0.8, pad * 0.8, pad * 0.8])
    add_coordinate_axes(p, origin=axis_origin, length=axis_len)

    # ── Title ────────────────────────────────────────────────────────
    p.add_text(
        f"{pose_path.name}\nRed (Start) -> Blue (End)",
        position="upper_left", font_size=10, color="black",
    )
    p.add_legend(face=None)

    # ── Camera: 3D perspective showing all axes ──────────────────────
    p.camera.position = (
        center[0] + max_extent * 1.2,
        center[1] - max_extent * 0.8,
        center[2] + max_extent * 1.0,
    )
    p.camera.focal_point = tuple(center)
    p.camera.up = (0, 0, 1)  # Y (up) is the vertical axis in display

    # ── Save ─────────────────────────────────────────────────────────
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = pose_path.with_name(f"{pose_path.stem}_trajectory.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    p.screenshot(str(output_path))
    p.close()

    print(f"Saved trajectory visualization to: {output_path}")


if __name__ == "__main__":
    main()

"""
python agents/visualize_custom_trajectory.py \
  --pose_path path/to/pose.npz \
  --step_size 5 \
  --frustum_size 0.03 \
  --resolution 1920 1080 \
  --output_path custom_output.png
"""
