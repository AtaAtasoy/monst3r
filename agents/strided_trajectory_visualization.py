import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from dreifus.matrix import CameraCoordinateConvention, Pose, PoseType

from lift_uv_traj_to_3d import cam_to_world, load_tum_traj, world_to_cam
from visualize_generated_trajectory_topdown import (
    build_frustum_world_points,
    build_topdown_camera,
    create_video_from_frames,
    draw_frustum_wireframe,
    list_frame_plys,
    load_intrinsics,
    project_topdown,
    render_scene_frame_topdown,
)


def lift_strided_poses(
    *,
    strided_positions: dict[str, dict[str, list[float]]],
    semantic_metadata: dict[str, Any],
    source_traj_path: Path,
) -> list[tuple[str, int, Pose, Pose]]:
    source_poses = load_tum_traj(source_traj_path)

    cam2world = np.asarray(semantic_metadata["orthographic_camera_to_world_4x4"], dtype=np.float64)
    params = semantic_metadata["orthographic_cam_parameters"]
    fx, fy = map(float, params["focal_length_xy"])
    cx, cy = map(float, params["principal_point_xy"])

    source_world = np.stack([pose["t"] for pose in source_poses], axis=0)
    source_cam = world_to_cam(source_world, cam2world)
    source_depths = source_cam[:, 2]

    lifted_poses: list[tuple[str, int, Pose, Pose]] = []
    for key in sorted(strided_positions, key=lambda item: int(item.split("_", 1)[1])):
        frame_index = int(key.split("_", 1)[1])
        u, v = strided_positions[key]["position"]
        src = source_poses[min(frame_index, len(source_poses) - 1)]
        z_val = float(source_depths[min(frame_index, len(source_depths) - 1)])
        cam_point = np.array(
            [[(u - cx) / max(fx, 1e-8), (v - cy) / max(fy, 1e-8), z_val]],
            dtype=np.float64,
        )
        world_point = cam_to_world(cam_point, cam2world)[0]

        qw, qx, qy, qz = [float(value) for value in src["q"]]
        quaternion_xyzw = [qx, qy, qz, qw]
        generated_pose = Pose.from_quaternion(
            quaternion=quaternion_xyzw,
            translation=world_point.tolist(),
            pose_type=PoseType.CAM_2_WORLD,
            camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
        )
        source_pose = Pose.from_quaternion(
            quaternion=quaternion_xyzw,
            translation=src["t"].astype(float).tolist(),
            pose_type=PoseType.CAM_2_WORLD,
            camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
        )
        lifted_poses.append((key, frame_index, generated_pose, source_pose))
    return lifted_poses


def project_pose_centers(
    poses: list[Pose],
    *,
    topdown_camera: Any,
    device: torch.device,
    img_h: int,
    img_w: int,
) -> list[tuple[float, float]]:
    if not poses:
        return []
    world_xyz = np.stack(
        [np.asarray(pose.get_translation(), dtype=np.float64) for pose in poses],
        axis=0,
    )
    screen = project_topdown(
        world_xyz,
        topdown_camera,
        device,
        img_h,
        img_w,
    )
    return [(float(x), float(y)) for x, y in screen]


def draw_center_path(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    color: tuple[int, int, int],
    width: int,
) -> None:
    if len(points) < 2:
        return
    draw.line(points, fill=color, width=width)


def visualize_strided_trajectory(
    *,
    output_video_path: Path,
    strided_positions: dict[str, dict[str, list[float]]],
    semantic_metadata: dict[str, Any],
    source_traj_path: Path,
    intrinsics_path: Path,
    scene_ply_dir: Path,
    fps: int = 4,
) -> None:
    img_h, img_w = map(int, semantic_metadata["image_size"])
    cam2world = np.asarray(semantic_metadata["orthographic_camera_to_world_4x4"], dtype=np.float64)
    fx, fy = map(float, semantic_metadata["orthographic_cam_parameters"]["focal_length_xy"])
    cx, cy = map(float, semantic_metadata["orthographic_cam_parameters"]["principal_point_xy"])
    view_half_extent = semantic_metadata["orthographic_cam_parameters"]["view_half_extent_xy"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    topdown_camera = build_topdown_camera(
        cam2world=cam2world,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        img_h=img_h,
        img_w=img_w,
        device=device,
    )

    intrinsics = load_intrinsics(intrinsics_path)
    scene_ply_files = list_frame_plys(scene_ply_dir)
    if not scene_ply_files:
        raise ValueError(f"No frame_*.ply files found in {scene_ply_dir}")

    sparse_poses = lift_strided_poses(
        strided_positions=strided_positions,
        semantic_metadata=semantic_metadata,
        source_traj_path=source_traj_path,
    )
    generated_poses = [generated_pose for _, _, generated_pose, _ in sparse_poses]
    source_poses = [source_pose for _, _, _, source_pose in sparse_poses]
    generated_centers = project_pose_centers(
        generated_poses,
        topdown_camera=topdown_camera,
        device=device,
        img_h=img_h,
        img_w=img_w,
    )
    source_centers = project_pose_centers(
        source_poses,
        topdown_camera=topdown_camera,
        device=device,
        img_h=img_h,
        img_w=img_w,
    )
    depth_scale = 0.08 * max(float(view_half_extent[0]), float(view_half_extent[1]))
    frames_dir = output_video_path.parent / f"{output_video_path.stem}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    for index, (key, frame_index, generated_pose, source_pose) in enumerate(sparse_poses):
        ply_path = scene_ply_files[min(frame_index, len(scene_ply_files) - 1)]
        bg_np = render_scene_frame_topdown(
            ply_path=ply_path,
            camera=topdown_camera,
            img_h=img_h,
            img_w=img_w,
            device=device,
        )
        base = Image.fromarray(bg_np)
        draw = ImageDraw.Draw(base)

        draw_center_path(draw, source_centers[: index + 1], color=(0, 255, 255), width=2)
        draw_center_path(draw, generated_centers[: index + 1], color=(255, 176, 66), width=3)

        intr = intrinsics[min(frame_index, len(intrinsics) - 1)]

        generated_frustum_world = build_frustum_world_points(
            generated_pose,
            intr,
            depth_scale=depth_scale,
        )
        generated_frustum_screen = project_topdown(
            generated_frustum_world,
            topdown_camera,
            device,
            img_h,
            img_w,
        )
        generated_frustum_px = [
            (int(round(x)), int(round(y))) for x, y in generated_frustum_screen
        ]
        draw_frustum_wireframe(draw, generated_frustum_px, color=(255, 176, 66), width=2)

        source_frustum_world = build_frustum_world_points(
            source_pose,
            intr,
            depth_scale=depth_scale,
        )
        source_frustum_screen = project_topdown(
            source_frustum_world,
            topdown_camera,
            device,
            img_h,
            img_w,
        )
        source_frustum_px = [(int(round(x)), int(round(y))) for x, y in source_frustum_screen]
        draw_frustum_wireframe(draw, source_frustum_px, color=(0, 255, 255), width=2)

        if index < len(generated_centers):
            center_x, center_y = generated_centers[index]
            draw.ellipse(
                (center_x - 4, center_y - 4, center_x + 4, center_y + 4),
                fill=(255, 96, 96),
                outline=(0, 0, 0),
            )
            draw.text((center_x + 8, center_y - 10), key, fill=(255, 255, 255), font=font)

        if index < len(source_centers):
            src_x, src_y = source_centers[index]
            draw.ellipse(
                (src_x - 4, src_y - 4, src_x + 4, src_y + 4),
                fill=(0, 255, 255),
                outline=(0, 0, 0),
            )

        base.save(frames_dir / f"{index:04d}.png")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    create_video_from_frames(
        str(frames_dir / "%04d.png"),
        output_video_path,
        fps=fps,
    )
