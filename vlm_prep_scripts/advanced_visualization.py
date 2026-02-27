import argparse
from pathlib import Path
import subprocess
import shutil
import json

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pytorch3d.io import IO
from pytorch3d.renderer import (
    AlphaCompositor,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    FoVOrthographicCameras,
)
from pytorch3d.structures import Pointclouds
from dreifus.matrix import CameraCoordinateConvention, Intrinsics, Pose, PoseType

from vlm_prep_scripts.render_top_down import make_topdown_pose, load_intrinsics, load_tum_poses, _make_topdown_camera


def _load_topdown_camera_from_json(json_path: Path, device: torch.device):
    """
    Load top-down camera parameters from JSON file exported by render_top_down.py.
    
    Returns: (camera, intr, img_w, img_h, look_at)
    """
    with open(json_path, 'r') as f:
        camera_data = json.load(f)
    
    projection_type = camera_data["projection_type"]
    image_size = camera_data["image_size"]
    img_h, img_w = image_size
    
    # Load extrinsics
    c2w_matrix = np.array(camera_data["extrinsics"]["camera_to_world_4x4"])
    look_at = np.array(camera_data["extrinsics"]["look_at_point"])
    
    # Load intrinsics
    intrinsics_data = camera_data["intrinsics"]
    intr = Intrinsics(matrix_or_fx=np.array(intrinsics_data["matrix_3x3"]))
    
    # Convert c2w matrix to Pose object in OpenCV convention
    topdown_pose = Pose(
        c2w_matrix,
        pose_type=PoseType.CAM_2_WORLD,
        camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
    )
    
    # Convert to PyTorch3D convention
    pytorch3d_c2w_pose = topdown_pose.change_camera_coordinate_convention(
        new_camera_coordinate_convention=CameraCoordinateConvention.PYTORCH_3D
    )
    R = pytorch3d_c2w_pose.get_rotation_matrix()
    pytorch3d_w2c_pose = pytorch3d_c2w_pose.change_pose_type(
        new_pose_type=PoseType.WORLD_2_CAM, inplace=False
    )
    T = pytorch3d_w2c_pose.get_translation()
    
    # Create camera based on projection type
    projection_params = camera_data["projection_params"]
    
    if projection_type == "perspective":
        fov_y = projection_params["fov_y_degrees"]
        camera = FoVPerspectiveCameras(
            R=R[None, :, :],
            T=T[None, :],
            fov=fov_y,
            degrees=True,
            device=device,
        )
    else:  # orthographic
        camera = FoVOrthographicCameras(
            R=R[None, :, :],
            T=T[None, :],
            min_x=projection_params["min_x"],
            max_x=projection_params["max_x"],
            min_y=projection_params["min_y"],
            max_y=projection_params["max_y"],
            device=device,
        )
    
    return camera, intr, img_w, img_h, look_at


def _list_frame_plys(ply_dir: Path) -> list[Path]:
    """List all frame PLY files in a directory."""
    return sorted([f for f in ply_dir.glob("frame_*.ply") if "_masked" not in f.name])


def _load_points(ply_path: Path, io: IO, device: torch.device) -> torch.Tensor:
    """Load point cloud from PLY file."""
    if ply_path is None or not ply_path.exists():
        return None

    cloud = io.load_pointcloud(str(ply_path), device=device)
    points = cloud.points_list()[0]
    if points.numel() == 0:
        return None
    
    return points


def _create_bev_occupancy(
    points: torch.Tensor,
    img_w: int,
    img_h: int,
    look_at: np.ndarray
) -> np.ndarray:
    """
    Create Bird's Eye View (BEV) Occupancy visualization.
    
    Maps X and Z coordinates to U and V pixel coordinates.
    Encodes Y (height) in Red channel, density in Green channel, presence in Blue channel.
    """
    bev_image = np.zeros((img_h, img_w, 3), dtype=np.float32)
    
    if points is None or points.numel() == 0:
        return (bev_image * 255).astype(np.uint8)
    
    points_np = points.cpu().numpy()
    
    x_vals = points_np[:, 0]
    y_vals = points_np[:, 1]  # Height
    z_vals = points_np[:, 2]
    
    # Map to pixel coordinates (X->U, Z->V)
    u = ((x_vals - look_at[0]) / 2.0 + 0.5) * img_w
    v = ((z_vals - look_at[2]) / 2.0 + 0.5) * img_h
    
    # Clamp to valid pixel range
    valid_mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    u = u[valid_mask].astype(int)
    v = v[valid_mask].astype(int)
    y_valid = y_vals[valid_mask]
    
    if len(u) == 0:
        return (bev_image * 255).astype(np.uint8)
    
    # Red channel: height encoding (normalize Y to [0, 1])
    y_min, y_max = y_valid.min(), y_valid.max()
    if y_max > y_min:
        y_normalized = (y_valid - y_min) / (y_max - y_min)
    else:
        y_normalized = np.zeros_like(y_valid)
    
    bev_image[v, u, 0] = np.maximum(bev_image[v, u, 0], y_normalized)
    
    # Green channel: density (increment for each point)
    np.add.at(bev_image[:, :, 1], (v, u), 1.0)
    if bev_image[:, :, 1].max() > 0:
        bev_image[:, :, 1] = np.clip(bev_image[:, :, 1] / bev_image[:, :, 1].max(), 0, 1)
    
    # Blue channel: presence (binary mask)
    bev_image[v, u, 2] = 1.0
    
    return (bev_image * 255).astype(np.uint8)


def _create_semantic_time_coding(
    all_points_frames: list[torch.Tensor],
    img_w: int,
    img_h: int,
    look_at: np.ndarray,
    cmap_name: str = "plasma"
) -> np.ndarray:
    """
    Create Semantic Time-Coding visualization.
    
    Assigns colors based on temporal sequence, creating a motion trail effect.
    Older frames have lower opacity (ghosting), newer frames are more prominent.
    """
    time_coded = np.zeros((img_h, img_w, 4), dtype=np.float32)  # RGBA
    
    cmap = plt.get_cmap(cmap_name)
    n_frames = len(all_points_frames)
    
    for frame_idx, points in enumerate(all_points_frames):
        if points is None or points.numel() == 0:
            continue
        
        # Get color from colormap based on frame position
        color = cmap(frame_idx / max(1, n_frames - 1))
        
        # Opacity decreases for older frames (temporal ghosting)
        alpha = (frame_idx + 1) / n_frames
        
        points_np = points.cpu().numpy()
        x_vals = points_np[:, 0]
        z_vals = points_np[:, 2]
        
        # Map to pixel coordinates
        u = ((x_vals - look_at[0]) / 2.0 + 0.5) * img_w
        v = ((z_vals - look_at[2]) / 2.0 + 0.5) * img_h
        
        # Clamp to valid pixel range
        valid_mask = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
        u = u[valid_mask].astype(int)
        v = v[valid_mask].astype(int)
        
        # Blend colors with alpha compositing
        for i in range(3):  # RGB
            time_coded[v, u, i] = (
                time_coded[v, u, i] * (1 - alpha) + color[i] * alpha
            )
        time_coded[v, u, 3] = np.maximum(time_coded[v, u, 3], alpha)
    
    # Normalize to [0, 255]
    result = time_coded[:, :, :3].copy()
    result = (result * 255).astype(np.uint8)
    
    return result


def _create_video_from_frames(frame_pattern: str, output_video: Path, fps: int = 30) -> None:
    """
    Create MP4 video from PNG frames using ffmpeg with yuv420p encoding.
    
    Args:
        frame_pattern: Pattern like "/path/to/prefix_%04d.png"
        output_video: Path to output MP4 file
        fps: Frames per second
    """
    cmd = [
        "ffmpeg",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-y",  # overwrite output file
        str(output_video),
    ]
    
    print(f"  Creating video: {output_video.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr}")
        raise RuntimeError(f"ffmpeg failed to create video {output_video}")


def _create_composite_frames(
    render_dir: Path,
    n_frames: int,
    img_w: int,
    img_h: int,
    enable_rgb: bool = True,
    enable_bev: bool = True,
    enable_time_coding: bool = True,
) -> list[Image.Image]:
    """
    Create composite images (3-way or 2-way split) from the generated visualizations.
    
    Note: BEV and time-coding frames are expected in the parent of render_dir,
    which is where they're generated by this script.
    
    Returns list of PIL Images.
    """
    composite_frames = []
    output_dir = render_dir.parent if render_dir != Path(render_dir).parent else render_dir
    
    for idx in range(n_frames):
        images_to_combine = []
        
        if enable_rgb:
            rgb_path = render_dir / f"top-down-render_{idx:04d}.png"
            if rgb_path.exists():
                images_to_combine.append(Image.open(rgb_path).convert("RGB"))
        
        if enable_bev:
            bev_path = output_dir / f"bev-occupancy_{idx:04d}.png"
            if bev_path.exists():
                images_to_combine.append(Image.open(bev_path).convert("RGB"))
        
        # For time-coding, we use the same composite for all frames
        if enable_time_coding:
            time_coding_path = output_dir / "semantic-time-coding.png"
            if time_coding_path.exists():
                images_to_combine.append(Image.open(time_coding_path).convert("RGB"))
        
        if not images_to_combine:
            continue
        
        # Create composite: arrange images in a grid
        if len(images_to_combine) == 3:
            composite_width = img_w * 3
            composite_height = img_h
            composite = Image.new("RGB", (composite_width, composite_height))
            composite.paste(images_to_combine[0], (0, 0))
            composite.paste(images_to_combine[1], (img_w, 0))
            composite.paste(images_to_combine[2], (img_w * 2, 0))
        elif len(images_to_combine) == 2:
            composite_width = img_w * 2
            composite_height = img_h
            composite = Image.new("RGB", (composite_width, composite_height))
            composite.paste(images_to_combine[0], (0, 0))
            composite.paste(images_to_combine[1], (img_w, 0))
        else:
            composite = images_to_combine[0]
        
        composite_frames.append(composite)
    
    return composite_frames


def _bbox_center(ply_files: list[Path]) -> np.ndarray:
    """Compute bounding box center from PLY files."""
    if not ply_files:
        return None
    
    io = IO()
    mins = []
    maxs = []
    
    for pth in ply_files:
        cloud = io.load_pointcloud(str(pth), device="cpu")
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


def main(args):
    """Main visualization function using PyTorch3D for all rendering."""
    root = Path(args.data_dir)
    fg_dir = root / "foreground_points"
    bg_dir = root / "background_points"
    
    # Load camera from JSON
    rgb_render_dir = Path(args.rgb_render_dir)
    json_path = rgb_render_dir / "topdown_camera.json"
    if not json_path.exists():
        raise FileNotFoundError(f"topdown_camera.json not found in {rgb_render_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading camera from {json_path}")
    camera, intr, img_w, img_h, look_at = _load_topdown_camera_from_json(json_path, device)
    projection = "ortho" if "ortho" in rgb_render_dir.name else "perspective"
    print(f"[camera] Loaded from JSON with projection={projection}")
    
    # Create output directory with render dir reference
    render_dir_name = rgb_render_dir.name
    out_root = root / f"master_visualizations-{render_dir_name}"
    ghost_dir = out_root / "temporal_ghosting"
    bev_dir = out_root / "bev_occupancy"
    
    for d in [ghost_dir, bev_dir]:
        d.mkdir(parents=True, exist_ok=True)

    fg_files = sorted(list(fg_dir.glob("frame_*.ply")))
    bg_files = sorted(list(bg_dir.glob("frame_*.ply")))
    num_frames = len(fg_files)
    print(f"Starting Master Render: {num_frames} frames found.")

    intrinsics_path = root / "pred_intrinsics.txt"
    intrinsics_list = load_intrinsics(intrinsics_path)
    intr = intrinsics_list[0] if intrinsics_list else None

    # Load source poses if available
    traj_path = root / "pred_traj.txt"
    source_poses = load_tum_poses(traj_path) if traj_path.exists() else []

    print(f"[look_at] Using look_at from loaded camera parameters: {look_at}")

    # Set up rasterization settings
    raster_settings = PointsRasterizationSettings(
        image_size=(img_h, img_w),
        bin_size=0,
        radius=0.003,
        points_per_pixel=10
    )

    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
        compositor=AlphaCompositor(),
    )

    io = IO()

    # Temporal ghosting visualization (foreground overlaid on background)
    for idx in range(num_frames):
        print(f"Rendering temporal ghosting {idx}/{num_frames} ...", end="\r")
        
        all_points = []
        all_features = []
        
        # Add background points with reduced opacity (0.3)
        if idx < len(bg_files):
            bg_pcd = io.load_pointcloud(str(bg_files[idx]), device=device)
            bg_pts_list = bg_pcd.points_list()
            if bg_pts_list[0].numel() > 0:
                bg_pts = bg_pts_list[0]
                bg_feat_list = bg_pcd.features_list()
                if bg_feat_list[0] is not None:
                    # Use original features (colors) from background points
                    bg_feat = bg_feat_list[0]
                    if bg_feat.dtype != torch.float32:
                        bg_feat = bg_feat.float()
                else:
                    # If no features, create white ones
                    bg_feat = torch.ones((bg_pts.shape[0], 3), device=device, dtype=torch.float32)
                
                # Reduce opacity to 0.3 by scaling colors
                bg_feat = bg_feat * 0.5
                all_points.append(bg_pts)
                all_features.append(bg_feat)
        
        # Add foreground points with time-dependent coloring (no ghosting)
        cmap = plt.get_cmap("plasma")
        # Map frame index to colormap (0 = oldest, 1 = newest)
        t = idx / max(1, num_frames - 1)
        color = cmap(t)  # Returns (R, G, B, A)
        
        fg_pcd = io.load_pointcloud(str(fg_files[idx]), device=device)
        fg_pts = fg_pcd.points_list()[0]
        if fg_pts.numel() > 0:
            # Create feature tensor with the temporal color
            color_tensor = torch.tensor(
                [color[0], color[1], color[2]], device=device, dtype=torch.float32
            )
            fg_feat = torch.ones((fg_pts.shape[0], 3), device=device, dtype=torch.float32) * color_tensor
            all_points.append(fg_pts)
            all_features.append(fg_feat)
        
        if not all_points:
            img_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        else:
            frame_pc = Pointclouds(
                points=[torch.cat(all_points, dim=0)],
                features=[torch.cat(all_features, dim=0)],
            )
            images = renderer(frame_pc)
            image = images[0, ..., :3].detach().cpu().numpy().clip(0, 1)
            img_np = (image * 255).astype(np.uint8)
        
        Image.fromarray(img_np).save(ghost_dir / f"ghost_{idx:04d}.png")

    # BEV occupancy visualization
    for idx in range(num_frames):
        print(f"Rendering BEV occupancy {idx}/{num_frames} ...", end="\r")
        pts = io.load_pointcloud(str(fg_files[idx]), device=device).points_list()[0]
        if pts.numel() == 0:
            img_np = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        else:
            img_np = _create_bev_occupancy(pts, img_w, img_h, look_at)
        Image.fromarray(img_np).save(bev_dir / f"bev_{idx:04d}.png")

    print(f"\nSuccess! Visualizations saved to: {out_root}")
    
    # Create MP4 videos from rendered frames
    print("\nGenerating MP4 videos...")
    video_dir = out_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Temporal ghosting video
    print("  Creating temporal ghosting video...")
    ghost_pattern = str(ghost_dir / "ghost_%04d.png")
    ghost_video = video_dir / "temporal_ghosting.mp4"
    try:
        _create_video_from_frames(ghost_pattern, ghost_video, fps=25)
    except RuntimeError as e:
        print(f"  Warning: Failed to create temporal ghosting video: {e}")
    
    # BEV occupancy video
    print("  Creating BEV occupancy video...")
    bev_pattern = str(bev_dir / "bev_%04d.png")
    bev_video = video_dir / "bev_occupancy.mp4"
    try:
        _create_video_from_frames(bev_pattern, bev_video, fps=25)
    except RuntimeError as e:
        print(f"  Warning: Failed to create BEV occupancy video: {e}")
    
    print(f"  Videos saved to {video_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Visualization Script for MoNST3R Scenes")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to normalized/ directory")
    parser.add_argument("--rgb_render_dir", type=str, required=True, help="Path to top-down RGB render directory (required, loads camera from topdown_camera.json)")
    parser.add_argument("--window", type=int, default=15, help="Number of frames for temporal accumulation")
    
    args = parser.parse_args()
    main(args)
