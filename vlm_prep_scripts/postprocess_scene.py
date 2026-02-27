"""
Advanced point cloud filtering with foreground/background separation.
Filtering can be enabled/disabled. When enabled, both foreground and background
support confidence and/or distance-to-cluster-mean filtering.

All filters are applied in the (H, W) pixel grid so that the 1:1 correspondence
between points, confidence, and mask pixels is maintained.

Usage:
    python filter_points.py --base_dir <base_dir> \
        --apply_filtering \
        --mask_prefix enlarged_dynamic_mask \
        --fg_filter_method distance \
        --bg_filter_method distance \
        --bg_dist_std_factor 3.0 \
"""

import os
import glob
import json
import shutil
import argparse
import numpy as np
import trimesh
import cv2
from tqdm import tqdm
from plyfile import PlyData, PlyElement


# ──────────────────────────────────────────────────────────────
# Core filtering
# ──────────────────────────────────────────────────────────────


def load_tum_trajectory(traj_path: str) -> list[dict]:
    """
    Load full TUM trajectory (timestamp, translation, quaternion).
    Returns list of dicts with 'timestamp', 'translation', 'qw', 'qx', 'qy', 'qz'.
    """
    entries = []
    with open(traj_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = list(map(float, line.split()))
            if len(parts) < 8:
                continue
            entries.append({
                "timestamp": parts[0],
                "tx": parts[1], "ty": parts[2], "tz": parts[3],
                "qw": parts[4], "qx": parts[5], "qy": parts[6], "qz": parts[7],
            })
    return entries


def normalize_scene(
    base_dir: str,
    suffix: str = "",
):
    """
    Normalize all_points/, foreground_points/, and background_points/
    into a unit cube [-0.5, 0.5] centred at origin.

    Outputs go to base_dir/normalized/ with sub-dirs:
        normalized/all_points/
        normalized/foreground_points/
        normalized/background_points/
        normalized/pred_traj.txt
        normalized/pred_intrinsics.txt
        normalized/normalization_params.json

    suffix: optional suffix matching the filtered directories (e.g. _conf0p1_fg0p0).
    """
    base_dir = os.path.abspath(base_dir)
    out_root = os.path.join(base_dir, f"normalized{suffix}")
    os.makedirs(out_root, exist_ok=True)

    traj_path = os.path.join(base_dir, "pred_traj.txt")
    intrinsics_path = os.path.join(base_dir, "pred_intrinsics.txt")

    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")

    traj = load_tum_trajectory(traj_path)
    cam_centers = np.array([[e["tx"], e["ty"], e["tz"]] for e in traj])

    # ── Compute bounding box from all_points PLYs + camera centres ──
    print("\nNormalization: computing scene bounding box …")
    scene_min = np.full(3, np.inf)
    scene_max = np.full(3, -np.inf)

    all_points_dir = os.path.join(base_dir, f"all_points{suffix}")
    ply_files = sorted(glob.glob(os.path.join(all_points_dir, "frame_*.ply")))

    for ply_path in tqdm(ply_files, desc="  bbox"):
        pcd = trimesh.load(ply_path, process=False)
        verts = _get_vertices(pcd)
        if verts is None or len(verts) == 0:
            continue
        scene_min = np.minimum(scene_min, verts.min(axis=0))
        scene_max = np.maximum(scene_max, verts.max(axis=0))

    # Include camera centres
    scene_min = np.minimum(scene_min, cam_centers.min(axis=0))
    scene_max = np.maximum(scene_max, cam_centers.max(axis=0))

    center = (scene_min + scene_max) / 2.0
    scale = 1.0 / (scene_max - scene_min).max()

    print(f"  Scene center: {center}")
    print(f"  Scene scale:  {scale:.6f}")

    # ── Save normalization params ────────────────────────────
    norm_params = {"center": center.tolist(), "scale": float(scale)}
    with open(os.path.join(out_root, "normalization_params.json"), "w") as f:
        json.dump(norm_params, f, indent=4)

    # ── Normalize + save trajectory ──────────────────────────
    with open(os.path.join(out_root, "pred_traj.txt"), "w") as f:
        for e in traj:
            t = np.array([e["tx"], e["ty"], e["tz"]])
            t_norm = (t - center) * scale
            f.write(
                f"{e['timestamp']:.6f} {t_norm[0]:.8f} {t_norm[1]:.8f} {t_norm[2]:.8f} "
                f"{e['qw']:.8f} {e['qx']:.8f} {e['qy']:.8f} {e['qz']:.8f}\n"
            )

    # ── Copy intrinsics (unchanged by normalization) ─────────
    if os.path.exists(intrinsics_path):
        shutil.copy(intrinsics_path, os.path.join(out_root, "pred_intrinsics.txt"))

    # ── Normalize PLY files in each sub-directory ────────────
    sub_dirs = ["all_points", "foreground_points", "background_points"]
    for sub in sub_dirs:
        src_dir = os.path.join(base_dir, f"{sub}{suffix}")
        dst_dir = os.path.join(out_root, sub)
        if not os.path.isdir(src_dir):
            continue
        os.makedirs(dst_dir, exist_ok=True)

        src_plys = sorted(glob.glob(os.path.join(src_dir, "frame_*.ply")))
        for ply_path in tqdm(src_plys, desc=f"  normalizing {sub}"):
            fname = os.path.basename(ply_path)
            ply_data = PlyData.read(ply_path)
            vertex = ply_data["vertex"]
            new_data = vertex.data.copy()

            pts = np.stack(
                [new_data["x"], new_data["y"], new_data["z"]], axis=-1
            ).astype(np.float64)
            pts_norm = (pts - center) * scale

            new_data["x"] = pts_norm[:, 0].astype(np.float32)
            new_data["y"] = pts_norm[:, 1].astype(np.float32)
            new_data["z"] = pts_norm[:, 2].astype(np.float32)

            new_vertex = PlyElement.describe(new_data, "vertex")
            PlyData([new_vertex], text=ply_data.text).write(
                os.path.join(dst_dir, fname)
            )

    # ── Copy images / depth / confidence ─────────────────────
    num_frames = len(traj)
    for i in range(num_frames):
        # PNG
        img_src = os.path.join(base_dir, f"frame_{i:04d}.png")
        if os.path.exists(img_src):
            shutil.copy(img_src, os.path.join(out_root, f"frame_{i:04d}.png"))

        # Depth NPY (scale depth by normalization scale)
        depth_src = os.path.join(base_dir, f"frame_{i:04d}.npy")
        if os.path.exists(depth_src):
            depth = np.load(depth_src)
            np.save(os.path.join(out_root, f"frame_{i:04d}.npy"), depth * scale)

        # Confidence NPY (unchanged)
        conf_src = os.path.join(base_dir, f"conf_{i}.npy")
        if os.path.exists(conf_src):
            shutil.copy(conf_src, os.path.join(out_root, f"conf_{i}.npy"))

    print(f"  Normalized scene saved to {out_root}")
    return norm_params


def filter_point_clouds(
    base_dir: str,
    apply_filtering: bool = False,
    conf_thre: float = 0.1,
    fg_conf_thre: float = 0.0,
    mask_prefix: str = "enlarged_dynamic_mask",
    fg_filter_method: str = "distance",
    bg_filter_method: str = "distance",
    fg_dist_max: float | None = None,
    fg_dist_std_factor: float = 3.0,
    bg_dist_max: float | None = None,
    bg_dist_std_factor: float = 3.0,
):
    base_dir = os.path.abspath(base_dir)
    use_init_conf = len(glob.glob(os.path.join(base_dir, "init_conf_*.npy"))) > 0

    suffix = _build_suffix(
        apply_filtering=apply_filtering,
        conf_thre=conf_thre,
        fg_conf_thre=fg_conf_thre,
        init_conf=use_init_conf,
        fg_filter_method=fg_filter_method,
        bg_filter_method=bg_filter_method,
        fg_dist_max=fg_dist_max,
        fg_dist_std_factor=fg_dist_std_factor,
        bg_dist_max=bg_dist_max,
        bg_dist_std_factor=bg_dist_std_factor,
    )

    # Output directories
    fg_dir = os.path.join(base_dir, f"foreground_points{suffix}")
    bg_dir = os.path.join(base_dir, f"background_points{suffix}")
    all_points_dir = os.path.join(base_dir, f"all_points{suffix}")
    os.makedirs(fg_dir, exist_ok=True)
    os.makedirs(bg_dir, exist_ok=True)
    os.makedirs(all_points_dir, exist_ok=True)

    # ── Discover PLY files ───────────────────────────────────
    ply_files = sorted(glob.glob(os.path.join(base_dir, "frame_*.ply")))
    ply_files = [f for f in ply_files if "_masked" not in os.path.basename(f)]
    print(f"Found {len(ply_files)} point cloud files.\n")

    # ── Collect per-frame stats for summary ──────────────────
    stats = {
        "total_points": 0,
        "removed_total": 0,
        "removed_fg_conf": 0,
        "removed_bg_conf": 0,
        "removed_fg_dist": 0,
        "removed_bg_dist": 0,
        "survived_total": 0,
        "fg_points": 0,
        "bg_points": 0,
    }

    # ── Single pass: visualization-aligned confidence + mask ─
    print("Splitting points into FG / BG and applying optional filtering …")
    for ply_path in tqdm(ply_files, desc="  filtering"):
        filename = os.path.basename(ply_path)
        idx = int(filename.split("_")[1].split(".")[0])

        # ── Load point cloud ─────────────────────────────────
        pcd = trimesh.load(ply_path, process=False)
        vertices = _get_vertices(pcd)
        if vertices is None:
            continue
        colors = _get_colors(pcd)

        # ── Load segmentation mask (visualizer semantics) ────
        # If mask is missing, visualizer treats all pixels as foreground.
        mask_path = os.path.join(base_dir, f"{mask_prefix}_{idx}.png")
        mask_flat = np.ones(vertices.shape[0], dtype=bool)
        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                H, W = mask_img.shape
                if vertices.shape[0] == H * W:
                    mask_flat = (mask_img.reshape(-1) > 0)
                else:
                    print(
                        f"  Shape mismatch frame {idx}: {vertices.shape[0]} pts vs "
                        f"{H}×{W}={H*W} px. Treating as all-foreground mask."
                    )
        else:
            # Try alternate prefix before defaulting to all-foreground
            alt_prefix = "dynamic_mask" if mask_prefix != "dynamic_mask" else "enlarged_dynamic_mask"
            alt_mask_path = os.path.join(base_dir, f"{alt_prefix}_{idx}.png")
            if os.path.exists(alt_mask_path):
                mask_img = cv2.imread(alt_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    H, W = mask_img.shape
                    if vertices.shape[0] == H * W:
                        mask_flat = (mask_img.reshape(-1) > 0)
                    else:
                        print(
                            f"  Shape mismatch frame {idx}: {vertices.shape[0]} pts vs "
                            f"{H}×{W}={H*W} px. Treating as all-foreground mask."
                        )

        # ── Load confidence (raw, clipped like visualizer) ───
        conf_path = os.path.join(base_dir, f"conf_{idx}.npy")
        if os.path.exists(conf_path):
            conf_raw = np.load(conf_path)
            conf_flat = np.clip(conf_raw.reshape(-1), 0.0001, 99999.0)
            if conf_flat.shape[0] != vertices.shape[0]:
                print(
                    f"  Shape mismatch frame {idx}: confidence has {conf_flat.shape[0]} "
                    f"values but point cloud has {vertices.shape[0]} points. Skipping."
                )
                continue
        else:
            # No confidence file → don't filter by confidence
            conf_flat = np.ones(vertices.shape[0])

        # Foreground confidence map: init_conf when present,
        # otherwise fall back to conf (same behavior as visualizer loader).
        init_conf_flat = conf_flat
        if use_init_conf:
            init_conf_path = os.path.join(base_dir, f"init_conf_{idx}.npy")
            if os.path.exists(init_conf_path):
                init_conf_raw = np.load(init_conf_path)
                init_conf_flat = np.clip(init_conf_raw.reshape(-1), 0.0001, 99999.0)
                if init_conf_flat.shape[0] != vertices.shape[0]:
                    print(
                        f"  Shape mismatch frame {idx}: init_conf has {init_conf_flat.shape[0]} "
                        f"values but point cloud has {vertices.shape[0]} points. "
                        "Falling back to conf map for foreground."
                    )
                    init_conf_flat = conf_flat

        n_total = vertices.shape[0]
        stats["total_points"] += n_total

        finite_conf = np.isfinite(conf_flat)
        finite_init = np.isfinite(init_conf_flat)
        conf_mask = finite_conf & (conf_flat > conf_thre)
        fg_conf_mask = finite_init & (init_conf_flat > fg_conf_thre)

        fg_region = mask_flat
        bg_region = ~mask_flat
        n_removed_fg_dist = 0
        n_removed_bg_dist = 0

        if not apply_filtering:
            fg_mask = fg_region
            bg_mask = bg_region
        else:
            # Foreground filtering
            if fg_filter_method == "confidence":
                fg_mask = fg_conf_mask & fg_region
            else:
                fg_dist_mask, _ = _compute_region_distance_mask(
                    vertices=vertices,
                    region_mask=fg_region,
                    dist_max=fg_dist_max,
                    dist_std_factor=fg_dist_std_factor,
                )
                if fg_filter_method == "distance":
                    fg_mask = fg_dist_mask & fg_region
                else:
                    fg_mask = fg_conf_mask & fg_dist_mask & fg_region
                n_removed_fg_dist = int((fg_region & ~fg_dist_mask).sum())

        # Background can be filtered by confidence, distance-to-cloud-mean, or both.
            if bg_filter_method == "confidence":
                bg_mask = conf_mask & bg_region
            else:
                bg_dist_mask, _ = _compute_region_distance_mask(
                    vertices=vertices,
                    region_mask=bg_region,
                    dist_max=bg_dist_max,
                    dist_std_factor=bg_dist_std_factor,
                )
                if bg_filter_method == "distance":
                    bg_mask = bg_dist_mask & bg_region
                else:
                    bg_mask = conf_mask & bg_dist_mask & bg_region
                n_removed_bg_dist = int((bg_region & ~bg_dist_mask).sum())

        valid = fg_mask | bg_mask

        n_fg = int(fg_mask.sum())
        n_bg = int(bg_mask.sum())
        n_survived = n_fg + n_bg
        n_removed = n_total - n_survived

        if apply_filtering:
            n_removed_fg_conf = int((fg_region & ~fg_conf_mask).sum())
            n_removed_bg_conf = int((bg_region & ~conf_mask).sum())
        else:
            n_removed_fg_conf = 0
            n_removed_bg_conf = 0

        stats["removed_total"] += n_removed
        stats["removed_fg_conf"] += n_removed_fg_conf
        stats["removed_bg_conf"] += n_removed_bg_conf
        stats["removed_fg_dist"] += n_removed_fg_dist
        stats["removed_bg_dist"] += n_removed_bg_dist
        stats["survived_total"] += n_survived
        stats["fg_points"] += n_fg
        stats["bg_points"] += n_bg

        if n_fg > 0:
            _save_ply(fg_dir, filename, vertices[fg_mask], colors[fg_mask] if colors is not None else None)
        if n_bg > 0:
            _save_ply(bg_dir, filename, vertices[bg_mask], colors[bg_mask] if colors is not None else None)

        # Save combined FG+BG points
        if n_survived > 0:
            _save_ply(all_points_dir, filename, vertices[valid], colors[valid] if colors is not None else None)

    # ── Print summary ────────────────────────────────────────
    _print_summary(
        stats=stats,
        apply_filtering=apply_filtering,
        conf_thre=conf_thre,
        fg_conf_thre=fg_conf_thre,
        init_conf=use_init_conf,
        fg_filter_method=fg_filter_method,
        bg_filter_method=bg_filter_method,
        fg_dist_max=fg_dist_max,
        fg_dist_std_factor=fg_dist_std_factor,
        bg_dist_max=bg_dist_max,
        bg_dist_std_factor=bg_dist_std_factor,
    )

    # ── Normalize scene + copy images/depth/conf ─────────────
    normalize_scene(
        base_dir=base_dir,
        suffix=suffix,
    )


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _format_float_for_suffix(val: float) -> str:
    """Format float for filesystem-safe suffix (95.0→95p0, 0.5→0p5)."""
    s = f"{val:g}"
    return s.replace("-", "m").replace(".", "p")


def _build_suffix(
    apply_filtering: bool,
    conf_thre: float,
    fg_conf_thre: float,
    init_conf: bool,
    fg_filter_method: str,
    bg_filter_method: str,
    fg_dist_max: float | None,
    fg_dist_std_factor: float,
    bg_dist_max: float | None,
    bg_dist_std_factor: float,
) -> str:
    """Build a suffix encoding filtering choices."""
    if not apply_filtering:
        return "_nofilter"
    conf_s = _format_float_for_suffix(conf_thre)
    fg_s = _format_float_for_suffix(fg_conf_thre)
    fg_mode_s = {
        "confidence": "fgconf",
        "distance": "fgdist",
        "confidence_and_distance": "fgconfdist",
    }[fg_filter_method]
    bg_mode_s = {
        "confidence": "bgconf",
        "distance": "bgdist",
        "confidence_and_distance": "bgconfdist",
    }[bg_filter_method]
    if fg_dist_max is not None:
        fg_dist_s = f"_fgdmax{_format_float_for_suffix(fg_dist_max)}"
    else:
        fg_dist_s = f"_fgdstd{_format_float_for_suffix(fg_dist_std_factor)}"
    if bg_dist_max is not None:
        bg_dist_s = f"_dmax{_format_float_for_suffix(bg_dist_max)}"
    else:
        bg_dist_s = f"_dstd{_format_float_for_suffix(bg_dist_std_factor)}"
    init_s = "_initconf" if init_conf else ""
    return f"_conf{conf_s}_fg{fg_s}_{fg_mode_s}{fg_dist_s}_{bg_mode_s}{bg_dist_s}{init_s}"


def _compute_region_distance_mask(
    vertices: np.ndarray,
    region_mask: np.ndarray,
    dist_max: float | None,
    dist_std_factor: float,
) -> tuple[np.ndarray, float]:
    """Compute distance filter mask using region-specific mean center."""
    dist_mask = np.zeros(vertices.shape[0], dtype=bool)
    region_pts = vertices[region_mask]
    if region_pts.shape[0] == 0:
        return dist_mask, np.inf

    center = region_pts.mean(axis=0)
    dists = np.linalg.norm(vertices - center, axis=1)
    finite_dist = np.isfinite(dists)
    region_finite = finite_dist & region_mask

    if dist_max is not None:
        dist_cutoff = dist_max
    else:
        finite_dists = dists[region_finite]
        if finite_dists.size == 0:
            dist_cutoff = np.inf
        else:
            dist_cutoff = float(finite_dists.mean() + dist_std_factor * finite_dists.std())

    dist_mask = finite_dist & (dists <= dist_cutoff)
    return dist_mask, dist_cutoff


def _get_vertices(pcd) -> np.ndarray | None:
    """Extract vertices from a trimesh object (PointCloud or Mesh)."""
    if isinstance(pcd, trimesh.points.PointCloud):
        return np.asarray(pcd.vertices)
    if hasattr(pcd, "vertices"):
        return np.asarray(pcd.vertices)
    return None


def _get_colors(pcd) -> np.ndarray | None:
    """Extract RGBA colors from a trimesh object."""
    if hasattr(pcd, "colors") and pcd.colors is not None and len(pcd.colors) > 0:
        return np.asarray(pcd.colors)
    if hasattr(pcd, "visual") and hasattr(pcd.visual, "vertex_colors"):
        return np.asarray(pcd.visual.vertex_colors)
    return None


def _save_ply(out_dir: str, filename: str, points: np.ndarray, colors: np.ndarray | None):
    """Save a point cloud to PLY."""
    pcd = trimesh.points.PointCloud(points, colors=colors)
    pcd.export(os.path.join(out_dir, filename))


def _print_summary(
    stats: dict,
    apply_filtering: bool,
    conf_thre: float,
    fg_conf_thre: float,
    init_conf: bool,
    fg_filter_method: str,
    bg_filter_method: str,
    fg_dist_max: float | None,
    fg_dist_std_factor: float,
    bg_dist_max: float | None,
    bg_dist_std_factor: float,
):
    total = stats["total_points"]
    if total == 0:
        print("No points processed.")
        return

    pct = lambda n: f"{n:>12,}  ({100*n/total:5.1f}%)"

    print("\n" + "=" * 62)
    print("  FILTERING SUMMARY")
    print("=" * 62)
    print(f"  Filtering enabled    : {apply_filtering}")
    print(f"  Foreground method    : {fg_filter_method}")
    print(f"  Background method    : {bg_filter_method}")
    print(f"  Foreground conf_thre : > {fg_conf_thre} ({'init_conf' if init_conf else 'conf'})")
    print(f"  Background conf_thre : > {conf_thre}")
    if fg_dist_max is not None:
        print(f"  Foreground dist max  : <= {fg_dist_max}")
    else:
        print(f"  Foreground dist max  : <= mean + {fg_dist_std_factor} * std")
    if bg_dist_max is not None:
        print(f"  Background dist max  : <= {bg_dist_max}")
    else:
        print(f"  Background dist max  : <= mean + {bg_dist_std_factor} * std")
    print("-" * 62)
    print(f"  Total points         : {total:>12,}")
    print(f"  Removed (total)      : {pct(stats['removed_total'])}")
    print(f"  Removed in FG region : {pct(stats['removed_fg_conf'])}")
    print(f"  Removed FG by dist   : {pct(stats['removed_fg_dist'])}")
    print(f"  Removed in BG region : {pct(stats['removed_bg_conf'])}")
    print(f"  Removed BG by dist   : {pct(stats['removed_bg_dist'])}")
    print(f"  Survived (valid)     : {pct(stats['survived_total'])}")
    print("-" * 62)
    print(f"  Foreground points    : {pct(stats['fg_points'])}")
    print(f"  Background points    : {pct(stats['bg_points'])}")
    print("=" * 62)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split point clouds into FG/BG via masks with optional FG/BG filtering, "
                    "then normalize outputs."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        help="Path to directory containing PLY files, conf NPYs, masks, and pred_traj.txt",
    )
    parser.add_argument(
        "--apply_filtering",
        action="store_true",
        help="Enable FG/BG point filtering; otherwise only split by mask",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.1,
        help="Background confidence threshold (raw conf > threshold, default: 0.1)",
    )
    parser.add_argument(
        "--fg_conf_threshold",
        type=float,
        default=0.0,
        help="Foreground confidence threshold (default: 0.0)",
    )
    parser.add_argument(
        "--mask_prefix",
        type=str,
        default="enlarged_dynamic_mask",
        help="Prefix for segmentation mask files (default: enlarged_dynamic_mask)",
    )
    parser.add_argument(
        "--fg_filter_method",
        type=str,
        default="distance",
        choices=["confidence", "distance", "confidence_and_distance"],
        help="Foreground filtering method (default: distance)",
    )
    parser.add_argument(
        "--bg_filter_method",
        type=str,
        default="distance",
        choices=["confidence", "distance", "confidence_and_distance"],
        help="Background filtering method (default: distance)",
    )
    parser.add_argument(
        "--fg_dist_max",
        type=float,
        default=None,
        help="Absolute distance threshold for foreground filtering (optional)",
    )
    parser.add_argument(
        "--fg_dist_std_factor",
        type=float,
        default=3.0,
        help="Foreground distance threshold as mean + factor*std when --fg_dist_max is unset (default: 3.0)",
    )
    parser.add_argument(
        "--bg_dist_max",
        type=float,
        default=None,
        help="Absolute distance threshold for background filtering (optional)",
    )
    parser.add_argument(
        "--bg_dist_std_factor",
        type=float,
        default=3.0,
        help="Distance threshold as mean + factor*std when --bg_dist_max is unset (default: 3.0)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.base_dir):
        print(f"Error: Directory not found: {args.base_dir}")
        exit(1)

    filter_point_clouds(
        base_dir=args.base_dir,
        apply_filtering=args.apply_filtering,
        conf_thre=args.conf_threshold,
        fg_conf_thre=args.fg_conf_threshold,
        mask_prefix=args.mask_prefix,
        fg_filter_method=args.fg_filter_method,
        bg_filter_method=args.bg_filter_method,
        fg_dist_max=args.fg_dist_max,
        fg_dist_std_factor=args.fg_dist_std_factor,
        bg_dist_max=args.bg_dist_max,
        bg_dist_std_factor=args.bg_dist_std_factor,
    )
