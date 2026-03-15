#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


def build_common_args(args, output_dir: Path) -> list[str]:
    cmd = [
        "python",
        "agents/visualize_generated_trajectory_topdown.py",
        "--semantic_metadata_json",
        str(args.semantic_metadata_json),
        "--output_dir",
        str(output_dir),
        "--fps",
        str(args.fps),
    ]
    if args.source_traj_txt:
        cmd.extend(["--source_traj_txt", str(args.source_traj_txt)])
    if args.scene_ply_dir:
        cmd.extend(["--scene_ply_dir", str(args.scene_ply_dir)])
    if args.intrinsics_path:
        cmd.extend(["--intrinsics_path", str(args.intrinsics_path)])
    return cmd


def run_visualization(label: str, cmd: list[str]) -> None:
    print(f"[{label}] Running:\n  {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{label} visualization failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render top-down visualizations for multiple trajectory variants."
    )
    parser.add_argument(
        "--iter-dir",
        required=True,
        type=str,
        help="Iteration directory containing 02_lift artifacts.",
    )
    parser.add_argument(
        "--semantic_metadata_json",
        type=str,
        default="demo_tmp/davis/tennis/normalized_nofilter/top-down-semantic-ortho-all-scene/semantic_topdown_metadata.json",
        help="Path to semantic_topdown_metadata.json",
    )
    parser.add_argument(
        "--source_traj_txt",
        type=str,
        default="demo_tmp/davis/tennis/normalized_nofilter/pred_traj.txt",
        help="Optional source trajectory for comparison/overlay.",
    )
    parser.add_argument(
        "--intrinsics_path",
        type=str,
        default="demo_tmp/davis/tennis/normalized_nofilter/pred_intrinsics.txt",
        help="Path to intrinsics file for frustum overlays.",
    )
    parser.add_argument(
        "--scene_ply_dir",
        type=str,
        default="demo_tmp/davis/tennis/normalized_nofilter/all_points",
        help="Scene point-cloud directory used for top-down background rendering.",
    )
    parser.add_argument(
        "--vlm-tum",
        type=str,
        default=None,
        help="Explicit lifted VLM TUM trajectory path. Default: 02_lift/lifted_pred_traj.txt under iter-dir.",
    )
    parser.add_argument(
        "--vlm-npz",
        type=str,
        default=None,
        help="Explicit lifted VLM NPZ trajectory path. Default: 02_lift/lifted_pred_traj.npz under iter-dir.",
    )
    parser.add_argument(
        "--gen3c-dir",
        type=str,
        default=None,
        help="Explicit Gen3C output dir containing pose/ and intrinsics/.npz. Default: 02_lift/gen3c_output under iter-dir.",
    )
    parser.add_argument(
        "--gen3c-pose-npz",
        type=str,
        default=None,
        help="Optional explicit Gen3C pose npz (used with --gen3c-dir).",
    )
    parser.add_argument(
        "--gen3c-intrinsics-npz",
        type=str,
        default=None,
        help="Optional explicit Gen3C intrinsics npz (used with --gen3c-dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root output directory for combined top-down videos. Default: <iter-dir>/05_topdown_variant_comparison",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Output video frame-rate.",
    )
    parser.add_argument(
        "--skip-vlm-tum",
        action="store_true",
        help="Skip VLM lifted TUM visualization.",
    )
    parser.add_argument(
        "--skip-vlm-npz",
        action="store_true",
        help="Skip VLM lifted NPZ visualization.",
    )
    parser.add_argument(
        "--skip-gen3c",
        action="store_true",
        help="Skip Gen3C visualization.",
    )
    args = parser.parse_args()

    iter_dir = Path(args.iter_dir)
    output_dir = Path(args.output_dir or (iter_dir / "05_topdown_variant_comparison"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not iter_dir.exists():
        raise FileNotFoundError(f"Missing iter dir: {iter_dir}")

    executed = []
    skipped = []

    vlm_tum = Path(args.vlm_tum) if args.vlm_tum else iter_dir / "02_lift" / "lifted_pred_traj.txt"
    if (not args.skip_vlm_tum) and vlm_tum.exists():
        run_visualization(
            "VLM_TUM",
            build_common_args(args, output_dir / "vlm_tum")
            + ["--generated_traj_txt", str(vlm_tum)],
        )
        executed.append("vlm_tum")
    else:
        skipped.append(f"vlm_tum ({vlm_tum})")

    vlm_npz = Path(args.vlm_npz) if args.vlm_npz else iter_dir / "02_lift" / "lifted_pred_traj.npz"
    if (not args.skip_vlm_npz) and vlm_npz.exists():
        run_visualization(
            "VLM_NPZ",
            build_common_args(args, output_dir / "vlm_npz")
            + ["--generated_traj_npz", str(vlm_npz)],
        )
        executed.append("vlm_npz")
    else:
        skipped.append(f"vlm_npz ({vlm_npz})")

    gen3c_dir = Path(args.gen3c_dir) if args.gen3c_dir else iter_dir / "02_lift" / "gen3c_output"
    has_gen3c = gen3c_dir.exists() and gen3c_dir.is_dir() and (gen3c_dir / "pose").exists()
    if not args.skip_gen3c and has_gen3c:
        gen3c_cmd = build_common_args(args, output_dir / "gen3c")
        gen3c_cmd.extend(["--gen3c-output-dir", str(gen3c_dir)])
        if args.gen3c_pose_npz:
            gen3c_cmd.extend(["--gen3c-pose-npz", str(args.gen3c_pose_npz)])
        if args.gen3c_intrinsics_npz:
            gen3c_cmd.extend(["--gen3c-intrinsics-npz", str(args.gen3c_intrinsics_npz)])
        run_visualization("GEN3C", gen3c_cmd)
        executed.append("gen3c")
    elif args.skip_gen3c:
        skipped.append("gen3c (requested)")
    else:
        skipped.append(f"gen3c (missing dir: {gen3c_dir})")

    print("Executed variants:", ", ".join(executed) if executed else "none")
    if skipped:
        print("Skipped variants:", ", ".join(skipped))

    if not executed:
        raise RuntimeError("No variants were rendered. Check that the iteration has supported trajectory artifacts.")


if __name__ == "__main__":
    main()
