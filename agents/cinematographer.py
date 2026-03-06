import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from transformers.video_utils import load_video

from .strided_trajectory_visualization import visualize_strided_trajectory
from .vlm_response_utils import extract_json_from_response, is_missing_response_content
from .io import open_image_as_base64, image_to_base64, save_base64_png


PROJECT_DIR = Path(str(os.environ.get("PROJECT_DIR")))

def render_system_prompt(
    *,
    num_frames: int,
    sparse_frame_keys: list[str],
    prompt_file_path: str = "cinematographer.txt",
) -> str:
    template = (Path(__file__).resolve().parent / prompt_file_path).read_text(
        encoding="utf-8"
    )
    replacements = {
        "[Insert Number]": str(num_frames),
        "[Insert Sparse Number]": str(len(sparse_frame_keys)),
        "[Insert Sparse Keys]": ", ".join(sparse_frame_keys),
        "[Insert Last Frame Index]": str(num_frames - 1),
    }
    prompt = template
    for old, new in replacements.items():
        prompt = prompt.replace(old, new)
    return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sparse VLM camera trajectory generation and save structured outputs."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--user-demand", type=str, required=True)
    parser.add_argument("--num-frames", type=int, default=70)
    parser.add_argument(
        "--sparse-frame-divisor",
        type=int,
        choices=(2, 4),
        default=4,
        help="Generate sparse control points at roughly N/divisor frames before interpolation.",
    )
    parser.add_argument("--strided-trajectory-out", type=Path, required=True)
    parser.add_argument("--dense-trajectory-out", type=Path, required=True)
    parser.add_argument("--trajectory-metadata-out", type=Path, required=True)
    parser.add_argument("--reasoning-out", type=Path, required=True)
    parser.add_argument("--user-demand-out", type=Path, required=True)
    parser.add_argument("--critic-feedback-json", type=Path, default=None)
    parser.add_argument("--prior-strided-trajectory-json", type=Path, default=None)
    parser.add_argument("--prior-trajectory-metadata-json", type=Path, default=None)
    return parser.parse_args()


def compute_sparse_frame_indices(num_frames: int, divisor: int) -> list[int]:
    if num_frames <= 0:
        raise ValueError("`num_frames` must be positive.")
    if num_frames == 1:
        return [0]

    sparse_count = max(2, int(math.ceil(num_frames / divisor)))
    raw_indices = np.linspace(0, num_frames - 1, num=sparse_count)
    indices: list[int] = []
    for raw_index in raw_indices:
        index = int(round(float(raw_index)))
        if not indices or indices[-1] != index:
            indices.append(index)

    if indices[0] != 0:
        indices.insert(0, 0)
    if indices[-1] != num_frames - 1:
        indices.append(num_frames - 1)
    return indices


def frame_keys_from_indices(indices: list[int]) -> list[str]:
    return [f"t_{index}" for index in indices]


def extract_trajectory_positions(
    data: dict[str, Any],
    *,
    top_level_key: str,
) -> dict[str, dict[str, list[float]]]:
    trajectory = data.get(top_level_key)
    if not isinstance(trajectory, dict):
        raise RuntimeError(f"Missing `{top_level_key}` object in model response.")

    normalized: dict[str, dict[str, list[float]]] = {}
    for key, value in trajectory.items():
        if not isinstance(key, str) or not key.startswith("t_"):
            continue
        if not isinstance(value, dict) or "position" not in value:
            continue
        position = value["position"]
        if not isinstance(position, list) or len(position) != 2:
            continue
        normalized[key] = {
            "position": [float(position[0]), float(position[1])],
        }
    return normalized


def get_missing_required_keys(
    positions: dict[str, dict[str, list[float]]], required_keys: list[str]
) -> list[str]:
    return [key for key in required_keys if key not in positions]


def interpolate_dense_trajectory(
    strided_positions: dict[str, dict[str, list[float]]],
    *,
    num_frames: int,
) -> dict[str, dict[str, list[float]]]:
    anchor_indices = sorted(int(key.split("_", 1)[1]) for key in strided_positions)
    anchor_x = [float(strided_positions[f"t_{index}"]["position"][0]) for index in anchor_indices]
    anchor_y = [float(strided_positions[f"t_{index}"]["position"][1]) for index in anchor_indices]
    frame_indices = np.arange(num_frames, dtype=np.float64)
    dense_x = np.interp(frame_indices, anchor_indices, anchor_x)
    dense_y = np.interp(frame_indices, anchor_indices, anchor_y)

    dense_positions: dict[str, dict[str, list[float]]] = {}
    for frame_index in range(num_frames):
        dense_positions[f"t_{frame_index}"] = {
            "position": [float(dense_x[frame_index]), float(dense_y[frame_index])]
        }

    for anchor_index in anchor_indices:
        key = f"t_{anchor_index}"
        dense_positions[key] = {
            "position": list(strided_positions[key]["position"])
        }
    return dense_positions


def build_trajectory_metadata(
    *,
    num_frames: int,
    sparse_frame_divisor: int,
    sparse_frame_indices: list[int],
    img_w: int,
    img_h: int,
) -> dict[str, Any]:
    return {
        "coordinate_space": "pixel_xy",
        "image_size": [img_h, img_w],
        "bounds_xy": {
            "x": [0.0, float(img_w - 1)],
            "y": [0.0, float(img_h - 1)],
        },
        "num_frames": num_frames,
        "sparse_frame_divisor": sparse_frame_divisor,
        "sparse_frame_count": len(sparse_frame_indices),
        "sparse_frame_indices": sparse_frame_indices,
        "sparse_frame_keys": frame_keys_from_indices(sparse_frame_indices),
        "interpolation_method": "linear_uv",
        "dense_output_key": "camera_trajectory_pixels",
        "strided_output_key": "strided_camera_trajectory_pixels",
    }

def main() -> None:
    args = parse_args()
    user_demand = args.user_demand.strip()
    model_name = args.model.strip()
    if not user_demand:
        raise ValueError("`--user-demand` cannot be empty.")
    if not model_name:
        raise ValueError("`--model` cannot be empty.")
    if args.num_frames <= 0:
        raise ValueError("`--num-frames` must be > 0.")
    if args.critic_feedback_json is not None and not args.critic_feedback_json.exists():
        raise FileNotFoundError(f"Missing critic feedback json: {args.critic_feedback_json}")
    if (
        args.prior_strided_trajectory_json is not None
        and not args.prior_strided_trajectory_json.exists()
    ):
        raise FileNotFoundError(
            f"Missing prior strided trajectory json: {args.prior_strided_trajectory_json}"
        )
    if (
        args.prior_trajectory_metadata_json is not None
        and not args.prior_trajectory_metadata_json.exists()
    ):
        raise FileNotFoundError(
            "Missing prior trajectory metadata json: "
            f"{args.prior_trajectory_metadata_json}"
        )

    refinement_requested = any(
        path is not None
        for path in (
            args.critic_feedback_json,
            args.prior_strided_trajectory_json,
            args.prior_trajectory_metadata_json,
        )
    )
    if refinement_requested and not all(
        path is not None
        for path in (
            args.critic_feedback_json,
            args.prior_strided_trajectory_json,
            args.prior_trajectory_metadata_json,
        )
    ):
        raise ValueError(
            "Refinement requires --critic-feedback-json, --prior-strided-trajectory-json, "
            "and --prior-trajectory-metadata-json together."
        )

    sparse_frame_indices = compute_sparse_frame_indices(
        num_frames=args.num_frames,
        divisor=args.sparse_frame_divisor,
    )
    sparse_frame_keys = frame_keys_from_indices(sparse_frame_indices)

    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=3600)
    system_prompt = render_system_prompt(
        prompt_file_path="cinematographer.txt",
        num_frames=args.num_frames,
        sparse_frame_keys=sparse_frame_keys,
    )

    scene_root = PROJECT_DIR / "monst3r/demo_tmp/davis/tennis/normalized_nofilter"
    topdown_scene_dir = scene_root / "top-down-semantic-ortho-all-scene"
    capture_cameras_json = topdown_scene_dir / "capture_cameras_pixels.json"
    moving_bbox_json = topdown_scene_dir / "moving_bbox_pixels.json"
    semantic_metadata_json = topdown_scene_dir / "semantic_topdown_metadata.json"

    capture_cameras_data = json.loads(capture_cameras_json.read_text(encoding="utf-8"))
    moving_bbox_data = json.loads(moving_bbox_json.read_text(encoding="utf-8"))
    semantic_metadata = json.loads(semantic_metadata_json.read_text(encoding="utf-8"))

    img_h, img_w = map(int, semantic_metadata["image_size"])
    metadata_json = build_trajectory_metadata(
        num_frames=args.num_frames,
        sparse_frame_divisor=args.sparse_frame_divisor,
        sparse_frame_indices=sparse_frame_indices,
        img_w=img_w,
        img_h=img_h,
    )
    metadata_text = json.dumps(metadata_json, indent=2, ensure_ascii=True)

    video_object, _ = load_video(
        str(
            scene_root
            / "top-down-semantic-ortho-all-camera_centroid/vis-videos/motion_overlay.mp4"
        )  # ty:ignore[invalid-argument-type]
    )

    capture_camera_first_frame = open_image_as_base64(scene_root / "renders_all/0000.png")
    video_frames = video_object[[0, -1]]
    video_frames = [image_to_base64(frame) for frame in video_frames]

    artifact_manifest_lines = [
        "Artifact Manifest:",
        f"- User Demand: {user_demand}",
        "- Rendered Images: capture camera first frame, motion overlay start frame, motion overlay end frame",
        "- Foreground Motion Path (Pixel Coordinates): moving_bbox_pixels.json",
        "- Source Camera Positions: capture_cameras_pixels.json",
        "- Extra Metadata: semantic_topdown_metadata.json",
        f"- Sparse Control Frames Requested: {', '.join(sparse_frame_keys)}",
        f"- Dense Full Sequence Length: {args.num_frames}",
    ]

    refinement_block_text = ""
    critic_feedback_text = ""
    prior_strided_trajectory_text = ""
    prior_trajectory_metadata_text = ""
    if refinement_requested:
        critic_data = json.loads(args.critic_feedback_json.read_text(encoding="utf-8"))
        prior_strided_data = json.loads(
            args.prior_strided_trajectory_json.read_text(encoding="utf-8")
        )
        prior_metadata_data = json.loads(
            args.prior_trajectory_metadata_json.read_text(encoding="utf-8")
        )
        critic_feedback_text = json.dumps(critic_data, indent=2, ensure_ascii=True)
        prior_strided_trajectory_text = json.dumps(
            prior_strided_data, indent=2, ensure_ascii=True
        )
        prior_trajectory_metadata_text = json.dumps(
            prior_metadata_data, indent=2, ensure_ascii=True
        )

        analysis = str(critic_data.get("analysis", ""))
        actionable_feedback = str(critic_data.get("actionable_feedback", ""))
        failure_modes = critic_data.get("failure_modes", [])
        frame_level_notes = critic_data.get("frame_level_notes", {})
        overall_quality_score = critic_data.get("overall_quality_score", "")
        refinement_block_text = "\n".join(
            [
                "Refinement Context From Previous Iteration:",
                f"- Previous overall_quality_score: {overall_quality_score}",
                f"- Previous analysis: {analysis}",
                f"- Previous actionable_feedback: {actionable_feedback}",
                f"- Previous failure_modes: {json.dumps(failure_modes, ensure_ascii=True)}",
                f"- Previous frame_level_notes: {json.dumps(frame_level_notes, ensure_ascii=True)}",
                (
                    "- Previous sparse control trajectory is the editable representation. "
                    "The prior dense trajectory was derived from interpolation and should be "
                    "treated as a consequence of the sparse controls."
                ),
                "Improve the new sparse control trajectory using this feedback while still prioritizing the current user demand.",
            ]
        )
        artifact_manifest_lines.extend(
            [
                "- Critic Feedback (Previous Iteration): critic_result.json",
                "- Prior Sparse Camera Trajectory (Previous Iteration): strided_camera_trajectory_pixels.json",
                "- Prior Trajectory Metadata (Previous Iteration): trajectory_metadata.json",
            ]
        )
    artifact_manifest_text = "\n".join(artifact_manifest_lines)

    capture_cameras_text = json.dumps(capture_cameras_data, indent=2, ensure_ascii=True)
    moving_bbox_text = json.dumps(moving_bbox_data, indent=2, ensure_ascii=True)

    sparse_requirement_text = "\n".join(
        [
            "Sparse Control Point Requirements:",
            f"- Generate exactly {len(sparse_frame_keys)} sparse control points.",
            f"- Use only these frame keys: {', '.join(sparse_frame_keys)}",
            (
                f"- The full dense sequence has {args.num_frames} frames, but you must only "
                "author the sparse control points listed above."
            ),
            "- Dense per-frame cameras will be computed later by linear interpolation in UV space.",
        ]
    )

    user_text_blocks = [
        artifact_manifest_text,
        f"User Demand: {user_demand}",
        sparse_requirement_text,
        "Rendered Image 1 (capture camera first frame):",
        "Rendered Image 2 (motion overlay start frame):",
        "Rendered Image 3 (motion overlay end frame):",
        "Source Camera Positions (Pixel Coordinates) - capture_cameras_pixels.json:",
        capture_cameras_text,
        "Foreground Motion Path (Pixel Coordinates) - moving_bbox_pixels.json:",
        moving_bbox_text,
        "Current Trajectory Metadata Contract:",
        metadata_text,
    ]
    if refinement_block_text:
        user_text_blocks.extend(
            [
                refinement_block_text,
                "Previous Iteration Critic Feedback JSON:",
                critic_feedback_text,
                "Previous Iteration Sparse Trajectory JSON:",
                prior_strided_trajectory_text,
                "Previous Iteration Trajectory Metadata JSON:",
                prior_trajectory_metadata_text,
            ]
        )
    user_text_blocks.append(
        "Return JSON only, using the exact schema specified in the system prompt."
    )

    vlm_dir = args.strided_trajectory_out.parent
    vlm_inputs_dir = vlm_dir / "inputs"
    visual_inputs_dir = vlm_inputs_dir / "visual"
    language_inputs_dir = vlm_inputs_dir / "language"
    visual_inputs_dir.mkdir(parents=True, exist_ok=True)
    language_inputs_dir.mkdir(parents=True, exist_ok=True)

    visual_input_records: list[dict[str, int | str]] = [
        {
            "order": 0,
            "file_name": "input_000__capture_camera__frame_0000.png",
            "role": "capture_camera_first_frame",
            "source": "renders_all/0000.png",
            "frame_index": 0,
            "image_base64": capture_camera_first_frame,
        },
        {
            "order": 1,
            "file_name": "input_001__motion_overlay__frame_0000.png",
            "role": "motion_overlay_frame",
            "source": "motion_overlay.mp4",
            "frame_index": 0,
            "image_base64": video_frames[0],
        },
        {
            "order": 2,
            "file_name": "input_002__motion_overlay__frame_last.png",
            "role": "motion_overlay_frame",
            "source": "motion_overlay.mp4",
            "frame_index": -1,
            "image_base64": video_frames[1],
        },
    ]

    for record in visual_input_records:
        save_base64_png(
            str(record["image_base64"]),
            visual_inputs_dir / str(record["file_name"]),
        )

    visual_manifest = [
        {
            "order": int(record["order"]),
            "file_name": str(record["file_name"]),
            "role": str(record["role"]),
            "source": str(record["source"]),
            "frame_index": int(record["frame_index"]),
        }
        for record in visual_input_records
    ]
    (visual_inputs_dir / "visual_inputs_manifest.json").write_text(
        json.dumps(visual_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (language_inputs_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    (language_inputs_dir / "artifact_manifest.txt").write_text(
        f"{artifact_manifest_text}\n", encoding="utf-8"
    )
    (language_inputs_dir / "user_text_blocks.json").write_text(
        json.dumps(user_text_blocks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (language_inputs_dir / "capture_cameras_pixels.json").write_text(
        capture_cameras_text, encoding="utf-8"
    )
    (language_inputs_dir / "moving_bbox_pixels.json").write_text(
        moving_bbox_text, encoding="utf-8"
    )
    (language_inputs_dir / "trajectory_metadata.json").write_text(
        metadata_text, encoding="utf-8"
    )
    (language_inputs_dir / "request_config.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "num_frames": args.num_frames,
                "sparse_frame_divisor": args.sparse_frame_divisor,
                "sparse_frame_indices": sparse_frame_indices,
                "sparse_frame_keys": sparse_frame_keys,
                "response_format": {"type": "json_object"},
                "max_tokens": 4096 * 4,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if refinement_block_text:
        (language_inputs_dir / "critic_feedback.json").write_text(
            critic_feedback_text, encoding="utf-8"
        )
        (language_inputs_dir / "prior_strided_trajectory.json").write_text(
            prior_strided_trajectory_text, encoding="utf-8"
        )
        (language_inputs_dir / "prior_trajectory_metadata.json").write_text(
            prior_trajectory_metadata_text, encoding="utf-8"
        )
    print(f"Saved VLM inputs to: {vlm_inputs_dir}")

    user_content = [
        {
            "type": "text",
            "text": artifact_manifest_text,
        },
        {"type": "text", "text": f"User Demand: {user_demand}"},
        {"type": "text", "text": sparse_requirement_text},
        {"type": "text", "text": "Rendered Image 1 (capture camera first frame):"},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{capture_camera_first_frame}"
            },
        },
        {"type": "text", "text": "Rendered Image 2 (motion overlay start frame):"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{video_frames[0]}"},
        },
        {"type": "text", "text": "Rendered Image 3 (motion overlay end frame):"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{video_frames[1]}"},
        },
        {
            "type": "text",
            "text": "Source Camera Positions (Pixel Coordinates) - capture_cameras_pixels.json:",
        },
        {
            "type": "text",
            "text": capture_cameras_text,
        },
        {
            "type": "text",
            "text": "Foreground Motion Path (Pixel Coordinates) - moving_bbox_pixels.json:",
        },
        {
            "type": "text",
            "text": moving_bbox_text,
        },
        {
            "type": "text",
            "text": "Current Trajectory Metadata Contract:",
        },
        {
            "type": "text",
            "text": metadata_text,
        },
    ]
    if refinement_block_text:
        user_content.extend(
            [
                {"type": "text", "text": refinement_block_text},
                {"type": "text", "text": "Previous Iteration Critic Feedback JSON:"},
                {"type": "text", "text": critic_feedback_text},
                {"type": "text", "text": "Previous Iteration Sparse Trajectory JSON:"},
                {"type": "text", "text": prior_strided_trajectory_text},
                {"type": "text", "text": "Previous Iteration Trajectory Metadata JSON:"},
                {"type": "text", "text": prior_trajectory_metadata_text},
            ]
        )
    user_content.append(
        {
            "type": "text",
            "text": "Return JSON only, using the exact schema specified in the system prompt.",
        }
    )

    messages = cast(
        list[ChatCompletionMessageParam],
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
    )

    start = time.time()
    response = client.chat.completions.create(
        model=model_name,
        max_tokens=8192,
        response_format={"type": "json_object"},
        messages=messages,
    )
    print(f"Response costs: {time.time() - start:.2f}s")

    response_dict: dict[str, Any] = response.choices[0].message.to_dict()
    reasoning = response_dict.get("reasoning", "No reasoning provided.")
    content = response_dict.get("content", {})
    should_rerun = False
    rerun_reason = ""
    strided_trajectory_json: dict[str, Any] | None = None
    normalized_strided_positions: dict[str, dict[str, list[float]]] = {}
    if is_missing_response_content(content):
        should_rerun = True
        rerun_reason = (
            "Detected empty sparse trajectory output; rerunning generation with a stronger "
            "sparse coverage reminder."
        )
    else:
        strided_trajectory_json = extract_json_from_response(
            client=client,
            model_name=model_name,
            content=content,
            output_dir=args.strided_trajectory_out.parent,
            base_name="strided_trajectory",
            finish_reason=response.choices[0].finish_reason,
            required_top_level_key="strided_camera_trajectory_pixels",
            label="strided trajectory",
        )
        normalized_strided_positions = extract_trajectory_positions(
            strided_trajectory_json,
            top_level_key="strided_camera_trajectory_pixels",
        )
        missing_frame_keys = get_missing_required_keys(
            normalized_strided_positions, sparse_frame_keys
        )
        if missing_frame_keys:
            should_rerun = True
            rerun_reason = (
                "Detected incomplete sparse trajectory output; rerunning generation with a stronger "
                "sparse coverage reminder."
            )

    if should_rerun:
        print(rerun_reason)
        rerun_user_content = list(user_content)
        rerun_user_content.append(
            {
                "type": "text",
                "text": (
                    "Your previous response was empty or omitted required sparse frame keys. "
                    f"Return exactly {len(sparse_frame_keys)} sparse camera positions using only "
                    f"these keys: {', '.join(sparse_frame_keys)}. Do not include dense per-frame "
                    "trajectory entries."
                ),
            }
        )
        rerun_messages = cast(
            list[ChatCompletionMessageParam],
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": rerun_user_content,
                },
            ],
        )

        rerun_start = time.time()
        rerun_response = client.chat.completions.create(
            model=model_name,
            max_tokens=8192,
            response_format={"type": "json_object"},
            messages=rerun_messages,
        )
        print(f"Rerun response costs: {time.time() - rerun_start:.2f}s")

        rerun_response_dict = cast(dict[str, Any], rerun_response.choices[0].message.to_dict())
        reasoning = rerun_response_dict.get("reasoning", reasoning)
        rerun_content = rerun_response_dict.get("content", {})
        if is_missing_response_content(rerun_content):
            raise RuntimeError("Model response was empty (`None`/`null`) after rerun.")

        strided_trajectory_json = extract_json_from_response(
            client=client,
            model_name=model_name,
            content=rerun_content,
            output_dir=args.strided_trajectory_out.parent,
            base_name="strided_trajectory_rerun",
            finish_reason=rerun_response.choices[0].finish_reason,
            required_top_level_key="strided_camera_trajectory_pixels",
            label="strided trajectory rerun",
        )
        normalized_strided_positions = extract_trajectory_positions(
            strided_trajectory_json,
            top_level_key="strided_camera_trajectory_pixels",
        )
        missing_frame_keys = get_missing_required_keys(
            normalized_strided_positions, sparse_frame_keys
        )
        if missing_frame_keys:
            raise RuntimeError(
                "Model response is missing required sparse frame keys after rerun: "
                + ", ".join(missing_frame_keys[:10])
                + (" ..." if len(missing_frame_keys) > 10 else "")
            )

    strided_output = {
        "strided_camera_trajectory_pixels": {
            key: {"position": list(value["position"])}
            for key, value in sorted(
                normalized_strided_positions.items(),
                key=lambda item: int(item[0].split("_", 1)[1]),
            )
            if key in sparse_frame_keys
        }
    }
    dense_output = {
        "camera_trajectory_pixels": interpolate_dense_trajectory(
            strided_output["strided_camera_trajectory_pixels"],
            num_frames=args.num_frames,
        )
    }

    strided_viz_path = args.strided_trajectory_out.parent / "trajectory_strided_topdown.mp4"
    visualize_strided_trajectory(
        output_video_path=strided_viz_path,
        strided_positions=strided_output["strided_camera_trajectory_pixels"],
        semantic_metadata=semantic_metadata,
        source_traj_path=scene_root / "pred_traj.txt",
        intrinsics_path=scene_root / "pred_intrinsics.txt",
        scene_ply_dir=scene_root / "all_points",
        fps=4,
    )

    args.strided_trajectory_out.parent.mkdir(parents=True, exist_ok=True)
    args.dense_trajectory_out.parent.mkdir(parents=True, exist_ok=True)
    args.trajectory_metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.reasoning_out.parent.mkdir(parents=True, exist_ok=True)
    args.user_demand_out.parent.mkdir(parents=True, exist_ok=True)

    args.strided_trajectory_out.write_text(
        json.dumps(strided_output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved strided trajectory JSON to: {args.strided_trajectory_out}")

    args.dense_trajectory_out.write_text(
        json.dumps(dense_output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved dense trajectory JSON to: {args.dense_trajectory_out}")

    args.trajectory_metadata_out.write_text(
        json.dumps(metadata_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved trajectory metadata JSON to: {args.trajectory_metadata_out}")
    print(f"Saved strided trajectory topdown video to: {strided_viz_path}")

    args.reasoning_out.write_text(str(reasoning), encoding="utf-8")
    print(f"Saved reasoning text to: {args.reasoning_out}")

    args.user_demand_out.write_text(f"{user_demand}\n", encoding="utf-8")
    print(f"Saved user demand text to: {args.user_demand_out}")


if __name__ == "__main__":
    main()
