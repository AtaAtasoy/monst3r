import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .vlm_response_utils import is_missing_response_content
from .io import save_base64_png, read_text, load_video_keyframes


def render_system_prompt(*, num_frames: int, sparse_frame_keys: list[str]) -> str:
    template = (Path(__file__).resolve().parent / "critic.txt").read_text(encoding="utf-8")
    replacements = {
        "[Insert Number]": str(num_frames - 1),
        "[Insert Sparse Keys]": ", ".join(sparse_frame_keys),
    }
    prompt = template
    for old, new in replacements.items():
        prompt = prompt.replace(old, new)
    return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a critic VLM pass on generated sparse camera trajectory outputs."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--user-demand", type=str, required=True)
    parser.add_argument("--strided-trajectory-json", type=Path, required=True)
    parser.add_argument("--trajectory-metadata-json", type=Path, required=True)
    parser.add_argument("--lifted-traj-tum", type=Path, required=True)
    parser.add_argument("--render-video", type=Path, required=True)
    parser.add_argument("--topdown-video", type=Path, default=None)
    parser.add_argument("--critic-out", type=Path, required=True)
    parser.add_argument("--reasoning-out", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=None)
    return parser.parse_args()

def validate_critic_output(
    data: dict[str, Any],
    *,
    allowed_sparse_keys: set[str],
) -> None:
    required = [
        "intent_matched",
        "analysis",
        "actionable_feedback",
        "failure_modes",
        "frame_level_notes",
        "overall_quality_score",
        "refinement_suggestions",
    ]
    for key in required:
        if key not in data:
            raise RuntimeError(f"Missing required key in critic output: {key}")

    if not isinstance(data["intent_matched"], bool):
        raise RuntimeError("`intent_matched` must be bool")
    if not isinstance(data["analysis"], str):
        raise RuntimeError("`analysis` must be string")
    if not isinstance(data["actionable_feedback"], str):
        raise RuntimeError("`actionable_feedback` must be string")
    if not isinstance(data["failure_modes"], list):
        raise RuntimeError("`failure_modes` must be array")
    if not all(isinstance(item, str) for item in data["failure_modes"]):
        raise RuntimeError("`failure_modes` entries must be strings")
    if not isinstance(data["refinement_suggestions"], list):
        raise RuntimeError("`refinement_suggestions` must be array")
    if not all(isinstance(item, str) for item in data["refinement_suggestions"]):
        raise RuntimeError("`refinement_suggestions` entries must be strings")
    if not isinstance(data["frame_level_notes"], dict):
        raise RuntimeError("`frame_level_notes` must be object")
    for key, value in data["frame_level_notes"].items():
        if not isinstance(key, str) or not key.startswith("t_"):
            raise RuntimeError("`frame_level_notes` keys must be t_i format")
        if key not in allowed_sparse_keys:
            raise RuntimeError("`frame_level_notes` keys must come from sparse control keys")
        if not isinstance(value, str):
            raise RuntimeError("`frame_level_notes` values must be strings")

    actionable_feedback = data["actionable_feedback"]
    if data["intent_matched"] is True:
        if actionable_feedback != "None":
            raise RuntimeError(
                "`actionable_feedback` must be 'None' when `intent_matched` is true"
            )
    else:
        sparse_key_mentions = set(re.findall(r"t_\d+", actionable_feedback))
        if not sparse_key_mentions:
            raise RuntimeError(
                "`actionable_feedback` must reference sparse control keys when intent is false"
            )
        if sparse_key_mentions.isdisjoint(allowed_sparse_keys):
            raise RuntimeError(
                "`actionable_feedback` must reference at least one provided sparse control key"
            )

    score = data["overall_quality_score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise RuntimeError("`overall_quality_score` must be numeric in [0, 10]")
    score_f = float(score)
    if score_f < 0.0 or score_f > 10.0:
        raise RuntimeError("`overall_quality_score` must be within [0, 10]")


def main() -> None:
    args = parse_args()
    user_demand = args.user_demand.strip()
    model_name = args.model.strip()
    if not user_demand:
        raise ValueError("`--user-demand` cannot be empty.")
    if not model_name:
        raise ValueError("`--model` cannot be empty.")

    if not args.strided_trajectory_json.exists():
        raise FileNotFoundError(
            f"Missing strided trajectory json: {args.strided_trajectory_json}"
        )
    if not args.trajectory_metadata_json.exists():
        raise FileNotFoundError(
            f"Missing trajectory metadata json: {args.trajectory_metadata_json}"
        )
    if not args.lifted_traj_tum.exists():
        raise FileNotFoundError(f"Missing lifted traj txt: {args.lifted_traj_tum}")
    if not args.render_video.exists():
        raise FileNotFoundError(f"Missing render video: {args.render_video}")

    topdown_available = args.topdown_video is not None and args.topdown_video.exists()

    metadata = json.loads(read_text(args.trajectory_metadata_json))
    sparse_frame_keys = metadata.get("sparse_frame_keys")
    if not isinstance(sparse_frame_keys, list) or not all(
        isinstance(key, str) and key.startswith("t_") for key in sparse_frame_keys
    ):
        raise RuntimeError(
            "`trajectory_metadata_json` must contain `sparse_frame_keys` as a list of t_i keys"
        )
    allowed_sparse_keys = set(cast(list[str], sparse_frame_keys))

    if args.num_frames is None:
        num_frames = int(metadata.get("num_frames", 0))
    else:
        num_frames = int(args.num_frames)
    if num_frames <= 0:
        raise RuntimeError("Unable to determine positive `num_frames` for critic prompt")

    system_prompt = render_system_prompt(
        num_frames=num_frames,
        sparse_frame_keys=cast(list[str], sparse_frame_keys),
    )
    strided_trajectory_text = read_text(args.strided_trajectory_json)
    metadata_text = read_text(args.trajectory_metadata_json)
    lifted_tum_text = read_text(args.lifted_traj_tum)
    render_first, render_last = load_video_keyframes(args.render_video)

    topdown_first = None
    topdown_last = None
    if topdown_available and args.topdown_video is not None:
        topdown_first, topdown_last = load_video_keyframes(args.topdown_video)

    strided_viz_path = args.strided_trajectory_json.parent / "trajectory_strided_topdown.mp4"
    strided_viz_available = strided_viz_path.exists()
    strided_viz_first = None
    strided_viz_last = None
    if strided_viz_available:
        strided_viz_first, strided_viz_last = load_video_keyframes(strided_viz_path)

    artifact_manifest_lines = [
        "Artifact Manifest:",
        f"- User Demand: {user_demand}",
        f"- Sparse 2D Trajectory JSON: {args.strided_trajectory_json.name}",
        f"- Trajectory Metadata JSON: {args.trajectory_metadata_json.name}",
        f"- Lifted 3D Trajectory TUM: {args.lifted_traj_tum.name}",
        f"- Rendered Video: {args.render_video.name} (keyframes: first,last)",
        (
            f"- Sparse Trajectory Topdown Video: {strided_viz_path.name}"
            if strided_viz_available
            else "- Sparse Trajectory Topdown Video: N/A"
        ),
        (
            f"- Topdown Generated Trajectory Video: {args.topdown_video.name}"
            if topdown_available and args.topdown_video is not None
            else "- Topdown Generated Trajectory Video: N/A"
        ),
    ]
    artifact_manifest_text = "\n".join(artifact_manifest_lines)

    visual_inputs_dir = args.inputs_dir / "visual"
    language_inputs_dir = args.inputs_dir / "language"
    visual_inputs_dir.mkdir(parents=True, exist_ok=True)
    language_inputs_dir.mkdir(parents=True, exist_ok=True)

    visual_input_records: list[dict[str, int | str | bool]] = [
        {
            "order": 0,
            "file_name": "input_000__rendered_video__render_mp4__keyframe_0000.png",
            "role": "rendered_video_keyframe",
            "source": str(args.render_video.name),
            "frame_index": 0,
            "available": True,
            "image_base64": render_first,
        },
        {
            "order": 1,
            "file_name": "input_001__rendered_video__render_mp4__keyframe_last.png",
            "role": "rendered_video_keyframe",
            "source": str(args.render_video.name),
            "frame_index": -1,
            "available": True,
            "image_base64": render_last,
        },
        {
            "order": 2,
            "file_name": "input_002__strided_topdown_viz__trajectory_strided_topdown_mp4__keyframe_0000.png",
            "role": "strided_topdown_video_keyframe",
            "source": str(strided_viz_path.name),
            "frame_index": 0,
            "available": strided_viz_available,
            "image_base64": strided_viz_first if strided_viz_first is not None else "",
            "note": (
                ""
                if strided_viz_available
                else "Sparse topdown video missing; critic will use sparse JSON + metadata."
            ),
        },
        {
            "order": 3,
            "file_name": "input_003__strided_topdown_viz__trajectory_strided_topdown_mp4__keyframe_last.png",
            "role": "strided_topdown_video_keyframe",
            "source": str(strided_viz_path.name),
            "frame_index": -1,
            "available": strided_viz_available,
            "image_base64": strided_viz_last if strided_viz_last is not None else "",
            "note": (
                ""
                if strided_viz_available
                else "Sparse topdown video missing; critic will use sparse JSON + metadata."
            ),
        },
    ]

    if topdown_available and topdown_first is not None and topdown_last is not None:
        visual_input_records.extend(
            [
                {
                    "order": 4,
                    "file_name": "input_004__topdown_viz__generated_topdown_trajectory_mp4__keyframe_0000.png",
                    "role": "topdown_video_keyframe",
                    "source": str(args.topdown_video.name if args.topdown_video else "N/A"),
                    "frame_index": 0,
                    "available": True,
                    "image_base64": topdown_first,
                },
                {
                    "order": 5,
                    "file_name": "input_005__topdown_viz__generated_topdown_trajectory_mp4__keyframe_last.png",
                    "role": "topdown_video_keyframe",
                    "source": str(args.topdown_video.name if args.topdown_video else "N/A"),
                    "frame_index": -1,
                    "available": True,
                    "image_base64": topdown_last,
                },
            ]
        )
    else:
        visual_input_records.extend(
            [
                {
                    "order": 4,
                    "file_name": "input_004__topdown_viz__generated_topdown_trajectory_mp4__keyframe_0000.png",
                    "role": "topdown_video_keyframe",
                    "source": str(args.topdown_video.name if args.topdown_video else "N/A"),
                    "frame_index": 0,
                    "available": False,
                    "note": "Topdown video missing; skipped visual attachment.",
                },
                {
                    "order": 5,
                    "file_name": "input_005__topdown_viz__generated_topdown_trajectory_mp4__keyframe_last.png",
                    "role": "topdown_video_keyframe",
                    "source": str(args.topdown_video.name if args.topdown_video else "N/A"),
                    "frame_index": -1,
                    "available": False,
                    "note": "Topdown video missing; skipped visual attachment.",
                },
            ]
        )

    for record in visual_input_records:
        if bool(record.get("available", False)):
            save_base64_png(
                str(record["image_base64"]),
                visual_inputs_dir / str(record["file_name"]),
            )

    visual_manifest: list[dict[str, int | str | bool]] = []
    for record in visual_input_records:
        visual_manifest.append(
            {
                "order": int(record["order"]),
                "file_name": str(record["file_name"]),
                "role": str(record["role"]),
                "source": str(record["source"]),
                "frame_index": int(record["frame_index"]),
                "available": bool(record.get("available", False)),
                "note": str(record.get("note", "")),
            }
        )
    (visual_inputs_dir / "visual_inputs_manifest.json").write_text(
        json.dumps(visual_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    user_text_blocks = [
        artifact_manifest_text,
        f"User Demand: {user_demand}",
        "Sparse 2D Camera Control Trajectory JSON:",
        strided_trajectory_text,
        "Trajectory Metadata JSON:",
        metadata_text,
        "Lifted 3D Trajectory TUM:",
        lifted_tum_text,
        "Return JSON only, using the exact schema specified in the system prompt.",
    ]
    (language_inputs_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    (language_inputs_dir / "artifact_manifest.txt").write_text(
        f"{artifact_manifest_text}\n", encoding="utf-8"
    )
    (language_inputs_dir / "strided_camera_trajectory_pixels.json").write_text(
        strided_trajectory_text, encoding="utf-8"
    )
    (language_inputs_dir / "trajectory_metadata.json").write_text(
        metadata_text, encoding="utf-8"
    )
    (language_inputs_dir / "lifted_pred_traj.txt").write_text(
        lifted_tum_text, encoding="utf-8"
    )
    (language_inputs_dir / "user_text_blocks.json").write_text(
        json.dumps(user_text_blocks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (language_inputs_dir / "request_config.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "response_format": {"type": "json_object"},
                "max_tokens": 4096 * 2,
                "topdown_video_available": topdown_available,
                "strided_topdown_video_available": strided_viz_available,
                "sparse_frame_keys": sparse_frame_keys,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved critic inputs to: {args.inputs_dir}")

    user_content: list[dict[str, object]] = [
        {"type": "text", "text": artifact_manifest_text},
        {"type": "text", "text": f"User Demand: {user_demand}"},
        {"type": "text", "text": "Rendered Video Keyframe 1 (first):"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{render_first}"}},
        {"type": "text", "text": "Rendered Video Keyframe 2 (last):"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{render_last}"}},
    ]

    if (
        strided_viz_available
        and strided_viz_first is not None
        and strided_viz_last is not None
    ):
        user_content.extend(
            [
                {"type": "text", "text": "Sparse Trajectory Topdown Video Keyframe 1 (first):"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{strided_viz_first}"},
                },
                {"type": "text", "text": "Sparse Trajectory Topdown Video Keyframe 2 (last):"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{strided_viz_last}"},
                },
            ]
        )
    else:
        user_content.append(
            {
                "type": "text",
                "text": "Sparse trajectory topdown video is unavailable. Evaluate the sparse path from the JSON and metadata directly.",
            }
        )

    if topdown_available and topdown_first is not None and topdown_last is not None:
        user_content.extend(
            [
                {"type": "text", "text": "Topdown Trajectory Keyframe 1 (first):"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{topdown_first}"},
                },
                {"type": "text", "text": "Topdown Trajectory Keyframe 2 (last):"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{topdown_last}"},
                },
            ]
        )
    else:
        user_content.append(
            {
                "type": "text",
                "text": "Topdown trajectory video is unavailable for this run. Evaluate using rendered video and sparse trajectory artifacts.",
            }
        )

    user_content.extend(
        [
            {"type": "text", "text": "Sparse 2D Camera Control Trajectory JSON:"},
            {"type": "text", "text": strided_trajectory_text},
            {"type": "text", "text": "Trajectory Metadata JSON:"},
            {"type": "text", "text": metadata_text},
            {"type": "text", "text": "Lifted 3D Trajectory TUM:"},
            {"type": "text", "text": lifted_tum_text},
            {
                "type": "text",
                "text": "Return JSON only, using the exact schema specified in the system prompt.",
            },
        ]
    )

    messages = cast(
        list[ChatCompletionMessageParam],
        [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ],
    )

    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=3600)

    start = time.time()
    response = client.chat.completions.create(
        model=model_name,
        max_tokens=4096 * 2,
        response_format={"type": "json_object"},
        messages=messages,
    )
    print(f"Response costs: {time.time() - start:.2f}s")

    response_dict: dict[str, Any] = response.choices[0].message.to_dict()
    reasoning = response_dict.get("reasoning", "No reasoning provided.")
    content = response_dict.get("content", {})

    raw_debug_path = args.critic_out.parent / "critic_raw_response.txt"
    rerun_raw_debug_path = args.critic_out.parent / "critic_rerun_raw_response.txt"
    raw_debug_path.parent.mkdir(parents=True, exist_ok=True)
    raw_debug_path.write_text(str(content), encoding="utf-8")

    if is_missing_response_content(content):
        print("Detected empty critic output; rerunning critic with a stricter JSON reminder.")
        rerun_user_content = list(user_content)
        rerun_user_content.append(
            {
                "type": "text",
                "text": (
                    "Your previous response was empty. Return exactly one JSON object using "
                    "the required schema from the system prompt. Do not return None, null, or "
                    "an empty response."
                ),
            }
        )
        rerun_messages = cast(
            list[ChatCompletionMessageParam],
            [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": rerun_user_content},
            ],
        )

        rerun_start = time.time()
        rerun_response = client.chat.completions.create(
            model=model_name,
            max_tokens=4096 * 2,
            response_format={"type": "json_object"},
            messages=rerun_messages,
        )
        print(f"Rerun response costs: {time.time() - rerun_start:.2f}s")

        rerun_response_dict = cast(dict[str, Any], rerun_response.choices[0].message.to_dict())
        reasoning = rerun_response_dict.get("reasoning", reasoning)
        content = rerun_response_dict.get("content", {})
        rerun_raw_debug_path.write_text(str(content), encoding="utf-8")
        if is_missing_response_content(content):
            raise RuntimeError("Critic response was empty (`None`/`null`) after rerun.")

    if isinstance(content, str):
        try:
            critic_json: dict[str, Any] = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Critic response content is not valid JSON: {exc}") from exc
    elif isinstance(content, dict):
        critic_json = cast(dict[str, Any], content)
    else:
        raise RuntimeError(f"Unsupported response content type: {type(content).__name__}")

    validate_critic_output(critic_json, allowed_sparse_keys=allowed_sparse_keys)

    args.critic_out.parent.mkdir(parents=True, exist_ok=True)
    args.reasoning_out.parent.mkdir(parents=True, exist_ok=True)

    args.critic_out.write_text(
        json.dumps(critic_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved critic result JSON to: {args.critic_out}")

    args.reasoning_out.write_text(str(reasoning), encoding="utf-8")
    print(f"Saved critic reasoning to: {args.reasoning_out}")
    print(f"Saved raw critic response to: {raw_debug_path}")


if __name__ == "__main__":
    main()
