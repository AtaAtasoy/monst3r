import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .vlm_response_utils import is_missing_response_content
from .io import read_text, save_sampled_video_frames


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
    parser.add_argument("--sparse-trajectory-json", type=Path, required=True)
    parser.add_argument("--generated-cameras-render-video", type=Path, required=True)
    parser.add_argument("--generated-cameras-topdown-video", type=Path, required=True)
    parser.add_argument("--critic-out", type=Path, required=True)
    parser.add_argument("--reasoning-out", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=None)
    return parser.parse_args()


def extract_sparse_frame_keys(sparse_trajectory_text: str) -> list[str]:
    data = json.loads(sparse_trajectory_text)
    traj = data.get("sparse_camera_trajectory")
    if not isinstance(traj, dict):
        raise RuntimeError(
            "`strided_trajectory_json` must contain `sparse_camera_trajectory` as an object"
        )

    indexed_keys: list[tuple[int, str]] = []
    for key, value in traj.items():
        if not isinstance(key, str) or not key.startswith("t_"):
            continue
        if not isinstance(value, dict):
            continue
        try:
            frame_idx = int(key[2:])
        except ValueError:
            continue
        indexed_keys.append((frame_idx, key))

    if not indexed_keys:
        raise RuntimeError(
            "`strided_trajectory_json` must contain at least one sparse `t_i` entry"
        )

    indexed_keys.sort(key=lambda item: item[0])
    return [key for _, key in indexed_keys]


def link_or_copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


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

    if not args.sparse_trajectory_json.exists():
        raise FileNotFoundError(
            f"Missing sparse trajectory json: {args.sparse_trajectory_json}"
        )
    if not args.generated_cameras_render_video.exists():
        raise FileNotFoundError(
            f"Missing render video: {args.generated_cameras_render_video}"
        )
    if not args.generated_cameras_topdown_video.exists():
        raise FileNotFoundError(
            f"Missing topdown video: {args.generated_cameras_topdown_video}"
        )

    sparse_trajectory_text = read_text(args.sparse_trajectory_json)
    sparse_frame_keys = extract_sparse_frame_keys(sparse_trajectory_text)
    allowed_sparse_keys = set(sparse_frame_keys)

    if args.num_frames is None:
        num_frames = max(int(key[2:]) for key in sparse_frame_keys) + 1
    else:
        num_frames = int(args.num_frames)
    if num_frames <= 0:
        raise RuntimeError("Unable to determine positive `num_frames` for critic prompt")

    system_prompt = render_system_prompt(
        num_frames=num_frames,
        sparse_frame_keys=sparse_frame_keys,
    )
    sparse_viz_path = args.sparse_trajectory_json.parent / "trajectory_strided_topdown.mp4"
    sparse_viz_available = sparse_viz_path.exists()

    # artifact_manifest_lines = [
    #     "Artifact Manifest:",
    #     f"- User Demand: {user_demand}",
    #     f"- Sparse 2D Trajectory JSON: {args.sparse_trajectory_json.name}",
    #     f"- Generated Camera Renderings Video: {args.generated_cameras_render_video.name}",
    #     f"- Generated Cameras Topdown Video: {args.generated_cameras_topdown_video.name}",
    #     (
    #         f"- Sparse Trajectory Topdown Video: {sparse_viz_path.name}"
    #         if sparse_viz_available
    #         else "- Sparse Trajectory Topdown Video: N/A"
    #     ),
    # ]
    # artifact_manifest_text = "\n".join(artifact_manifest_lines)

    visual_inputs_dir = args.inputs_dir / "visual"
    language_inputs_dir = args.inputs_dir / "language"
    visual_inputs_dir.mkdir(parents=True, exist_ok=True)
    language_inputs_dir.mkdir(parents=True, exist_ok=True)

    visual_input_records: list[dict[str, int | str | bool]] = [
        {
            "order": 0,
            "file_type": "video",
            "file_name": "generated_cameras_render.mp4",
            "role": "generated_cameras_render_video",
            "source": str(args.generated_cameras_render_video),
            "available": True,
        },
        {
            "order": 1,
            "file_type": "video",
            "file_name": "generated_cameras_topdown.mp4",
            "role": "generated_cameras_topdown_video",
            "source": str(args.generated_cameras_topdown_video),
            "available": True,
        },
        {
            "order": 2,
            "file_type": "video",
            "file_name": "sparse_trajectory_topdown.mp4",
            "role": "sparse_trajectory_topdown_video",
            "source": str(sparse_viz_path),
            "available": sparse_viz_available,
            "note": (
                ""
                if sparse_viz_available
                else "Sparse topdown video missing; critic will use sparse JSON directly."
            ),
        },
    ]

    for record in visual_input_records:
        if not bool(record.get("available", False)):
            continue
        source_video = Path(str(record["source"]))
        target_video = visual_inputs_dir / str(record["file_name"])
        link_or_copy_file(source_video, target_video)
        save_sampled_video_frames(
            target_video,
            visual_inputs_dir / f"{target_video.stem}__sampled_frames",
            fps=2.0,
            backend="opencv",
        )

    visual_manifest: list[dict[str, int | str | bool]] = []
    for record in visual_input_records:
        visual_manifest.append(
            {
                "order": int(record["order"]),
                "file_type": str(record["file_type"]),
                "file_name": str(record["file_name"]),
                "role": str(record["role"]),
                "source": str(record["source"]),
                "available": bool(record.get("available", False)),
                "note": str(record.get("note", "")),
            }
        )
    (visual_inputs_dir / "visual_inputs_manifest.json").write_text(
        json.dumps(visual_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # user_text_blocks = [
    #     artifact_manifest_text,
    #     f"User Demand: {user_demand}",
    #     "Sparse 2D Camera Control Trajectory JSON:",
    #     sparse_trajectory_text,
    #     "Return JSON only, using the exact schema specified in the system prompt.",
    # ]
    (language_inputs_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
    # (language_inputs_dir / "artifact_manifest.txt").write_text(
    #     f"{artifact_manifest_text}\n", encoding="utf-8"
    # )
    (language_inputs_dir / "sparse_camera_trajectory.json").write_text(
        sparse_trajectory_text, encoding="utf-8"
    )
    # (language_inputs_dir / "user_text_blocks.json").write_text(
    #     json.dumps(user_text_blocks, indent=2, ensure_ascii=False),
    #     encoding="utf-8",
    # )
    (language_inputs_dir / "request_config.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "response_format": {"type": "json_object"},
                "max_tokens": 4096 * 2,
                "topdown_video_available": True,
                "strided_topdown_video_available": sparse_viz_available,
                "mm_processor_kwargs": {"fps": 2.0, "do_sample_frames": True},
                "sparse_frame_keys": sparse_frame_keys,
                "num_frames": num_frames,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved critic inputs to: {args.inputs_dir}")

    generated_render_video = (visual_inputs_dir / "generated_cameras_render.mp4").resolve()
    generated_topdown_video = (visual_inputs_dir / "generated_cameras_topdown.mp4").resolve()
    sparse_topdown_video = (visual_inputs_dir / "sparse_trajectory_topdown.mp4").resolve()

    user_content: list[dict[str, object]] = [
        # {"type": "text", "text": artifact_manifest_text},
        {"type": "text", "text": f"User Demand: {user_demand}"},
        {
            "type": "text",
            "text": "Generated camera renderings video from the generated camera trajectory:",
        },
        {"type": "video_url", "video_url": {"url": generated_render_video.as_uri()}},
        {
            "type": "text",
            "text": "Top-down video of generated cameras trajectory:",
        },
        {"type": "video_url", "video_url": {"url": generated_topdown_video.as_uri()}},
    ]

    if sparse_viz_available:
        user_content.extend(
            [
                {"type": "text", "text": "Top-down video of sparse trajectory control points:"},
                {
                    "type": "video_url",
                    "video_url": {"url": sparse_topdown_video.as_uri()},
                },
            ]
        )
    else:
        user_content.append(
            {
                "type": "text",
                "text": "Sparse trajectory topdown video is unavailable. Evaluate the sparse path from the JSON directly.",
            }
        )

    user_content.extend(
        [
            {"type": "text", "text": "Sparse 2D Camera Control Trajectory JSON:"},
            {"type": "text", "text": sparse_trajectory_text},
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
        extra_body={
            "mm_processor_kwargs": {"fps": 2.0, "do_sample_frames": True},
        },
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
            extra_body={
                "mm_processor_kwargs": {"fps": 2.0, "do_sample_frames": True},
            },
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
