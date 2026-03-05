import argparse
import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from PIL import Image
from transformers.video_utils import load_video


PROJECT_DIR = Path(os.environ["PROJECT_DIR"])


def open_imgage_as_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode("utf-8")


def image_to_base64(image: np.ndarray) -> str:
    image_pil = Image.fromarray(image)
    buffered = BytesIO()
    image_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def save_base64_png(image_base64: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(image_base64))


def json_to_text(json_path: Path) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return json.dumps(data, indent=2, ensure_ascii=True)


def render_system_prompt(
    num_frames: int | None = None, prompt_file_path: str = "cinematographer.txt"
) -> str:
    template: str = (Path(__file__).resolve().parent / prompt_file_path).read_text(
        encoding="utf-8"
    )
    prompt: str = template
    if num_frames is not None:
        prompt = prompt.replace("[Insert Number]", str(num_frames))
    return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VLM camera trajectory generation and save structured outputs."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--user-demand", type=str, required=True)
    parser.add_argument("--num-frames", type=int, default=70)
    parser.add_argument("--trajectory-out", type=Path, required=True)
    parser.add_argument("--reasoning-out", type=Path, required=True)
    parser.add_argument("--user-demand-out", type=Path, required=True)
    return parser.parse_args()


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

    client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1", timeout=3600)
    system_prompt = render_system_prompt(
        prompt_file_path="cinematographer.txt",
        num_frames=args.num_frames,
    )

    scene_root = (
        PROJECT_DIR / "monst3r/demo_tmp/davis/tennis/normalized_nofilter"
    )
    topdown_scene_dir = scene_root / "top-down-semantic-ortho-all-scene"
    capture_cameras_json = topdown_scene_dir / "capture_cameras_pixels.json"
    moving_bbox_json = topdown_scene_dir / "moving_bbox_pixels.json"
    semantic_metadata_json = topdown_scene_dir / "semantic_topdown_metadata.json"
    topdown_camera_json = topdown_scene_dir / "topdown_camera_pixels.json"

    video_object, _ = load_video(
        str(
            scene_root
            / "top-down-semantic-ortho-all-camera_centroid/vis-videos/motion_overlay.mp4"
        )  # ty:ignore[invalid-argument-type]
    )

    capture_camera_first_frame = open_imgage_as_base64(scene_root / "renders_all/0000.png")
    video_frames = video_object[[0, -1]]
    video_frames = [image_to_base64(frame) for frame in video_frames]

    artifact_manifest_lines = [
        "Artifact Manifest:",
        f"- User Demand: {user_demand}",
        "- Rendered Images: capture camera first frame, motion overlay start frame, motion overlay end frame",
        "- Foreground Motion Path (Pixel Coordinates): moving_bbox_pixels.json",
        "- Input Camera Trajectory (Pixel Coordinates): topdown_camera_pixels.json",
        "- Source Camera Positions: capture_cameras_pixels.json",
        "- Extra Metadata: semantic_topdown_metadata.json",
    ]
    artifact_manifest_text = "\n".join(artifact_manifest_lines)

    capture_cameras_text = json_to_text(capture_cameras_json)
    moving_bbox_text = json_to_text(moving_bbox_json)
    semantic_metadata_text = json_to_text(semantic_metadata_json)
    topdown_camera_text = json_to_text(topdown_camera_json)

    user_text_blocks = [
        artifact_manifest_text,
        f"User Demand: {user_demand}",
        "Rendered Image 1 (capture camera first frame):",
        "Rendered Image 2 (motion overlay start frame):",
        "Rendered Image 3 (motion overlay end frame):",
        "Source Camera Positions (Pixel Coordinates) - capture_cameras_pixels.json:",
        capture_cameras_text,
        "Foreground Motion Path (Pixel Coordinates) - moving_bbox_pixels.json:",
        moving_bbox_text,
        "Extra Metadata - semantic_topdown_metadata.json:",
        semantic_metadata_text,
        "Input Camera Trajectory (Pixel Coordinates) - topdown_camera_pixels.json:",
        topdown_camera_text,
        "Return JSON only, using the exact schema specified in the system prompt.",
    ]

    # Save all VLM inputs under 01_vlm/inputs for reproducible experiments.
    vlm_dir = args.trajectory_out.parent
    vlm_inputs_dir = vlm_dir / "inputs"
    visual_inputs_dir = vlm_inputs_dir / "visual"
    language_inputs_dir = vlm_inputs_dir / "language"
    visual_inputs_dir.mkdir(parents=True, exist_ok=True)
    language_inputs_dir.mkdir(parents=True, exist_ok=True)

    visual_input_records: list[dict[str, object]] = [
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
    (language_inputs_dir / "semantic_topdown_metadata.json").write_text(
        semantic_metadata_text, encoding="utf-8"
    )
    (language_inputs_dir / "topdown_camera_pixels.json").write_text(
        topdown_camera_text, encoding="utf-8"
    )
    (language_inputs_dir / "request_config.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "num_frames": args.num_frames,
                "response_format": {"type": "json_object"},
                "max_tokens": 4096 * 4,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved VLM inputs to: {vlm_inputs_dir}")

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": artifact_manifest_text,
                },
                {"type": "text", "text": f"User Demand: {user_demand}"},
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
                    "text": "Extra Metadata - semantic_topdown_metadata.json:",
                },
                {
                    "type": "text",
                    "text": semantic_metadata_text,
                },
                {
                    "type": "text",
                    "text": "Input Camera Trajectory (Pixel Coordinates) - topdown_camera_pixels.json:",
                },
                {
                    "type": "text",
                    "text": topdown_camera_text,
                },
                {
                    "type": "text",
                    "text": "Return JSON only, using the exact schema specified in the system prompt.",
                },
            ],
        },
    ]

    start = time.time()
    response = client.chat.completions.create(
        model=model_name,
        max_tokens=4096 * 4,
        response_format={"type": "json_object"},
        messages=messages,
    )
    print(f"Response costs: {time.time() - start:.2f}s")

    response_dict: dict[str, object] = response.choices[0].message.to_dict()
    reasoning = response_dict.get("reasoning", "No reasoning provided.")
    content = response_dict.get("content", {})

    if isinstance(content, str):
        try:
            trajectory_json: dict[str, object] = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Model response content is not valid JSON: {exc}") from exc
    elif isinstance(content, dict):
        trajectory_json = content
    else:
        raise RuntimeError(f"Unsupported response content type: {type(content).__name__}")

    if not isinstance(trajectory_json.get("camera_trajectory_pixels"), dict):
        raise RuntimeError("Missing `camera_trajectory_pixels` object in model response.")

    args.trajectory_out.parent.mkdir(parents=True, exist_ok=True)
    args.reasoning_out.parent.mkdir(parents=True, exist_ok=True)
    args.user_demand_out.parent.mkdir(parents=True, exist_ok=True)

    args.trajectory_out.write_text(
        json.dumps(trajectory_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved trajectory JSON to: {args.trajectory_out}")

    args.reasoning_out.write_text(str(reasoning), encoding="utf-8")
    print(f"Saved reasoning text to: {args.reasoning_out}")

    args.user_demand_out.write_text(f"{user_demand}\n", encoding="utf-8")
    print(f"Saved user demand text to: {args.user_demand_out}")


if __name__ == "__main__":
    main()
