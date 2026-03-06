import json
import time
from pathlib import Path
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


def strip_markdown_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)
    raise json.JSONDecodeError("Could not find a complete JSON object", text, 0)


def parse_json_object_text(text: str) -> dict[str, Any]:
    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    unwrapped = strip_markdown_code_fences(text)
    if unwrapped and unwrapped not in candidates:
        candidates.append(unwrapped)

    parse_errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            parse_errors.append(f"json.loads: {exc}")
        else:
            if isinstance(parsed, dict):
                return cast(dict[str, Any], parsed)
            raise RuntimeError("Model response JSON is valid but not a JSON object.")

        try:
            return extract_first_json_object(candidate)
        except json.JSONDecodeError as exc:
            parse_errors.append(f"extract_first_json_object: {exc}")

    raise RuntimeError(
        "Unable to parse model response as a JSON object. " + " | ".join(parse_errors)
    )


def request_json_repair(
    client: OpenAI,
    model_name: str,
    malformed_text: str,
    *,
    required_top_level_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repair_messages = cast(
        list[ChatCompletionMessageParam],
        [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You repair malformed JSON outputs. Return a single valid JSON object "
                            "only. Do not include markdown fences or any explanation."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Repair the malformed JSON below. Preserve the original intent and "
                            "content when possible. The result must contain the top-level key "
                            f"'{required_top_level_key}' and must be a single JSON object. If the "
                            "text appears truncated, infer the minimal completion needed to make "
                            "it valid JSON.\n\nMalformed text:\n"
                            f"{malformed_text}"
                        ),
                    }
                ],
            },
        ],
    )

    repair_response = client.chat.completions.create(
        model=model_name,
        max_tokens=8192,
        response_format={"type": "json_object"},
        messages=repair_messages,
    )
    repair_dict = cast(dict[str, Any], repair_response.choices[0].message.to_dict())
    repair_content = repair_dict.get("content", {})

    if isinstance(repair_content, str):
        repaired_json = parse_json_object_text(repair_content)
    elif isinstance(repair_content, dict):
        repaired_json = cast(dict[str, Any], repair_content)
    else:
        raise RuntimeError(
            f"Unsupported repair response content type: {type(repair_content).__name__}"
        )

    return repaired_json, repair_dict


def extract_json_from_response(
    client: OpenAI,
    model_name: str,
    content: Any,
    output_dir: Path,
    base_name: str,
    finish_reason: Any,
    *,
    required_top_level_key: str,
    label: str,
) -> dict[str, Any]:
    raw_debug_path = output_dir / f"{base_name}_raw_response.txt"
    response_meta_path = output_dir / f"{base_name}_response_meta.json"
    raw_debug_path.parent.mkdir(parents=True, exist_ok=True)
    raw_debug_path.write_text(str(content), encoding="utf-8")
    response_meta_path.write_text(
        json.dumps(
            {
                "finish_reason": finish_reason,
                "content_type": type(content).__name__,
                "model": model_name,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved raw {label} response to: {raw_debug_path}")
    print(f"Saved {label} response metadata to: {response_meta_path}")

    if isinstance(content, str):
        try:
            return parse_json_object_text(content)
        except RuntimeError:
            repaired_path = output_dir / f"{base_name}_repaired_response.txt"
            repaired_meta_path = output_dir / f"{base_name}_repair_meta.json"
            repaired_start = time.time()
            parsed_json, repair_response_dict = request_json_repair(
                client=client,
                model_name=model_name,
                malformed_text=content,
                required_top_level_key=required_top_level_key,
            )
            repaired_content = repair_response_dict.get("content", {})
            repaired_path.write_text(str(repaired_content), encoding="utf-8")
            repaired_meta_path.write_text(
                json.dumps(
                    {
                        "finish_reason": repair_response_dict.get("finish_reason"),
                        "content_type": type(repaired_content).__name__,
                        "model": model_name,
                        "repair_elapsed_seconds": time.time() - repaired_start,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"Saved repaired {label} response to: {repaired_path}")
            print(f"Saved repair metadata to: {repaired_meta_path}")
            return parsed_json
    if isinstance(content, dict):
        return cast(dict[str, Any], content)
    raise RuntimeError(f"Unsupported response content type: {type(content).__name__}")


def is_missing_response_content(content: Any) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        normalized = content.strip().lower()
        return normalized in {"", "none", "null"}
    return False


def get_missing_frame_keys(trajectory_json: dict[str, Any], num_frames: int) -> list[str]:
    trajectory_pixels = trajectory_json.get("camera_trajectory_pixels")
    if not isinstance(trajectory_pixels, dict):
        raise RuntimeError("Missing `camera_trajectory_pixels` object in model response.")

    return [
        f"t_{frame_index}"
        for frame_index in range(num_frames)
        if f"t_{frame_index}" not in trajectory_pixels
    ]
