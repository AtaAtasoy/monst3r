from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from pathlib import Path
from transformers.video_utils import load_video
from PIL import Image
import numpy as np
import json
import os

project_dir = Path(os.environ["PROJECT_DIR"])
dataset_dir = Path(os.environ["DATASET_DIR"])
    
def render_system_prompt(num_frames: int | None = None) -> str:
    template: str = (Path(__file__).resolve().parent / "cinematographer.txt").read_text(
        encoding="utf-8"
    )
    prompt: str = template
    if num_frames is not None:
        prompt: str = prompt.replace("[Insert Number]", str(num_frames))
    return prompt


model: Qwen3VLForConditionalGeneration = (
    Qwen3VLForConditionalGeneration.from_pretrained(
        dataset_dir / "checkpoints/agents/Qwen3-VL-8B-Instruct",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
)

processor = AutoProcessor.from_pretrained(
    dataset_dir / "checkpoints/agents/Qwen3-VL-8B-Instruct"
)

user_demand = "Zoom in on the tennis player and track their movement."
system_prompt: str = render_system_prompt(num_frames=5)

video_object, _ = load_video(
    project_dir / "monst3r/demo_tmp/davis/tennis/normalized_nofilter/top-down-semantic-ortho-all-camera_centroid/vis-videos/motion_overlay.mp4"  # ty:ignore[invalid-argument-type]
)

capture_camera_first_frame = np.array(
    Image.open(project_dir / "monst3r/demo_tmp/davis/tennis/normalized_nofilter/renders_all/0000.png")
)

print(f"Video obj shape: {video_object.shape}, dtype: {video_object.dtype}")
# use the first and the last frames of the video for testing
video_frames = video_object[[0, -1]]
print(f"Selected video frames shape: {video_frames.shape}, dtype: {video_frames.dtype}")


messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": system_prompt}],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": f"User Demand: {user_demand}"},
            {"type": "image", "image": capture_camera_first_frame},
            {"type": "image", "image": video_frames[0]},
            {"type": "image", "image": video_frames[1]},
        ],
    },
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=256)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
)

output_json = json.loads(output_text[0])
print(json.dumps(output_json, indent=4))