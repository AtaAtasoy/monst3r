import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
import numpy as np
from transformers.video_utils import load_video

def open_image_as_base64(image_path: Path) -> str:
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
    
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def load_video_keyframes(video_path: Path) -> tuple[str, str]:
    video_object, _ = load_video(str(video_path))  # ty: ignore[invalid-argument-type]
    if len(video_object) == 0:
        raise RuntimeError(f"No frames loaded from video: {video_path}")
    first = image_to_base64(video_object[0])
    last = image_to_base64(video_object[-1])
    return first, last