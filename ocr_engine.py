from __future__ import annotations

import base64
import io
import re
import time
from pathlib import Path
from typing import Callable, List, Optional

try:
    from openai import OpenAI
    from PIL import Image
    IMPORT_ERROR: Optional[str] = None
except ImportError as exc:  # pragma: no cover
    OpenAI = None
    Image = None
    IMPORT_ERROR = str(exc)


OCR_MODELS = [
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]

OCR_PROMPT = (
    "This image contains a handwritten answer written by a student. "
    "Extract ALL the text exactly as written, preserving line breaks. "
    "Output only the extracted text — no commentary, no labels, no formatting."
)

DEFAULT_RETRY_WAIT_SECONDS = 60
DEFAULT_DELAY_BETWEEN_IMAGES = 10
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def make_client(api_key: str):
    """Builds an OpenAI-SDK client pointed at OpenRouter."""
    if IMPORT_ERROR:
        raise RuntimeError(
            f"Missing OCR dependencies: {IMPORT_ERROR}\nRun: pip install openai pillow"
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def image_to_base64(image_path: Path) -> tuple:
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


def try_model(image_path: Path, client, model: str) -> tuple:
    try:
        b64_data, media_type = image_to_base64(image_path)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{b64_data}"},
                        },
                        {"type": "text", "text": OCR_PROMPT},
                    ],
                }
            ],
        )
        return response.choices[0].message.content.strip(), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def extract_text(
    image_path: Path,
    client,
    log: Callable[[str], None] = lambda msg: None,
    models: List[str] = OCR_MODELS,
    retry_wait_default: int = DEFAULT_RETRY_WAIT_SECONDS,
) -> dict:
    """Runs OCR on a single image, falling back across models on failure.

    Returns a result dict with keys: image, path, extracted_text, status,
    model_used (on success) or error (on failure).
    """
    for model in models:
        text, error = try_model(image_path, client, model)

        if text is not None:
            return {
                "image": image_path.name,
                "path": str(image_path.resolve()),
                "extracted_text": text,
                "status": "success",
                "model_used": model,
                "char_count": len(text),
            }

        if error and "429" in error:
            wait = retry_wait_default
            match = re.search(r"seconds[\":\s]+(\d+)", error)
            if match:
                wait = int(match.group(1)) + 5
            log(f"  rate limited on {model}, waiting {wait}s...")
            time.sleep(wait)

            text, error = try_model(image_path, client, model)
            if text is not None:
                return {
                    "image": image_path.name,
                    "path": str(image_path.resolve()),
                    "extracted_text": text,
                    "status": "success",
                    "model_used": model,
                    "char_count": len(text),
                }

        log(f"  {model} failed, trying next model...")

    return {
        "image": image_path.name,
        "path": str(image_path.resolve()),
        "extracted_text": None,
        "status": "error",
        "error": "All models failed. Check your API key or try again later.",
    }


def extract_text_from_images(
    image_paths: List[Path],
    api_key: str,
    log: Callable[[str], None] = lambda msg: None,
    progress: Callable[[int, int], None] = lambda current, total: None,
    delay_between_images: int = DEFAULT_DELAY_BETWEEN_IMAGES,
) -> List[dict]:
    """Runs OCR over a list of images in order, returning a list of result dicts."""
    client = make_client(api_key)
    results = []
    total = len(image_paths)

    for i, img_path in enumerate(image_paths, 1):
        log(f"[{i}/{total}] OCR: {img_path.name}")
        result = extract_text(img_path, client, log=log)
        results.append(result)
        progress(i, total)

        if result["status"] == "success":
            preview = (result["extracted_text"] or "")[:60].replace("\n", " ")
            log(f'  done: "{preview}..."')
        else:
            log(f"  failed: {result.get('error')}")

        if i < total:
            time.sleep(delay_between_images)

    return results


if __name__ == "__main__":
    import json
    import os
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_engine.py <images_dir> [output.json]")
        sys.exit(1)

    images_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output.json")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("Set OPENROUTER_API_KEY first.")
        sys.exit(1)

    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    paths = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in supported)
    if not paths:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    all_results = extract_text_from_images(paths, key, log=print)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {output_path}")
