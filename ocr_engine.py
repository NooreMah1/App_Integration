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
DEFAULT_DELAY_BETWEEN_IMAGES = 3  # OpenRouter free tier allows 20 requests/minute = 1 every 3s
MAX_PARALLEL_REQUESTS = 3         # keep concurrency well under that 20/min ceiling
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
    # Resize down if very large — full-resolution isn't needed for text
    # recognition, and smaller images upload + process faster.
    max_dimension = 1200
    if max(img.size) > max_dimension:
        ratio = max_dimension / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
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
    start_index: int = 0,
) -> dict:
    """Runs OCR on a single image, falling back across models on failure.

    `start_index` controls WHICH model in the list this call starts with
    (round-robin) — it still falls back through all remaining models in
    order if the starting one fails, just doesn't always begin at index 0.

    Returns a result dict with keys: image, path, extracted_text, status,
    model_used (on success) or error (on failure).
    """
    # Rotate the list so we start at `start_index` but still try every model
    rotated_models = models[start_index:] + models[:start_index]

    for model in rotated_models:
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
    max_workers: int = MAX_PARALLEL_REQUESTS,
) -> List[dict]:
    """Runs OCR over a list of images CONCURRENTLY (up to `max_workers` at once),
    while still returning results in the SAME ORDER as image_paths.

    IMPORTANT: order must be preserved because marking_app.py maps
    list position -> answer_no. We do this by pre-allocating a
    results list of the right size and writing into it by index,
    regardless of which thread finishes first.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = make_client(api_key)
    total = len(image_paths)
    results: List[Optional[dict]] = [None] * total
    completed = 0
    lock = threading.Lock()

    def worker(index: int, img_path: Path) -> None:
        nonlocal completed
        log(f"[{index + 1}/{total}] OCR: {img_path.name}")
        # Round-robin: image 0 starts with model[0], image 1 with model[1],
        # image 2 with model[2], image 3 wraps back to model[0], etc.
        start_index = index % len(OCR_MODELS)
        result = extract_text(img_path, client, log=log, start_index=start_index)
        results[index] = result

        with lock:
            completed += 1
            current = completed
        progress(current, total)

        if result["status"] == "success":
            preview = (result["extracted_text"] or "")[:60].replace("\n", " ")
            log(f'  done ({img_path.name}): "{preview}..."')
        else:
            log(f"  failed ({img_path.name}): {result.get('error')}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, img_path in enumerate(image_paths):
            futures.append(executor.submit(worker, i, img_path))
            # Small stagger so we don't fire all requests in the same instant
            time.sleep(delay_between_images / max_workers)
        for future in as_completed(futures):
            future.result()  # re-raises any exception from the worker thread

    return results


if __name__ == "__main__":
    import json
    import os
    import sys
    from dotenv import load_dotenv

    load_dotenv()  # reads OPENROUTER_API_KEY from your .env file

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