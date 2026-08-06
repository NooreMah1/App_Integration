from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

try:
    import cv2
    import fitz  # PyMuPDF
    import numpy as np
    IMPORT_ERROR: Optional[str] = None
except ImportError as exc:  # pragma: no cover
    cv2 = None
    fitz = None
    np = None
    IMPORT_ERROR = str(exc)


DEFAULT_DPI = 200
DEFAULT_IMAGES_DIR_NAME = "images"

MIN_BOX_AREA_FRACTION = 0.003
MAX_BOX_AREA_FRACTION = 0.95
MERGE_TOLERANCE_PX = 8
CROP_PADDING_PX = 4


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    def padded(self, pad: int, max_w: int, max_h: int) -> tuple:
        x0 = max(0, self.x - pad)
        y0 = max(0, self.y - pad)
        x1 = min(max_w, self.x + self.w + pad)
        y1 = min(max_h, self.y + self.h + pad)
        return x0, y0, x1, y1

    def is_close_to(self, other: "BoundingBox", tol: int) -> bool:
        return (
            abs(self.x - other.x) <= tol
            and abs(self.y - other.y) <= tol
            and abs(self.w - other.w) <= tol
            and abs(self.h - other.h) <= tol
        )

    @staticmethod
    def union(boxes: List["BoundingBox"]) -> "BoundingBox":
        x = min(b.x for b in boxes)
        y = min(b.y for b in boxes)
        x2 = max(b.x + b.w for b in boxes)
        y2 = max(b.y + b.h for b in boxes)
        return BoundingBox(x, y, x2 - x, y2 - y)


def render_page(page, dpi: int):
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    array = np.frombuffer(pixmap.samples, dtype=np.uint8)
    rgb = array.reshape(pixmap.height, pixmap.width, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def preprocess(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.Canny(closed, 30, 120)


def is_valid_box(box: BoundingBox, page_area: int) -> bool:
    relative_area = box.area / page_area
    if not (MIN_BOX_AREA_FRACTION <= relative_area <= MAX_BOX_AREA_FRACTION):
        return False
    return box.area > 0


def merge_duplicates(boxes: List[BoundingBox], tol: int) -> List[BoundingBox]:
    used = [False] * len(boxes)
    merged: List[BoundingBox] = []
    for i, box in enumerate(boxes):
        if used[i]:
            continue
        group = [box]
        for j in range(i + 1, len(boxes)):
            if not used[j] and box.is_close_to(boxes[j], tol):
                group.append(boxes[j])
                used[j] = True
        merged.append(BoundingBox.union(group))
        used[i] = True
    return merged


def detect_boxes(bgr) -> List[BoundingBox]:
    h, w = bgr.shape[:2]
    page_area = h * w
    edges = preprocess(bgr)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        box = BoundingBox(x, y, bw, bh)
        cnt_area = cv2.contourArea(cnt)
        solidity = cnt_area / box.area if box.area > 0 else 0
        if is_valid_box(box, page_area) and solidity >= 0.05:
            raw_boxes.append(box)

    merged = merge_duplicates(raw_boxes, tol=MERGE_TOLERANCE_PX)
    return sorted(merged, key=lambda b: (b.y // 50, b.x))


def save_crop(bgr, box: BoundingBox, path: Path) -> None:
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = box.padded(CROP_PADDING_PX, w, h)
    crop = bgr[y0:y1, x0:x1]
    cv2.imwrite(str(path), crop)


def extract_answer_images(
    pdf_path: str,
    output_dir: str = DEFAULT_IMAGES_DIR_NAME,
    dpi: int = DEFAULT_DPI,
    log: Callable[[str], None] = lambda msg: None,
) -> List[Path]:
    if IMPORT_ERROR:
        raise RuntimeError(
            f"Missing PDF/image-processing dependencies: {IMPORT_ERROR}\n"
            "Run: pip install opencv-python PyMuPDF numpy"
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    saved_paths: List[Path] = []

    try:
        for page_num, page in enumerate(doc, start=1):
            log(f"Page {page_num}: rendering...")
            bgr = render_page(page, dpi)
            boxes = detect_boxes(bgr)
            log(f"Page {page_num}: {len(boxes)} answer box(es) detected")

            for box_num, box in enumerate(boxes, start=1):
                filename = out / f"page{page_num:02d}_answer{box_num:02d}.jpg"
                save_crop(bgr, box, filename)
                saved_paths.append(filename)
                log(f"  saved {filename.name}")
    finally:
        doc.close()

    return saved_paths


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <pdf_path> [output_dir]")
        sys.exit(1)

    pdf_arg = sys.argv[1]
    out_arg = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_IMAGES_DIR_NAME
    paths = extract_answer_images(pdf_arg, out_arg, log=print)
    print(f"\nDone — {len(paths)} image(s) saved to '{out_arg}/'")
