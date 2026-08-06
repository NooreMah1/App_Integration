import io
import textwrap
from dataclasses import dataclass

import cv2
import fitz
import numpy as np
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color

DPI = 200
MIN_BOX_AREA_FRACTION = 0.003   #The box is said to be actual box only if its area is greater than 0.3% of the page
MAX_BOX_AREA_FRACTION = 0.95    #The box is said to be actual box only if its area is smaller than 95% of the page
MERGE_TOLERANCE_PX = 8          #If the difference between boundaries (gap) of 2 box is less then 8px, consider them a single box and merge.

MARK_FONT = "Helvetica-Bold"
MARK_FONT_SIZE = 11
MARK_COLOR = Color(0.75, 0.05, 0.05, alpha=0.55)
REMARKS_FONT = "Helvetica"
REMARKS_FONT_SIZE = 8.5
REMARKS_COLOR = Color(0.05, 0.05, 0.55, alpha=0.55)
LINE_SPACING = 11
VERTICAL_GAP_BELOW_LABEL = 16
HORIZONTAL_GAP_FROM_LEFT_BORDER = 15
WRAP_WIDTH_CHARS = 78


@dataclass(frozen=True) # it is a decorator.it creates the --init-- function (constructor) automatically
# frozen=True -----After the creation of object,values of object cant change(immutable)
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @property #it is a decorator- that allows you to use a function like a variable(no need to write brackets when calling function)
    def area(self): #this is a property method
        return self.w * self.h
     #this is an instance method - It is a boolean method. It return true or false
    def is_close_to(self, other, tol): #tol is tolerance- same as one we declared above MERGE_TOLERANCE_PX = 8
        return (abs(self.x - other.x) <= tol and abs(self.y - other.y) <= tol #abs is absolute difference function that convert -ve values,+ve
                and abs(self.w - other.w) <= tol and abs(self.h - other.h) <= tol)
    #now all conditions will be checked out. if the difference between any of the parameter of boxes is less the tolerance (8px) , then both are close enough boxes and must be merged as a single one.

    @staticmethod       #Unlike above 2 methods types that uses object to call i.e obj1.area or obj1.is_close_to() means they need self parameter
    #this method is isolated. It uses class to be refer or call i.e BondingBox.union---It don't require self parameter.
    def union(boxes): #box is actually a list of boxes
        x = min(b.x for b in boxes)
        y = min(b.y for b in boxes)
        x2 = max(b.x + b.w for b in boxes)
        y2 = max(b.y + b.h for b in boxes)
        return BoundingBox(x, y, x2 - x, y2 - y)

def render_page_to_image(page, dpi):
    matrix = fitz.Matrix(dpi / 72, dpi / 72)   # 1 point of pdf is equal to how many pixes of image
    pixmap = page.get_pixmap(matrix=matrix, alpha=False) # pixmap is an object having raw pixel data
    array = np.frombuffer(pixmap.samples, dtype=np.uint8) #pixmap.sample is a long 1D flat, raw pixel byte string array and np.buffer converts those raw data to an array without copying(efficient)
    #np.unit8-> hr pixel value (0-255) 8 unsigned bits ki hogi-> 1D array of numbers jo sary pixel values ko store krta hy.
    rgb = array.reshape(pixmap.height, pixmap.width, 3) #reshape converts 1D->3D array, pixmap.eight and width will shows no of rows and columns and 3 shows each pixel has 3 values
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def preprocess(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0) #window size(3X3)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)  #har pixel ko check karo — agar uski brightness 200 se zyada hai (matlab light/white), to usay black (0) bana do; agar kam hai (dark), to white (255) bana do (INV = inverted).
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) #create a 3X3 rectangle that will be use as a brush in next step
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2) #chhoti gaps/holes ko band (fill) kar deta hai lines mein.
    return cv2.Canny(closed, 30, 120)


def is_valid_box(box, page_area):
    relative_area = box.area / page_area
    return MIN_BOX_AREA_FRACTION <= relative_area <= MAX_BOX_AREA_FRACTION and box.area > 0


def merge_duplicates(boxes, tol):
    used = [False] * len(boxes)
    merged = []
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


def detect_boxes(bgr):
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


def find_answer_anchor_points_by_box(pdf_path, wanted_answer_numbers, dpi=DPI):
    """Global counter — page 1's boxes come first (1,2,3...), then page 2's."""
    scale = 72.0 / dpi
    anchors = {}
    doc = fitz.open(pdf_path)
    global_box_counter = 0
    for page_index, page in enumerate(doc):
        page_height_pt = page.rect.height
        bgr = render_page_to_image(page, dpi)
        boxes = detect_boxes(bgr)
        for box in boxes:
            global_box_counter += 1
            answer_no = global_box_counter
            if answer_no not in wanted_answer_numbers:
                continue
            anchors[answer_no] = {
                "page_index": page_index,
                "x0": box.x * scale,
                "top": box.y * scale,
                "bottom": (box.y + box.h) * scale,
                "page_height": page_height_pt,
            }
    doc.close()
    return anchors


def draw_mark_and_remarks(c, anchor, data):
    x = anchor["x0"] + HORIZONTAL_GAP_FROM_LEFT_BORDER
    page_height = anchor["page_height"]
    top_y = page_height - anchor["top"]
    cursor_y = top_y - VERTICAL_GAP_BELOW_LABEL

    c.setFont(MARK_FONT, MARK_FONT_SIZE)
    c.setFillColor(MARK_COLOR)
    c.drawString(x, cursor_y, f"Score: {data['score']} / 10")
    cursor_y -= LINE_SPACING + 4

    c.setFont(REMARKS_FONT, REMARKS_FONT_SIZE)
    c.setFillColor(REMARKS_COLOR)
    for remark in data["remarks"]:
        bullet_text = f"- {remark}"
        for line in textwrap.wrap(bullet_text, width=WRAP_WIDTH_CHARS):
            c.drawString(x, cursor_y, line)
            cursor_y -= LINE_SPACING


def build_overlay_pdf(original_pdf_path, anchors, scoring_data):
    reader = PdfReader(original_pdf_path)
    anchors_by_page = {}
    for answer_no, anchor in anchors.items():
        anchors_by_page.setdefault(anchor["page_index"], []).append((answer_no, anchor))

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)
    for page_index, page in enumerate(reader.pages):
        page_width = float(page.mediabox.width)
        page_height = float(page.mediabox.height)
        c.setPageSize((page_width, page_height))
        for answer_no, anchor in anchors_by_page.get(page_index, []):
            draw_mark_and_remarks(c, anchor, scoring_data[answer_no])
        c.showPage()
    c.save()
    buffer.seek(0)
    return PdfReader(buffer)


def merge_and_save(original_pdf_path, overlay_reader, output_path):
    original_reader = PdfReader(original_pdf_path)
    writer = PdfWriter()
    for page_index, original_page in enumerate(original_reader.pages):
        overlay_page = overlay_reader.pages[page_index]
        original_page.merge_page(overlay_page)
        writer.add_page(original_page)
    with open(output_path, "wb") as f:
        writer.write(f)


def generate_marked_pdf(pdf_path, scoring_list, output_path, log=lambda msg: None):
    """
    Main entry point — GUI yahi function call karega.

    pdf_path      : original PDF ka path
    scoring_list  : [{"answer_no": 1, "score": 8, "remarks": [...]}, ...]
    output_path   : final marked PDF kahan save hogi
    """
    scoring_data = {item["answer_no"]: {"score": item["score"], "remarks": item["remarks"]}
                     for item in scoring_list}

    log(f"Detecting boxes in '{pdf_path}' ...")
    anchors = find_answer_anchor_points_by_box(pdf_path, wanted_answer_numbers=set(scoring_data.keys()))

    missing = set(scoring_data.keys()) - set(anchors.keys())
    if missing:
        log(f"WARNING: boxes not found for answers: {sorted(missing)}")

    log("Building overlay ...")
    overlay_reader = build_overlay_pdf(pdf_path, anchors, scoring_data)

    log(f"Saving marked PDF to '{output_path}' ...")
    merge_and_save(pdf_path, overlay_reader, output_path)
    log("Done.")
    return output_path