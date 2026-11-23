"""Draw citation rectangles onto a copy of the source PDF."""
import io

from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Highlight
from pypdf.generic import ArrayObject, FloatObject


def _quad(x0: float, y0: float, x1: float, y1: float) -> ArrayObject:
    """PDF highlight quad: top-left, top-right, bottom-left, bottom-right."""
    top = max(y0, y1)
    bottom = min(y0, y1)
    left = min(x0, x1)
    right = max(x0, x1)
    return ArrayObject([
        FloatObject(left), FloatObject(top),
        FloatObject(right), FloatObject(top),
        FloatObject(left), FloatObject(bottom),
        FloatObject(right), FloatObject(bottom),
    ])


def _boxes_from_citations(citations):
    boxes = []
    for item in citations or []:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else item
        box = metadata.get("bbox") if isinstance(metadata.get("bbox"), dict) else metadata
        coords = [box.get(key) for key in ("x0", "y0", "x1", "y1")]
        page = box.get("page", metadata.get("page"))
        if page is None or not all(isinstance(value, (int, float)) for value in coords):
            continue
        boxes.append((int(page), [float(value) for value in coords]))
    return boxes


def paint_citation_boxes(pdf_bytes: bytes, citations) -> bytes:
    """Return a PDF with a highlight annotation for every citation that has a box."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    writer.append(reader)
    for page_number, (x0, y0, x1, y1) in _boxes_from_citations(citations):
        page_index = page_number - 1
        if page_index < 0 or page_index >= len(writer.pages):
            continue
        left, right = min(x0, x1), max(x0, x1)
        bottom, top = min(y0, y1), max(y0, y1)
        annotation = Highlight(
            rect=(left, bottom, right, top),
            quad_points=_quad(left, bottom, right, top),
            highlight_color="ffeb3b",
        )
        writer.add_annotation(page_number=page_index, annotation=annotation)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()
