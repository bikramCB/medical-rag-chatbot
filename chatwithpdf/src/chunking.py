import datetime
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.schema import Document
from unstructured.partition.pdf import partition_pdf

_docling_converter = None

_DOCLING_ELEMENT_TYPES = {
    "section_header": "Title",
    "title": "Title",
    "page_header": "Header",
    "table": "Table",
    "text": "NarrativeText",
    "list_item": "NarrativeText",
    "caption": "NarrativeText",
    "footnote": "NarrativeText",
    "formula": "NarrativeText",
    "code": "NarrativeText",
}
_DOCLING_SKIP = {"page_footer"}
_FIGURE_PROMPT = (
    "Read this one figure from a medical paper. Reply in plain text, under 180 words. "
    "Name the figure type. List every number you can see: percentages, time points, "
    "hazard ratios, confidence intervals, p-values, sample sizes, and counts. "
    "For a survival curve, name the groups and any numbers at risk. "
    "For a flow diagram, list each box and its count. "
    "Do not invent a number. If the image is a logo, decoration, or unreadable, reply SKIP."
)

# First matching label wins. "Results and Discussion" stays Results.
_SECTION_PREFIXES = (
    ("References", ("references", "bibliography", "literature cited")),
    ("Funding/COI", (
        "funding",
        "competing interest",
        "conflict of interest",
        "conflicts of interest",
        "disclosure",
        "acknowledgement",
        "acknowledgment",
    )),
    ("Limitations", ("limitation", "strengths and limitations")),
    ("Discussion", ("discussion",)),
    ("Results", ("result", "finding")),
    ("Methods", (
        "method",
        "materials and methods",
        "study design",
        "experimental procedure",
        "patients and methods",
    )),
    ("Introduction", ("introduction", "background")),
    ("Abstract", ("abstract", "summary")),
)


def estimate_tokens(text: str) -> int:
    """Word count times 1.3. Closer to model tokens than a raw character split."""
    words = re.findall(r"\S+", text or "")
    if not words:
        return 0
    return max(1, math.ceil(len(words) * 1.3))


def overlap_tail(text: str, max_sentences: int = 2, max_words: int = 40) -> str:
    """Carry the last sentences into the next chunk, capped so a long paragraph is not copied whole."""
    compact = re.sub(r"\s+", " ", (text or "").strip())
    if not compact:
        return ""
    sentences = re.findall(r".+?[.!?](?=\s|$)|[^.!?]+$", compact)
    tail = " ".join(part.strip() for part in sentences[-max_sentences:] if part.strip())
    words = tail.split()
    if len(words) > max_words:
        tail = " ".join(words[-max_words:])
    return tail


def classify_section(heading: str):
    """Map a heading line to a section label. Unrecognized headings return None."""
    if not heading or not heading.strip():
        return None
    first_line = heading.strip().splitlines()[0]
    normalized = re.sub(
        r"^(?:\d+|[IVXLC]+)(?:\s*[\.\)\:\-])?\s+",
        "",
        first_line,
        flags=re.IGNORECASE,
    )
    normalized = normalized.strip(" .:-").lower()
    normalized = re.sub(r"\s+", " ", normalized)
    for label, prefixes in _SECTION_PREFIXES:
        for prefix in prefixes:
            if (
                normalized == prefix
                or normalized.startswith(prefix + " ")
                or normalized.startswith(prefix + ":")
                or normalized.startswith(prefix + "s")
            ):
                return label
    return None


def docling_label(item) -> str:
    label = getattr(item, "label", "")
    name = getattr(label, "value", None) or getattr(label, "name", None) or label
    return str(name).strip().lower()


def docling_bbox(item):
    """Page number and an axis-aligned box. Docling's origin can be the bottom-left corner."""
    prov = getattr(item, "prov", None) or []
    if not prov:
        return None, {}
    first = prov[0]
    page = getattr(first, "page_no", None)
    box = getattr(first, "bbox", None)
    if box is None:
        return page, {}
    if isinstance(box, dict):
        left, top, right, bottom = box.get("l"), box.get("t"), box.get("r"), box.get("b")
    else:
        left, top, right, bottom = getattr(box, "l", None), getattr(box, "t", None), getattr(box, "r", None), getattr(box, "b", None)
    coords = (left, top, right, bottom)
    if not all(isinstance(value, (int, float)) for value in coords):
        return page, {}
    return page, {
        "x0": float(min(left, right)),
        "y0": float(min(top, bottom)),
        "x1": float(max(left, right)),
        "y1": float(max(top, bottom)),
    }


def docling_item_text(item, document) -> str:
    if docling_label(item) != "table":
        return str(getattr(item, "text", "") or "")
    export_markdown = getattr(item, "export_to_markdown", None)
    if export_markdown is not None:
        try:
            return str(export_markdown(doc=document) or "")
        except TypeError:
            return str(export_markdown() or "")
    export_frame = getattr(item, "export_to_dataframe", None)
    if export_frame is not None:
        frame = export_frame(doc=document)
        return frame.to_markdown(index=False)
    return str(getattr(item, "text", "") or "")


def picture_image(item, document):
    """Return the figure image Docling cropped, or None when the item has no pixels."""
    getter = getattr(item, "get_image", None)
    if getter is None:
        return None
    try:
        return getter(document)
    except TypeError:
        try:
            return getter()
        except Exception:
            return None
    except Exception:
        return None


def _figure_is_worth_describing(image) -> bool:
    if image is None:
        return False
    width, height = getattr(image, "size", (0, 0))
    return width >= 80 and height >= 80


def _vision_settings():
    """One Vertex login shared by every figure call in a paper."""
    enabled = os.getenv("FIGURE_VISION", "true").strip().lower()
    if enabled in ("0", "false", "no"):
        return None
    import google.auth
    from google.auth.transport.requests import Request as AuthRequest

    credentials, detected_project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(AuthRequest())
    project = os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or detected_project
    if not project:
        print("Figure vision skipped: set VERTEX_PROJECT or GOOGLE_CLOUD_PROJECT.")
        return None
    # Gemini 3.5 Flash is served from the global endpoint. us-central1 returns 404.
    location = os.getenv("VERTEX_VISION_LOCATION", "global")
    model = os.getenv("VERTEX_VISION_MODEL", "gemini-3.5-flash")
    host = "https://aiplatform.googleapis.com" if location == "global" else f"https://{location}-aiplatform.googleapis.com"
    url = (
        f"{host}/v1/projects/{project}/locations/{location}"
        f"/publishers/google/models/{model}:generateContent"
    )
    return url, credentials.token


def _describe_one_figure(image, url, token) -> str:
    """One Gemini call. Retries a rate limit so a parallel batch can finish."""
    import base64
    import time
    from io import BytesIO

    import requests

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(buffer.getvalue()).decode("ascii")}},
                {"text": _FIGURE_PROMPT},
            ],
        }],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 512},
    }
    response = None
    for attempt in range(3):
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}"},
            json=body,
            timeout=120,
        )
        if response.status_code != 429 or attempt == 2:
            break
        time.sleep(20 * (attempt + 1))
    if response.status_code >= 400:
        print(f"Figure vision returned {response.status_code}. The figure caption is still indexed.")
        return ""
    payload = response.json()
    parts = payload["candidates"][0]["content"]["parts"]
    text = " ".join(part.get("text", "") for part in parts).strip()
    if not text or text.upper() == "SKIP":
        return ""
    print("Described one figure with Gemini Flash.")
    return text


def describe_figure(image) -> str:
    """Ask Vertex Gemini Flash to write the numbers visible in one figure.

    Logos and failed calls return an empty string so the rest of the PDF still indexes.
    """
    if not _figure_is_worth_describing(image):
        return ""
    try:
        settings = _vision_settings()
        if settings is None:
            return ""
        url, token = settings
        return _describe_one_figure(image, url, token)
    except Exception as exc:
        print(f"Figure vision skipped ({exc}).")
        return ""


def _describe_figures_together(images):
    """Send every real figure at once. The result list lines up with images."""
    descriptions = [""] * len(images)
    pending = [index for index, image in enumerate(images) if _figure_is_worth_describing(image)]
    if not pending:
        return descriptions
    try:
        settings = _vision_settings()
        if settings is None:
            return descriptions
        url, token = settings
        workers = int(os.getenv("FIGURE_VISION_WORKERS", "5"))
        workers = max(1, min(workers, len(pending)))
        print(f"Describing {len(pending)} figures in parallel.")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_describe_one_figure, images[index], url, token): index
                for index in pending
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    descriptions[index] = future.result()
                except Exception as exc:
                    print(f"Figure vision skipped ({exc}).")
                    descriptions[index] = ""
    except Exception as exc:
        print(f"Figure vision skipped ({exc}).")
    return descriptions


def _element_from_item(item, document, source: str):
    """One non-picture Docling item, or None when it should not be stored."""
    element_type = _DOCLING_ELEMENT_TYPES.get(docling_label(item), "NarrativeText")
    text = docling_item_text(item, document).strip()
    if not text:
        return None
    page, box = docling_bbox(item)
    metadata = {
        "source": source,
        "element_type": element_type,
        "page_number": page or 1,
    }
    metadata.update(box)
    return Document(page_content=text, metadata=metadata)


def documents_from_docling(document, source: str):
    """Walk Docling's reading order into the same element records the chunker already uses.

    Picture crops are collected first and sent to Gemini together. Text keeps its
    original place in the reading order.
    """
    ordered = []
    pictures = []
    for item, _level in document.iterate_items():
        label = docling_label(item)
        if label == "picture":
            ordered.append(("picture", len(pictures)))
            pictures.append((item, picture_image(item, document)))
            continue
        if label in _DOCLING_SKIP:
            continue
        element = _element_from_item(item, document, source)
        if element is not None:
            ordered.append(("element", element))

    descriptions = _describe_figures_together([image for _item, image in pictures])
    docs = []
    for kind, payload in ordered:
        if kind == "element":
            docs.append(payload)
            continue
        description = descriptions[payload]
        if not description:
            continue
        item, _image = pictures[payload]
        page, box = docling_bbox(item)
        metadata = {
            "source": source,
            "element_type": "Figure",
            "page_number": page or 1,
        }
        metadata.update(box)
        docs.append(Document(page_content=f"Figure. {description}", metadata=metadata))
    return docs


def _get_docling_converter():
    global _docling_converter
    if _docling_converter is None:
        from docling.document_converter import DocumentConverter

        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption

            options = PdfPipelineOptions()
            options.generate_picture_images = True
            options.images_scale = 1.0
            _docling_converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        except Exception as exc:
            print(f"Docling picture images are unavailable ({exc}). Figures will not be described.")
            _docling_converter = DocumentConverter()
    return _docling_converter


def element_bbox(element) -> dict:
    """Axis-aligned box from an Unstructured element's corner points."""
    coords = getattr(getattr(element, "metadata", None), "coordinates", None)
    points = getattr(coords, "points", None) if coords is not None else None
    if not points:
        return {}
    xs = []
    ys = []
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
    if not xs:
        return {}
    return {"x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys)}


def union_bbox(current: dict, incoming: dict) -> dict:
    keys = ("x0", "y0", "x1", "y1")
    if not all(key in incoming for key in keys):
        return current
    if not current:
        return {key: incoming[key] for key in keys}
    return {
        "x0": min(current["x0"], incoming["x0"]),
        "y0": min(current["y0"], incoming["y0"]),
        "x1": max(current["x1"], incoming["x1"]),
        "y1": max(current["y1"], incoming["y1"]),
    }


def html_table_to_markdown(html: str) -> str:
    """Turn a simple HTML table into Markdown. Empty string means the HTML was not a table."""
    if not html:
        return ""
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.IGNORECASE | re.DOTALL)
    parsed = []
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.IGNORECASE | re.DOTALL)
        cleaned = []
        for cell in cells:
            text = re.sub(r"<[^>]+>", "", cell)
            text = re.sub(r"\s+", " ", text).strip().replace("|", "\\|")
            cleaned.append(text)
        if any(cleaned):
            parsed.append(cleaned)
    if not parsed:
        return ""
    width = max(len(row) for row in parsed)
    parsed = [row + [""] * (width - len(row)) for row in parsed]

    def format_row(row):
        return "| " + " | ".join(row) + " |"

    header = format_row(parsed[0])
    separator = "| " + " | ".join("---" for _ in parsed[0]) + " |"
    body = [format_row(row) for row in parsed[1:]]
    return "\n".join([header, separator, *body])


class ChunkingMixin:
    def _load_and_process_documents(self):
        def count_tokens(text: str) -> int:
            return estimate_tokens(text)

        print("\n--- Starting new processing pipeline. ---")

        # === CHANGED: Use the memory file we downloaded in __init__ ===
        if not self.pdf_file_bytes:
            print("Error: PDF bytes were not downloaded during initialization.")
            return

        # Reset pointer to start of file
        self.pdf_file_bytes.seek(0)

        # --- Step 1: Parse the PDF into raw elements (from memory) ---
        self.pdf_file_bytes.seek(0)
        pdf_payload = self.pdf_file_bytes.read()
        self.pdf_file_bytes.seek(0)
        raw_docs = self._parse_pdf(pdf_payload)

        if not raw_docs:
            print("No documents were loaded successfully. Exiting.")
            return

        # Sort the documents by page number before processing
        raw_docs.sort(key=lambda x: x.metadata.get("page_number", 0))

        # --- Step 2: NEW LOGIC - Process elements sequentially ---
        final_chunks = []
        text_buffer = ""
        CHUNK_TOKEN_LIMIT = 600
        TEXT_LIKE_ELEMENTS = ["NarrativeText", "Title", "Header"]

        # NEW STATE VARIABLES for complex overlap rules
        current_page_number = raw_docs[0].metadata.get("page_number") if raw_docs else None
        last_metadata_for_page = None
        # This stores the overlap from the last TEXT chunk, used for INTER-PAGE overlap
        global_last_text_overlap = ""
        # This stores the overlap for the NEXT chunk on the CURRENT page, used for INTRA-PAGE overlap
        intra_page_overlap = ""

        creation_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def process_text_buffer(buffer, metadata, initial_overlap):
            """
            Chunks a given text buffer, managing intra-chunk overlap.
            It returns the new chunks and the overlap from the VERY LAST chunk it created.
            """
            nonlocal global_last_text_overlap

            new_chunks = []
            current_overlap = initial_overlap
            final_overlap_from_this_call = ""

            if not buffer or not metadata:
                return [], initial_overlap

            def stamp_chunk(content, chunk_metadata):
                enriched_metadata = chunk_metadata.copy()
                enriched_metadata["creation_date_time"] = creation_timestamp
                enriched_metadata["token_size"] = count_tokens(content)
                summary = enriched_metadata.get("embed_text")
                if isinstance(summary, str) and summary.strip():
                    enriched_metadata["embed_text"] = summary.strip() + "\n" + content[:400]
                new_chunks.append(Document(page_content=content, metadata=enriched_metadata))

            while count_tokens(buffer) > 0:
                final_chunk_content = current_overlap + buffer

                # Check if the combined content fits in one chunk
                if count_tokens(final_chunk_content) <= CHUNK_TOKEN_LIMIT:
                    stamp_chunk(final_chunk_content, metadata)

                    # This is a text chunk, so its overlap is eligible for all future overlaps
                    final_overlap_from_this_call = overlap_tail(buffer)
                    global_last_text_overlap = final_overlap_from_this_call
                    buffer = ""
                else:
                    # Buffer is too large, must split
                    estimated_split_char = (CHUNK_TOKEN_LIMIT - count_tokens(current_overlap)) * 4
                    split_index = -1
                    for delimiter in [".", "\n", " "]:
                        found_index = buffer.rfind(delimiter, 0, estimated_split_char)
                        if found_index != -1:
                            split_index = found_index
                            break

                    if split_index == -1:
                        split_index = estimated_split_char

                    text_for_this_chunk = buffer[:split_index + 1]
                    chunk_to_append_content = current_overlap + text_for_this_chunk
                    stamp_chunk(chunk_to_append_content, metadata)

                    # The overlap for the NEXT chunk is from the text we just added
                    current_overlap = overlap_tail(text_for_this_chunk)
                    final_overlap_from_this_call = current_overlap
                    global_last_text_overlap = final_overlap_from_this_call

                    buffer = buffer[split_index + 1:]

            return new_chunks, final_overlap_from_this_call

        current_section = "Front matter"
        chunk_bbox = {}

        def flush_text_buffer():
            nonlocal text_buffer, intra_page_overlap, chunk_bbox, last_metadata_for_page
            if text_buffer and last_metadata_for_page:
                metadata = dict(last_metadata_for_page)
                metadata.update(chunk_bbox)
                chunks, last_overlap = process_text_buffer(text_buffer, metadata, intra_page_overlap)
                final_chunks.extend(chunks)
                intra_page_overlap = last_overlap
            text_buffer = ""
            chunk_bbox = {}

        # --- Main Processing Loop ---
        for i, element in enumerate(raw_docs):
            element_page = element.metadata.get("page_number")
            element_type = element.metadata.get("element_type")
            section_changed = False

            # A Title that names a real section starts a new chunk. Running headers are Header, not Title.
            if element_type == "Title":
                new_section = classify_section(element.page_content)
                if new_section and new_section != current_section:
                    flush_text_buffer()
                    current_section = new_section
                    intra_page_overlap = ""
                    section_changed = True

            # === PAGE BREAK LOGIC ===
            if element_page != current_page_number:
                flush_text_buffer()
                current_page_number = element_page
                # Keep a short overlap across pages of the same section. A new section starts clean.
                if not section_changed:
                    intra_page_overlap = global_last_text_overlap

            element.metadata["section_category"] = current_section

            # A table is its own chunk. The vector is a two-sentence summary; the prompt receives the Markdown table.
            if element_type == "Table":
                flush_text_buffer()
                table_meta = dict(element.metadata)
                summarizer = getattr(self, "_summarize_table", None)
                summary = summarizer(element.page_content) if summarizer else ""
                if summary:
                    table_meta["embed_text"] = summary
                chunks, _ = process_text_buffer(element.page_content, table_meta, "")
                final_chunks.extend(chunks)
                intra_page_overlap = ""
                continue

            # A figure description is its own chunk. The vision call already happened at parse time.
            if element_type == "Figure":
                flush_text_buffer()
                chunks, _ = process_text_buffer(element.page_content, dict(element.metadata), "")
                final_chunks.extend(chunks)
                intra_page_overlap = ""
                continue

            chunk_bbox = union_bbox(chunk_bbox, element.metadata)

            # === ELEMENT HANDLING LOGIC (MODIFIED) ===
            text_buffer += "\n\n" + element.page_content
            if last_metadata_for_page is None or element_type in TEXT_LIKE_ELEMENTS:
                last_metadata_for_page = element.metadata

        # After the loop, process the very last buffer for the final page
        flush_text_buffer()

        study_id = (getattr(self, "study_id", "") or "").strip()
        raw_role = (getattr(self, "doc_role", "main") or "main").strip().lower()
        doc_role = raw_role if raw_role in {"main", "supplement", "protocol"} else "main"

        from biomedical_nern import stamp_biomedical_metadata

        for idx, doc in enumerate(final_chunks):
            doc.metadata["chunk_index"] = idx
            if isinstance(doc.metadata, dict):
                if study_id:
                    doc.metadata["study_id"] = study_id
                doc.metadata["doc_role"] = doc_role
                stamp_biomedical_metadata(doc)

        self.all_documents = final_chunks
        print(f"Chunking complete. Original elements: {len(raw_docs)}, Final intelligent chunks: {len(self.all_documents)}")

        self._index_chunks_in_weaviate()
        self.pdf_file_bytes = None

    def _parse_pdf(self, pdf_payload: bytes):
        """Docling first. Unstructured hi_res remains the fallback if Docling cannot read the file."""
        try:
            from io import BytesIO

            from docling.datamodel.base_models import DocumentStream

            print("Parsing PDF from memory with Docling...")
            stream = DocumentStream(name="paper.pdf", stream=BytesIO(pdf_payload))
            result = _get_docling_converter().convert(stream)
            raw_docs = documents_from_docling(result.document, self.normalized_pdf_url)
            print(f"Docling parsed {len(raw_docs)} elements from the PDF.")
            if raw_docs:
                return raw_docs
            print("Docling returned no text. Falling back to Unstructured hi_res.")
        except Exception as e:
            print(f"Docling parse failed ({e}). Falling back to Unstructured hi_res.")
        return self._parse_with_unstructured(pdf_payload)

    def _parse_with_unstructured(self, pdf_payload: bytes):
        from io import BytesIO

        raw_docs = []
        try:
            print("Parsing PDF from memory with unstructured.io (hi_res, tables preserved)...")
            elements = partition_pdf(
                file=BytesIO(pdf_payload),
                strategy="hi_res",
                infer_table_structure=True,
            )
            for el in elements:
                if not el.text or el.text.isspace():
                    continue
                element_type = type(el).__name__
                text = str(el.text)
                if element_type == "Table":
                    table_html = getattr(getattr(el, "metadata", None), "text_as_html", None)
                    markdown = html_table_to_markdown(table_html) if table_html else ""
                    if markdown:
                        text = f"{text}\n\n{markdown}"
                    elif table_html:
                        text = f"{text}\n\n{table_html}"
                metadata = {
                    "source": self.normalized_pdf_url,
                    "element_type": element_type,
                    "page_number": el.metadata.page_number,
                }
                metadata.update(element_bbox(el))
                raw_docs.append(Document(page_content=text, metadata=metadata))
            print(f"Unstructured parsed {len(elements)} elements from the PDF.")
        except Exception as e:
            print(f"Error parsing PDF from memory with unstructured: {e}.")
        return raw_docs

    def get_all_documents(self):
        return self.all_documents