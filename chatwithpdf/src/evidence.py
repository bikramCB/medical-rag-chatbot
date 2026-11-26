"""Clinical scorecard for one indexed paper.

The card is extracted once from labeled chunks and cached by the PDF content hash.
A value is kept only when its quote appears in the chunk it cites.
"""
import json
import re
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from stats_audit import audit_estimate

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?(?:\s*%)?")
_PREFERRED_SECTIONS = {"Abstract", "Methods", "Results", "Funding/COI"}
_SCORECARD_DIR = Path(__file__).resolve().parent.parent / "scorecards"
_FIELD_NAMES = (
    "population",
    "sample_size",
    "inclusion",
    "exclusion",
    "demographics",
    "intervention",
    "dosage",
    "route",
    "schedule",
    "comparator",
    "study_design",
    "trial_id",
    "funding",
    "conflicts_of_interest",
    "median_os_pfs",
    "grade_3_plus_ae",
    "key_limitations",
)


class SourcedValue(BaseModel):
    value: str = ""
    verbatim_quote: str = ""
    chunk_id: int = -1


class OutcomeValue(BaseModel):
    endpoint: str = ""
    role: str = ""
    effect: str = ""
    ci: str = ""
    p_value: str = ""
    verbatim_quote: str = ""
    chunk_id: int = -1


class EvidenceCardModel(BaseModel):
    population: SourcedValue = Field(default_factory=SourcedValue)
    sample_size: SourcedValue = Field(default_factory=SourcedValue)
    inclusion: SourcedValue = Field(default_factory=SourcedValue)
    exclusion: SourcedValue = Field(default_factory=SourcedValue)
    demographics: SourcedValue = Field(default_factory=SourcedValue)
    intervention: SourcedValue = Field(default_factory=SourcedValue)
    dosage: SourcedValue = Field(default_factory=SourcedValue)
    route: SourcedValue = Field(default_factory=SourcedValue)
    schedule: SourcedValue = Field(default_factory=SourcedValue)
    comparator: SourcedValue = Field(default_factory=SourcedValue)
    study_design: SourcedValue = Field(default_factory=SourcedValue)
    trial_id: SourcedValue = Field(default_factory=SourcedValue)
    funding: SourcedValue = Field(default_factory=SourcedValue)
    conflicts_of_interest: SourcedValue = Field(default_factory=SourcedValue)
    outcomes: List[OutcomeValue] = Field(default_factory=list)
    median_os_pfs: SourcedValue = Field(default_factory=SourcedValue)
    grade_3_plus_ae: SourcedValue = Field(default_factory=SourcedValue)
    key_limitations: SourcedValue = Field(default_factory=SourcedValue)

def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _numbers_in_text(value: str, source: str) -> bool:
    compact_source = re.sub(r"\s+", "", source or "")
    for number in _NUMBER_RE.findall(value or ""):
        compact = re.sub(r"\s+", "", number)
        if compact and compact not in compact_source:
            return False
    return True


def empty_source():
    return {
        "value": "",
        "verbatim_quote": "",
        "quote_found": False,
        "page": None,
        "chunk_index": None,
        "section_category": "",
        "bbox": None,
    }


def bbox_from_metadata(metadata: dict):
    box = {}
    for key in ("x0", "y0", "x1", "y1"):
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            box[key] = value
    if not box:
        return None
    return {"page": metadata.get("page_number"), **box}


def _document_order(documents):
    return sorted(documents, key=lambda doc: (
        doc.metadata.get("chunk_index") is None,
        doc.metadata.get("chunk_index") if doc.metadata.get("chunk_index") is not None else 0,
        doc.metadata.get("page_number") or 0,
    ))


def select_evidence_chunks(documents, limit: int = 24):
    """Keep tables even when the Methods section is long, then fill with preferred sections."""
    tables = [doc for doc in documents if (doc.metadata.get("element_type") or "") == "Table"]
    preferred = [
        doc for doc in documents
        if doc not in tables and (doc.metadata.get("section_category") or "") in _PREFERRED_SECTIONS
    ]
    chosen = _document_order(tables)[:8]
    room = max(0, limit - len(chosen))
    chosen.extend(_document_order(preferred)[:room])
    if not chosen:
        chosen = list(documents)
    return _document_order(chosen)[:limit]


def render_evidence_context(documents, chars_per_chunk: int = 1200) -> str:
    parts = []
    for index, doc in enumerate(documents):
        section = doc.metadata.get("section_category", "Front matter")
        page = doc.metadata.get("page_number", "N/A")
        limit = 4000 if (doc.metadata.get("element_type") or "") == "Table" else chars_per_chunk
        text = (doc.page_content or "")[:limit]
        parts.append(f"### CHUNK_{index}\nSection: {section}\nPage: {page}\n{text}")
    return "\n\n".join(parts)


def locate_quote(quote: str, documents):
    norm_quote = _normalize_ws(quote)
    if not norm_quote:
        return None
    for index, doc in enumerate(documents):
        if norm_quote in _normalize_ws(doc.page_content):
            return index
    return None


def ground_source(raw, documents):
    """Keep a field only when the quote is in a chunk and every number in the value is too."""
    if isinstance(raw, dict):
        value = (raw.get("value") or "").strip()
        quote = (raw.get("verbatim_quote") or "").strip()
        chunk_id = raw.get("chunk_id", -1)
    else:
        value = (getattr(raw, "value", "") or "").strip()
        quote = (getattr(raw, "verbatim_quote", "") or "").strip()
        chunk_id = getattr(raw, "chunk_id", -1)
    if not value or not quote:
        return empty_source()
    matched = None
    if isinstance(chunk_id, int) and 0 <= chunk_id < len(documents):
        if _normalize_ws(quote) in _normalize_ws(documents[chunk_id].page_content):
            matched = chunk_id
    if matched is None:
        matched = locate_quote(quote, documents)
    if matched is None:
        return empty_source()
    doc = documents[matched]
    if not _numbers_in_text(value, doc.page_content):
        return empty_source()
    grounded = empty_source()
    grounded.update({
        "value": value,
        "verbatim_quote": quote,
        "quote_found": True,
        "page": doc.metadata.get("page_number"),
        "chunk_index": doc.metadata.get("chunk_index"),
        "section_category": doc.metadata.get("section_category", "Front matter"),
        "bbox": bbox_from_metadata(doc.metadata),
    })
    return grounded


def ground_outcome(raw, documents):
    if isinstance(raw, dict):
        endpoint = (raw.get("endpoint") or "").strip()
        role = (raw.get("role") or "").strip()
        effect = (raw.get("effect") or "").strip()
        ci = (raw.get("ci") or "").strip()
        p_value = (raw.get("p_value") or "").strip()
        quote = (raw.get("verbatim_quote") or "").strip()
        chunk_id = raw.get("chunk_id", -1)
    else:
        endpoint = (getattr(raw, "endpoint", "") or "").strip()
        role = (getattr(raw, "role", "") or "").strip()
        effect = (getattr(raw, "effect", "") or "").strip()
        ci = (getattr(raw, "ci", "") or "").strip()
        p_value = (getattr(raw, "p_value", "") or "").strip()
        quote = (getattr(raw, "verbatim_quote", "") or "").strip()
        chunk_id = getattr(raw, "chunk_id", -1)
    if not endpoint or not quote:
        return None
    matched = None
    if isinstance(chunk_id, int) and 0 <= chunk_id < len(documents):
        if _normalize_ws(quote) in _normalize_ws(documents[chunk_id].page_content):
            matched = chunk_id
    if matched is None:
        matched = locate_quote(quote, documents)
    if matched is None:
        return None
    doc = documents[matched]
    combined = " ".join([endpoint, effect, ci, p_value])
    if not _numbers_in_text(combined, doc.page_content):
        return None
    return {
        "endpoint": endpoint,
        "role": role,
        "effect": effect,
        "ci": ci,
        "p_value": p_value,
        "verbatim_quote": quote,
        "quote_found": True,
        "page": doc.metadata.get("page_number"),
        "chunk_index": doc.metadata.get("chunk_index"),
        "section_category": doc.metadata.get("section_category", "Front matter"),
        "bbox": bbox_from_metadata(doc.metadata),
        "statistics": audit_estimate(effect, ci, p_value, quote),
    }


def ground_evidence_card(raw_card, documents, source_pdf: str, content_hash: str) -> dict:
    if isinstance(raw_card, EvidenceCardModel):
        payload = raw_card.model_dump()
    elif isinstance(raw_card, dict):
        payload = raw_card
    else:
        payload = {}
    card = {
        "source_pdf": source_pdf,
        "content_hash": content_hash,
        "outcomes": [],
        "error": "",
    }
    for name in _FIELD_NAMES:
        card[name] = ground_source(payload.get(name) or {}, documents)
    for outcome in payload.get("outcomes") or []:
        grounded = ground_outcome(outcome, documents)
        if grounded:
            card["outcomes"].append(grounded)
    return card


def scorecard_path(content_hash: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9]", "", content_hash or "")
    return _SCORECARD_DIR / f"{safe}.json"


def load_cached_card(content_hash: str):
    path = scorecard_path(content_hash)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_card(content_hash: str, card: dict):
    if not content_hash or content_hash == "default_index":
        return
    _SCORECARD_DIR.mkdir(parents=True, exist_ok=True)
    scorecard_path(content_hash).write_text(json.dumps(card, indent=2), encoding="utf-8")


def extract_evidence_card(llm, documents, prompt_template: str, source_pdf: str, content_hash: str) -> dict:
    """One structured model call, then drop any field whose quote is not in a chunk."""
    selected = select_evidence_chunks(documents)
    context = render_evidence_context(selected)
    if not context.strip():
        card = ground_evidence_card({}, [], source_pdf, content_hash)
        card["error"] = "No chunks were available for the scorecard."
        return card
    try:
        from langchain.schema import HumanMessage, SystemMessage

        extractor = llm.with_structured_output(EvidenceCardModel)
        result = extractor.invoke([
            SystemMessage(content=prompt_template.format(context=context)),
            HumanMessage(content="Extract the clinical scorecard."),
        ])
        if isinstance(result, dict):
            result = EvidenceCardModel.model_validate(result)
        return ground_evidence_card(result, selected, source_pdf, content_hash)
    except Exception as exc:
        card = ground_evidence_card({}, [], source_pdf, content_hash)
        card["error"] = f"Scorecard extraction failed: {exc}"
        return card
