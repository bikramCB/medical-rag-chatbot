"""Automated checklist notes for one paper.

Reported concerns are kept only when the quote is in a chunk.
Red flags that come from the interval audit or from a direct text scan do not use the model.
This is not an official GRADE or Cochrane rating.
"""
import re

from evidence import locate_quote, render_evidence_context, select_evidence_chunks
from stats_audit import claims_significance

_CHECKLISTS = {
    "rct": (
        "RoB 2 domains",
        [
            "random sequence and allocation concealment",
            "blinding of participants and investigators",
            "missing outcome data and attrition",
            "outcome measurement",
            "selective reporting or a protocol change",
        ],
    ),
    "observational": (
        "STROBE reporting items",
        [
            "selection and eligibility",
            "control of confounding",
            "missing data",
            "outcome definition",
        ],
    ),
    "systematic_review": (
        "PRISMA reporting items",
        [
            "protocol or registration",
            "search strategy",
            "study selection",
            "risk of bias of the included studies",
        ],
    ),
    "other": (
        "limited reporting check",
        [
            "what the paper says it studied",
            "what the paper says it did not establish",
        ],
    ),
}
_SURROGATE = re.compile(
    r"\b(progression[- ]free|disease[- ]free|response rate|objective response|biomarker|tumor response|tumour response|ldl|hba1c)\b",
    re.I,
)
_CLINICAL = re.compile(r"\b(overall survival|mortality|death|hospitali[sz]ation|quality of life)\b", re.I)
_ATTRITION = re.compile(
    r"(?:dropout|dropped out|withdrew|withdrawn|lost to follow[- ]up|discontinued)[^\n\.]{0,80}?(\d+(?:\.\d+)?)\s*%|"
    r"(\d+(?:\.\d+)?)\s*%[^\n\.]{0,80}?(?:dropout|dropped out|withdrew|withdrawn|lost to follow[- ]up|discontinued)",
    re.I,
)
_OPEN_LABEL = re.compile(r"\bopen[- ]label\b", re.I)
_BLIND = re.compile(r"\b(blind|placebo)\b", re.I)


def study_type_from_design(design: str) -> str:
    text = (design or "").lower()
    if any(token in text for token in ("meta-analysis", "systematic review", "prisma")):
        return "systematic_review"
    if any(token in text for token in ("cohort", "case-control", "case control", "observational", "cross-sectional", "registry")):
        return "observational"
    if any(token in text for token in ("random", "rct", "phase", "trial")):
        return "rct"
    return "other"


def deterministic_red_flags(card: dict, documents) -> list:
    flags = []
    design = ((card.get("study_design") or {}).get("value")) or ""
    study_type = study_type_from_design(design)
    abstract_bits = []
    body_bits = []
    for doc in documents or []:
        text = getattr(doc, "page_content", "") or ""
        section = (doc.metadata.get("section_category") or "")
        body_bits.append(text)
        if section == "Abstract" or (doc.metadata.get("page_number") or 99) <= 2:
            abstract_bits.append(text)
    abstract = "\n".join(abstract_bits)
    body = "\n".join(body_bits)

    for outcome in card.get("outcomes") or []:
        stats = outcome.get("statistics") or {}
        label = outcome.get("endpoint") or "an endpoint"
        if stats.get("spin") or (stats.get("crosses_null") and claims_significance(abstract) and (outcome.get("role") or "").lower() == "primary"):
            flags.append({
                "flag": "interval_crosses_null",
                "detail": f"{label} has an interval that crosses the null while the wording treats it as significant.",
                "endpoint": label,
            })
        elif stats.get("crosses_null") and (outcome.get("role") or "").lower() == "primary":
            flags.append({
                "flag": "primary_interval_crosses_null",
                "detail": f"The primary interval for {label} includes the null value.",
                "endpoint": label,
            })
        if stats.get("ci_p_disagree"):
            flags.append({
                "flag": "ci_p_disagree",
                "detail": f"The interval and the p-value for {label} do not agree on significance at alpha 0.05.",
                "endpoint": label,
            })
        if stats.get("population") == "per-protocol" and (outcome.get("role") or "").lower() == "primary":
            flags.append({
                "flag": "per_protocol_primary",
                "detail": f"The primary result for {label} is described as per-protocol rather than intention-to-treat.",
                "endpoint": label,
            })
        endpoint = outcome.get("endpoint") or ""
        if (outcome.get("role") or "").lower() == "primary" and _SURROGATE.search(endpoint) and not _CLINICAL.search(endpoint):
            flags.append({
                "flag": "surrogate_primary_endpoint",
                "detail": f"The primary endpoint ({endpoint}) is a surrogate rather than a clinical outcome such as survival or hospitalization.",
                "endpoint": endpoint,
            })

    for match in _ATTRITION.finditer(body):
        raw = match.group(1) or match.group(2)
        try:
            rate = float(raw)
        except (TypeError, ValueError):
            continue
        if rate >= 20:
            flags.append({
                "flag": "high_attrition",
                "detail": f"The paper reports about {rate:g}% dropout, withdrawal, or loss to follow-up.",
                "endpoint": "",
            })
            break

    if study_type == "rct":
        if _OPEN_LABEL.search(body):
            flags.append({
                "flag": "open_label",
                "detail": "The trial is described as open-label, so participants and investigators were not blinded.",
                "endpoint": "",
            })
        elif not _BLIND.search(body):
            flags.append({
                "flag": "blinding_not_described",
                "detail": "The indexed text does not describe blinding or a placebo.",
                "endpoint": "",
            })
    return flags


def _ground_item(raw, documents):
    if isinstance(raw, dict):
        item = (raw.get("item") or "").strip()
        status = (raw.get("status") or "unclear").strip().lower()
        note = (raw.get("note") or "").strip()
        quote = (raw.get("verbatim_quote") or "").strip()
        chunk_id = raw.get("chunk_id", -1)
    else:
        item = (getattr(raw, "item", "") or "").strip()
        status = (getattr(raw, "status", "") or "unclear").strip().lower()
        note = (getattr(raw, "note", "") or "").strip()
        quote = (getattr(raw, "verbatim_quote", "") or "").strip()
        chunk_id = getattr(raw, "chunk_id", -1)
    if status not in {"reported", "concern", "not_reported", "unclear"}:
        status = "unclear"
    if not item:
        return None
    if status in {"reported", "concern"}:
        matched = None
        if isinstance(chunk_id, int) and 0 <= chunk_id < len(documents):
            if quote and re.sub(r"\s+", " ", quote).strip().lower() in re.sub(r"\s+", " ", documents[chunk_id].page_content or "").lower():
                matched = chunk_id
        if matched is None:
            matched = locate_quote(quote, documents)
        if matched is None:
            return None
        doc = documents[matched]
        return {
            "item": item,
            "status": status,
            "note": note,
            "verbatim_quote": quote,
            "quote_found": True,
            "page": doc.metadata.get("page_number"),
            "chunk_index": doc.metadata.get("chunk_index"),
        }
    return {"item": item, "status": status, "note": note, "verbatim_quote": "", "quote_found": False, "page": None, "chunk_index": None}


def build_appraisal(card: dict, documents, llm=None, prompt_template: str = "") -> dict:
    design = ((card.get("study_design") or {}).get("value")) or ""
    study_type = study_type_from_design(design)
    checklist, items = _CHECKLISTS[study_type]
    appraisal = {
        "study_type": study_type,
        "checklist": checklist,
        "official_grade": False,
        "items": [],
        "red_flags": deterministic_red_flags(card, documents),
        "error": "",
    }
    if llm is None or not prompt_template or not documents:
        return appraisal
    selected = select_evidence_chunks(documents)
    context = render_evidence_context(selected, chars_per_chunk=1200)
    if not context.strip():
        return appraisal
    try:
        from pydantic import BaseModel, Field
        from typing import List, Literal
        from langchain.schema import HumanMessage, SystemMessage

        class AppraisalItem(BaseModel):
            item: str = ""
            status: Literal["reported", "concern", "not_reported", "unclear"] = "unclear"
            note: str = ""
            verbatim_quote: str = ""
            chunk_id: int = -1

        class AppraisalModel(BaseModel):
            items: List[AppraisalItem] = Field(default_factory=list)

        prompt = prompt_template.format(
            checklist=checklist,
            study_design=design or "not stated",
            items="\n".join(f"- {item}" for item in items),
            context=context,
        )
        checker = llm.with_structured_output(AppraisalModel)
        result = checker.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Fill the checklist from the chunks."),
        ])
        raw_items = result.get("items") if isinstance(result, dict) else result.items
        grounded = []
        for raw in raw_items or []:
            kept = _ground_item(raw, selected)
            if kept:
                grounded.append(kept)
        appraisal["items"] = grounded
    except Exception as exc:
        appraisal["error"] = f"Checklist extraction failed: {exc}"
    return appraisal
