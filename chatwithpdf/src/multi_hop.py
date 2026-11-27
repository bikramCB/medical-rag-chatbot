"""Bounded multi-hop retrieval planner for cross-referenced clinical/medical chunks."""
import re
from typing import List, Sequence

from langchain.schema import Document

_TABLE_REF_RE = re.compile(r"\b(?:table|tbl\.?|fig(?:ure)?)\s*(s?\d+[a-z]?)\b", re.IGNORECASE)
_SECTION_REF_RE = re.compile(
    r"\b(?:section|sec\.?|methods?|methodology|statistical analysis|sap|appendix|supplementary)\b",
    re.IGNORECASE,
)
_SUBGROUP_REF_RE = re.compile(
    r"\b(?:subgroup|sub-group|stratified|cohort|mutant|wild-type|adverse event|grade\s*3|neutropenia|os|pfs)\b",
    re.IGNORECASE,
)


def _doc_key(doc: Document) -> tuple:
    meta = getattr(doc, "metadata", None) or {}
    return (
        meta.get("chunk_index"),
        meta.get("page_number"),
        (getattr(doc, "page_content", "") or "")[:120],
    )


def merge_unique_documents(existing: Sequence[Document], incoming: Sequence[Document], max_total: int = 16) -> List[Document]:
    """Append newly retrieved chunks while preserving order and deduplicating."""
    merged: List[Document] = []
    seen = set()
    for doc in list(existing or []) + list(incoming or []):
        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
        if len(merged) >= max_total:
            break
    return merged


def _table_label_covered(docs: Sequence[Document], label: str) -> bool:
    """A citation in prose is not coverage. The table or figure chunk itself has to be present."""
    needle = re.sub(r"\s+", " ", label).strip().lower()
    compact = re.sub(r"[\s.]", "", needle)
    for doc in docs or []:
        element = ((getattr(doc, "metadata", None) or {}).get("element_type") or "").lower()
        if element not in {"table", "figure"}:
            continue
        text = (getattr(doc, "page_content", "") or "").lower()
        if needle in text or compact in re.sub(r"[\s.]", "", text[:500]):
            return True
    return False


def _methods_already_retrieved(docs: Sequence[Document]) -> bool:
    for doc in docs or []:
        category = ((getattr(doc, "metadata", None) or {}).get("section_category") or "").lower()
        if any(token in category for token in ("method", "statistic", "appendix", "supplement")):
            return True
    return False


def plan_followup_queries(query: str, docs: Sequence[Document], max_queries: int = 2) -> List[str]:
    """Inspect retrieved chunks for cross-references not already covered and generate follow-up queries."""
    combined_text = "\n".join(getattr(d, "page_content", "") or "" for d in (docs or []))
    if not combined_text.strip():
        return []

    followups: List[str] = []
    seen_q = {(query or "").strip().lower()}

    def _add_q(q: str):
        cleaned = re.sub(r"\s+", " ", (q or "").strip())
        low = cleaned.lower()
        if cleaned and low not in seen_q and len(followups) < max_queries:
            seen_q.add(low)
            followups.append(cleaned)

    # 1. Table / figure labels cited in the retrieved text but not yet retrieved as a table or figure.
    for ref_match in _TABLE_REF_RE.finditer(combined_text):
        label = re.sub(r"\s+", " ", ref_match.group(0)).strip()
        if _table_label_covered(docs, label):
            continue
        _add_q(f"{label} {query}")

    # 2. Subgroup / safety questions need the table plus the definition, not only a results sentence.
    # Match the user question. Matching every retrieved chunk would hop on ordinary OS/PFS papers.
    if _SUBGROUP_REF_RE.search(query or ""):
        _add_q(f"{query} subgroup table stratification definition")

    # 3. Methods / appendix cross-references when that section is not already in the candidate set.
    if _SECTION_REF_RE.search(combined_text) and not _methods_already_retrieved(docs):
        _add_q(f"{query} methods subgroup eligibility statistical analysis")

    return followups