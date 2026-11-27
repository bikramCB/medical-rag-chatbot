"""Lightweight Biomedical Named Entity Recognition & Normalization (NERN)."""
import re
from typing import Dict, List, Set

# Canonical biomedical concept registry:
# concept_id -> {"type": ..., "canonical": ..., "aliases": [...]}
_CONCEPT_REGISTRY: Dict[str, dict] = {
    "RXNORM:OSIMERTINIB": {
        "type": "drug",
        "canonical": "osimertinib",
        "aliases": ["osimertinib", "azd9291", "azd-9291", "tagrisso", "mereletinib"],
    },
    "RXNORM:PEMBROLIZUMAB": {
        "type": "drug",
        "canonical": "pembrolizumab",
        "aliases": ["pembrolizumab", "mk-3475", "mk3475", "keytruda", "lambrolizumab"],
    },
    "RXNORM:NIVOLUMAB": {
        "type": "drug",
        "canonical": "nivolumab",
        "aliases": ["nivolumab", "bms-936558", "bms936558", "ono-4538", "opdivo"],
    },
    "RXNORM:TRASTUZUMAB": {
        "type": "drug",
        "canonical": "trastuzumab",
        "aliases": ["trastuzumab", "herceptin", "t-dm1", "ado-trastuzumab emtansine", "kadcyla", "t-dxd", "trastuzumab deruxtecan", "enhertu"],
    },
    "RXNORM:BEVACIZUMAB": {
        "type": "drug",
        "canonical": "bevacizumab",
        "aliases": ["bevacizumab", "avastin", "mvasi", "zyrabez"],
    },
    "RXNORM:SOTORASIB": {
        "type": "drug",
        "canonical": "sotorasib",
        "aliases": ["sotorasib", "amg 510", "amg-510", "amg510", "lumakras"],
    },
    "RXNORM:ADAGRASIB": {
        "type": "drug",
        "canonical": "adagrasib",
        "aliases": ["adagrasib", "mrtx849", "mrtx-849", "krazati"],
    },
    "HGNC:ERBB2": {
        "type": "gene",
        "canonical": "ERBB2",
        "aliases": ["erbb2", "erbb-2", "her2", "her2/neu", "her-2", "neu", "cd340"],
    },
    "HGNC:EGFR": {
        "type": "gene",
        "canonical": "EGFR",
        "aliases": ["egfr", "erbb1", "her1", "t790m", "l858r", "exon 19del", "exon 19 deletion"],
    },
    "HGNC:KRAS": {
        "type": "gene",
        "canonical": "KRAS",
        "aliases": ["kras", "k-ras", "kras g12c", "g12c"],
    },
    "HGNC:ALK": {
        "type": "gene",
        "canonical": "ALK",
        "aliases": ["alk", "anaplastic lymphoma kinase", "alk fusion", "eml4-alk"],
    },
    "HGNC:CD274": {
        "type": "gene",
        "canonical": "CD274",
        "aliases": ["cd274", "pd-l1", "pdl1", "programmed death-ligand 1", "b7-h1"],
    },
    "MESH:NSCLC": {
        "type": "disease",
        "canonical": "non-small cell lung cancer",
        "aliases": ["nsclc", "non-small cell lung cancer", "non-small-cell lung cancer", "non small cell lung carcinoma"],
    },
    "MESH:BREAST_CANCER": {
        "type": "disease",
        "canonical": "breast cancer",
        "aliases": ["breast cancer", "breast carcinoma", "tnbc", "triple-negative breast cancer", "hr+/her2-", "her2+ breast cancer"],
    },
}

# Build fast lookup table: normalized_alias -> concept_id
_ALIAS_TO_CONCEPT: Dict[str, str] = {}
_ALIAS_PATTERNS: List[tuple[str, re.Pattern]] = []

for _cid, _info in _CONCEPT_REGISTRY.items():
    for _alias in _info["aliases"]:
        _norm = re.sub(r"\s+", " ", _alias.strip().lower())
        if not _norm:
            continue
        _ALIAS_TO_CONCEPT[_norm] = _cid
        _escaped = re.escape(_norm).replace(r"\ ", r"[\s\-/]+")
        _pattern = re.compile(rf"(?<![a-z0-9]){_escaped}(?![a-z0-9])", re.IGNORECASE)
        _ALIAS_PATTERNS.append((_cid, _pattern))


def normalize_biomedical_text(text: str, max_extra_terms: int = 8) -> dict:
    """Extract matched biomedical concepts and return canonical IDs plus expansion aliases."""
    if not text or not text.strip():
        return {"concept_ids": [], "entities": [], "expansion_terms": []}

    matched_ids: List[str] = []
    seen_ids: Set[str] = set()
    matched_mentions: Set[str] = set()

    for cid, pattern in _ALIAS_PATTERNS:
        for match in pattern.finditer(text):
            matched_mentions.add(match.group(0).strip())
            if cid not in seen_ids:
                seen_ids.add(cid)
                matched_ids.append(cid)

    entities: List[dict] = []
    expansion_terms: List[str] = []
    seen_terms: Set[str] = {t.strip().lower() for t in re.findall(r"[a-z0-9\-+/]{2,}", text.lower())}

    for cid in matched_ids:
        entry = _CONCEPT_REGISTRY[cid]
        entities.append({
            "concept_id": cid,
            "type": entry["type"],
            "canonical": entry["canonical"],
        })
        for candidate in [entry["canonical"], *entry["aliases"]]:
            low = candidate.strip().lower()
            if low and low not in seen_terms:
                seen_terms.add(low)
                expansion_terms.append(candidate)
                if len(expansion_terms) >= max_extra_terms:
                    break
        if len(expansion_terms) >= max_extra_terms:
            break

    return {
        "concept_ids": matched_ids,
        "entities": entities,
        "matched_mentions": sorted(matched_mentions),
        "expansion_terms": expansion_terms[:max_extra_terms],
    }


def expand_query_with_nern(query: str, max_extra_terms: int = 6) -> str:
    """Append canonical names and key aliases for matched biomedical entities in a query."""
    cleaned = re.sub(r"\s+", " ", (query or "").strip())
    if not cleaned:
        return cleaned
    info = normalize_biomedical_text(cleaned, max_extra_terms=max_extra_terms)
    extras = info.get("expansion_terms") or []
    if not extras:
        return cleaned
    return f"{cleaned} {' '.join(extras)}"


def stamp_biomedical_metadata(doc, max_aliases: int = 8):
    """Write concept_ids and expansion aliases onto chunk metadata. Page text stays unchanged."""
    meta = getattr(doc, "metadata", None)
    if not isinstance(meta, dict):
        return doc
    info = normalize_biomedical_text(getattr(doc, "page_content", "") or "", max_extra_terms=max_aliases)
    meta["concept_ids"] = info.get("concept_ids") or []
    meta["entity_aliases"] = info.get("expansion_terms") or []
    return doc