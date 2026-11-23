"""Build a comparison table from scorecards that were already extracted.

A contradiction is recorded only when two papers report related endpoints and their
intervals disagree. Papers that were not loaded are not called contradictory.
"""
import re

_STOP = {"with", "from", "that", "this", "primary", "endpoint", "patients", "group", "versus", "after", "before", "among"}


def _value(card: dict, name: str) -> str:
    field = card.get(name) or {}
    if isinstance(field, dict):
        return field.get("value") or ""
    return ""


def _tokens(text: str):
    return {token for token in re.findall(r"[a-z]{4,}", (text or "").lower()) if token not in _STOP}


def _primary_outcomes(card: dict):
    outcomes = card.get("outcomes") or []
    primary = [item for item in outcomes if (item.get("role") or "").lower() == "primary"]
    return primary or outcomes[:1]


def _direction(stats: dict) -> str:
    if not stats:
        return ""
    if stats.get("crosses_null"):
        return "includes_null"
    if stats.get("significant_by_ci") and stats.get("direction") == "below_null":
        return "below_null"
    if stats.get("significant_by_ci") and stats.get("direction") == "above_null":
        return "above_null"
    return ""


def matrix_rows(cards) -> list:
    rows = []
    for card in cards or []:
        outcomes = []
        for outcome in _primary_outcomes(card):
            outcomes.append({
                "endpoint": outcome.get("endpoint") or "",
                "effect": outcome.get("effect") or "",
                "ci": outcome.get("ci") or "",
                "p_value": outcome.get("p_value") or "",
                "page": outcome.get("page"),
                "chunk_index": outcome.get("chunk_index"),
                "bbox": outcome.get("bbox"),
                "statistics": outcome.get("statistics") or {},
            })
        rows.append({
            "source_pdf": card.get("source_pdf") or "",
            "content_hash": card.get("content_hash") or "",
            "study_design": _value(card, "study_design"),
            "sample_size": _value(card, "sample_size"),
            "population": _value(card, "population"),
            "intervention": _value(card, "intervention"),
            "comparator": _value(card, "comparator"),
            "trial_id": _value(card, "trial_id"),
            "funding": _value(card, "funding"),
            "conflicts_of_interest": _value(card, "conflicts_of_interest"),
            "primary_outcomes": outcomes,
        })
    return rows


def compare_outcomes(rows) -> dict:
    paired = []
    for row in rows:
        for outcome in row.get("primary_outcomes") or []:
            direction = _direction(outcome.get("statistics") or {})
            if not direction:
                continue
            paired.append({
                "source_pdf": row.get("source_pdf"),
                "study_design": row.get("study_design"),
                "endpoint": outcome.get("endpoint") or "",
                "direction": direction,
                "tokens": _tokens(outcome.get("endpoint") or ""),
                "page": outcome.get("page"),
                "chunk_index": outcome.get("chunk_index"),
            })
    contradictions = []
    seen = set()
    for index, left in enumerate(paired):
        for right in paired[index + 1:]:
            if left["source_pdf"] == right["source_pdf"]:
                continue
            if not left["tokens"] or left["tokens"].isdisjoint(right["tokens"]):
                continue
            if left["direction"] == right["direction"]:
                continue
            key = tuple(sorted([left["source_pdf"], right["source_pdf"], left["endpoint"], right["endpoint"]]))
            if key in seen:
                continue
            seen.add(key)
            contradictions.append({
                "endpoints": [left["endpoint"], right["endpoint"]],
                "papers": [
                    {"source_pdf": left["source_pdf"], "study_design": left["study_design"], "direction": left["direction"], "page": left["page"], "chunk_index": left["chunk_index"]},
                    {"source_pdf": right["source_pdf"], "study_design": right["study_design"], "direction": right["direction"], "page": right["page"], "chunk_index": right["chunk_index"]},
                ],
            })
    groups = {}
    for item in paired:
        for token in item["tokens"]:
            groups.setdefault(token, []).append(item)
    consensus = []
    for token, items in groups.items():
        papers = {item["source_pdf"] for item in items}
        directions = {item["direction"] for item in items}
        if len(papers) < 2 or len(directions) != 1:
            continue
        consensus.append({
            "shared_term": token,
            "direction": items[0]["direction"],
            "papers": sorted(papers),
            "count": len(papers),
        })
    return {"contradictions": contradictions, "consensus": consensus}


def build_matrix(cards) -> dict:
    rows = matrix_rows(cards)
    compared = compare_outcomes(rows)
    return {
        "paper_count": len(rows),
        "rows": rows,
        "contradictions": compared["contradictions"],
        "consensus": compared["consensus"],
    }
