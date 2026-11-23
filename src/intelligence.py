"""Attach interval checks, checklist notes, and live lookups to one scorecard."""
import os

from appraisal import build_appraisal
from live_evidence import live_evidence_for_paper
from stats_audit import audit_estimate


def _live_enabled() -> bool:
    return os.getenv("LIVE_EVIDENCE", "true").strip().lower() not in {"0", "false", "no"}


def enrich_card(card: dict, documents, llm=None, appraisal_prompt: str = "", force: bool = False) -> bool:
    """Fill missing extras. Returns True when the card changed and should be saved."""
    changed = False
    for outcome in card.get("outcomes") or []:
        if force or "statistics" not in outcome:
            outcome["statistics"] = audit_estimate(
                outcome.get("effect") or "",
                outcome.get("ci") or "",
                outcome.get("p_value") or "",
                outcome.get("verbatim_quote") or "",
            )
            changed = True
    if force or "appraisal" not in card:
        card["appraisal"] = build_appraisal(card, documents, llm=llm, prompt_template=appraisal_prompt)
        changed = True
    if not _live_enabled():
        if "live_evidence" not in card:
            card["live_evidence"] = {"skipped": True, "errors": [], "retracted": False, "trials": [], "citations": None}
            changed = True
    elif force or "live_evidence" not in card:
        primary = [
            outcome.get("endpoint") or ""
            for outcome in (card.get("outcomes") or [])
            if (outcome.get("role") or "").lower() == "primary"
        ]
        trial_id = ((card.get("trial_id") or {}).get("value")) or ""
        card["live_evidence"] = live_evidence_for_paper(documents, trial_id, primary)
        changed = True
    return changed
