"""Look up retraction, trial registration, and newer citing papers for one article.

Each call has a short timeout. A failed service is recorded and the rest of the card still returns.
Newer citing papers are listed. They are not described as contradictions until those papers are read.
"""
import os
import re

import requests

_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.I)
_NCT_RE = re.compile(r"\bNCT\d{8}\b", re.I)
_TIMEOUT = 8
_HEADERS = {
    "User-Agent": os.getenv("EVIDENCE_HTTP_USER_AGENT", "chatwithpdf/1.0 (evidence-check)"),
    "Accept": "application/json",
}


def find_identifiers(documents, trial_id: str = ""):
    dois = []
    ncts = []
    blobs = [trial_id or ""]
    for doc in documents or []:
        blobs.append(getattr(doc, "page_content", "") or "")
    for blob in blobs:
        for match in _DOI_RE.findall(blob or ""):
            doi = match.rstrip(").,;")
            if doi not in dois:
                dois.append(doi)
        for match in _NCT_RE.findall(blob or ""):
            nct = match.upper()
            if nct not in ncts:
                ncts.append(nct)
    return {"dois": dois[:6], "ncts": ncts[:3]}


def _get_json(url: str, params=None):
    response = requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
    response.raise_for_status()
    return response.json()


def crossref_record(doi: str) -> dict:
    payload = _get_json(f"https://api.crossref.org/works/{requests.utils.quote(doi)}")
    message = payload.get("message") or {}
    retractions = []
    corrections = []
    for update in message.get("update-to") or []:
        kind = (update.get("type") or "").lower()
        item = {"type": kind, "doi": update.get("DOI") or "", "label": update.get("label") or ""}
        if "retract" in kind:
            retractions.append(item)
        elif kind:
            corrections.append(item)
    return {
        "doi": doi,
        "title": " ".join(message.get("title") or []),
        "retracted": bool(retractions),
        "retractions": retractions,
        "corrections": corrections,
        "referenced_by_count": message.get("is-referenced-by-count"),
    }


def pubmed_retraction(doi: str) -> dict:
    payload = _get_json(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": f"{doi} AND Retracted Publication[pt]",
            "retmode": "json",
        },
    )
    count = int(((payload.get("esearchresult") or {}).get("count")) or 0)
    return {"doi": doi, "retracted_publication_count": count, "retracted": count > 0}

def extract_protocol_updates(protocol: dict) -> dict:
    """Extract last update date and any protocol/amendment notes from protocolSection."""
    if not isinstance(protocol, dict):
        return {"last_update_date": "", "amendment_note": ""}
    status_mod = protocol.get("statusModule") or {}
    update_struct = status_mod.get("lastUpdatePostDateStruct") or {}
    last_update = (update_struct.get("date") or "").strip()

    # ClinicalTrials.gov v2 may expose design/protocol or amendment-like strings depending on record.
    design_mod = protocol.get("designModule") or {}
    notes = []
    if status_mod.get("expandedAccessInfo") and isinstance(status_mod["expandedAccessInfo"], dict):
        avail = status_mod["expandedAccessInfo"].get("hasExpandedAccess")
        if avail is not None:
            notes.append(f"Expanded access: {avail}")
    if design_mod.get("bioSpec") and isinstance(design_mod["bioSpec"], dict):
        desc = (design_mod["bioSpec"].get("description") or "").strip()
        if desc:
            notes.append(desc[:160])

    return {
        "last_update_date": last_update,
        "amendment_note": " | ".join(notes),
    }


def extract_adverse_events(results_section: dict) -> dict:
    """Safely extract posted serious and other adverse events from ClinicalTrials.gov v2 resultsSection."""
    if not isinstance(results_section, dict):
        return {"posted": False, "serious_summary": "", "top_events": []}
    ae_mod = results_section.get("adverseEventsModule")
    if not isinstance(ae_mod, dict) or not ae_mod:
        return {"posted": False, "serious_summary": "", "top_events": []}

    serious_summary = ""
    serious_groups = ae_mod.get("seriousEvents") or []
    other_groups = ae_mod.get("otherEvents") or []
    timeframe = (ae_mod.get("timeFrame") or "").strip()

    parts = []
    if timeframe:
        parts.append(f"Timeframe: {timeframe}")
    if serious_groups:
        parts.append(f"Serious AE groups listed: {len(serious_groups)}")
    if other_groups:
        parts.append(f"Other AE groups listed: {len(other_groups)}")
    serious_summary = "; ".join(parts)

    top_events = []
    for group in (serious_groups + other_groups)[:5]:
        if not isinstance(group, dict):
            continue
        term = (group.get("term") or group.get("organSystem") or "").strip()
        if not term:
            continue
        stats_list = group.get("stats") or []
        affected_total = 0
        for st_item in stats_list:
            if isinstance(st_item, dict) and isinstance(st_item.get("numAffected"), int):
                affected_total += st_item["numAffected"]
        top_events.append({
            "term": term,
            "num_affected_total": affected_total,
        })

    return {
        "posted": True,
        "serious_summary": serious_summary,
        "top_events": top_events,
    }


def clinical_trial(nct_id: str) -> dict:
    payload = _get_json(f"https://clinicaltrials.gov/api/v2/studies/{nct_id}")
    protocol = payload.get("protocolSection") or {}
    status = ((protocol.get("statusModule") or {}).get("overallStatus")) or ""
    outcomes = ((protocol.get("outcomesModule") or {}).get("primaryOutcomes")) or []
    measures = [item.get("measure") or "" for item in outcomes if item.get("measure")]
    updates = extract_protocol_updates(protocol)
    adverse_events = extract_adverse_events(payload.get("resultsSection") or {})
    return {"nct_id": nct_id, "status": status, "registered_primary_outcomes": measures}


def _tokens(text: str):
    stop = {"with", "from", "that", "this", "primary", "endpoint", "patients", "after", "before", "versus"}
    return {token for token in re.findall(r"[a-z]{5,}", (text or "").lower()) if token not in stop}


def outcome_switch(paper_endpoints, registered_measures) -> dict:
    """Flag a primary endpoint whose content words are not all in the registry entry."""
    paper = [item for item in paper_endpoints if item]
    registered = [item for item in registered_measures if item]
    if not paper or not registered:
        return {"flag": False, "reason": ""}
    registered_tokens = set()
    for measure in registered:
        registered_tokens.update(_tokens(measure))
    missed = []
    for endpoint in paper:
        tokens = _tokens(endpoint)
        if tokens and not tokens.issubset(registered_tokens):
            missed.append(endpoint)
    if not missed:
        return {"flag": False, "reason": ""}
    return {
        "flag": True,
        "reason": "A printed primary endpoint shares no content word with the registered primary outcomes.",
        "paper_endpoints": missed,
        "registered_primary_outcomes": registered,
    }


def openalex_forward(doi: str) -> dict:
    work = _get_json(f"https://api.openalex.org/works/https://doi.org/{doi}")
    work_id = work.get("id") or ""
    short_id = work_id.rsplit("/", 1)[-1]
    paper_year = work.get("publication_year")
    newer = []
    if short_id:
        citing = _get_json(
            "https://api.openalex.org/works",
            params={"filter": f"cites:{short_id}", "per-page": 5, "sort": "publication_year:desc"},
        )
        for item in citing.get("results") or []:
            year = item.get("publication_year")
            if paper_year and year and year < paper_year:
                continue
            newer.append({
                "title": item.get("display_name") or "",
                "year": year,
                "doi": (item.get("doi") or "").replace("https://doi.org/", ""),
            })
    return {
        "cited_by_count": work.get("cited_by_count"),
        "paper_year": paper_year,
        "newer_citing_papers": newer,
        "contradiction_judged": False,
    }


def _safe(label: str, func, errors: list):
    try:
        return func()
    except Exception as exc:
        errors.append(f"{label}: {exc}")
        return None


def live_evidence_for_paper(documents, trial_id: str = "", primary_endpoints=None) -> dict:
    found = find_identifiers(documents, trial_id)
    errors = []
    paper_doi = found["dois"][0] if found["dois"] else ""
    crossref = _safe("crossref", lambda: crossref_record(paper_doi), errors) if paper_doi else None
    pubmed = _safe("pubmed", lambda: pubmed_retraction(paper_doi), errors) if paper_doi else None
    citations = _safe("openalex", lambda: openalex_forward(paper_doi), errors) if paper_doi else None
    trials = []
    for nct in found["ncts"]:
        trial = _safe(nct, lambda nct_id=nct: clinical_trial(nct_id), errors)
        if not trial:
            continue
        trial["outcome_switch"] = outcome_switch(primary_endpoints or [], trial.get("registered_primary_outcomes") or [])
        trials.append(trial)
    cited_alerts = []
    for doi in found["dois"][1:4]:
        record = _safe(f"cited {doi}", lambda cited=doi: crossref_record(cited), errors)
        if record and (record["retracted"] or record["corrections"]):
            cited_alerts.append(record)
    retracted = bool(crossref and crossref["retracted"]) or bool(pubmed and pubmed["retracted"])
    return {
        "dois": found["dois"],
        "ncts": found["ncts"],
        "retracted": retracted,
        "crossref": crossref,
        "pubmed": pubmed,
        "trials": trials,
        "citations": citations,
        "cited_paper_alerts": cited_alerts,
        "errors": errors,
    }
